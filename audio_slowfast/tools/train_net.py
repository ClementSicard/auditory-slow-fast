#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train an audio classification model."""
import logging as lg
import pprint
from typing import Tuple

import numpy as np
import torch
import wandb
from fvcore.nn.precise_bn import update_bn_stats
from fvcore.common.config import CfgNode
from loguru import logger
from tqdm import tqdm
from audio_slowfast.models.build import build_model
from audio_slowfast.tools.eval_net import eval_epoch, eval_epoch_with_state
import audio_slowfast.models.losses as losses
import audio_slowfast.models.optimizer as optim
import audio_slowfast.utils.checkpoint as cu
import audio_slowfast.utils.distributed as du
import audio_slowfast.utils.logging as logging
import audio_slowfast.utils.metrics as metrics
import audio_slowfast.utils.misc as misc
import audio_slowfast.tools.train_utils as train_utils
import audio_slowfast.visualization.tensorboard_vis as tb
from audio_slowfast.datasets import loader
from audio_slowfast.utils.meters import (
    EPICTrainMeter,
    EPICTrainMeterWithState,
    EPICValMeter,
    EPICValMeterWithState,
    TrainMeter,
    ValMeter,
)
from src.utils import display_gpu_info
from src.dataset import load_nouns, load_all_verbs

# Silence minor loggers
numba_logger = lg.getLogger("numba")
numba_logger.setLevel(lg.ERROR)

jit_logger = lg.getLogger("fvcore.nn.jit_analysis")
jit_logger.setLevel(lg.ERROR)

bn_logger = lg.getLogger("fvcore.nn.precise_bn")
bn_logger.setLevel(lg.ERROR)

wandb_logger = lg.getLogger("wandb")
wandb_logger.setLevel(lg.ERROR)


def train_epoch_state(
    train_loader,
    model,
    optimizer,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
    wandb_log=False,
):
    """
    Perform the audio training for one epoch.
    Args:
        train_loader (loader): audio training loader.
        model (model): the audio model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    if cfg.BN.FREEZE:
        model.module.freeze_fn("bn_statistics") if cfg.NUM_GPUS > 1 else model.freeze_fn("bn_statistics")

    train_meter.iter_tic()
    data_size = len(train_loader)

    verb_names = load_all_verbs(cfg.EPICKITCHENS.VERBS_FILE)
    noun_names = load_nouns(cfg.EPICKITCHENS.NOUNS_FILE)

    for cur_iter, batch in enumerate(
        # Write to stderr
        tqdm(
            train_loader,
            desc="Epoch: {}/{}".format(
                cur_epoch + 1,
                cfg.SOLVER.MAX_EPOCH,
            ),
            unit="batch",
        ),
    ):
        if not "GRU" in cfg.TRAIN.DATASET:
            inputs, labels, _, _ = batch

            # Transfer the data to the current GPU device.
            if cfg.NUM_GPUS:
                # Transferthe data to the current GPU device.
                if isinstance(inputs, list):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                if isinstance(labels, dict):
                    labels = {k: v.cuda() for k, v in labels.items()}
                else:
                    labels = labels.cuda()

                if cur_iter % cfg.LOG_PERIOD == 0:
                    display_gpu_info()

            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)
            train_meter.data_toc()

            preds = model(x=inputs)
        else:
            inputs, lengths, labels, _, noun_embeddings, _ = batch
            # Transfer the data to the current GPU device.
            if cfg.NUM_GPUS:
                # Transferthe data to the current GPU device.
                if isinstance(inputs, list):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                if isinstance(labels, dict):
                    labels = {k: v.cuda() for k, v in labels.items()}
                else:
                    labels = labels.cuda()

                noun_embeddings = noun_embeddings.cuda()

                if cur_iter % cfg.LOG_PERIOD == 0:
                    display_gpu_info()

            # Update the learning rate.
            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)

            train_meter.data_toc()
            preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = model(
                x=inputs,
                lengths=lengths,
                noun_embeddings=noun_embeddings if not cfg.MODEL.ONLY_ACTION_RECOGNITION else None,
            )

        verb_preds, noun_preds, state_preds = preds

        labels["state"] = train_utils.prepare_state_labels(preds, labels, lengths)

        # # Check if the predictions look good or are weird. In this case, send an alert.
        # train_utils.check_predictions(preds=preds, labels=labels, threshold=0.1)

        if isinstance(labels, dict):
            loss, loss_verb, loss_noun, loss_state = train_utils.compute_loss_with_state(
                verb_preds=verb_preds,
                noun_preds=noun_preds,
                state_preds=state_preds,
                labels=labels,
                cfg=cfg,
            )

        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)

            # check Nan Loss.
            misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if isinstance(labels, (dict,)):
            """
            VERB METRICS/LOSS
            """
            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(verb_preds, labels["verb"], (1, 5))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce([loss_verb, verb_top1_acc, verb_top5_acc])

            # Copy the stats from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (
                loss_verb.item(),
                verb_top1_acc.item(),
                verb_top5_acc.item(),
            )

            """
            NOUN METRICS/LOSS
            """
            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(noun_preds, labels["noun"], (1, 5))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce([loss_noun, noun_top1_acc, noun_top5_acc])

            # Copy the stats from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = (
                loss_noun.item(),
                noun_top1_acc.item(),
                noun_top5_acc.item(),
            )

            """
            ACTION METRICS/LOSS
            """
            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                (verb_preds, noun_preds),
                (labels["verb"], labels["noun"]),
                (1, 5),
            )
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce([loss, action_top1_acc, action_top5_acc])

            # Copy the stats from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (
                loss.item(),
                action_top1_acc.item(),
                action_top5_acc.item(),
            )

            """
            UPDATE STATS
            """
            # Update and log stats.
            train_meter.update_stats(
                top1_acc=(
                    verb_top1_acc,
                    noun_top1_acc,
                    action_top1_acc,
                ),
                top5_acc=(
                    verb_top5_acc,
                    noun_top5_acc,
                    action_top5_acc,
                ),
                loss=(
                    loss_verb,
                    loss_noun,
                    loss_state.item(),
                    loss,
                ),
                lr=lr,
                mb_size=inputs[0].size(0)
                * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_acc": action_top1_acc,
                        "Train/Top5_acc": action_top5_acc,
                        "Train/verb/loss": loss_verb,
                        "Train/noun/loss": loss_noun,
                        "Train/state/loss": loss_state,
                        "Train/verb/Top1_acc": verb_top1_acc,
                        "Train/verb/Top5_acc": verb_top5_acc,
                        "Train/noun/Top1_acc": noun_top1_acc,
                        "Train/noun/Top5_acc": noun_top5_acc,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

            if wandb_log:
                wandb.log(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_acc": action_top1_acc,
                        "Train/Top5_acc": action_top5_acc,
                        "Train/verb/loss": loss_verb,
                        "Train/noun/loss": loss_noun,
                        "Train/state/loss": loss_state,
                        "Train/verb/Top1_acc": verb_top1_acc,
                        "Train/verb/Top5_acc": verb_top5_acc,
                        "Train/noun/Top1_acc": noun_top1_acc,
                        "Train/noun/Top5_acc": noun_top5_acc,
                        "train_step": data_size * cur_epoch + cur_iter,
                    },
                )
        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

            if wandb_log:
                wandb.log(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                        "train_step": data_size * cur_epoch + cur_iter,
                    },
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def train_epoch(
    train_loader,
    model,
    optimizer,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
    wandb_log=False,
):
    """
    Perform the audio training for one epoch.
    Args:
        train_loader (loader): audio training loader.
        model (model): the audio model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    if cfg.BN.FREEZE:
        model.module.freeze_fn("bn_statistics") if cfg.NUM_GPUS > 1 else model.freeze_fn("bn_statistics")

    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, batch in enumerate(
        # Write to stderr
        tqdm(
            train_loader,
            desc="Epoch: {}/{}".format(
                cur_epoch + 1,
                cfg.SOLVER.MAX_EPOCH,
            ),
            unit="batch",
        ),
    ):
        if not "GRU" in cfg.TRAIN.DATASET:
            inputs, labels, _, _ = batch

            # Transfer the data to the current GPU device.
            if cfg.NUM_GPUS:
                # Transferthe data to the current GPU device.
                if isinstance(inputs, list):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                if isinstance(labels, dict):
                    labels = {k: v.cuda() for k, v in labels.items()}
                else:
                    labels = labels.cuda()

                if cur_iter % cfg.LOG_PERIOD == 0:
                    display_gpu_info()

            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)
            train_meter.data_toc()

            preds = model(x=inputs)
        else:
            inputs, lengths, labels, _, noun_embeddings, _ = batch

            # Transfer the data to the current GPU device.
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list)):
                    for i in range(len(inputs)):
                        try:
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                        except Exception as e:
                            logger.exception(f"Error with inputs[{i}]: {inputs[i]}")
                            raise e
                else:
                    inputs = inputs.cuda(non_blocking=True)
                if isinstance(labels, dict):
                    labels = {k: v.cuda() for k, v in labels.items()}
                else:
                    labels = labels.cuda()

                if cur_iter % cfg.LOG_PERIOD == 0:
                    display_gpu_info()

                noun_embeddings = noun_embeddings.cuda()

            # Update the learning rate.
            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)

            train_meter.data_toc()

            preds = model(
                x=inputs,
                lengths=lengths,
                noun_embeddings=noun_embeddings,
            )

        verb_preds, noun_preds = preds

        if isinstance(labels, dict):
            loss, loss_verb, loss_noun = train_utils.compute_loss(
                verb_preds=verb_preds,
                noun_preds=noun_preds,
                labels=labels,
                cfg=cfg,
            )

        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)

            # check Nan Loss.
            misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if isinstance(labels, (dict,)):
            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(verb_preds, labels["verb"], (1, 5))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce([loss_verb, verb_top1_acc, verb_top5_acc])

            # Copy the stats from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (
                loss_verb.item(),
                verb_top1_acc.item(),
                verb_top5_acc.item(),
            )

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(noun_preds, labels["noun"], (1, 5))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce([loss_noun, noun_top1_acc, noun_top5_acc])

            # Copy the stats from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = (
                loss_noun.item(),
                noun_top1_acc.item(),
                noun_top5_acc.item(),
            )

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                (verb_preds, noun_preds),
                (labels["verb"], labels["noun"]),
                (1, 5),
            )
            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce([loss, action_top1_acc, action_top5_acc])

            # Copy the stats from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (
                loss.item(),
                action_top1_acc.item(),
                action_top5_acc.item(),
            )

            # Update and log stats.
            train_meter.update_stats(
                top1_acc=(
                    verb_top1_acc,
                    noun_top1_acc,
                    action_top1_acc,
                ),
                top5_acc=(
                    verb_top5_acc,
                    noun_top5_acc,
                    action_top5_acc,
                ),
                loss=(
                    loss_verb,
                    loss_noun,
                    loss,
                ),
                lr=lr,
                mb_size=inputs[0].size(0)
                * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_acc": action_top1_acc,
                        "Train/Top5_acc": action_top5_acc,
                        "Train/verb/loss": loss_verb,
                        "Train/noun/loss": loss_noun,
                        "Train/verb/Top1_acc": verb_top1_acc,
                        "Train/verb/Top5_acc": verb_top5_acc,
                        "Train/noun/Top1_acc": noun_top1_acc,
                        "Train/noun/Top5_acc": noun_top5_acc,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

            if wandb_log:
                wandb.log(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_acc": action_top1_acc,
                        "Train/Top5_acc": action_top5_acc,
                        "Train/verb/loss": loss_verb,
                        "Train/noun/loss": loss_noun,
                        "Train/verb/Top1_acc": verb_top1_acc,
                        "Train/verb/Top5_acc": verb_top5_acc,
                        "Train/noun/Top1_acc": noun_top1_acc,
                        "Train/noun/Top5_acc": noun_top5_acc,
                        "train_step": data_size * cur_epoch + cur_iter,
                    },
                )
        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

            if wandb_log:
                wandb.log(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                        "train_step": data_size * cur_epoch + cur_iter,
                    },
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg: CfgNode):
    """
    Train an audio model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(cfg)

    # Build the audio model and print model statistics.
    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg)

    if cfg.BN.FREEZE:
        model.module.freeze_fn("bn_parameters") if cfg.NUM_GPUS > 1 else model.freeze_fn("bn_parameters")

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the audio train and val loaders.
    if not cfg.TRAIN.DATASET.startswith("EpicKitchens") or not cfg.EPICKITCHENS.TRAIN_PLUS_VAL:
        train_loader = loader.construct_loader(cfg, "train")
        val_loader = loader.construct_loader(cfg, "val")
    else:
        train_loader = loader.construct_loader(cfg, "train+val")
        val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    if cfg.TRAIN.DATASET.startswith("EpicKitchens"):
        train_meter = (
            EPICTrainMeter(epoch_iters=len(train_loader), cfg=cfg)
            if cfg.MODEL.ONLY_ACTION_RECOGNITION
            else EPICTrainMeterWithState(epoch_iters=len(train_loader), cfg=cfg)
        )
        val_meter = (
            EPICValMeter(max_iter=len(val_loader), cfg=cfg)
            if cfg.MODEL.ONLY_ACTION_RECOGNITION
            else EPICValMeterWithState(max_iter=len(train_loader), cfg=cfg)
        )
    else:
        train_meter = TrainMeter(epoch_iters=len(train_loader), cfg=cfg)
        val_meter = ValMeter(max_iter=len(val_loader), cfg=cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    if cfg.WANDB.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        wandb_log = True
        project_name = cfg.MODEL.MODEL_NAME
        project_name += " + Augment" if cfg.EPICKITCHENS.AUGMENT.ENABLE else ""
        project_name += " + State" if not cfg.MODEL.ONLY_ACTION_RECOGNITION else ""
        project_name += " (from VGG-SOUND)" if "VGG" in cfg.TRAIN.CHECKPOINT_FILE_PATH else ""

        if cfg.TRAIN.AUTO_RESUME and cfg.WANDB.RUN_ID != "":
            wandb.init(
                project=project_name,
                config=cfg,
                sync_tensorboard=True,
                resume=cfg.WANDB.RUN_ID,
            )
        else:
            wandb.init(
                project=project_name,
                config=cfg,
                sync_tensorboard=True,
            )
        wandb.watch(model)

    else:
        wandb_log = False

    # Perform the training loop.
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        logger.info(f"Epoch {cur_epoch + 1} started. Shuffling dataset.")
        loader.shuffle_dataset(train_loader, cur_epoch)

        logger.info(f"Training for epoch {cur_epoch + 1}.")
        # Train for one epoch.
        if cfg.MODEL.ONLY_ACTION_RECOGNITION:
            train_epoch(
                train_loader,
                model,
                optimizer,
                train_meter,
                cur_epoch,
                cfg,
                writer,
                wandb_log,
            )
        else:
            train_epoch_state(
                train_loader,
                model,
                optimizer,
                train_meter,
                cur_epoch,
                cfg,
                writer,
                wandb_log,
            )
        logger.success(f"Done training for epoch {cur_epoch + 1}!")

        # Save
        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg,
            cur_epoch,
        )

        # Save a checkpoint.
        if is_checkp_epoch:
            logger.info(f"Saving a checkpoint for epoch {cur_epoch}.")
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
            logger.success(f"Done saving a checkpoint for epoch {cur_epoch}.")

        # Evaluate the model on validation set.
        if is_eval_epoch:
            logger.info(f"Evaluating the model for epoch {cur_epoch}.")
            is_best_epoch, _ = (
                eval_epoch(
                    val_loader=val_loader,
                    model=model,
                    val_meter=val_meter,
                    cur_epoch=cur_epoch,
                    cfg=cfg,
                    writer=writer,
                    wandb_log=wandb_log,
                )
                if cfg.MODEL.ONLY_ACTION_RECOGNITION
                else eval_epoch_with_state(
                    val_loader=val_loader,
                    model=model,
                    val_meter=val_meter,
                    cur_epoch=cur_epoch,
                    cfg=cfg,
                    writer=writer,
                    wandb_log=wandb_log,
                )
            )
            logger.success(f"Done evaluating the model for epoch {cur_epoch}!")
            if is_best_epoch:
                logger.success(f"Saving a best checkpoint for epoch {cur_epoch}.")
                cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    cur_epoch,
                    cfg,
                    is_best_epoch=is_best_epoch,
                )

    if writer is not None:
        writer.close()

    logger.success("Training complete! 🎉")
