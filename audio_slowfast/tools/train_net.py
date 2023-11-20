#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train an audio classification model."""

from typing import Dict, Tuple
import logging as lg
import pprint
import sys

import numpy as np
import torch
import wandb
from fvcore.common.config import CfgNode
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from loguru import logger
from tqdm import tqdm
from wandb import AlertLevel
from audio_slowfast.models.build import build_model

import audio_slowfast.models.losses as losses
import audio_slowfast.models.optimizer as optim
import audio_slowfast.utils.checkpoint as cu
import audio_slowfast.utils.distributed as du
import audio_slowfast.utils.logging as logging
import audio_slowfast.utils.metrics as metrics
import audio_slowfast.utils.misc as misc
import audio_slowfast.visualization.tensorboard_vis as tb
from audio_slowfast.datasets import loader
from audio_slowfast.utils.meters import (
    EPICTrainMeter,
    EPICValMeter,
    TrainMeter,
    ValMeter,
)
from src.utils import display_gpu_info

numba_logger = lg.getLogger("numba")
numba_logger.setLevel(lg.WARNING)

jit_logger = lg.getLogger("fvcore.nn.jit_analysis")
jit_logger.setLevel(lg.ERROR)

bn_logger = lg.getLogger("fvcore.nn.precise_bn")
bn_logger.setLevel(lg.ERROR)


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

    for cur_iter, (inputs, lengths, labels, _, noun_embeddings, _) in enumerate(
        # Write to stderr
        tqdm(
            train_loader,
            desc="Epoch: {}/{}".format(cur_epoch + 1, cfg.SOLVER.MAX_EPOCH),
            unit="batch",
            file=sys.stderr,
        ),
    ):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
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

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        preds = model(
            x=inputs,
            lengths=lengths,
            noun_embeddings=noun_embeddings,
        )
        preds = [pred.squeeze(1) for pred in preds]

        # Check if the predictions look good or are weird. In this case, send an alert.
        check_predictions(preds=preds, labels=labels, threshold=0.1)

        if isinstance(labels, dict):
            loss, loss_verb, loss_noun, loss_state = compute_loss(
                preds=preds,
                labels=labels,
                lengths=lengths,
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
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels["verb"], (1, 5))

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
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels["noun"], (1, 5))

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
                (preds[0], preds[1]), (labels["verb"], labels["noun"]), (1, 5)
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


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, wandb_log=False):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, lengths, labels, _, noun_embeddings, _) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

        val_meter.data_toc()

        preds = model(
            x=inputs,
            lengths=lengths,
            noun_embeddings=noun_embeddings,
        )
        preds = [pred.squeeze(1) for pred in preds]
        check_predictions(preds=preds, labels=labels, threshold=0.1)

        if isinstance(labels, dict):
            loss, loss_verb, loss_noun, loss_state = compute_loss(
                preds,
                labels,
                cfg,
            )

            # Compute the verb accuracies.
            verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels["verb"], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce([loss_verb, verb_top1_acc, verb_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss_verb, verb_top1_acc, verb_top5_acc = (
                loss_verb.item(),
                verb_top1_acc.item(),
                verb_top5_acc.item(),
            )

            # Compute the noun accuracies.
            noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels["noun"], (1, 5))

            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce([loss_noun, noun_top1_acc, noun_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss_noun, noun_top1_acc, noun_top5_acc = loss_noun.item(), noun_top1_acc.item(), noun_top5_acc.item()

            # Compute the action accuracies.
            action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                (preds[0], preds[1]), (labels["verb"], labels["noun"]), (1, 5)
            )
            # Combine the errors across the GPUs.
            if cfg.NUM_GPUS > 1:
                loss, action_top1_acc, action_top5_acc = du.all_reduce([loss, action_top1_acc, action_top5_acc])

            # Copy the errors from GPU to CPU (sync point).
            loss, action_top1_acc, action_top5_acc = (
                loss.item(),
                action_top1_acc.item(),
                action_top5_acc.item(),
            )

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
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
                mb_size=inputs[0].size(0)
                * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None and not wandb_log:
                writer.add_scalars(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                        "Val/state/loss": loss_state,
                    },
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            if wandb_log:
                wandb.log(
                    {
                        "Val/loss": loss,
                        "Val/Top1_acc": action_top1_acc,
                        "Val/Top5_acc": action_top5_acc,
                        "Val/verb/loss": loss_verb,
                        "Val/verb/Top1_acc": verb_top1_acc,
                        "Val/verb/Top5_acc": verb_top5_acc,
                        "Val/noun/loss": loss_noun,
                        "Val/noun/Top1_acc": noun_top1_acc,
                        "Val/noun/Top5_acc": noun_top5_acc,
                        "Val/state/loss": loss_state,
                        "val_step": len(val_loader) * cur_epoch + cur_iter,
                    },
                )

            val_meter.update_predictions(
                preds=(
                    preds[0],
                    preds[1],
                    preds[2],
                ),
                labels=(
                    labels["verb"],
                    labels["noun"],
                    labels["state"],
                ),
            )

        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            loss = loss_fun(preds, labels)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])

            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(cfg.NUM_GPUS, 1),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None and not wandb_log:
                    writer.add_scalars(
                        {
                            "Val/loss": loss,
                            "Val/Top1_err": top1_err,
                            "Val/Top5_err": top5_err,
                        },
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

                if wandb_log:
                    wandb.log(
                        {
                            "Val/loss": loss,
                            "Val/Top1_err": top1_err,
                            "Val/Top5_err": top5_err,
                            "val_step": len(val_loader) * cur_epoch + cur_iter,
                        },
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch, top1_dict = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        all_verb_preds = [pred.clone().detach() for pred in val_meter.all_verb_preds]
        all_verb_labels = [label.clone().detach() for label in val_meter.all_verb_labels]
        all_noun_preds = [pred.clone().detach() for pred in val_meter.all_noun_preds]
        all_noun_labels = [label.clone().detach() for label in val_meter.all_noun_labels]
        all_state_preds = [pred.clone().detach() for pred in val_meter.all_state_preds]
        all_state_labels = [label.clone().detach() for label in val_meter.all_state_labels]

        if cfg.NUM_GPUS:
            all_verb_preds = [pred.cpu() for pred in all_verb_preds]
            all_verb_labels = [label.cpu() for label in all_verb_labels]
            all_noun_preds = [pred.cpu() for pred in all_noun_preds]
            all_noun_labels = [label.cpu() for label in all_noun_labels]
            all_state_preds = [pred.cpu() for pred in all_state_preds]
            all_state_labels = [label.cpu() for label in all_state_labels]

        writer.plot_eval(preds=all_verb_preds, labels=all_verb_labels, global_step=cur_epoch)
        writer.plot_eval(preds=all_noun_preds, labels=all_noun_labels, global_step=cur_epoch)
        writer.plot_eval(preds=all_state_preds, labels=all_state_labels, global_step=cur_epoch)

    if writer is not None and not wandb_log:
        if "top1_acc" in top1_dict.keys():
            writer.add_scalars(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                },
                global_step=cur_epoch,
            )

        else:
            writer.add_scalars(
                {"Val/epoch/Top1_err": top1_dict["top1_err"]},
                global_step=cur_epoch,
            )

    if wandb_log:
        if "top1_acc" in top1_dict.keys():
            wandb.log(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                    "epoch": cur_epoch,
                },
            )

        else:
            wandb.log({"Val/epoch/Top1_err": top1_dict["top1_err"], "epoch": cur_epoch})

    top1 = top1_dict["top1_acc"] if "top1_acc" in top1_dict.keys() else top1_dict["top1_err"]
    val_meter.reset()
    return is_best_epoch, top1


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


def train(cfg):
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
    logger.info(pprint.pformat(cfg))

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
    if cfg.TRAIN.DATASET != "epickitchens" or not cfg.EPICKITCHENS.TRAIN_PLUS_VAL:
        train_loader = loader.construct_loader(cfg, "train")
        val_loader = loader.construct_loader(cfg, "val")
        precise_bn_loader = loader.construct_loader(cfg, "train") if cfg.BN.USE_PRECISE_STATS else None
    else:
        train_loader = loader.construct_loader(cfg, "train+val")
        val_loader = loader.construct_loader(cfg, "val")
        precise_bn_loader = loader.construct_loader(cfg, "train+val") if cfg.BN.USE_PRECISE_STATS else None

    # Create meters.
    if cfg.TRAIN.DATASET == "epickitchens":
        train_meter = EPICTrainMeter(epoch_iters=len(train_loader), cfg=cfg)
        val_meter = EPICValMeter(max_iter=len(val_loader), cfg=cfg)
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
        if cfg.TRAIN.AUTO_RESUME and cfg.WANDB.RUN_ID != "":
            wandb.init(
                project="slowfast",
                config=cfg,
                sync_tensorboard=True,
                resume=cfg.WANDB.RUN_ID,
            )
        else:
            wandb.init(project="slowfast", config=cfg, sync_tensorboard=True)
        wandb.watch(model)

    else:
        wandb_log = False

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        logger.info(f"Epoch {cur_epoch} started. Shuffling dataset.")
        loader.shuffle_dataset(train_loader, cur_epoch)
        logger.success("Done!")

        logger.info(f"Training for epoch {cur_epoch}.")
        # Train for one epoch.
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
        logger.success(f"Done training for epoch {cur_epoch}!")

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg,
            cur_epoch,
        )

        # Compute precise BN stats.
        if (is_checkp_epoch or is_eval_epoch) and cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            logger.info(f"Computing precise BN stats for epoch {cur_epoch}.")
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
            logger.info(f"Computing precise BN stats for epoch {cur_epoch}.")
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            logger.info(f"Saving a checkpoint for epoch {cur_epoch}.")
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
            logger.success(f"Done saving a checkpoint for epoch {cur_epoch}.")
        # Evaluate the model on validation set.
        if is_eval_epoch:
            logger.info(f"Evaluating the model for epoch {cur_epoch}.")
            is_best_epoch, _ = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer, wandb_log)
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


"""
Custom code for the new loss involving state.
"""


def compute_masked_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for the given predictions and labels.
    Args:
        preds (tensor): the logits from the model output.
        labels (tensor): the labels.
    Returns:
        loss (tensor): the loss.
    """
    assert preds.shape == labels.shape, f"Shapes must match: {preds.shape=} {labels.shape=}"

    bce = losses.get_loss_func("bce")(reduction="mean")
    mse = losses.get_loss_func("mse")(reduction="mean")

    abs_preds = torch.abs(preds)
    abs_labels = torch.abs(labels)

    # Get the indices where the abs_labels are 1
    pos_mask_indices = abs_labels.nonzero(as_tuple=True)

    bce_term = bce(abs_preds, abs_labels)
    mse_term = mse(preds[pos_mask_indices], labels[pos_mask_indices])

    return bce_term + mse_term


def compute_loss(preds: torch.Tensor, labels: Dict[str, torch.Tensor], cfg: CfgNode) -> Tuple[torch.Tensor, ...]:
    logger.warning(f"Preds shapes: {[pred.shape for pred in preds]}")
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    masked_loss_fun = losses.get_loss_func(cfg.MODEL.STATE_LOSS_FUNC)(reduction="mean")

    loss_verb = loss_fun(preds[0], labels["verb"])
    loss_noun = loss_fun(preds[1], labels["noun"])

    #
    loss_state = masked_loss_fun(preds[2], labels["state"])

    # Use torch.mean to average the losses over all GPUs for logging purposes.
    loss_vec = torch.stack(
        [
            loss_verb,
            loss_noun,
            loss_state,
        ]
    )

    loss = torch.mean(loss_vec)

    # check Nan Loss.
    misc.check_nan_losses(loss)

    return loss, loss_verb, loss_noun, loss_state


def _check_prediction(pred: torch.Tensor, threshold: float = 0.1) -> bool:
    return torch.all(torch.abs(pred) <= threshold)


def check_predictions(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.1):
    """
    Check if the state predictions are within the threshold and actually look like something.

    Parameters
    ----------
    `preds`: `torch.Tensor`
        The predictions from the model.

    `labels`: `torch.Tensor`
        The labels for the predictions.

    `threshold`: `float`
        The threshold to check the predictions against.
    """
    if _check_prediction(pred=preds[2], threshold=threshold):
        text = f"State < 0.1\n\nPreds:{preds[2]}\nLabels:{labels[2]}"
        logger.warning(text)
        wandb.alert(
            title="State looking strange",
            text=text,
            level=AlertLevel.WARN,
        )
