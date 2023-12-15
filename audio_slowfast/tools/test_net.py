#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test an audio classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager

import audio_slowfast.utils.checkpoint as cu
import audio_slowfast.utils.distributed as du
import audio_slowfast.utils.logging as logging
import audio_slowfast.utils.misc as misc
import audio_slowfast.visualization.tensorboard_vis as tb
from audio_slowfast.datasets import loader
from audio_slowfast.models import build_model
from audio_slowfast.utils.meters import EPICTestMeterWithState, TestMeter, EPICTestMeter
from audio_slowfast.utils.vggsound_metrics import get_stats

from loguru import logger
from tqdm import tqdm


@torch.no_grad()
def perform_test_with_state(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from an audio along
    its temporal axis. Softmax scores are averaged across all N views to
    form an audio-level prediction. All audio predictions are compared to
    ground-truth labels and the final testing performance is logged.
    Args:
        test_loader (loader): audio testing loader.
        model (model): the pretrained audio model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    raise NotImplementedError("Need to update this code to work with the new test meter")

    for cur_iter, (inputs, labels, audio_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()
            audio_idx = audio_idx.cuda()
        test_meter.data_toc()

        # Perform the forward pass.
        preds = model(inputs)

        if isinstance(labels, (dict,)):
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                verb_preds, verb_labels, audio_idx = du.all_gather([preds[0], labels["verb"], audio_idx])

                noun_preds, noun_labels, audio_idx = du.all_gather([preds[1], labels["noun"], audio_idx])
                meta = du.all_gather_unaligned(meta)
                metadata = {"narration_id": []}
                for i in range(len(meta)):
                    metadata["narration_id"].extend(meta[i]["narration_id"])
            else:
                metadata = meta
                verb_preds, verb_labels, audio_idx = preds[0], labels["verb"], audio_idx
                noun_preds, noun_labels, audio_idx = preds[1], labels["noun"], audio_idx
            if cfg.NUM_GPUS:
                verb_preds = verb_preds.cpu()
                verb_labels = verb_labels.cpu()
                noun_preds = noun_preds.cpu()
                noun_labels = noun_labels.cpu()
                audio_idx = audio_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                (verb_preds.detach(), noun_preds.detach()),
                (verb_labels.detach(), noun_labels.detach()),
                metadata,
                audio_idx.detach(),
            )

            test_meter.log_iter_stats(cur_iter)
        else:
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, audio_idx = du.all_gather([preds, labels, audio_idx])
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                audio_idx = audio_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds.detach(), labels.detach(), audio_idx.detach())
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.TEST.DATASET.startswith("EpicKtichens"):
        all_preds = test_meter.audio_preds.clone().detach()
        all_labels = test_meter.audio_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with PathManager.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))

    preds, preds_clips, labels, metadata = test_meter.finalize_metrics()
    return test_meter, preds, preds_clips, labels, metadata


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from an audio along
    its temporal axis. Softmax scores are averaged across all N views to
    form an audio-level prediction. All audio predictions are compared to
    ground-truth labels and the final testing performance is logged.
    Args:
        test_loader (loader): audio testing loader.
        model (model): the pretrained audio model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    for cur_iter, batch in enumerate(
        tqdm(
            test_loader,
            desc="Testing",
            unit="batch",
        )
    ):
        if not "GRU" in cfg.TEST.DATASET:
            inputs, labels, audio_idx, meta = batch

            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)

                # Transfer the data to the current GPU device.
                if isinstance(labels, (dict,)):
                    labels = {k: v.cuda() for k, v in labels.items()}
                else:
                    labels = labels.cuda()
                audio_idx = audio_idx.cuda()

            test_meter.data_toc()

            # Perform the forward pass.
            preds = model(inputs)

        else:
            inputs, lengths, labels, audio_idx, noun_embeddings, meta = batch
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

            test_meter.data_toc()

            preds = model(
                x=inputs,
                lengths=lengths,
                noun_embeddings=noun_embeddings,
            )

        if isinstance(labels, (dict,)):
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                verb_preds, verb_labels, audio_idx = du.all_gather([preds[0], labels["verb"], audio_idx])

                noun_preds, noun_labels, audio_idx = du.all_gather([preds[1], labels["noun"], audio_idx])
                meta = du.all_gather_unaligned(meta)
                metadata = {"narration_id": []}
                for i in range(len(meta)):
                    metadata["narration_id"].extend(meta[i]["narration_id"])
            else:
                metadata = meta
                verb_preds, verb_labels, audio_idx = preds[0], labels["verb"], audio_idx
                noun_preds, noun_labels, audio_idx = preds[1], labels["noun"], audio_idx
            if cfg.NUM_GPUS:
                verb_preds = verb_preds.cpu()
                verb_labels = verb_labels.cpu()
                noun_preds = noun_preds.cpu()
                noun_labels = noun_labels.cpu()
                audio_idx = audio_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                (verb_preds.detach(), noun_preds.detach()),
                (verb_labels.detach(), noun_labels.detach()),
                metadata,
                audio_idx.detach(),
            )

            # test_meter.log_iter_stats(cur_iter)
        else:
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, audio_idx = du.all_gather([preds, labels, audio_idx])
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                audio_idx = audio_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds.detach(), labels.detach(), audio_idx.detach())
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.TEST.DATASET.startswith("EpicKitchens"):
        all_preds = test_meter.audio_preds.clone().detach()
        all_labels = test_meter.audio_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with PathManager.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))

    preds, preds_clips, labels, metadata = test_meter.finalize_metrics()
    return test_meter, preds, preds_clips, labels, metadata


def test(cfg):
    """
    Perform multi-view testing on the pretrained audio model.
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
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the audio model and print model statistics.
    model = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg)

    cu.load_test_checkpoint(cfg, model)

    # Create audio testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if not "GRU" in cfg.TEST.DATASET:
        assert len(test_loader.dataset) % cfg.TEST.NUM_ENSEMBLE_VIEWS == 0

    # Create meters for multi-view testing.
    if cfg.TEST.DATASET.startswith("EpicKitchens"):
        test_meter = (
            EPICTestMeter(
                len(test_loader.dataset) // cfg.TEST.NUM_ENSEMBLE_VIEWS
                if not "GRU" in cfg.TEST.DATASET
                else len(test_loader.dataset),
                cfg.TEST.NUM_ENSEMBLE_VIEWS if not "GRU" in cfg.TEST.DATASET else 1,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                cfg.DATA.ENSEMBLE_METHOD,
            )
            if cfg.MODEL.ONLY_ACTION_RECOGNITION
            else EPICTestMeterWithState(
                len(test_loader.dataset) // cfg.TEST.NUM_ENSEMBLE_VIEWS
                if not "GRU" in cfg.TEST.DATASET
                else len(test_loader.dataset),
                cfg.TEST.NUM_ENSEMBLE_VIEWS if not "GRU" in cfg.TEST.DATASET else 1,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                cfg.DATA.ENSEMBLE_METHOD,
            )
        )

    else:
        test_meter = TestMeter(
            len(test_loader.dataset) // cfg.TEST.NUM_ENSEMBLE_VIEWS,
            cfg.TEST.NUM_ENSEMBLE_VIEWS,
            cfg.MODEL.NUM_CLASSES[0],
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform multi-view test on the entire dataset.
    test_meter, preds, preds_clips, labels, metadata = (
        perform_test(
            test_loader=test_loader,
            model=model,
            test_meter=test_meter,
            cfg=cfg,
            writer=writer,
        )
        if cfg.MODEL.ONLY_ACTION_RECOGNITION
        else perform_test_with_state(
            test_loader=test_loader,
            model=model,
            test_meter=test_meter,
            cfg=cfg,
            writer=writer,
        )
    )

    if du.is_master_proc():
        if cfg.TEST.DATASET.startswith("EpicKitchens"):
            if cfg.MODEL.ONLY_ACTION_RECOGNITION:
                results = {
                    "verb_output": preds[0],
                    "noun_output": preds[1],
                    "narration_id": metadata,
                }
            else:
                results = {
                    "verb_output": preds[0],
                    "noun_output": preds[1],
                    "state_output": preds[2],
                    "narration_id": metadata,
                }

            scores_path = os.path.join(cfg.OUTPUT_DIR, "scores")
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            file_path = os.path.join(scores_path, cfg.EPICKITCHENS.TEST_SPLIT + ".pkl")
            pickle.dump(results, open(file_path, "wb"))
        else:
            if cfg.TEST.DATASET == "vggsound":
                get_stats(preds, labels)
            results = {"scores": preds, "labels": labels}
            scores_path = os.path.join(cfg.OUTPUT_DIR, "scores")
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            file_path = os.path.join(scores_path, "test.pkl")
            pickle.dump(results, open(file_path, "wb"))

    if writer is not None:
        writer.close()
