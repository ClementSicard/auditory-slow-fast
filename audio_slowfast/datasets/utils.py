#!/usr/bin/env python3

import logging
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
from fvcore.common.config import CfgNode
from loguru import logger


def pack_pathway_output(cfg, spectrogram):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        spectrogram (tensor): frames of spectrograms sampled from the complete spectrogram. The
            dimension is `channel` x `num frames` x `num frequencies`.
    Returns:
        spectrogram_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `num frequencies`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        spectrogram_list = [spectrogram]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = spectrogram
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            spectrogram,
            1,
            torch.linspace(0, spectrogram.shape[1] - 1, spectrogram.shape[1] // cfg.SLOWFAST.ALPHA).long(),
        )
        spectrogram_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return spectrogram_list


def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            audio_slowfast/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    return sampler


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None


def get_num_spectrogram_frames(duration: float, cfg: CfgNode) -> int:
    window_length_ms = cfg.AUDIO_DATA.WINDOW_LENGTH  # in milliseconds
    hop_length_ms = cfg.AUDIO_DATA.HOP_LENGTH  # in milliseconds
    sampling_rate = cfg.AUDIO_DATA.SAMPLING_RATE  # samples per second

    # Convert window length and hop length to samples
    window_length_samples = int(window_length_ms / 1000 * sampling_rate)
    hop_length_samples = int(hop_length_ms / 1000 * sampling_rate)

    # Calculate the number of frames
    num_frames = (duration * sampling_rate + 1 - window_length_samples) / hop_length_samples + 1
    return int(np.ceil(num_frames))
