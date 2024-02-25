#!/usr/bin/env python3
import random
import time
from datetime import timedelta

import numpy as np
import torch
from fvcore.common.config import CfgNode
from torch.utils.data.distributed import DistributedSampler


def get_start_end_idx(audio_size, clip_size, clip_idx, num_clips, start_sample=0):
    """
    Sample a clip of size clip_size from an audio of size audio_size and
    return the indices of the first and last sample of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the audio to
    num_clips clips, and select the start and end index of clip_idx-th audio
    clip.
    Args:
        audio_size (int): number of overall samples.
        clip_size (int): size of the clip to sample from the samples.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the audio to num_clips
            clips, and select the start and end index of the clip_idx-th audio
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given audio for testing.
    Returns:
        start_idx (int): the start sample index.
        end_idx (int): the end sample index.
    """
    delta = max(audio_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = np.linspace(0, delta, num=num_clips)[clip_idx]
    end_idx = start_idx + clip_size - 1
    return start_sample + start_idx, start_sample + end_idx


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


def timestamp_to_sec(timestamp):
    time_parts = timestamp.split(".")
    base_time = time_parts[0]
    microsecond_part = time_parts[1].rstrip("0") if len(time_parts) > 1 else "0"

    if not microsecond_part:
        microsecond_part = "0"

    x = time.strptime(base_time, "%H:%M:%S")

    # Calculate the divisor based on the length of the microsecond part
    divisor = 10 ** len(microsecond_part)

    sec = (
        float(
            timedelta(
                hours=x.tm_hour,
                minutes=x.tm_min,
                seconds=x.tm_sec,
                microseconds=int(microsecond_part),
            ).total_seconds()
        )
        + int(microsecond_part) / divisor
    )
    return sec
