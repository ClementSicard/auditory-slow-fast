#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""


from loguru import logger
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from . import utils as utils
from .build import build_dataset


def epickitchens_collate_fn(batch):
    # Unzip the batch into separate lists
    spectrograms, labels, indices, noun_embeddings, metadata = zip(*batch)

    slow_spectrograms, fast_spectrograms = zip(*spectrograms)

    # Find the maximum num_spectrogram for each type of spectrogram
    max_num_spectrogram_slow = max([s.shape[0] for s in slow_spectrograms])
    max_num_spectrogram_fast = max([s.shape[0] for s in fast_spectrograms])

    assert (
        max_num_spectrogram_slow == max_num_spectrogram_fast
    ), f"{max_num_spectrogram_slow=} != {max_num_spectrogram_fast=}"

    # 0-pad the first dimension to be of length max_num_spectrogram
    padded_spectrograms1 = torch.zeros(
        (len(slow_spectrograms), max_num_spectrogram_slow, *slow_spectrograms[0].shape[1:])
    )
    padded_spectrograms2 = torch.zeros(
        (len(fast_spectrograms), max_num_spectrogram_fast, *fast_spectrograms[0].shape[1:])
    )

    for i, (s1, s2) in enumerate(zip(slow_spectrograms, fast_spectrograms)):
        padded_spectrograms1[i, : s1.shape[0], :, :] = s1
        padded_spectrograms2[i, : s2.shape[0], :, :] = s2

    # Combine padded spectrograms into a tuple
    padded_spectrograms = (padded_spectrograms1, padded_spectrograms2)

    # Convert other elements of the batch to tensors or appropriate formats
    indices = torch.tensor(indices)

    return padded_spectrograms, labels, indices, torch.stack(noun_embeddings, dim=0), metadata


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            audio_slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test", "train+val"]
    if split in ["train", "train+val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    sampler = utils.create_sampler(dataset, shuffle, cfg)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=epickitchens_collate_fn if split in ["train", "train+val"] else None,
        worker_init_fn=utils.loader_worker_init_fn(dataset),
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(sampler, (RandomSampler, DistributedSampler)), "Sampler type '{}' not supported".format(
        type(sampler)
    )
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
