#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""


from typing import Any, List
from loguru import logger
import pandas as pd
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from . import utils as utils
from .build import build_dataset


def epickitchens_collate_fn(batch: List[Any]):
    # Unzip the batch into separate lists
    spectrograms, labels, indices, noun_embeddings, metadata = zip(*batch)

    slow_spectrograms, fast_spectrograms = zip(*spectrograms)

    lengths = [slow_spectrogram.shape[0] for slow_spectrogram in slow_spectrograms]

    padded_slow_spectrograms = torch.nn.utils.rnn.pad_sequence(
        slow_spectrograms,
        batch_first=True,
        padding_value=0.0,
    )
    padded_fast_spectrograms = torch.nn.utils.rnn.pad_sequence(
        fast_spectrograms,
        batch_first=True,
        padding_value=0.0,
    )

    # Combine padded spectrograms into a tuple
    padded_spectrograms = (padded_slow_spectrograms, padded_fast_spectrograms)

    # Convert other elements of the batch to tensors or appropriate formats
    indices = torch.tensor(indices)

    # Stack noun embeddings
    stacked_noun_embeddings = torch.stack(noun_embeddings, dim=0)

    # Create a pandas DataFrame for labels
    grouped_labels = {
        k: torch.stack(v, dim=0) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
        for k, v in pd.DataFrame(labels).to_dict("list").items()
    }

    metadata = pd.DataFrame(metadata).to_dict("list")

    logger.error("ICIIII")
    return_tuple = (
        padded_spectrograms,
        lengths,
        grouped_labels,
        indices,
        stacked_noun_embeddings,
        metadata,
    )

    return return_tuple


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
        shuffle=False if sampler else shuffle,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=(
            lambda batch: epickitchens_collate_fn(
                batch=batch,
            )
        )
        if "gru" in cfg.TRAIN.DATASET.lower()
        else None,
        worker_init_fn=utils.loader_worker_init_fn(dataset),
    )

    return loader


def shuffle_dataset(loader, cur_epoch):
    """
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
