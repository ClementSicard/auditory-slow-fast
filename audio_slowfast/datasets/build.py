#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry
from torch.utils.data import Dataset

from fvcore.common.config import CfgNode

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name: str, cfg: CfgNode, split: str) -> Dataset:
    """
    Build a dataset, defined by `dataset_name`.

    Parameters
    ----------
    `dataset_name` : `str`
        The name of the dataset to be constructed.

    `cfg` : `CfgNode`
        The configs. Details can be found in
        `audio_slowfast/config/defaults.py`.

    `split` : `str`
        The split of the data loader. Options include `train`,
        `val`, and `test`.

    Returns
    -------
    `Dataset`
        A constructed dataset specified by dataset_name.
    """

    return DATASET_REGISTRY.get(dataset_name)(cfg, split)
