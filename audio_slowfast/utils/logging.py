#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import atexit
import builtins
import decimal
import functools
import logging
import os
import sys
from typing import Optional

import simplejson
from fvcore.common.file_io import PathManager
from loguru import logger

import audio_slowfast.utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = PathManager.open(filename, "a", buffering=1024)
    atexit.register(io.close)
    return io


def setup_logging(output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if du.is_master_proc():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if du.is_master_proc():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats, it: Optional[int] = None):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """

    stats = {k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v for k, v in stats.items()}
    json_stats = simplejson.dumps(
        stats,
        sort_keys=True,
        use_decimal=True,
        indent=2 * " ",
    )
    logger.debug(f"Stats: {json_stats}")

    file_name = os.getenv("TRAIN_STATS")
    if not file_name:
        file_name = "logs/stats.json"
        logger.warning(f"Environment variable TRAIN_STATS is not set. Writing to '{file_name}' instead.")
