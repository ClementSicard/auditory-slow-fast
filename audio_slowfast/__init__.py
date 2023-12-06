#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .tools.test_net import test  # noqa
from .tools.train_net import train  # noqa
from .utils.env import setup_environment

setup_environment()
