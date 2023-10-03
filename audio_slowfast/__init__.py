#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .utils.env import setup_environment
from .audio_slowfast import AudioSlowFast, CustomResNetBasicHead
from .tools.train_net import train
from .tools.test_net import test

setup_environment()
