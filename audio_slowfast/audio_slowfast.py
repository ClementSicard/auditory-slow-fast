"""
This file contains the wrapper around the Auditory SlowFast model.

It changes the original head defined in audio_slowfast/models/head_helper.py
to be able to return 3 outputs: the recognized action along the pre- and post-conditions.

It was largely inspired by https://github.com/VIDA-NYU/ptg-server-ml/blob/main/ptgprocess/audio_slowfast.py
"""

import os
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from fvcore.common.config import CfgNode
from loguru import logger
from torch import nn

import audio_slowfast.models.optimizer as optim
import audio_slowfast.utils.checkpoint as cu
from audio_slowfast.config.defaults import get_cfg
from audio_slowfast.models.build import MODEL_REGISTRY

from .custom_resnet_head import CustomResNetBasicHead

MODEL_DIR = os.getenv("MODEL_DIR") or "models/asf/weights"

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "EPIC-KITCHENS/SLOWFAST_R50.yaml")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "SLOWFAST_EPIC_fixed.pyth")


class AudioSlowFast(nn.Module):
    """
    Wrapper around the Auditory SlowFast model.
    """

    cfg: CfgNode
    eps: float = 1e-6
    vocab: List[str]
    win_size: int
    hop_size: int
    num_frames: int
    num_classes: int | List[int]
    model: nn.Module

    def __init__(
        self,
        checkpoint: str = DEFAULT_MODEL,
        cfg_file_path: str = DEFAULT_CONFIG,
        cfg: Optional[CfgNode] = None,
        train: bool = False,
    ):
        super().__init__()
        # init config
        if cfg is None:
            self.cfg = cfg = get_cfg()
            cfg.merge_from_file(cfg_file_path)

            logger.success(f"Loaded config from {cfg_file_path}")
        else:
            self.cfg = cfg
        # get vocab classes
        self.vocab = []
        if cfg.MODEL.VOCAB_FILE:
            import json

            self.vocab = json.load(open(cfg.MODEL.VOCAB_FILE))
            logger.success(f"Loaded vocab from {cfg.MODEL.VOCAB_FILE}")

        # Load pre-condition vocab
        self.vocab_prec = []
        if cfg.MODEL.VOCAB_PDDL_PRE_CONDITIONS:
            self.vocab_prec = pd.read_csv(cfg.MODEL.VOCAB_PDDL_PRE_CONDITIONS)["attribute"].to_list()
            logger.success(f"Loaded pre-conditions vocab from {cfg.MODEL.VOCAB_PDDL_PRE_CONDITIONS}")
            self.cfg.MODEL.NUM_CLASSES.append(len(self.vocab_prec))

        # Load post-condition vocab
        self.vocab_postc = []
        if cfg.MODEL.VOCAB_PDDL_POST_CONDITIONS:
            self.vocab_postc = pd.read_csv(cfg.MODEL.VOCAB_PDDL_POST_CONDITIONS)["attribute"].to_list()
            logger.success(f"Loaded post-conditions vocab from {cfg.MODEL.VOCAB_PDDL_POST_CONDITIONS}")
            self.cfg.MODEL.NUM_CLASSES.append(len(self.vocab_postc))

        # window params
        window_size = cfg.AUDIO_DATA.WINDOW_LENGTH
        step_size = cfg.AUDIO_DATA.HOP_LENGTH
        self.win_size = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
        self.hop_size = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
        self.num_frames = cfg.AUDIO_DATA.NUM_FRAMES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        logger.info(f"Window size: {self.win_size}")
        logger.info(f"Hop size: {self.hop_size}")
        logger.info(f"Step size: {step_size}")
        logger.info(f"Window size: {window_size}")
        logger.info(f"Number of frames: {self.num_frames}")
        logger.info(f"Number of classes: {self.num_classes}")

        # build and load model
        # cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint
        cfg.NUM_GPUS = min(cfg.NUM_GPUS, torch.cuda.device_count())
        self.model = build_model(cfg)

        if train:
            if cfg.BN.FREEZE:
                self.model.module.freeze_fn("bn_parameters") if cfg.NUM_GPUS > 1 else self.model.freeze_fn(
                    "bn_parameters"
                )

            # Construct the optimizer.
            self.optimizer = optim.construct_optimizer(self, cfg)
            # Load a checkpoint to resume training if applicable.
            self.start_epoch = cu.load_train_checkpoint(cfg, self, self.optimizer)
        else:
            cu.load_test_checkpoint(cfg, self)

        self.model.head.__class__ = CustomResNetBasicHead

    def prepare_audio(self, y: np.ndarray, sr: int = 24_000) -> np.ndarray:
        """
        Prepare audio for inference.

        Parameters
        ----------
        `y` : `np.ndarray`
            The input signal (audio time series).
        `sr` : `int`, optional
            Sampling rate, by default `24_000`

        Returns
        -------
        `np.ndarray`
            The prepared audio signal.
        """
        spec = librosa.stft(
            y,
            n_fft=2048,
            window="hann",
            hop_length=self.hop_size,
            win_length=self.win_size,
            pad_mode="constant",
        )
        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=2048,
            n_mels=128,
            htk=True,
            norm=None,
        )
        spec = np.dot(mel_basis, np.abs(spec))
        spec = np.log(spec + self.eps)

        npad = max(0, self.num_frames - spec.shape[-1])
        spec = np.pad(spec, ((0, npad), (0, 0)), "edge")
        spec = librosa.util.frame(spec, frame_length=self.num_frames, hop_length=int(self.num_frames // 4))
        spec = spec.transpose((2, 1, 0))[:, None]

        spec = torch.tensor(spec, dtype=torch.float)  # mono to stereo
        spec = pack_pathway_output(self.cfg, spec)
        return spec

    def forward(
        self,
        specs: torch.Tensor,
        return_embedding: bool = False,
        noun_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Wrapper around the model's forward function.

        Parameters
        ----------
        `specs` : `torch.Tensor`
            The prepared audio signal.

        `return_embedding` : `bool`, optional
            Flag to return the embedding or not, by default `False`

        Returns
        -------
        `torch.Tensor`
            The output tensor.
        """
        z = self.model(specs)
        y = self.model.head.project_pre_post_conditions(z)

        if self.model.training:
            y = [x.view((len(x), -1, s)) for x, s in zip(y, self.num_classes)]

        if return_embedding:
            return y, z[:, 0, 0]

        return y


def pack_pathway_output(cfg: CfgNode, spec: torch.Tensor) -> List[torch.Tensor]:
    """
    [ch X time X freq] -> [ [ch X slow time X freq], [ch X fast time X freq] ]
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        return [spec]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        # Perform temporal sampling from the fast pathway.
        T = spec.shape[-2]
        i = torch.linspace(0, T - 1, T // cfg.SLOWFAST.ALPHA).long()
        slow = torch.index_select(spec, -2, i)
        return [slow, spec]

    raise NotImplementedError(
        f"Model arch {cfg.MODEL.ARCH} is not in {cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH}"
    )


def build_model(cfg, gpu_id=None):
    """
    Builds the audio model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in audio_audio_slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert cfg.NUM_GPUS <= torch.cuda.device_count(), "Cannot use more GPU devices than available"
    else:
        assert cfg.NUM_GPUS == 0, "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        if cfg.NUM_GPUS > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(device=cur_device)

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=True,
        )
    return model
