"""
This file contains the wrapper around the Auditory SlowFast model.

It changes the original head defined in audio_slowfast/models/head_helper.py
to be able to return 3 outputs: the recognized action along the pre- and post-conditions.

It was largely inspired by https://github.com/VIDA-NYU/ptg-server-ml/blob/main/ptgprocess/audio_slowfast.py
"""

import os
from typing import Any, Dict, List

import audio_slowfast
import audio_slowfast.utils.checkpoint as cu
import librosa
import numpy as np
import pandas as pd
import torch
from audio_slowfast.config.defaults import get_cfg
from audio_slowfast.models import build_model
from fvcore.common.config import CfgNode
from loguru import logger
from torch import nn

MODEL_DIR = os.getenv("MODEL_DIR") or "models/asf/weights"

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "EPIC-KITCHENS/SLOWFAST_R50.yaml")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "SLOWFAST_EPIC.pyth")


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
    ):
        super().__init__()
        # init config
        self.cfg = cfg = get_cfg()
        cfg.merge_from_file(cfg_file_path)
        logger.success(f"Loaded config from {cfg_file_path}")
        # get vocab classes
        self.vocab = []
        if cfg.MODEL.VOCAB_FILE:
            import json

            cfg.MODEL.VOCAB_FILE = os.path.join(
                os.path.dirname(cfg_file_path), cfg.MODEL.VOCAB_FILE
            )
            self.vocab = json.load(open(cfg.MODEL.VOCAB_FILE))
            logger.success(f"Loaded vocab from {cfg.MODEL.VOCAB_FILE}")

        # window params
        window_size = cfg.AUDIO_DATA.WINDOW_LENGTH
        step_size = cfg.AUDIO_DATA.HOP_LENGTH
        self.win_size = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
        self.hop_size = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
        self.num_frames = 400  # cfg.AUDIO_DATA.NUM_FRAMES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        logger.info(f"Window size: {self.win_size}")
        logger.info(f"Hop size: {self.hop_size}")
        logger.info(f"Step size: {step_size}")
        logger.info(f"Window size: {window_size}")
        logger.info(f"Number of frames: {self.num_frames}")
        logger.info(f"Number of classes: {self.num_classes}")

        # build and load model
        cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint
        cfg.NUM_GPUS = min(cfg.NUM_GPUS, torch.cuda.device_count())
        self.model = build_model(cfg)
        cu.load_test_checkpoint(cfg, self.model)
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
        spec = librosa.util.frame(
            spec, frame_length=self.num_frames, hop_length=int(self.num_frames // 4)
        )
        spec = spec.transpose((2, 1, 0))[:, None]

        spec = torch.tensor(spec, dtype=torch.float)  # mono to stereo
        spec = pack_pathway_output(self.cfg, spec)
        return spec

    def forward(
        self,
        specs,
        return_embedding: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Wrapper around the model's forward function.

        Parameters
        ----------
        `specs` : `np.ndarray`
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
        # y = self.model.head.project_verb_noun(z)
        if self.model.training:
            y = [x.view((len(x), -1, s)) for x, s in zip(y, self.num_classes)]
        if return_embedding:
            return y, z[:, 0, 0]
        return y


class CustomResNetBasicHead(audio_slowfast.models.head_helper.ResNetBasicHead):
    """
    ResNet basic head for audio classification.

    Overrides the original head defined in
    auditory_slow_fast/audio_slowfast/models/head_helper.py
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the ResNet basic head.

        Parameters
        ----------
        `inputs` : `list` of `torch.Tensor`
            List of input tensors.

        Returns
        -------
        `torch.Tensor`
            Output tensor.
        """
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H) -> (N, T, H, C).
        x = x.permute((0, 2, 3, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x

    def _proj(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """
        Projection function

        Parameters
        ----------
        `x` : `torch.Tensor`
            Input tensor.
        `proj` : `nn.Module`
            Projection function

        Returns
        -------
        `torch.Tensor`
            Projected view of input tensor.
        """
        x_v = proj(x)
        # Performs fully convolutional inference.
        if not self.training:
            x_v: torch.Tensor = self.act(x_v)
            x_v = x_v.mean([1, 2])
        return x_v.view(x_v.shape[0], -1)

    def project_verb_noun(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """
        Project the input tensor to verb and noun classes.

        Parameters
        ----------
        `x` : `torch.Tensor`
            Input tensor.

        Returns
        -------
        `torch.Tensor | List[torch.Tensor]`
            Projected view of input tensor.
        """
        if isinstance(self.num_classes, (list, tuple)):
            return (
                self._proj(x, self.projection_verb),
                self._proj(x, self.projection_noun),
            )
        return self._proj(x, self.projection)

    def project_pre_post_conditions(
        self, x: torch.Tensor
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Project the input tensor to pre- and post-conditions respective classes.

        Parameters
        ----------
        `x` : `torch.Tensor`
            Input tensor.

        Returns
        -------
        `torch.Tensor | List[torch.Tensor]`
            Projected view of input tensor.
        """

        if isinstance(self.num_classes, (list, tuple)):
            return (
                self._proj(x, self.projection_verb),
                self._proj(x, self.projection_noun),
                self._proj(x, self.projection_prec),
                self._proj(x, self.projection_postc),
            )

        return self._proj(x, self.projection)


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


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: Dict[str, Any]) -> None:
    """
    Main function to run inference on Auditory SlowFast model.

    Parameters
    ----------
    `input_path` : `str`
        Path to audio file to run inference on.
    """
    if not os.path.exists(args["input_path"]):
        logger.error(f"Input path {args['input_path']} does not exist.")
        exit(1)
    model = AudioSlowFast(checkpoint=args["weights"], cfg_file_path=args["config"])

    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    vocab_verb, vocab_noun = model.vocab
    vocab_prec = pd.read_csv(args["prec_vocab"])["precondition"].to_list()
    vocab_postc = pd.read_csv(args["postc_vocab"])["postcondition"].to_list()

    logger.info(f"Loading input audio from {args['input_path']}")

    y, sr = librosa.load(args["input_path"], sr=None)
    spec = model.prepare_audio(y, sr)

    logger.debug(f"Spec shapes: {[x.shape for x in spec]}")
    verb, noun, prec, postc = model([x.to(device) for x in spec])
    i_vs, i_ns, i_pres, i_poss = (
        torch.argmax(verb, dim=-1),
        torch.argmax(noun, dim=-1),
        torch.argmax(prec, dim=-1),
        torch.argmax(postc, dim=-1),
    )

    for v, n, pre, pos, i_v, i_n, i_pre, i_pos in zip(
        verb, noun, prec, postc, i_vs, i_ns, i_pres, i_poss
    ):
        logger.debug(
            f"{vocab_verb[i_v]:>12}: ({v[i_v]:.2%})"
            f"{vocab_noun[i_n]:>12}: ({n[i_n]:.2%})"
            f"{vocab_prec[i_pre]:>12}: ({pre[i_pre]:.2%})"
            f"{vocab_postc[i_pos]:>12}: ({pos[i_pos]:.2%})"
        )


def parse_args() -> Dict[str, Any]:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="input_path",
        help="Path to the input audio file",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--weights",
        dest="weights",
        help="Path to the model weights",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="The path to the config file",
        required=True,
    )
    parser.add_argument(
        "--prec_vocab",
        dest="prec_vocab",
        help="Path to the preconditions vocab file",
        required=True,
    )
    parser.add_argument(
        "--postc_vocab",
        dest="postc_vocab",
        help="Path to the postconditions vocab file",
        required=True,
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
