from typing import Optional
from loguru import logger
import torch
from audiomentations.core.transforms_interface import BaseWaveformTransform
from fvcore.common.config import CfgNode
from h5py._hl.files import File

from .epickitchens_record_gru_pddl import EpicKitchensAudioRecordGRUwithPDDL
from audio_slowfast.datasets.utils import get_start_end_idx
from audio_slowfast.datasets.audio_loader_epic import _extract_sound_feature


def pack_audio_gru(
    cfg: CfgNode,
    audio_dataset: File,
    audio_record: EpicKitchensAudioRecordGRUwithPDDL,
    temporal_sample_index: int,
    transform: Optional[BaseWaveformTransform] = None,
    start_offset: float = 0.0,
) -> torch.Tensor:
    """
    Extracts sound features from audio samples of an Epic Kitchens audio record based on the given configuration and
    temporal sample index.

    Parameters
    ----------
    `cfg`: `CfgNode`
        The configuration node.
    `audio_dataset`: `File`
        The HDF5 file containing the audio samples.
    `audio_record`: `EpicKitchensAudioRecord`
        The audio record.
    `temporal_sample_index`: `int`
        The temporal sample index.
    `transform`: `Optional[BaseWaveformTransform]`
        The audio transform. By default `None`.
    `start_offset`: `float`
        The start offset. By default `0.0`.

    Returns
    -------
    `torch.Tensor`
        The sound features, transformed if `transform` is not `None`.
    """
    samples = audio_dataset[audio_record.untrimmed_video_name][()]
    start_sample = audio_record.start_audio_sample + start_offset * cfg.AUDIO_DATA.SAMPLING_RATE

    start_idx, end_idx = get_start_end_idx(
        audio_record.num_audio_samples,
        int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)),
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        start_sample=start_sample,
    )
    start_idx, end_idx = int(start_idx), int(end_idx)
    spectrogram = _extract_sound_feature(
        cfg=cfg,
        samples=samples,
        audio_record=audio_record,
        start_idx=start_idx,
        end_idx=end_idx,
        transform=transform,
    )
    return spectrogram
