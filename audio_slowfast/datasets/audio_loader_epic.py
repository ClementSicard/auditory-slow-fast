from typing import Optional
from loguru import logger

import numpy as np
import torch
from audiomentations.core.transforms_interface import BaseWaveformTransform
from fvcore.common.config import CfgNode
from h5py._hl.files import File

from audio_slowfast.datasets.utils import get_start_end_idx


from .epickitchens_record import EpicKitchensAudioRecord


def pack_audio(
    cfg: CfgNode,
    audio_dataset: File,
    audio_record: EpicKitchensAudioRecord,
    temporal_sample_index: int,
    transform: Optional[BaseWaveformTransform] = None,
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

    Returns
    -------
    `torch.Tensor`
        The sound features, transformed if `transform` is not `None`.
    """
    samples = audio_dataset[audio_record.untrimmed_video_name][()]
    start_idx, end_idx = get_start_end_idx(
        audio_record.num_audio_samples,
        int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)),
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        start_sample=audio_record.start_audio_sample,
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


def _log_specgram(cfg, audio, window_size=10, step_size=5, eps=1e-6):
    stft_window_size = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    stft_hop_size = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
    from librosa import filters, stft

    assert (
        stft_window_size > stft_hop_size
    ), f"nperseg ({stft_window_size}) must be greater than noverlap ({stft_hop_size})."
    stft_hop_size = stft_window_size - stft_hop_size

    # mel-spec
    spec = stft(
        audio,
        n_fft=2048,
        window="hann",
        hop_length=stft_hop_size,
        win_length=stft_window_size,
        pad_mode="constant",
    )
    mel_basis = filters.mel(
        sr=cfg.AUDIO_DATA.SAMPLING_RATE,
        n_fft=cfg.AUDIO_DATA.N_FFT,
        n_mels=cfg.AUDIO_DATA.NUM_FREQUENCIES,
        htk=True,
        norm=None,
    )
    mel_spec = np.dot(mel_basis, np.abs(spec))

    # log-mel-spec
    log_mel_spec = np.log(mel_spec + eps)
    return log_mel_spec.T


def _extract_sound_feature(
    cfg: CfgNode,
    samples: np.ndarray,
    audio_record: EpicKitchensAudioRecord,
    start_idx: int,
    end_idx: int,
    transform: Optional[BaseWaveformTransform] = None,
):
    """
    Extracts sound features from audio samples of an Epic Kitchens audio record based on the given configuration,
    start and end indices, and waveform transform object.

    Parameters
    ----------
    `cfg`: `CfgNode`
        The configuration node.
    `samples`: `np.ndarray`
        The audio samples **for the full video**.
    `audio_record`: `EpicKitchensAudioRecord`
        The audio record.
    `start_idx`: `int`
        The start index.
    `end_idx`: `int`
        The end index.
    `transform`: `Optional[BaseWaveformTransform]`
        The audio transform. By default `None`.

    Returns
    -------
    `torch.Tensor`
        The sound features, transformed if `transform` is not `None`.
    """
    # 1st case: the audio clip is shorter than the desired length.
    if audio_record.num_audio_samples < int(round(cfg.AUDIO_DATA.SAMPLING_RATE * cfg.AUDIO_DATA.CLIP_SECS)):
        samples = samples[audio_record.start_audio_sample : audio_record.end_audio_sample]

    # 2nd case: the audio clip is longer than the desired length.
    else:
        samples = samples[start_idx:end_idx]

    # In case the overlaps goes beyond the end of the audio clip, pad the audio clip with copies of the last sample.
    if transform is not None:
        samples = transform(samples, sample_rate=cfg.AUDIO_DATA.SAMPLING_RATE)

    spectrogram = _log_specgram(
        cfg=cfg,
        audio=samples,
        window_size=cfg.AUDIO_DATA.WINDOW_LENGTH,
        step_size=cfg.AUDIO_DATA.HOP_LENGTH,
    )

    num_timesteps_to_pad = cfg.AUDIO_DATA.NUM_FRAMES - spectrogram.shape[0]
    if num_timesteps_to_pad > 0:
        # logger.warning(f"Padded spectrogram {audio_record._index} with copies of the last sample.")
        spectrogram = np.pad(spectrogram, ((0, num_timesteps_to_pad), (0, 0)), "edge")

    return torch.tensor(spectrogram).unsqueeze(0)
