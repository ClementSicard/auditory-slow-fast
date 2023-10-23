from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Waveform",
    xlim: Tuple[float, float] = None,
):
    """
    Plots waveform

    Parameters
    ----------
    `waveform` : `np.ndarray`
        The waveform array
    `sample_rate` : `int`
        Sampling rate of the waveform
    `title` : `str`, optional
        Title of the plot, by default `"Waveform"`
    `xlim` : `Tuple[float, float]`, optional
        Xlim behavior, by default `None`
    """
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show()


def plot_specgram(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Spectrogram",
    xlim: Tuple[float, float] = None,
):
    """
    Plot spectrogram

    Parameters
    ----------
    `waveform` : `np.ndarray`
        Waveform array
    `sample_rate` : `int`
        Sampling rate of the waveform
    `title` : `str`, optional
        Title of the plot, by default `"Spectrogram"`
    `xlim` : `Tuple[float, float]`, optional
        Xlim behavior of the plot, by default `None`
    """
    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
