from typing import Tuple
from loguru import logger
import h5py
import sys
import os
import argparse
import plotly.express as px
import torch
import matplotlib.pyplot as plt
import librosa

from audio_slowfast.config.defaults import get_cfg
from audio_slowfast.datasets.epickitchens_gru_pddl import EpicKitchensGRUwithPDDL
from audio_slowfast.datasets.epickitchens import EpicKitchens


def run(index: int) -> None:
    cfg = get_cfg()
    cfg.merge_from_file("models/asf/config/SLOWFAST_R50.yaml")

    audio_dataset = EpicKitchens(cfg, "train")
    sample = audio_dataset[index]
    plot_spectrograms(sample[0], index=index, d_type="regular")

    audio_dataset = EpicKitchensGRUwithPDDL(cfg, "train")
    sample = audio_dataset[index]
    plot_spectrograms(sample[0], index=index, d_type="gru")


def plot_spectrograms(
    spectrograms: Tuple[torch.Tensor, torch.Tensor],
    index: int,
    prefix: str = "",
    d_type: str = "regular",
) -> None:
    slow_sg, fast_sg = spectrograms

    # Get rid of the channel dimension
    if d_type == "regular":
        slow_sg = slow_sg[0, :, :]
        fast_sg = fast_sg[0, :, :]
    elif d_type == "gru":
        slow_sg = slow_sg[0, 0, :, :]
        fast_sg = fast_sg[0, 0, :, :]

    _plot_spectrogram(
        spectrogram=slow_sg,
        index=index,
        prefix=prefix,
        s_type="Slow",
        d_type=d_type,
    )
    _plot_spectrogram(
        spectrogram=fast_sg,
        index=index,
        prefix=prefix,
        s_type="Fast",
        d_type=d_type,
    )


def _plot_spectrogram(
    spectrogram: torch.Tensor,
    index: int,
    prefix: str = "",
    s_type: str = "Slow",
    d_type: str = "gru",
) -> None:
    spectrogram = spectrogram.T.numpy()
    output_path = f"res/dataloader/{d_type}"
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(10, 10))
    librosa.display.specshow(spectrogram, vmin=-6, vmax=3, cmap="magma")
    plt.colorbar(label="dB")
    plt.title(f"Mel spectrogram for sample {index} ({prefix})", fontdict=dict(size=18))
    plt.xlabel("Time", fontdict=dict(size=15))
    plt.ylabel("Mel-Frequency bins", fontdict=dict(size=15))
    plt.savefig(f"{output_path}/{s_type.lower()}_{index}_{prefix}_{d_type}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=int, required=True)
    args = parser.parse_args()
    run(index=args.index)
