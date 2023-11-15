import sys
from loguru import logger
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.config.defaults import get_cfg
import audio_slowfast.datasets.loader as loader
import audio_slowfast.datasets.utils as utils


def run() -> None:
    cfg = get_cfg()
    cfg.merge_from_file("models/asf/config/SLOWFAST_R50.yaml")

    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.TRAIN.BATCH_SIZE = 2

    train_loader: DataLoader = loader.construct_loader(cfg, "train")

    # Expects [(batch_size, 1, T_slow, F), (batch_size, 1, T_fast, F)]
    C = 1
    F = cfg.AUDIO_DATA.NUM_FREQUENCIES
    T_fast = cfg.AUDIO_DATA.NUM_FRAMES
    T_slow = cfg.AUDIO_DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA

    for batch in train_loader:
        for i, b in enumerate(batch):
            match i:
                case 0:
                    logger.warning(f"Spectrogram shape: {[bb.shape for bb in b]}")
                case 1:
                    logger.warning(f"Labels: {b}")
                case 2:
                    logger.warning(f"Indices shape: {b}")
                case 3:
                    logger.warning(f"Metadata: {b}")
        break


if __name__ == "__main__":
    run()
