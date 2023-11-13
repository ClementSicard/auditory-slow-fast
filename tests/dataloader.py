import sys
from loguru import logger
import os
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.config.defaults import get_cfg
import audio_slowfast.datasets.loader as loader


def run() -> None:
    cfg = get_cfg()
    cfg.merge_from_file("models/asf/config/SLOWFAST_R50.yaml")

    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.TRAIN.BATCH_SIZE = 10

    train_loader: DataLoader = loader.construct_loader(cfg, "train")

    for batch in train_loader:
        for i, b in enumerate(batch):
            try:
                logger.warning(f"{i}: {b.shape}")
            except:
                logger.warning(f"{i}: {len(b)}")
        break


if __name__ == "__main__":
    run()
