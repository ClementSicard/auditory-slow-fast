import os
import sys

from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.config.defaults import get_cfg
from audio_slowfast.datasets.epickitchens_slide import EpicKitchensSlide

CONFIG_PATH = "models/asf/config/asf-slide.yaml"


def run() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)

    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.TRAIN.BATCH_SIZE = 1

    logger.info("Testing EpicKitchens dataloader")
    dataset = EpicKitchensSlide(cfg=cfg, mode="test")
    sample = dataset[0]

    logger.debug(f"{len(sample)=}")


if __name__ == "__main__":
    run()
