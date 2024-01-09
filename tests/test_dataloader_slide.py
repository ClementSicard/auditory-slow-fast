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
    cfg.TEST.BATCH_SIZE = 1

    logger.info("Testing EpicKitchensSlide dataloader with sliding over the whole video")
    cfg.TEST.SLIDE.PER_ACTION_INSTANCE = False
    cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS = False
    dataset = EpicKitchensSlide(cfg=cfg, mode="test")
    sample = dataset[0]

    print()

    logger.info("Testing EpicKitchensSlide dataloader with sliding over the whole action")
    cfg.TEST.SLIDE.PER_ACTION_INSTANCE = False
    cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS = True
    dataset = EpicKitchensSlide(cfg=cfg, mode="test")
    sample = dataset[0]

    print()

    logger.info("Testing EpicKitchensSlide dataloader with one record per action")
    cfg.TEST.SLIDE.PER_ACTION_INSTANCE = True
    cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS = True
    dataset = EpicKitchensSlide(cfg=cfg, mode="test")
    sample = dataset[0]


if __name__ == "__main__":
    run()
