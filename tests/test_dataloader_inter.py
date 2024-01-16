import os
import sys

from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.config.defaults import get_cfg
from audio_slowfast.datasets.epickitchens_inter_es import EpicKitchensInterES

CONFIG_PATH = "models/asf/config/inter/asf-original-inter.yaml"


def run() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)

    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TEST.BATCH_SIZE = 128
    # cfg.EPICKITCHENS.SINGLE_BATCH = False

    logger.info("Testing EpicKitchensSlide dataloader with sliding over the whole video")
    cfg.TEST.SLIDE.PER_ACTION_INSTANCE = False
    cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS = False
    dataset = EpicKitchensInterES(cfg=cfg, mode="train")
    sample = dataset[0]


if __name__ == "__main__":
    run()
