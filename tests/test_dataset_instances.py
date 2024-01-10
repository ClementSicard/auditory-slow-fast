import os
import sys

from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.config.defaults import get_cfg
from audio_slowfast.datasets.epickitchens import EpicKitchens
from audio_slowfast.datasets.epickitchens_gru import EpicKitchensGRU
from audio_slowfast.datasets.epickitchens_gru_pddl import EpicKitchensGRUwithPDDL


CONFIG_PATH = "models/asf/config/asf-original.yaml"


def run() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)

    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.TRAIN.BATCH_SIZE = 1

    logger.info("Testing EpicKitchens dataloader")
    dataset = EpicKitchens(cfg=cfg, mode="train")
    sample = dataset[0]

    # logger.debug(f"{len(sample)=}")

    # logger.info("Testing EpicKitchensGRU dataloader")
    # dataset = EpicKitchensGRU(cfg=cfg, mode="train")
    # sample = dataset[0]

    # logger.info("Testing EpicKitchensGRU dataloader")
    # dataset = EpicKitchensGRUwithPDDL(cfg=cfg, mode="train")
    # sample = dataset[0]


if __name__ == "__main__":
    run()
