from loguru import logger
import h5py
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from audio_slowfast.config.defaults import get_cfg


from audio_slowfast.datasets.epickitchens import EpicKitchens


def run(index: int) -> None:
    cfg = get_cfg()
    cfg.merge_from_file("models/asf/config/SLOWFAST_R50.yaml")

    audio_dataset = EpicKitchens(cfg, "train")

    sample = audio_dataset[index]

    logger.debug(f"{[s.shape for s in sample[0]]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=int, required=True)
    args = parser.parse_args()
    run(index=args.index)
