from collections import OrderedDict
from typing import Any
import torch
from loguru import logger
import os

from audio_slowfast.config.defaults import get_cfg


def main() -> None:
    checkpoint = load_checkpoint("models/asf/weights/SLOWFAST_EPIC.pyth")

    cfg = get_cfg()
    cfg.merge_from_file("./models/asf/config/SLOWFAST_R50.yaml")

    cp_model_state_new = OrderedDict()
    for k in checkpoint["model_state"].keys():
        logger.info(f"Fixing weights for {k}")
        cp_model_state_new[f"model.{k}"] = checkpoint["model_state"][k]

    checkpoint["model_state"] = cp_model_state_new

    logger.debug(f"{checkpoint['model_state'].keys()}")
    with open("models/asf/weights/SLOWFAST_EPIC_fixed.pyth", "wb") as f:
        torch.save(checkpoint, f)


def load_checkpoint(path: str) -> OrderedDict[str, Any]:
    if not os.path.exists(path=path):
        logger.error(f"Checkpoint '{path}' does not exist")
        exit(1)

    with open(path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    return checkpoint


if __name__ == "__main__":
    main()
