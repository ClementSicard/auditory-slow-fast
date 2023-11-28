import sys
from loguru import logger
import torch
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.utils.loss import MaskedLoss


def run():
    loss = MaskedLoss()

    labels = torch.tensor(
        [
            [
                [1, -1, 0, 0, 1, -10],
            ],
            [
                [1, 0, -10, -10, -10, -10],
            ],
        ],
        dtype=torch.float,
    )
    a = torch.rand_like(labels)

    l = loss(a, labels)
    logger.debug(f"{l=}")


if __name__ == "__main__":
    run()
