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
                [0.2, 0.3, -10],
                [0.1, 0.5, -10],
            ],
            [
                [0.2, -10, -10],
                [0.1, -10, -10],
            ],
        ]
    )
    a = torch.rand_like(labels)

    loss(a, labels)


if __name__ == "__main__":
    run()
