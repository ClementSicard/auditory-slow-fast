from loguru import logger
import torch
import time


def run():
    # Create a tensor that will take 5 Go of memory
    x = torch.randn(100000000, 10).cuda()

    while True:
        x = x.matmul(x)
        x = x / 2
        time.sleep(1)


if __name__ == "__main__":
    logger.info("Starting GPU stress test")
    run()
