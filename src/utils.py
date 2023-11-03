import os
from datetime import datetime

import torch
from loguru import logger


def setup_run() -> None:
    """
    Sets up the run by adding a logger and displaying the GPU info.
    """
    add_logger()


def add_logger() -> None:
    """
    Adds a logger to the script. It will output the logs to the file
    `logs/YYYY-MM-DD_HH-MM-SS.log`.
    """
    os.makedirs("logs", exist_ok=True, mode=0o744)
    # Get date in the format YYYY-MM-DD_HH:MM:SS
    date = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    file_name = f"logs/{date}.log"
    os.environ["TRAIN_STATS"] = file_name

    logger.add(
        file_name,
        level="DEBUG",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Writing logs to '{file_name}'")


def display_gpu_info() -> None:
    """
    Display information about the GPU usage. It uses schedule library, this
    command is ran every 2 minutes during training.
    """
    try:
        free, available = torch.cuda.mem_get_info()
        used = available - free
        gpu_name = torch.cuda.get_device_name(0)
        logger.warning(
            f"{gpu_name}\tvRAM used: {used / 1024 / 1024:.2f} MB\tvRAM available: "
            f"{free / 1_024 / 1_024:.2f}/{available / 1024 / 1024:.2f} MB ({free / available * 100:.2f}%))"
        )

    except Exception as e:
        logger.error(f"Error when getting GPU info: {e}")
