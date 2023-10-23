from datetime import datetime
import os
from loguru import logger


def add_logger() -> None:
    """
    Adds a logger to the script. It will output the logs to the file
    `logs/YYYY-MM-DD_HH-MM-SS.log`.
    """
    os.makedirs("logs", exist_ok=True, mode=0o744)
    # Get date in the format YYYY-MM-DD_HH:MM:SS
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
