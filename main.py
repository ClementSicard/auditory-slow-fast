import argparse
import json
import os
from typing import Any, Dict, List
from time import sleep
import librosa
import pandas as pd
import torch
import yaml
from loguru import logger
from audio_slowfast.models.build import build_model

import src.utils
from audio_slowfast import test, train
from audio_slowfast.config.defaults import get_cfg
from audio_slowfast.utils.discretize import discretize
from audio_slowfast.utils.misc import launch_job
from src.dataset import prepare_dataset
from src.pddl import Predicate


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration file.

    Parameters
    ----------
    `config_path` : `str`
        The path to the configuration file.

    Returns
    -------
    `Dict[str, Any]`
        The configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parse_args() -> Dict[str, Any]:
    """
    Parse the arguments passed to the script.

    Returns
    -------
    `Dict[str, Any]`
        The arguments passed to the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--example", type=str, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    return vars(args)


def validate_args(args: Dict[str, Any]) -> None:
    """
    Validate the arguments passed to the script.

    Parameters
    ----------
    `args` : `Dict[str, Any]`
        The arguments passed to the script.
    """
    logger.debug(f"Arguments:\n{json.dumps(args, indent=4)}")

    if not os.path.exists(args["config"]):
        logger.error(f"Config file {args['config']} does not exist")
        exit(1)

    if args["example"] and not os.path.exists(args["example"]):
        logger.error(f"Example file {args['example']} does not exist")
        exit(1)


def main(args: Dict[str, Any]) -> None:
    """
    Main function of the script.

    Parameters
    ----------
    `args` : `Dict[str, Any]`
        The arguments passed to the script.

    Raises
    ------
    `ValueError`
        In case the povided model is not supported.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args["config"])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args["train"]:
        if not torch.cuda.is_available():
            logger.warning("No GPU found. Running on CPU. Also deactivating W&B reports.")

            # Modify config for debug training
            cfg.NUM_GPUS = 0
            cfg.WANDB.ENABLE = False
            cfg.TENSORBOARD.ENABLE = False
            cfg.DATA_LOADER.NUM_WORKERS = 4
            cfg.TRAIN.BATCH_SIZE = 2

        # Prepare the dataset
        if not cfg.EPICKITCHENS.SKIP_PREPARATION:
            prepare_dataset(cfg=cfg)
        else:
            if not os.path.exists(cfg.EPICKITCHENS.PROCESSED_TRAIN_LIST):
                logger.error(f"Train list {cfg.EPICKITCHENS.PROCESSED_TRAIN_LIST} does not exist")
                exit(1)
            if not os.path.exists(cfg.EPICKITCHENS.PROCESSED_VAL_LIST):
                logger.error(f"Val list {cfg.EPICKITCHENS.PROCESSED_VAL_LIST} does not exist")
                exit(1)

        sleep(1)
        launch_job(cfg=cfg, init_method=None, func=train)
        launch_job(cfg=cfg, init_method=None, func=test)

    elif args["test"]:
        launch_job(cfg=cfg, init_method=None, func=test)


def example(
    model: torch.nn.Module,
    attributes: List[str],
    file_path: str,
    device: str = "cuda",
) -> None:
    logger.warning(model)
    model.eval()
    vocab_verb, vocab_noun = model.vocab

    logger.info(f"Loading input audio from {args['example']}")

    y, sr = librosa.load(file_path, sr=24_000)
    spec = model.prepare_audio(y, sr)
    logger.debug(f"Spec shapes: {[x.shape for x in spec]}")

    verb, noun, prec, postc = model([x.to(device) for x in spec])

    i_vs, i_ns = (
        torch.argmax(verb, dim=-1),
        torch.argmax(noun, dim=-1),
    )
    i_pres, i_poss = prec[0], postc[0]
    logger.info(f"{i_pres=}")
    logger.info(f"{i_poss=}")
    logger.debug(f"Discrete pre: {discretize(i_pres)}")
    logger.debug(f"Discrete posts: {discretize(i_poss)}")

    for v, n, _, _, i_v, i_n, i_pre, i_pos in zip(verb, noun, prec, postc, i_vs, i_ns, prec, postc):
        logger.debug(
            f"\nVerb: {vocab_verb[i_v]} ({v[i_v]:.2%})\n"
            f"Noun: {vocab_noun[i_n]} ({n[i_n]:.2%})\n"
            f"Preconditions: {Predicate.predicates_from_vector(vector=discretize(i_pre), attributes=attributes, to_str=True,)}\n"
            f"Postconditions: {Predicate.predicates_from_vector(vector=discretize(i_pos), attributes=attributes, to_str=True,)}\n"
        )


if __name__ == "__main__":
    src.utils.setup_run()
    args = parse_args()
    validate_args(args=args)
    main(args=args)
