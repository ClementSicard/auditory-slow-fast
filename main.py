import torch
from typing import Dict, Any
from loguru import logger
import os
import librosa
import pandas as pd
from audio_slowfast import AudioSlowFast

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: Dict[str, Any]) -> None:
    """
    Main function to run inference on Auditory SlowFast model.

    Parameters
    ----------
    `input_path` : `str`
        Path to audio file to run inference on.
    """
    if not os.path.exists(args["input_path"]):
        logger.error(f"Input path {args['input_path']} does not exist.")
        exit(1)
    if not os.path.exists(args["config"]):
        logger.error(f"Config path {args['config']} does not exist.")
        exit(1)
    model = AudioSlowFast(
        checkpoint=args["weights"],
        cfg_file_path=args["config"],
    )

    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    vocab_verb, vocab_noun = model.vocab
    vocab_prec, vocab_postc = model.vocab_prec, model.vocab_postc

    logger.info(f"Loading input audio from {args['input_path']}")

    y, sr = librosa.load(args["input_path"], sr=None)
    spec = model.prepare_audio(y, sr)

    logger.debug(f"Spec shapes: {[x.shape for x in spec]}")
    verb, noun, prec, postc = model([x.to(device) for x in spec])
    i_vs, i_ns, i_pres, i_poss = (
        torch.argmax(verb, dim=-1),
        torch.argmax(noun, dim=-1),
        torch.argmax(prec, dim=-1),
        torch.argmax(postc, dim=-1),
    )

    for v, n, pre, pos, i_v, i_n, i_pre, i_pos in zip(
        verb, noun, prec, postc, i_vs, i_ns, i_pres, i_poss
    ):
        logger.debug(
            f"{vocab_verb[i_v]:>12}: ({v[i_v]:.2%})"
            f"{vocab_noun[i_n]:>12}: ({n[i_n]:.2%})"
            f"{vocab_prec[i_pre]:>12}: ({pre[i_pre]:.2%})"
            f"{vocab_postc[i_pos]:>12}: ({pos[i_pos]:.2%})"
        )


def parse_args() -> Dict[str, Any]:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="input_path",
        help="Path to the input audio file",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--weights",
        dest="weights",
        help="Path to the model weights",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="The path to the config file",
        required=True,
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
