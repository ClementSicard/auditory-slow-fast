import sys
from loguru import logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.config.defaults import get_cfg
import audio_slowfast.datasets.loader as loader
import audio_slowfast.datasets.utils as utils
from fvcore.common.config import CfgNode

CONFIG_PATH = "models/asf/config/SLOWFAST_R50.yaml"


def test_dataloader() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)

    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.TRAIN.BATCH_SIZE = 1

    train_loader: DataLoader = loader.construct_loader(cfg, "train")

    if "gru" in cfg.TRAIN.DATASET.lower():
        gru_dataloader(train_loader, cfg)
    else:
        regular_dataloader(train_loader, cfg)

    logger.success(f"DataLoader seems to work as expected! 🥳")


def gru_dataloader(train_loader: DataLoader, cfg: CfgNode):
    # Expects [(batch_size, 1, T_slow, F), (batch_size, 1, T_fast, F)]
    C = 1
    F = cfg.AUDIO_DATA.NUM_FREQUENCIES
    T_fast = cfg.AUDIO_DATA.NUM_FRAMES
    T_slow = cfg.AUDIO_DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA
    CLIP_EMBEDDING_SIZE = 512

    for _, batch in enumerate(
        tqdm(
            train_loader,
            desc=f"Testing {train_loader.dataset.__class__.__name__} dataloader",
            total=len(train_loader),
            unit="batch",
        ),
    ):
        for j, b in enumerate(batch):
            match j:
                case 0:
                    assert len(b) == 2, f"Expected 2 spectrograms, got {len(b)}"

                    assert all([len(bb.shape) == 5 for bb in b]), f"Expected 5D tensors, got {[bb.shape for bb in b]}"

                    assert (
                        b[0].shape[0] == b[1].shape[0] == cfg.TRAIN.BATCH_SIZE
                    ), f"Expected batch size {cfg.TRAIN.BATCH_SIZE}, got {b[0].shape[0]} and {b[1].shape[0]}"

                    assert (
                        b[0].shape[1] == b[1].shape[1]
                    ), f"Expected list of spectrograms to be of the same size, got {b[0].shape[1]} and {b[1].shape[1]}"

                    assert (
                        b[0].shape[2] == C == b[1].shape[2]
                    ), f"Expected spectrograms to have {C} channels, got {b[0].shape[2]} and {b[1].shape[2]}"

                    assert (
                        b[0].shape[3] == T_slow
                    ), f"Expected slow spectrogram to have {T_slow} frames, got {b[0].shape[3]}"

                    assert (
                        b[1].shape[3] == T_fast
                    ), f"Expected fast spectrogram to have {T_fast} frames, got {b[1].shape[3]}"

                    assert (
                        b[0].shape[4] == b[1].shape[4] == F
                    ), f"Expected spectrograms to have {F} frequencies, got {b[0].shape[4]} and {b[1].shape[4]}"

                case 1:
                    assert all([bb >= 1 for bb in b]), f"Expected at least 1 label, got {b}"

                # case 2:
                #     logger.warning(f"Labels: {b}")

                # case 3:
                #     logger.warning(f"Indices: {b}")

                case 4:
                    assert b.shape == (
                        cfg.TRAIN.BATCH_SIZE,
                        CLIP_EMBEDDING_SIZE,
                    ), f"Expected shape {(cfg.TRAIN.BATCH_SIZE, CLIP_EMBEDDING_SIZE)}, got {b.shape}"

                # case 5:
                #     logger.warning(f"Metadata: {b}")


def regular_dataloader(train_loader: DataLoader, cfg: CfgNode):
    for _, __ in enumerate(
        tqdm(
            train_loader,
            desc=f"Testing {train_loader.dataset.__class__.__name__} dataloader",
            total=len(train_loader),
            unit="batch",
        ),
    ):
        continue


if __name__ == "__main__":
    test_dataloader()
