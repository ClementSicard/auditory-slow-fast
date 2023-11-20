import sys
from loguru import logger
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from audio_slowfast.config.defaults import get_cfg
import audio_slowfast.datasets.loader as loader
import audio_slowfast.datasets.utils as utils


def test_dataloader() -> None:
    cfg = get_cfg()
    cfg.merge_from_file("models/asf/config/SLOWFAST_R50.yaml")

    cfg.DATA_LOADER.NUM_WORKERS = 4
    cfg.TRAIN.BATCH_SIZE = 2

    train_loader: DataLoader = loader.construct_loader(cfg, "train")

    # Expects [(batch_size, 1, T_slow, F), (batch_size, 1, T_fast, F)]
    C = 1
    F = cfg.AUDIO_DATA.NUM_FREQUENCIES
    T_fast = cfg.AUDIO_DATA.NUM_FRAMES
    T_slow = cfg.AUDIO_DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA
    CLIP_EMBEDDING_SIZE = 512

    for batch in train_loader:
        logger.warning(batch)
        # for i, b in enumerate(batch):
        #     match i:
        #         case 0:
        #             logger.warning(f"Spectrogram shape: {[bb.shape for bb in b]}")

        #             assert len(b) == 2, f"Expected 2 spectrograms, got {len(b)}"

        #             assert all([len(bb.shape) == 5 for bb in b]), f"Expected 5D tensors, got {[bb.shape for bb in b]}"

        #             assert (
        #                 b[0].shape[0] == b[1].shape[0] == cfg.TRAIN.BATCH_SIZE
        #             ), f"Expected batch size {cfg.TRAIN.BATCH_SIZE}, got {b[0].shape[0]} and {b[1].shape[0]}"

        #             assert (
        #                 b[0].shape[1] == b[1].shape[1]
        #             ), f"Expected list of spectrograms to be of the same size, got {b[0].shape[1]} and {b[1].shape[1]}"

        #             assert (
        #                 b[0].shape[2] == C == b[1].shape[2]
        #             ), f"Expected spectrograms to have {C} channels, got {b[0].shape[2]} and {b[1].shape[2]}"

        #             assert (
        #                 b[0].shape[3] == T_slow
        #             ), f"Expected slow spectrogram to have {T_slow} frames, got {b[0].shape[3]}"

        #             assert (
        #                 b[1].shape[3] == T_fast
        #             ), f"Expected fast spectrogram to have {T_fast} frames, got {b[1].shape[3]}"

        #             assert (
        #                 b[0].shape[4] == b[1].shape[4] == F
        #             ), f"Expected spectrograms to have {F} frequencies, got {b[0].shape[4]} and {b[1].shape[4]}"

        #         case 1:
        #             logger.warning(f"Lengths: {b}")

        #         case 2:
        #             logger.warning(f"Labels: {b}")

        #         case 3:
        #             logger.warning(f"Indices: {b}")

        #         case 4:
        #             logger.warning(f"Noun embedding: {b.shape}")

        #             assert b.shape == (
        #                 cfg.TRAIN.BATCH_SIZE,
        #                 CLIP_EMBEDDING_SIZE,
        #             ), f"Expected shape {(cfg.TRAIN.BATCH_SIZE, CLIP_EMBEDDING_SIZE)}, got {b.shape}"

        #         case 5:
        #             logger.warning(f"Metadata: {b}")

        break


if __name__ == "__main__":
    test_dataloader()
