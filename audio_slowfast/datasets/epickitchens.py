import os

import h5py
import pandas as pd
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
from loguru import logger

from src.transforms import get_transforms

from . import utils as utils
from .audio_loader_epic import pack_audio
from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensAudioRecord
from .spec_augment import combined_transforms


@DATASET_REGISTRY.register()
class Epickitchens(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        assert mode in [
            "train",
            "val",
            "test",
            "train+val",
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        self.audio_dataset = None
        logger.info("Constructing EPIC-KITCHENS Audio {}...".format(mode))
        self._construct_loader()

        self.transforms = get_transforms()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.TRAIN_LIST,
                )
            ]
        elif self.mode == "val":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.VAL_LIST,
                )
            ]
        elif self.mode == "test":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.TEST_LIST,
                )
            ]
        else:
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                for file in [
                    self.cfg.EPICKITCHENS.TRAIN_LIST,
                    self.cfg.EPICKITCHENS.VAL_LIST,
                ]
            ]

        for file in path_annotations_pickle:
            assert PathManager.exists(file), "{} dir not found".format(file)

        self._audio_records = []
        self._temporal_idx = []
        for file in path_annotations_pickle:
            file_df = pd.read_pickle(file)
            for tup in file_df.iterrows():
                for idx in range(self._num_clips):
                    self._audio_records.append(EpicKitchensAudioRecord(tup))
                    self._temporal_idx.append(idx)
        assert len(self._audio_records) > 0, "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._audio_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the audio index, return the spectrogram, label, audio
        index, and metadata.
        Args:
            index (int): the audio index provided by the pytorch sampler.
        Returns:
            spectrogram (tensor): the spectrogram sampled from the audio. The dimension
                is `channel` x `num frames` x `num frequencies`.
            label (int): the label of the current audio.
            index (int): Return the index of the audio.
        """
        if self.audio_dataset is None:
            self.audio_dataset = h5py.File(self.cfg.EPICKITCHENS.AUDIO_DATA_FILE, "r")

        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        transformation = self._audio_records[index].transformation

        if transformation is not None:
            logger.debug(f"Transforming spectrogram with {transformation}: {self.transforms[transformation]}")

        spectrogram = pack_audio(
            cfg=self.cfg,
            audio_dataset=self.audio_dataset,
            audio_record=self._audio_records[index],
            temporal_sample_index=temporal_sample_index,
            transform=self.transforms[transformation] if transformation is not None else None,
        )

        logger.warning(f"spectrogram shape: {spectrogram.shape}")
        # Normalization.
        spectrogram = spectrogram.float()

        if self.mode in ["train", "train+val"]:
            # Data augmentation.
            # C T F -> C F T
            spectrogram = spectrogram.permute(0, 2, 1)
            # SpecAugment
            spectrogram = combined_transforms(spectrogram)
            # C F T -> C T F
            spectrogram = spectrogram.permute(0, 2, 1)
        label = self._audio_records[index].label
        spectrogram = utils.pack_pathway_output(self.cfg, spectrogram)
        metadata = self._audio_records[index].metadata

        return spectrogram, label, index, metadata

    def __len__(self):
        return len(self._audio_records)
