from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensAudioRecord, timestamp_to_sec
from .epickitchens import EpicKitchens
from fvcore.common.config import CfgNode
from . import utils as utils
import os

from fvcore.common.file_io import PathManager

import pandas as pd
from loguru import logger
from tqdm import tqdm


@DATASET_REGISTRY.register()
class EpicKitchensInterES(EpicKitchens):
    def __init__(
        self,
        cfg: CfgNode,
        mode: str,
    ):
        super().__init__(
            cfg=cfg,
            mode=mode,
            record_type=EpicKitchensAudioRecord,
            gru_format=False,
            modes=["train", "val", "test"],
        )

    def _construct_loader(self):
        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.PROCESSED_TRAIN_LIST,
                )
            ]
            es_path = self.cfg.EPICSOUNDS.TRAIN_LIST
        elif self.mode == "val":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.PROCESSED_VAL_LIST,
                )
            ]
            es_path = self.cfg.EPICSOUNDS.VAL_LIST
        elif self.mode == "test":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.PROCESSED_TEST_LIST,
                )
            ]
            es_path = self.cfg.EPICSOUNDS.TEST_LIST

        else:
            raise ValueError("Unknown mode {}".format(self.mode))

        for file in path_annotations_pickle:
            assert PathManager.exists(file), "{} dir not found".format(file)

        assert PathManager.exists(es_path), "{} dir not found".format(es_path)

        self._audio_records = []
        self._temporal_idx = []

        es_file_df = pd.read_csv(es_path, header=0)
        filtered_duration = 0

        for file in path_annotations_pickle:
            file_df = pd.read_pickle(file)
            file_df["start_s"] = file_df["start_timestamp"].apply(timestamp_to_sec)
            file_df["stop_s"] = file_df["stop_timestamp"].apply(timestamp_to_sec)
            file_df["duration"] = file_df["stop_s"] - file_df["start_s"]
            logger.warning(f"Total dataset duration is {file_df['duration'].sum():,} seconds")

            for i, action in tqdm(
                file_df.iterrows() if not self.unique_batch else file_df[: self.cfg.TRAIN.BATCH_SIZE].iterrows(),
                total=file_df.shape[0],
                unit=" annotation(s)",
                desc="Getting intersection between EK-100 and ES",
            ):
                # Check if sound annotations are available for these timestamps
                # If not, skip the annotation
                s_k, e_k = action["start_timestamp"], action["stop_timestamp"]
                video_id = action["video_id"]

                # Get éoose overlaps
                corresponding_sounds = es_file_df[es_file_df["video_id"] == video_id]
                overlapping_sounds = corresponding_sounds[
                    (corresponding_sounds["start_timestamp"] <= e_k) & (s_k <= corresponding_sounds["stop_timestamp"])
                ]

                if overlapping_sounds.shape[0] == 0:
                    continue

                for _, overlapping_sound in overlapping_sounds.iterrows():
                    copy = action.copy()
                    copy["start_timestamp"] = max(
                        s_k,
                        overlapping_sound["start_timestamp"],
                    )
                    copy["stop_timestamp"] = min(
                        e_k,
                        overlapping_sound["stop_timestamp"],
                    )

                    filtered_duration += timestamp_to_sec(copy["stop_timestamp"]) - timestamp_to_sec(
                        copy["start_timestamp"]
                    )

                    for idx in range(self._num_clips):
                        self._audio_records.append(
                            self.record_type((i, copy), cfg=self.cfg),
                        )
                        self._temporal_idx.append(idx)

        assert len(self._audio_records) > 0, "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing {} dataloader (size: {:,}) from {} for a total duration of {:,} seconds".format(
                self.__class__.__name__,
                len(self._audio_records),
                path_annotations_pickle,
                filtered_duration,
            )
        )
