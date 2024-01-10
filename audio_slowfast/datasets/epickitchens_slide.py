import os
from typing import Type

import h5py
import numpy as np
import pandas as pd
import datetime

from loguru import logger
from fvcore.common.file_io import PathManager
from tqdm import tqdm


from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensAudioRecord, timestamp_to_sec
from .epickitchens import EpicKitchens
from fvcore.common.config import CfgNode
from . import utils as utils


@DATASET_REGISTRY.register()
class EpicKitchensSlide(EpicKitchens):
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
            modes=["test"],
        )

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        self.path_annotations_pickle = [
            os.path.join(
                self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                self.cfg.EPICKITCHENS.PROCESSED_TEST_LIST,
            )
        ]

        for file in self.path_annotations_pickle:
            assert PathManager.exists(file), "{} dir not found".format(file)

        self._audio_records = []
        self._temporal_idx = []

        if not self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE and not self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS:
            self._construct_loader_whole_video()
        elif self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS:
            self._construct_loader_action_bounds(per_instance=self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE)
        else:
            raise NotImplementedError("Only whole video mode is supported for now")

    def _construct_loader_whole_video(self) -> None:
        """
        Constructs the datalaoder by sliding over the whole video.
        It works as follows:

        1. Get the video durations
        2. Loop over all videos:
            - For each video, slide over the whole video with a window size of `self.cfg.TEST.SLIDE.WIN_SIZE` and a hop size of `self.cfg.TEST.SLIDE.HOP_SIZE`
            - For each window, create an empty audio record
        3. Load the annotations
            - For each annotation, get the video id and start/stop timestamps
            - Get the annotations for this video and this time window
            - If there are annotations, fill the audio record with the annotation
        """

        logger.info(f"Constructing dataloader for whole video mode")
        # get the video duration
        video_durs = pd.read_csv(
            os.path.join(
                self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                self.cfg.EPICKITCHENS.VIDEO_DURS,
            ),
        )
        video_durs["n_samples"] = video_durs["video_id"].map(lambda x: self.audio_dataset[x].shape[0])

        for file in self.path_annotations_pickle:
            # Load the annotations
            file_df = pd.read_pickle(file)
            file_df = file_df.sort_values(by=["video_id", "start_timestamp", "stop_timestamp"])
            file_df["start_s"] = file_df["start_timestamp"].map(timestamp_to_sec)
            file_df["stop_s"] = file_df["stop_timestamp"].map(timestamp_to_sec)

            # Only evaluate on videos that are in the dataset
            video_durs = video_durs[video_durs["video_id"].isin(file_df["video_id"].unique())]

            # Loop over all videos
            for i, video in tqdm(
                video_durs.iterrows(),
                unit=" video(s)",
                total=video_durs.shape[0],
                desc="Creating empty audio records for all videos",
            ):
                start = 0
                end = self.cfg.TEST.SLIDE.WIN_SIZE

                # Slide over the dataset
                while (start + end) / 2 < video.duration:
                    end = min(end, video.duration)  # TODO: should we keep this or 0-pad instead?
                    ek_ann = {
                        "video_id": video.video_id,
                        "start_timestamp": (datetime.datetime.min + datetime.timedelta(seconds=start)).strftime(
                            "%H:%M:%S.%f"
                        ),
                        "stop_timestamp": (datetime.datetime.min + datetime.timedelta(seconds=end)).strftime(
                            "%H:%M:%S.%f"
                        ),
                    }
                    new_record = EpicKitchensAudioRecord((i, ek_ann), cfg=self.cfg)
                    self._audio_records.append(new_record)
                    self._temporal_idx.append(0)  # TODO: Do we still want this?

                    start += self.cfg.TEST.SLIDE.HOP_SIZE
                    end = start + self.cfg.TEST.SLIDE.WIN_SIZE

            assert len(self._audio_records) > 0, "Failed to load {} split {} from {}".format(
                self.__class__.__name__,
                self.mode,
                self.path_annotations_pickle,
            )

            nb_annotations = 0
            max_overlaps = 0

            # Loop over all audio_records
            for i in tqdm(
                range(len(self._audio_records)),
                unit=" record(s)",
                total=len(self._audio_records),
                desc="Filling empty records with actual annotations",
            ):
                current_record = self._audio_records[i]

                # Get the video_id and start/stop timestamps
                video_id = current_record._series["video_id"]
                start_timestamp = current_record._series["start_timestamp"]
                stop_timestamp = current_record._series["stop_timestamp"]

                middle_frame_in_s = (timestamp_to_sec(start_timestamp) + timestamp_to_sec(stop_timestamp)) / 2

                # Get the annotations for this video
                video_df = file_df[file_df["video_id"] == video_id]

                assert video_df.shape[0] > 0, f"No annotations for {video_id}"

                # Get the annotations for this video and this time window based on the labels for the middle frame
                video_df: pd.DataFrame = video_df[
                    (video_df["start_s"] <= middle_frame_in_s) & (middle_frame_in_s <= video_df["stop_s"])
                ]

                if video_df.shape[0] == 0:
                    continue

                # Deals with overlaps by adding every pair of verb/noun classes.
                # For evaluation, the accuracy will be computed on checking the presence in a zip() of the two lists
                current_record._series["verb_class"] = video_df["verb_class"].to_numpy()
                current_record._series["noun_class"] = video_df["noun_class"].to_numpy()
                current_record._series["participant_id"] = video_df["participant_id"].to_numpy()

                max_overlaps = max(max_overlaps, video_df.shape[0])

                self._audio_records[i] = current_record

        logger.info(
            "Constructing {} dataloader (size: {:,}) from {} with {:,} completed annotations".format(
                self.__class__.__name__,
                len(self._audio_records),
                self.path_annotations_pickle,
                nb_annotations,
            )
        )
        logger.info(f"Max overlap is {max_overlaps}")

    def _construct_loader_action_bounds(self, per_instance: bool = False) -> None:
        """
        Generating the dataloader within action bounds.

        Parameters
        ----------
        `per_instance`: `bool`
            If `True`, we create one record per instance. Otherwise, we create one record per window, sliding the window over the whole annotation duration.
        """
        logger.info(f"Constructing dataloader for action bounds mode" + (" per instance" if per_instance else ""))

        for file in self.path_annotations_pickle:
            # Load the annotations
            file_df = pd.read_pickle(file)

            # Get the timestams in seconds
            file_df["start_s"] = file_df["start_timestamp"].map(timestamp_to_sec)
            file_df["stop_s"] = file_df["stop_timestamp"].map(timestamp_to_sec)

            for i, annotation in tqdm(
                file_df.iterrows() if not self.unique_batch else file_df[: self.cfg.TRAIN.BATCH_SIZE].iterrows(),
                total=file_df.shape[0],
                unit=" annotation(s)",
                desc="Parsing annotations",
            ):
                action_end = annotation["stop_s"]
                start = annotation["start_s"]

                # If the action duration is shorted than the window size, we just add one
                # sample of the full length of the action, as well as the case for
                # per-instance
                if per_instance:
                    new_record = EpicKitchensAudioRecord((i, annotation), cfg=self.cfg)
                    self._audio_records.append(new_record)
                    self._temporal_idx.append(0)

                # Otherwise, we slide a window over the whole action
                else:
                    action_end = annotation["stop_s"]
                    end = start + self.cfg.TEST.SLIDE.WIN_SIZE

                    if action_end - start < self.cfg.TEST.SLIDE.WIN_SIZE:
                        new_record = EpicKitchensAudioRecord((i, annotation), cfg=self.cfg)
                        self._audio_records.append(new_record)
                        self._temporal_idx.append(0)

                    while (start + end) / 2 <= action_end:
                        # Here, we use min of end and action_end for indexing because
                        # 0-padding will happen in the pack_audio function when creating
                        # spectrograms of the right length
                        end = min(end, action_end)

                        new_record = EpicKitchensAudioRecord((i, annotation), cfg=self.cfg)
                        new_record._series["start_timestamp"] = (
                            datetime.datetime.min + datetime.timedelta(seconds=start)
                        ).strftime("%H:%M:%S.%f")
                        new_record._series["stop_timestamp"] = (
                            datetime.datetime.min + datetime.timedelta(seconds=end)
                        ).strftime("%H:%M:%S.%f")

                        self._audio_records.append(new_record)
                        self._temporal_idx.append(0)

                        # Update the start and end of the sliding window
                        start += self.cfg.TEST.SLIDE.HOP_SIZE
                        end = start + self.cfg.TEST.SLIDE.WIN_SIZE

            logger.info(
                "Constructing {} dataloader (size: {:,}) from {}".format(
                    self.__class__.__name__,
                    len(self._audio_records),
                    self.path_annotations_pickle,
                )
            )

    def __len__(self):
        return len(self._audio_records)
