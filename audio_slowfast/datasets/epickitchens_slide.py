import os
from typing import Type

import h5py
import pandas as pd
import torch
import torch.utils.data
import numpy as np
import datetime

from loguru import logger
from fvcore.common.file_io import PathManager


from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensAudioRecord, timestamp_to_sec
from .epickitchens import EpicKitchens
from fvcore.common.config import CfgNode
from . import utils as utils
from .audio_loader_epic import pack_audio
from tqdm import tqdm


@DATASET_REGISTRY.register()
class EpicKitchensSlide(EpicKitchens):
    def __init__(self, cfg: CfgNode, mode: str, record_type: Type[EpicKitchensAudioRecord] = EpicKitchensAudioRecord):
        assert mode in ["test"], "Split '{}' not supported for {}".format(mode, self.__class__.__name__)
        self.cfg = cfg
        self.mode = mode
        self.record_type = record_type

        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS if not "GRU" in cfg.TEST.DATASET else 1

        self.audio_dataset = None
        logger.info("Constructing {} Audio {}...".format(self.__class__.__name__, mode))
        self.unique_batch = cfg.EPICKITCHENS.SINGLE_BATCH
        if self.unique_batch:
            logger.warning("Using a SINGLE batch for debugging.")

        self._construct_loader(cfg)

    def _construct_loader(self, cfg: CfgNode):
        """
        Construct the video loader.
        """
        logger.error(f"EPIC-KITCHENS for class {self.__class__.__name__}")
        path_annotations_pickle = [
            os.path.join(
                self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                self.cfg.EPICKITCHENS.PROCESSED_TEST_LIST,
            )
        ]

        for file in path_annotations_pickle:
            assert PathManager.exists(file), "{} dir not found".format(file)

        self._audio_records = []
        self._temporal_idx = []
        iii = 0

        # get the video duration
        video_durs = pd.read_csv(
            os.path.join(
                self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                self.cfg.EPICKITCHENS.VIDEO_DURS,
            )
        )
        video_durs = dict(zip(video_durs["video_id"], video_durs["duration"]))

        if not self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE:
            video_times = {}

        for file in path_annotations_pickle:
            file_df = pd.read_pickle(file)

            for tup in tqdm(file_df.iterrows(), unit="row", total=len(file_df)):
                # get the action start and end
                action_start_sec = timestamp_to_sec(tup[1]["start_timestamp"])
                action_stop_sec = timestamp_to_sec(tup[1]["stop_timestamp"])

                if self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS:
                    win_start_sec = self.cfg.TEST.SLIDE.HOP_SIZE * np.ceil(
                        action_start_sec / self.cfg.TEST.SLIDE.HOP_SIZE
                    )
                    win_stop_sec = win_start_sec + self.cfg.TEST.SLIDE.WIN_SIZE
                    win_label_sec = win_start_sec + self.cfg.TEST.SLIDE.LABEL_FRAME * self.cfg.TEST.SLIDE.WIN_SIZE
                else:
                    win_label_sec = self.cfg.TEST.SLIDE.HOP_SIZE * np.ceil(
                        action_start_sec / self.cfg.TEST.SLIDE.HOP_SIZE
                    )
                    win_start_sec = win_label_sec - self.cfg.TEST.SLIDE.LABEL_FRAME * self.cfg.TEST.SLIDE.WIN_SIZE
                    win_stop_sec = win_label_sec + (1 - self.cfg.TEST.SLIDE.LABEL_FRAME) * self.cfg.TEST.SLIDE.WIN_SIZE
                win_start_sec = win_start_sec if win_start_sec > 0.0 else 0.0
                win_label_sec = win_label_sec if win_label_sec > 0.0 else 0.0

                while (win_label_sec <= action_stop_sec) if not self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS else True:
                    ek_ann = tup[1].copy()
                    ek_ann["start_timestamp"] = (
                        datetime.datetime.min + datetime.timedelta(seconds=win_start_sec)
                    ).strftime("%H:%M:%S.%f")
                    ek_ann["stop_timestamp"] = (
                        datetime.datetime.min + datetime.timedelta(seconds=win_stop_sec)
                    ).strftime("%H:%M:%S.%f")
                    # TODO: FOR AUDIO, REPLACE TARGET FPS BY AUDIO FRAMES PER SECOND
                    ek_ann["action_stop_frame"] = action_stop_sec * cfg.AUDIO_DATA.SAMPLING_RATE

                    if not self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE:
                        # up to three sim actions per frame in validation set
                        ek_ann["verb_class"] = [ek_ann["verb_class"]] * 3
                        ek_ann["noun_class"] = [ek_ann["noun_class"]] * 3
                        ek_ann["verb"] = [ek_ann["verb"]] * 3
                        ek_ann["noun"] = [ek_ann["noun"]] * 3
                        video_timestamp = f'{ek_ann["video_id"]}_{win_label_sec:.2f}'
                        if video_timestamp not in video_times:
                            video_times[video_timestamp] = iii
                            self._audio_records.append(EpicKitchensAudioRecord((tup[0], ek_ann), cfg=cfg))
                            self._audio_records[-1].time_end = video_durs[ek_ann["video_id"]]
                            self._temporal_idx.append(0)
                            iii += 1
                        else:
                            video_record = self._audio_records[video_times[video_timestamp]]
                            unique_verb_noun = list(
                                set(zip(video_record._series["verb_class"], video_record._series["noun_class"]))
                            )
                            if (ek_ann["verb"], ek_ann["noun"]) not in unique_verb_noun:
                                video_record._series["verb_class"][len(unique_verb_noun)] = ek_ann["verb_class"][0]
                                video_record._series["noun_class"][len(unique_verb_noun)] = ek_ann["noun_class"][0]
                                video_record._series["verb"][len(unique_verb_noun)] = ek_ann["verb"][0]
                                video_record._series["noun"][len(unique_verb_noun)] = ek_ann["noun"][0]
                            self._audio_records[video_times[video_timestamp]] = video_record
                        if win_stop_sec > action_stop_sec and self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS:
                            break
                        win_stop_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                        win_label_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                        win_start_sec = win_stop_sec - self.cfg.TEST.SLIDE.WIN_SIZE
                        win_start_sec = win_start_sec if win_start_sec > 0.0 else 0.0
                        win_label_sec = win_label_sec if win_label_sec > 0.0 else 0.0
                        continue
                    self._audio_records.append(EpicKitchensAudioRecord((tup[0], ek_ann), cfg=cfg))
                    self._audio_records[-1].time_end = video_durs[ek_ann["video_id"]]
                    self._temporal_idx.append(0)
                    if win_stop_sec > action_stop_sec and self.cfg.TEST.SLIDE.INSIDE_ACTION_BOUNDS:
                        break
                    win_stop_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                    win_label_sec += self.cfg.TEST.SLIDE.HOP_SIZE
                    win_start_sec = win_stop_sec - self.cfg.TEST.SLIDE.WIN_SIZE
                    win_start_sec = win_start_sec if win_start_sec > 0.0 else 0.0
                    win_label_sec = win_label_sec if win_label_sec > 0.0 else 0.0

        if self.cfg.TEST.SLIDE.ENABLE and not self.cfg.TEST.SLIDE.PER_ACTION_INSTANCE:
            for iii in range(len(self._audio_records)):
                self._audio_records[iii]._series["noun_class"] = np.array(
                    self._audio_records[iii]._series["noun_class"]
                )
                self._audio_records[iii]._series["verb_class"] = np.array(
                    self._audio_records[iii]._series["verb_class"]
                )

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
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = self._temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        frames = pack_frames_to_video_clip(self.cfg, self._audio_records[index], temporal_sample_index)

        if self.cfg.MODEL.MODEL_NAME == "SlowFast":
            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            frames = frames / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )

            label = self._audio_records[index].label
            frames = utils.pack_pathway_output(self.cfg, frames)
            metadata = self._audio_records[index].metadata
            return frames, label, index, metadata

    def __len__(self):
        return len(self._audio_records)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            # assert len({min_scale, max_scale, crop_size}) == 1
            # frames, _ = transform.random_short_side_scale_jitter(
            #    frames, min_scale, max_scale
            # )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
