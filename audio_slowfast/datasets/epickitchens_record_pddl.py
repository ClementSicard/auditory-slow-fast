import time
from datetime import timedelta

import numpy as np

from .audio_record import AudioRecord, timestamp_to_sec
from .utils import get_num_spectrogram_frames
from fvcore.common.config import CfgNode


class EpicKitchensAudioRecordWithPDDL(AudioRecord):
    def __init__(self, tup, cfg: CfgNode):
        self.cfg = cfg
        self._index = str(tup[0])
        self._series = tup[1]
        self._sampling_rate = cfg.AUDIO_DATA.SAMPLING_RATE
        self._spectrogram_overlap = cfg.AUDIO_DATA.SPECTROGRAM_OVERLAP
        self._num_overlap_frames = get_num_spectrogram_frames(self._spectrogram_overlap, self.cfg)

    @property
    def participant(self):
        return self._series["participant_id"]

    @property
    def untrimmed_video_name(self):
        return self._series["video_id"]

    @property
    def start_audio_sample(self):
        return int(round(timestamp_to_sec(self._series["start_timestamp"]) * self._sampling_rate))

    @property
    def end_audio_sample(self):
        return int(round(timestamp_to_sec(self._series["stop_timestamp"]) * self._sampling_rate))

    @property
    def num_audio_samples(self):
        return self.end_audio_sample - self.start_audio_sample

    @property
    def transformation(self):
        return self._series["transformation"] if "transformation" in self._series else "none"

    @property
    def label(self):
        return {
            "verb": self._series["verb_class"],
            "noun": self._series["noun_class"],
            "precs": self._series["precs_vec"],
            "posts": self._series["posts_vec"],
        }

    @property
    def noun_embedding(self):
        # (1, 512) -> (512,)
        return self._series["noun_embedding"].reshape(-1)

    @property
    def metadata(self):
        return {"narration_id": self._index}
