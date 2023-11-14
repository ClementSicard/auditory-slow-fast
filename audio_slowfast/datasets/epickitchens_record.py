import time
from datetime import timedelta

import numpy as np

from .audio_record import AudioRecord
from .utils import get_num_spectrogram_frames
from fvcore.common.config import CfgNode


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, "%H:%M:%S.%f")
    sec = (
        float(timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds())
        + float(timestamp.split(".")[-1]) / 100
    )
    return sec


class EpicKitchensAudioRecord(AudioRecord):
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
    def length_in_s(self):
        return self.num_audio_samples / self._sampling_rate

    @property
    def transformation(self):
        return self._series["transformation"]

    @property
    def num_spectrograms(self):
        """
        Calculate the number of `cfg.AUDIO_DATA.CLIP_SECS`-second spectrograms needed for the audio segment
        with a `cfg.AUDIO_DATA.SPECTROGRAM_OVERLAP`-second overlap between consecutive spectrograms.

        Also include
        """
        return int(
            np.ceil(
                max(
                    (self.length_in_s - self._spectrogram_overlap)
                    / (self.cfg.AUDIO_DATA.CLIP_SECS - self._spectrogram_overlap),
                    1,
                )
            )
        )

    @property
    def label(self):
        return {
            "verb": self._series["verb_class"] if "verb_class" in self._series else -1,
            "noun": self._series["noun_class"] if "noun_class" in self._series else -1,
            "precs": self._series["precs_vec"] if "precs_vec" in self._series else -1,
            "posts": self._series["posts_vec"] if "posts_vec" in self._series else -1,
        }

    @property
    def metadata(self):
        return {"narration_id": self._index}
