import time
from datetime import timedelta

from audio_slowfast.datasets.utils import timestamp_to_sec
from .audio_record import AudioRecord
from fvcore.common.config import CfgNode


class EpicKitchensAudioRecord(AudioRecord):
    def __init__(self, tup, cfg: CfgNode):
        self.cfg = cfg
        self._index = str(tup[0])
        self._series = tup[1]
        self._sampling_rate = cfg.AUDIO_DATA.SAMPLING_RATE

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
            # "verb": self._series["verb_class"] if "verb_class" in self._series else -1,
            "verb": self._series["verb_class"],
            # "noun": self._series["noun_class"] if "noun_class" in self._series else -1,
            "noun": self._series["noun_class"],
        }

    @property
    def metadata(self):
        return {"narration_id": self._index}
