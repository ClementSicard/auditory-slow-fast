import time
from datetime import timedelta


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, "%H:%M:%S.%f")
    sec = (
        float(timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds())
        + float(timestamp.split(".")[-1]) / 100
    )
    return sec


class AudioRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def participant(self):
        return NotImplementedError()

    @property
    def untrimmed_video_name(self):
        return NotImplementedError()

    @property
    def start_audio_sample(self):
        return NotImplementedError()

    @property
    def end_audio_sample(self):
        return NotImplementedError()

    @property
    def num_audio_samples(self):
        return NotImplementedError()

    @property
    def label(self):
        return NotImplementedError()

    @property
    def metadata(self):
        return NotImplementedError()
