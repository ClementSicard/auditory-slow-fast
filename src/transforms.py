from typing import Dict

from audiomentations import AddGaussianNoise, PitchShift, PolarityInversion, TimeStretch
from audiomentations.core.transforms_interface import BaseWaveformTransform


def get_transforms(p: float = 1.0) -> Dict[str, BaseWaveformTransform]:
    """
    This function returns a list of transformations to augment dataset.

    Returns
    -------
    `List[BaseWaveformTransform]`
        The list of transforms.
    """
    assert p <= 1.0, f"{p=} must be smaller than 1.0"
    return {
        "polarity_inversion": PolarityInversion(p=p),
        "gaussian_noise": AddGaussianNoise(p=p),
        # "time_stretch": TimeStretch(p=p),
        "pitch_shift": PitchShift(p=p),
    }
