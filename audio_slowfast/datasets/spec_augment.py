import random
from loguru import logger

import torch

from .sparse_image_warp import sparse_image_warp


def time_warp(spec: torch.Tensor, W: int = 5) -> torch.Tensor:
    """
    Warps the input spectrogram along the time axis in the range (0, W).

    Parameters
    ----------
    `spec`: `torch.Tensor`
        The input spectrogram.

    `W`: `int`, optional (default = 5)
        The maximum width of each time warp.

    Returns
    -------
    `torch.Tensor`
        The warped spectrogram.
    """

    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (
        torch.tensor([[[y, point_to_warp]]], device=device),
        torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device),
    )
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def freq_mask(
    spec: torch.Tensor,
    F: int = 27,
    num_masks: int = 1,
    replace_with_zero: bool = False,
):
    """
    Creates a mask of size (num_masks, F), where F is the number of mel frequency channels,

    Parameters
    ----------
    `spec`: `torch.Tensor`
        The input spectrogram.

    `F`: `int`, optional (default = 27)
        The maximum width of each frequency mask.

    `num_masks`: `int`, optional (default = 1)
        The number of frequency masks to apply.

    `replace_with_zero`: `bool`, optional (default = `False`)
        Whether to replace the masked region with zeros or the mean value of the spectrogram.

    Returns
    -------
    `torch.Tensor`
        The masked spectrogram.
    """
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()  # Mask == mean spec value

    return cloned


def time_mask(
    spec: torch.Tensor,
    T: int = 25,
    num_masks: int = 1,
    replace_with_zero: bool = False,
) -> torch.Tensor:
    """
    Create a mask of size T x num_masks, where T is the number of time steps
    and num_masks is the number of masks to apply.

    Parameters
    ----------
    `spec`: `torch.Tensor`
        The input spectrogram.

    `T`: `int`, optional (default = 25)
        The maximum width of each time mask.

    `num_masks`: `int`, optional (default = 1)
        The number of time masks to apply.

    `replace_with_zero`: `bool`, optional (default = `False`)
        Whether to replace the masked region with zeros or the mean value of the spectrogram.

    Returns
    -------
    `torch.Tensor`
        The masked spectrogram.
    """
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if replace_with_zero:
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()  # Mask == mean spec value
    return cloned


def spec_augment(
    spec: torch.Tensor,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    F: int = 27,
    T: int = 25,
    W: int = 5,
) -> torch.Tensor:
    """
    Performs SpecAugment (https://arxiv.org/abs/1904.08779) on the input spectrogram.

    Parameters
    ----------
    `spec`: `torch.Tensor`
        The input spectrogram.

    `num_freq_masks`: `int`, optional (default = 2)
        The number of frequency masks to apply.

    `num_time_masks`: `int`, optional (default = 2)
        The number of time masks to apply.

    `F`: `int`, optional (default = 27)
        The maximum width of each frequency mask.

    `T`: `int`, optional (default = 25)
        The maximum width of each time mask.

    `W`: `int`, optional (default = 5)
        The maximum width of each time warp.

    Returns
    -------
    `torch.Tensor`
        The masked spectrogram.
    """
    return time_mask(
        spec=freq_mask(
            spec=time_warp(
                spec=spec,
                W=W,
            ),
            F=F,
            num_masks=num_freq_masks,
        ),
        T=T,
        num_masks=num_time_masks,
    )
