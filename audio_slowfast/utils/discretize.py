import torch


def discretize(
    x: torch.Tensor,
    low_t: float = -0.5,
    high_t: float = 0.5,
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    """
    Discretize the tensor values to {low, 0, high}.

    Parameters
    ----------
    `x` : `torch.Tensor`
        Input tensor.
    `low_t` : `float`
        Lower threshold.
    `high_t` : `float`
        Upper threshold.
    `low` : `float`
        Lower value.
    `high` : `float`
        Upper value.

    Returns
    -------
    `torch.Tensor`
        Discretized tensor.
    """
    # Define the thresholds
    lower_threshold = low_t
    upper_threshold = high_t

    # Discretize the values
    x_discrete = torch.zeros_like(x)
    x_discrete[x < lower_threshold] = low
    x_discrete[x > upper_threshold] = high

    return x_discrete
