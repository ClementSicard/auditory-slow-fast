from typing import Dict, List, Tuple
from loguru import logger
import torch
import wandb
from wandb import AlertLevel
from fvcore.common.config import CfgNode

from audio_slowfast.models import losses
from audio_slowfast.utils import misc


def _check_prediction(pred: torch.Tensor, threshold: float = 0.1) -> bool:
    return torch.all(torch.abs(pred) <= threshold)


def check_predictions(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.1):
    """
    Check if the state predictions are within the threshold and actually look like something.

    Parameters
    ----------
    `preds`: `torch.Tensor`
        The predictions from the model.

    `labels`: `torch.Tensor`
        The labels for the predictions.

    `threshold`: `float`
        The threshold to check the predictions against.
    """
    if _check_prediction(pred=preds[2], threshold=threshold):
        text = f"State < 0.1\n\nPreds:{preds[2]}\nLabels:{labels[2]}"
        logger.warning(text)
        wandb.alert(
            title="State looking strange",
            text=text,
            level=AlertLevel.WARN,
        )


def prepare_state_labels(preds, labels, lengths) -> torch.Tensor:
    """
    Returns the state labels for the given predictions and labels.
    It creates labels as follows:

    1. Sets value longer than audio segment length to -10
    2. Sets values up to length // 2 to the precondition label
    3. Sets values after length // 2 to length to the postcondition label
    """
    try:
        B, N, P = preds[2].shape
    except Exception as e:
        logger.debug(f"{preds[2].shape}")
        logger.warning(f"{[p.shape for p in preds]}")
        raise e

    state = labels["posts"].clone().unsqueeze(1).repeat(1, N, 1)

    for i, length in enumerate(lengths):
        state[i, length:, :] = -10
        state[i, : length // 2] = labels["precs"][i]

    return state


def compute_loss(
    verb_preds: torch.Tensor,
    noun_preds: torch.Tensor,
    state_preds: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    lengths: List[int],
    cfg: CfgNode,
) -> Tuple[torch.Tensor, ...]:
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    masked_loss_fun = losses.get_loss_func(cfg.MODEL.STATE_LOSS_FUNC)(reduction="mean")

    loss_verb = loss_fun(verb_preds, labels["verb"])
    loss_noun = loss_fun(noun_preds, labels["noun"])
    loss_state = masked_loss_fun(state_preds, labels["state"])

    # Use torch.mean to average the losses over all GPUs for logging purposes.
    loss_vec = torch.stack(
        [
            loss_verb,
            loss_noun,
            loss_state,
        ]
    )

    loss = torch.mean(loss_vec)

    # check Nan Loss.
    misc.check_nan_losses(loss)

    return loss, loss_verb, loss_noun, loss_state
