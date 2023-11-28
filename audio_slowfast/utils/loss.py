from typing import List
from loguru import logger
import torch
import torch.nn as nn


class MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Compute the loss for the given predictions and labels.

        Input is the full batch of predictions and labels for state vectors

        preds shape: (B, N, 12), labels shape: (B, N, 12)
        """
        indices_to_keep = labels != -10

        # Create mask for the loss
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[indices_to_keep] = True

        abs_preds = torch.abs(preds)
        abs_labels = torch.abs(labels)

        # Assert all values are in the range [0, 1]
        assert torch.all(abs_preds[mask] <= 1), "abs_preds not in range [0, 1]"

        # Get the indices where the abs_labels are 1
        # pos_mask_indices = abs_labels.nonzero(as_tuple=True)
        pos_mask_indices = torch.zeros_like(abs_labels, dtype=torch.bool)
        pos_mask_indices[abs_labels == 1] = True
        pos_mask_indices *= mask

        bce_term = self.bce(abs_preds[mask], abs_labels[mask])
        mse_term = self.mse(preds[pos_mask_indices], labels[pos_mask_indices])

        result = 0.5 * (bce_term.mean() + mse_term.mean())

        return result
