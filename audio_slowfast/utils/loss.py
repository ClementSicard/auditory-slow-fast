from typing import List
import torch
import torch.nn as nn


class MaskedLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Compute the loss for the given predictions and labels.
        """
        abs_preds = torch.abs(preds)
        abs_labels = torch.abs(labels)

        # Get the indices where the abs_labels are 1
        pos_mask_indices = abs_labels.nonzero(as_tuple=True)

        bce_term = self.bce(abs_preds, abs_labels)
        mse_term = self.mse(preds[pos_mask_indices], labels[pos_mask_indices])

        return bce_term + mse_term
