from typing import List
from loguru import logger
import torch
import torch.nn as nn
import wandb


class MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCELoss(reduction="mean")

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, current_iter: int = 0):
        """
        Compute the loss for the given predictions and labels.

        Input is the full batch of predictions and labels for state vectors

        preds shape: (B, N, 12), labels shape: (B, N, 12)
        """
        indices_to_keep = labels != -10

        for i in range(labels.shape[0]):
            assert not torch.all(labels[i] == -10), f"labels[{i}] == -10"

        # Create mask for the loss
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[indices_to_keep] = True

        abs_preds = torch.abs(preds)
        abs_labels = torch.abs(labels)

        # Assert all values are in the range [0, 1]
        assert torch.all(abs_preds[mask] <= 1), "abs_preds not in range [0, 1]"

        bce_term = self.bce(abs_preds[mask], abs_labels[mask])

        # logger.error(f"preds: {preds}")
        # logger.error(f"labels: {labels}")
        # logger.error(f"abs_preds: {abs_preds}")
        # logger.error(f"abs_labels: {abs_labels}")
        # logger.error(f"mask: {mask}")
        # logger.error(f"abs_preds[mask]: {abs_preds[mask]}")
        # logger.error(f"abs_labels[mask]: {abs_labels[mask]}")
        # logger.error(f"bce_term: {bce_term}")

        # Get the indices where the abs_labels are 1
        # pos_mask_indices = abs_labels.nonzero(as_tuple=True)
        pos_mask_indices = torch.zeros_like(abs_labels, dtype=torch.bool)
        pos_mask_indices[abs_labels == 1.0] = True
        pos_mask_indices *= mask

        # mse_term = self.mse(preds[pos_mask_indices], labels[pos_mask_indices])
        mse_term = self.mse(preds[pos_mask_indices], labels[pos_mask_indices])

        result = 0.5 * (bce_term + mse_term)

        if wandb.run is not None:
            # Log the loss values
            if current_iter % 10 == 0:
                wandb.log(
                    {
                        "Train/state/loss_bce": bce_term,
                        "Train/state/loss_mse": mse_term,
                    },
                    step=current_iter,
                )

            # Create a table of the prediction, label, abs_pred, abs_label, loss_bce, loss_mse, loss values for each batch element. Create a column for each label dimension
            if current_iter % 50 == 0:
                columns = [
                    "preds",
                    "abs_preds",
                    "labels",
                    "abs_labels",
                    "mask",
                    "abs_preds[mask]",
                    "abs_labels[mask]",
                    "loss_bce",
                    "loss_mse",
                    "loss",
                ]

                table = wandb.Table(columns=columns)

                for i in range(preds.shape[0]):
                    table.add_data(
                        preds[i].tolist(),
                        abs_preds[i].tolist(),
                        labels[i].tolist(),
                        abs_labels[i].tolist(),
                        mask[i].tolist(),
                        abs_preds[mask].tolist(),
                        abs_labels[mask].tolist(),
                        bce_term,
                        mse_term,
                        result.tolist(),
                    )
                wandb.log({"Train/state/loss_table": table}, step=current_iter)

            bce_threshold = 40
            if bce_term >= bce_threshold:
                logger.error(f"bce_term >= {bce_threshold}: {bce_term}")
                logger.error(f"preds: {preds}")
                logger.error(f"labels: {labels}")
                logger.error(f"abs_preds: {abs_preds}")
                logger.error(f"abs_labels: {abs_labels}")
                logger.error(f"mask: {mask}")
                logger.error(f"abs_preds[mask]: {abs_preds[mask]}")
                logger.error(f"abs_labels[mask]: {abs_labels[mask]}")
                logger.error(f"bce_term: {bce_term}")

                wandb.alert(
                    title=f"bce_term >= {bce_threshold}",
                    text=f"""bce_term >= {bce_threshold}: {bce_term}\n
                    preds: {preds.detach().cpu().numpy()}\n
                    labels: {labels.detach().cpu().numpy()}\n
                    abs_preds: {abs_preds.detach().cpu().numpy()}\n
                    abs_labels: {abs_labels.detach().cpu().numpy()}\n
                    mask: {mask.detach().cpu().numpy()}\n
                    abs_preds[mask]: {abs_preds[mask].detach().cpu().numpy()}\n
                    abs_labels[mask]: {abs_labels[mask].detach().cpu().numpy()}\n
                    bce_term: {bce_term}\n
                    mse_term: {mse_term} 
                    """,
                    level=wandb.AlertLevel.WARN,
                )
        return result
