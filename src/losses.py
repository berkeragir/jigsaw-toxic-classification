"""Multi-label focal loss with per-class alpha.

v6 used focal (gamma=2) AND BCEWithLogitsLoss(pos_weight=neg/pos). That
double-corrects rare classes and collapsed `threat`/`identity_hate` to
near-random. v7 drops pos_weight and bounds alpha via clamping to [0.25, 0.75].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-label focal loss (Lin et al. 2017) without pos_weight.

    Args:
        gamma: focusing parameter. Higher = stronger down-weighting of easy
            examples.
        alpha: per-class positive weight, shape [num_labels], values in (0, 1).
            alpha_c weights positives; (1 - alpha_c) weights negatives.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=torch.float)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_factor * bce
        else:
            loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_focal_alpha(labels: np.ndarray, clamp=(0.25, 0.75)) -> torch.Tensor:
    """Compute per-class alpha from positive rates.

    For a label with pos-rate r, alpha_c = 1 - r gives positives and negatives
    equal aggregate weight. Clamping bounds rare-class weighting so we don't
    recreate v6's overcorrection.

    Args:
        labels: shape [N, num_labels], values in {0, 1}.
        clamp: (lo, hi) bounds on alpha.

    Returns:
        Float tensor of shape [num_labels].
    """
    pos_rates = labels.mean(axis=0)
    alpha = np.clip(1.0 - pos_rates, clamp[0], clamp[1]).astype(np.float32)
    return torch.tensor(alpha, dtype=torch.float)
