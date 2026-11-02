"""Masked MAE loss and test metrics over valid targets."""

import numpy as np
import torch

ZERO_EPS = 5e-5


def valid_mask(target, missing):
    """Finite targets; with missing='zero', near-zero targets are also excluded."""
    valid = torch.isfinite(target)
    if missing == "zero":
        valid &= target.abs() > ZERO_EPS
    return valid


def masked_mae(prediction, target, mask):
    """Mean absolute error over valid targets."""
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Prediction contains non-finite values")
    if not mask.any():
        return prediction.sum() * 0
    return (prediction[mask] - target[mask]).abs().mean()


class Metrics:
    """Accumulates MAE, RMSE and MAPE over a whole split."""

    def __init__(self, missing):
        """Start with empty totals."""
        self.missing = missing
        self.abs_error = self.sq_error = self.pct_error = 0.0
        self.count = self.pct_count = 0

    def update(self, prediction, target, mask):
        """Add one batch in double precision."""
        prediction, target = prediction.double(), target.double()
        errors = (prediction[mask] - target[mask]).abs()
        self.abs_error += errors.sum().item()
        self.sq_error += errors.square().sum().item()
        self.count += errors.numel()
        pct_mask = mask & (target.abs() > ZERO_EPS)
        diff = (prediction[pct_mask] - target[pct_mask]).abs()
        self.pct_error += (diff / target[pct_mask].abs()).sum().item()
        self.pct_count += pct_mask.sum().item()

    def result(self):
        """MAE, RMSE, MAPE (%) and the number of valid targets."""
        return {
            "MAE": self.abs_error / self.count,
            "RMSE": float(np.sqrt(self.sq_error / self.count)),
            "MAPE": 100 * self.pct_error / self.pct_count if self.pct_count else None,
            "valid_targets": self.count,
        }
