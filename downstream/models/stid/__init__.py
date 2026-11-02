"""STID backbone."""

import torch

from .model import STID


def build(params, dataset, H, P):
    """Create STID for this dataset and horizon."""
    nodes = dataset.values("train").shape[1]
    return STID(nodes, H, P, dataset.steps_per_day, **params)


def predict(model, x, calendar):
    """Predict normalized targets from normalized inputs and calendar features."""
    return model(torch.cat([x, calendar], dim=-1))
