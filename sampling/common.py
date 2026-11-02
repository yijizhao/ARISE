"""Budget parsing and raw-data accounting shared by all samplers."""

from decimal import ROUND_FLOOR, Decimal, InvalidOperation

import numpy as np


class SamplingError(ValueError):
    """A budget or dataset for which no valid selection exists."""


def ratio_floor(ratio, total):
    """Return floor(ratio * total) for a decimal ratio in (0, 1]."""
    try:
        value = Decimal(str(ratio))
    except InvalidOperation as exc:
        raise SamplingError(f"Budget must be a number in (0, 1], got {ratio}") from exc
    if not value.is_finite() or not Decimal(0) < value <= Decimal(1):
        raise SamplingError(f"Budget must be a number in (0, 1], got {ratio}")
    return int((value * total).to_integral_value(rounding=ROUND_FLOOR))


def covered_steps(starts, W):
    """Number of distinct raw time steps |U(A)| used by the selected samples (Eq. 2)."""
    covered, end = 0, -1
    for start in np.sort(starts):
        start = int(start)
        covered += start + W - max(start, end)
        end = start + W
    return covered
