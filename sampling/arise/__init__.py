"""ARISE: Adaptive Residual-Informed Segment Extraction (Algorithm 1)."""

import numpy as np

from ..common import SamplingError, ratio_floor
from .objective import QuantizedW1, exact_w1
from .probe import residual_losses
from .refine import Refiner, is_valid
from .strata import choose_strata, initialize, time_strata


def window_starts(segments, W):
    """Sample starts I(Omega) whose windows lie entirely inside one segment (Eq. 8)."""
    return np.concatenate([np.arange(a, b - W + 1, dtype=np.int64) for a, b in sorted(segments)])


def select(train, budget, H, P, missing="nan", strata=16):
    """Select contiguous segments under the raw-data budget C = floor(beta * T)."""
    T = len(train)
    W = H + P
    C = ratio_floor(budget, T)
    info = {"C": C, "requested_strata": strata}

    if C == T:
        info.update(strata=None, segments=[[0, T]])
        return np.arange(T - W + 1, dtype=np.int64), info

    # Reduce G until a feasible initialization exists (Algorithm 1, line 5).
    G = choose_strata(T, C, W, strata)
    if G is None:
        raise SamplingError(
            f"Budget C={C} is too small: one segment needs H+P={W} steps "
            f"(beta >= {W / T:.4f} for this dataset)."
        )
    strata_bounds = time_strata(T, G)

    losses = residual_losses(train, H, P, missing)
    objective = QuantizedW1(losses, W)
    refiner = Refiner(objective, C, W, strata_bounds)
    segments, score = refiner.run(initialize(objective, C, W, strata_bounds))
    if not segments:
        raise SamplingError("No valid residual losses inside the initial segments.")
    assert is_valid(segments, strata_bounds, C, W)

    segments = [[a, b] for a, b, _ in segments]
    starts = window_starts(segments, W)
    selected = losses[starts]
    info.update(
        strata=G,
        segments=segments,
        objective_J=score,
        exact_W1=exact_w1(losses, selected),
        quantization_error=objective.quantization_error,
        bins=objective.K,
        evaluations=refiner.evaluations,
    )
    return starts, info
