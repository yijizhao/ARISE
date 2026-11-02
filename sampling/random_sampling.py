"""Random baseline: training samples drawn uniformly without replacement (Sect. 5.2)."""

import numpy as np

from .common import ratio_floor


def select_random(train, budget, H, P, seed=0):
    """Draw n = floor(q * N) sample starts (at least one) without replacement."""
    N = len(train) - H - P + 1
    n = max(1, ratio_floor(budget, N))
    if n == N:
        return np.arange(N, dtype=np.int64), {"seed": seed}
    starts = np.random.RandomState(seed).choice(N, n, replace=False)
    return np.sort(starts).astype(np.int64), {"seed": seed}
