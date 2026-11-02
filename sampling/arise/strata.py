"""Phase II: temporal strata and budgeted segment initialization (Sect. 4.2)."""

import numpy as np


def time_strata(T, G):
    """Split [0, T) into G nearly equal strata with one unselected step between them."""
    return [(j * T // G + int(j > 0), (j + 1) * T // G) for j in range(G)]


def is_feasible(T, C, W, G):
    """Check C >= G(H+P) and that every stratum can hold one segment."""
    capacity = [hi - lo for lo, hi in time_strata(T, G)]
    return min(capacity) >= W and G * W <= C <= sum(capacity)


def choose_strata(T, C, W, G):
    """Return the requested G, or the largest feasible G/2, G/4, ..., 1 if it is infeasible."""
    g = G
    while g >= 1:
        if is_feasible(T, C, W, g):
            return g
        g //= 2
    return None


def allocate_lengths(C, strata, W):
    """Split the budget C into one segment length S_i >= W per stratum."""
    capacity = np.asarray([hi - lo for lo, hi in strata], dtype=np.int64)

    # Largest common length L with sum(min(capacity, L)) <= C.
    low, high = W, int(capacity.max())
    while low < high:
        mid = (low + high + 1) // 2
        if int(np.minimum(capacity, mid).sum()) <= C:
            low = mid
        else:
            high = mid - 1
    lengths = np.minimum(capacity, low)

    # Hand out the remaining steps one by one from the first stratum.
    remaining = C - int(lengths.sum())
    for j in range(len(strata)):
        if remaining == 0:
            break
        if lengths[j] < capacity[j]:
            lengths[j] += 1
            remaining -= 1
    return [int(v) for v in lengths]


def initialize(objective, C, W, strata):
    """Place one segment per stratum with the most valid losses (ties: nearest the center)."""
    segments = []
    for k, ((lo, hi), length) in enumerate(zip(strata, allocate_lengths(C, strata, W))):
        positions = np.arange(lo, hi - length + 1, dtype=np.int64)
        ends = positions + length - W + 1
        valid = objective.valid_prefix[ends] - objective.valid_prefix[positions]
        best = positions[valid == valid.max()]
        center = (lo + hi - length) / 2
        a = int(best[np.argmin(np.abs(best - center))])
        segments.append((a, a + length, k))
    return segments
