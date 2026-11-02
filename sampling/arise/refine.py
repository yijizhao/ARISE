"""Phase III: Wasserstein-guided segment refinement (Sect. 4.3, Algorithm 1)."""

import numpy as np

EPS = 1e-12
CHUNK = 4096


def is_valid(segments, strata, C, W):
    """Check budget, minimum length, separation and one segment per stratum."""
    if not segments or sum(b - a for a, b, _ in segments) != C:
        return False
    covered = set()
    previous_end = -1
    for a, b, k in sorted(segments):
        if not 0 <= k < len(strata):
            return False
        lo, hi = strata[k]
        if not (lo <= a < b <= hi and b - a >= W and a > previous_end):
            return False
        covered.add(k)
        previous_end = b
    return len(covered) == len(strata)


def without_overlap(starts, length, others):
    """Mask of candidate starts whose segment does not touch any other segment."""
    mask = np.ones(len(starts), bool)
    for a, b, _ in others:
        mask &= (starts + length < a) | (starts > b)
    return mask


class Refiner:
    """Relocation, length transfer and splitting with strict improvement of J."""

    def __init__(self, objective, C, W, strata):
        """Store the objective, budget, sample width H+P and strata."""
        self.J = objective
        self.C = C
        self.W = W
        self.strata = strata
        self.evaluations = 0
        T = objective.T
        G = len(strata)
        # R_max (Eq. 23); W - 1 = H + P - 1.
        self.max_segments = min(T - C + 1, C // W, max(G, C // (2 * (W - 1))))

    def best_position(self, length, k, others, limit):
        """Best start for a segment of this length in stratum k, if it beats limit."""
        lo, hi = self.strata[k]
        if length > hi - lo:
            return None, limit
        base = self.J.total_counts([(a, b) for a, b, _ in others])
        best_start, best_score = None, float(limit)
        for first in range(lo, hi - length + 1, CHUNK):
            last = min(first + CHUNK, hi - length + 1)
            starts = np.arange(first, last, dtype=np.int64)
            starts = starts[without_overlap(starts, length, others)]
            if not len(starts):
                continue
            scores = self.J.score_counts(base[None, :] + self.J.counts(starts, length))
            self.evaluations += len(starts)
            i = int(np.argmin(scores))
            if scores[i] < best_score - EPS:
                best_start, best_score = int(starts[i]), float(scores[i])
        return best_start, best_score

    def relocate(self, segments, score):
        """Move each segment to its best position within its stratum (Eq. 20)."""
        for i, (a, b, k) in enumerate(segments):
            others = segments[:i] + segments[i + 1 :]
            start, value = self.best_position(b - a, k, others, score)
            if start is not None:
                segments = sorted(others + [(start, start + b - a, k)])
                score = value
        return segments, score

    def transfer(self, segments, score):
        """Best transfer of d steps from a donor to a recipient segment (Eq. 21)."""
        best_segments, best_score = segments, score
        for i, (a, b, k) in enumerate(segments):
            donor_len = b - a
            if donor_len <= self.W:
                continue
            lo_d, hi_d = self.strata[k]
            for j, (c, d, kr) in enumerate(segments):
                if i == j:
                    continue
                recipient_len = d - c
                lo_r, hi_r = self.strata[kr]
                max_amount = min(donor_len - self.W, hi_r - lo_r - recipient_len)
                if max_amount < 1:
                    continue
                others = [s for h, s in enumerate(segments) if h not in (i, j)]
                for first in range(1, max_amount + 1, CHUNK):
                    amount = np.arange(first, min(first + CHUNK, max_amount + 1), dtype=np.int64)
                    new_d = donor_len - amount
                    new_r = recipient_len + amount
                    # Keep both segments near their previous centers.
                    start_d = np.clip((a + b - new_d) // 2, lo_d, hi_d - new_d)
                    start_r = np.clip((c + d - new_r) // 2, lo_r, hi_r - new_r)
                    mask = without_overlap(start_d, new_d, others)
                    mask &= without_overlap(start_r, new_r, others)
                    mask &= (start_d + new_d < start_r) | (start_r + new_r < start_d)
                    if not mask.any():
                        continue
                    start_d, new_d, start_r, new_r = (
                        v[mask] for v in (start_d, new_d, start_r, new_r)
                    )
                    base = self.J.total_counts([(x, y) for x, y, _ in others])
                    counts = (
                        base[None, :]
                        + self.J.counts(start_d, new_d)
                        + self.J.counts(start_r, new_r)
                    )
                    scores = self.J.score_counts(counts)
                    self.evaluations += len(scores)
                    m = int(np.argmin(scores))
                    if scores[m] < best_score - EPS:
                        best_score = float(scores[m])
                        best_segments = sorted(
                            others
                            + [
                                (int(start_d[m]), int(start_d[m] + new_d[m]), k),
                                (int(start_r[m]), int(start_r[m] + new_r[m]), kr),
                            ]
                        )
        return best_segments, best_score

    def split(self, segments, score):
        """Best split of one segment into two within its stratum (Eq. 22)."""
        if len(segments) >= self.max_segments:
            return segments, score
        best_segments, best_score = segments, score
        for i, (a, b, k) in enumerate(segments):
            length = b - a
            if length < 2 * self.W + 1:
                continue
            sizes = sorted({self.W, length // 4, length // 2, 3 * length // 4, length - self.W})
            for size in sizes:
                if not self.W <= size <= length - self.W:
                    continue
                keep = length - size
                for kept in [(a, a + keep, k), (b - keep, b, k)]:
                    others = segments[:i] + segments[i + 1 :] + [kept]
                    start, value = self.best_position(size, k, others, best_score)
                    if start is not None and value < best_score - EPS:
                        candidate = sorted(others + [(start, start + size, k)])
                        if is_valid(candidate, self.strata, self.C, self.W):
                            best_segments, best_score = candidate, value
        return best_segments, best_score

    def run(self, segments):
        """Refine an initial view following Algorithm 1, lines 7-25."""
        score = self.J.score([(a, b) for a, b, _ in segments])
        if not np.isfinite(score):
            return [], float("inf")
        segments, score = self.relocate(segments, score)

        for _ in range(2):
            candidate, value = self.transfer(segments, score)
            if value >= score - EPS:
                break
            segments, score = self.relocate(candidate, value)

        while len(segments) < self.max_segments:
            candidate, value = self.split(segments, score)
            if value >= score - EPS:
                break
            segments, score = self.relocate(candidate, value)
            candidate, value = self.transfer(segments, score)
            if value < score - EPS:
                segments, score = self.relocate(candidate, value)
        return segments, float(score)
