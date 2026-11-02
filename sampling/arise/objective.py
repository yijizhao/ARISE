"""Quantized Wasserstein objective J(Omega) (Eqs. 17-18)."""

import numpy as np

MAX_BINS = 100


class QuantizedW1:
    """Quantile bins of the full residual losses and their prefix histograms."""

    def __init__(self, losses, W, max_bins=MAX_BINS):
        """Build bin centers c_k, full probabilities p_k and prefix counts B(u)."""
        losses = np.asarray(losses, dtype=np.float64)
        self.W = int(W)
        self.N = len(losses)
        self.T = self.N + self.W - 1

        valid = np.isfinite(losses)
        if not valid.any():
            raise ValueError("No valid residual losses in the training split")
        values = losses[valid]

        # Quantile bins; duplicate edges are merged and empty bins dropped.
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, max_bins + 1)))
        if len(edges) == 1:
            labels = np.zeros(len(values), dtype=np.int32)
            K = 1
        else:
            labels = np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)
            K = len(edges) - 1
        used = np.flatnonzero(np.bincount(labels, minlength=K))
        remap = np.full(K, -1, dtype=np.int32)
        remap[used] = np.arange(len(used))
        labels = remap[labels]
        self.K = len(used)

        counts = np.bincount(labels, minlength=self.K).astype(np.float64)
        self.centers = np.bincount(labels, weights=values, minlength=self.K) / counts
        self.gaps = np.diff(self.centers)
        self.full_cdf = np.cumsum(counts / counts.sum())[:-1]
        self.quantization_error = float(np.max(np.abs(values - self.centers[labels])))

        # B(u): bin counts of valid losses among sample starts s < u (Eq. 18).
        sample_labels = np.full(self.N, -1, dtype=np.int32)
        sample_labels[valid] = labels
        self.prefix = np.zeros((self.N + 1, self.K), dtype=np.int32)
        for k in range(self.K):
            self.prefix[1:, k] = np.cumsum(sample_labels == k, dtype=np.int32)
        self.valid_prefix = np.r_[0, np.cumsum(valid, dtype=np.int64)]

    def counts(self, starts, lengths):
        """Bin counts h_i of segments [a, a + S) for arrays of starts and lengths."""
        ends = starts + lengths - self.W + 1
        return (self.prefix[ends] - self.prefix[starts]).astype(np.float64)

    def total_counts(self, segments):
        """Summed bin counts of a list of segments."""
        total = np.zeros(self.K, dtype=np.float64)
        for a, b in segments:
            total += self.counts(a, b - a)
        return total

    def score_counts(self, counts):
        """J for one or more rows of bin counts (Eq. 17)."""
        counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
        n = counts.sum(axis=1)
        probs = np.divide(counts, n[:, None], out=np.zeros_like(counts), where=n[:, None] > 0)
        cdf_gap = np.abs(np.cumsum(probs, axis=1)[:, :-1] - self.full_cdf)
        score = np.sum(cdf_gap * self.gaps, axis=1)
        score[n <= 0] = np.inf
        return score

    def score(self, segments):
        """J for a list of segments."""
        return float(self.score_counts(self.total_counts(segments))[0])


def exact_w1(x, y):
    """Unquantized empirical W1 between two samples (Eq. 16), for reporting."""
    a = np.sort(np.asarray(x, float)[np.isfinite(x)])
    b = np.sort(np.asarray(y, float)[np.isfinite(y)])
    if not len(a) or not len(b):
        return float("nan")
    z = np.unique(np.r_[a, b])
    if len(z) < 2:
        return 0.0
    cdf_a = np.searchsorted(a, z[:-1], side="right") / len(a)
    cdf_b = np.searchsorted(b, z[:-1], side="right") / len(b)
    return float(np.dot(np.abs(cdf_a - cdf_b), np.diff(z)))
