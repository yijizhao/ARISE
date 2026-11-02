"""Phase I: residual losses of the rolling-mean probe (Eqs. 13-14)."""

import numpy as np

ZERO_EPS = 5e-5


def residual_losses(raw, H, P, missing="nan", block=128):
    """Return the residual loss r_s of every training sample (NaN if undefined).

    The probe predicts x_t from the mean of its H preceding observations.
    Missing and zero history values are left out of the mean; missing targets
    are left out of the loss.
    """
    T, V = raw.shape
    N = T - H - P + 1
    losses = np.full(N, np.nan, dtype=np.float64)

    for start in range(0, N, block):
        end = min(N, start + block)
        batch = end - start
        rows = np.asarray(raw[start : end + H + P - 1], dtype=np.float64)

        valid = np.isfinite(rows)
        if missing == "zero":
            valid &= np.abs(rows) > ZERO_EPS
        usable_history = valid & (rows != 0.0)

        # Rolling mean over the previous H steps via cumulative sums.
        sums = np.vstack([np.zeros((1, V)), np.cumsum(np.where(usable_history, rows, 0.0), axis=0)])
        counts = np.vstack(
            [
                np.zeros((1, V), dtype=np.int32),
                np.cumsum(usable_history, axis=0, dtype=np.int32),
            ]
        )
        steps = batch + P - 1
        history_sum = sums[H : H + steps] - sums[:steps]
        history_count = counts[H : H + steps] - counts[:steps]
        mean = np.divide(
            history_sum,
            history_count,
            out=np.zeros_like(history_sum),
            where=history_count > 0,
        )

        # One-step absolute errors, summed over nodes.
        target = rows[H : H + steps]
        good = valid[H : H + steps] & (history_count > 0)
        step_error = np.sum(np.where(good, np.abs(target - mean), 0.0), axis=1)
        step_count = np.sum(good, axis=1, dtype=np.int64)

        # Average over the P target steps of each sample (Eq. 14).
        error_prefix = np.r_[0.0, np.cumsum(step_error, dtype=np.float64)]
        count_prefix = np.r_[0, np.cumsum(step_count, dtype=np.int64)]
        total_error = error_prefix[P : P + batch] - error_prefix[:batch]
        total_count = count_prefix[P : P + batch] - count_prefix[:batch]
        losses[start:end] = np.divide(
            total_error,
            total_count,
            out=np.full(batch, np.nan),
            where=total_count > 0,
        )
    return losses
