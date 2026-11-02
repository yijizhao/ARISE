"""Dataset splits, training-split normalization and forecasting windows."""

import numpy as np

from project import load_values, project_path


class Dataset:
    """Prepared train/val/test arrays of one dataset."""

    def __init__(self, config):
        """Store the data folder, calendar resolution and missing-value rule."""
        self.name = config["name"]
        self.root = project_path(config["root"])
        self.steps_per_day = int(config["steps_per_day"])
        self.missing = config["missing"]
        self.norm_each_node = bool(config["norm_each_node"])
        self.has_coordinates = bool(config.get("has_coordinates", False))

    def values(self, split):
        """[time, node] values of a split."""
        return load_values(self.root, split)

    def timestamps(self, split):
        """Time-of-day and day-of-week features, each normalized to [0, 1)."""
        stamps = np.load(self.root / f"{split}_timestamps.npy", allow_pickle=False)[:, :2]
        index = np.rint(stamps * [self.steps_per_day, 7])
        if np.any(index < 0) or np.any(index >= [self.steps_per_day, 7]):
            raise ValueError(
                f"{split}_timestamps.npy does not match steps_per_day={self.steps_per_day}"
            )
        return stamps.astype(np.float32)

    def all_starts(self, split, H, P):
        """All valid sample starts of a split (an explicit start file is used if present)."""
        n = len(self.values(split)) - H - P + 1
        if n < 1:
            raise ValueError(f"The {split} split is shorter than H + P")
        path = self.root / f"{split}_starts_h{H}_p{P}.npy"
        return np.load(path, allow_pickle=False) if path.exists() else np.arange(n, dtype=np.int64)


class StandardScaler:
    """Z-score normalization fitted on the full training split."""

    def __init__(self, mean, std):
        """Store global or per-node statistics."""
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, raw, per_node=False):
        """Fit mean and std on finite training values."""
        axis = 0 if per_node else None
        count = np.zeros(raw.shape[1] if per_node else (), dtype=np.int64)
        total = np.zeros_like(count, dtype=np.float64)
        squares = np.zeros_like(total)
        for start in range(0, len(raw), 1024):
            values = np.asarray(raw[start : start + 1024], dtype=np.float64)
            valid = np.isfinite(values)
            values = np.where(valid, values, 0)
            count += valid.sum(axis=axis)
            total += values.sum(axis=axis)
            squares += np.square(values).sum(axis=axis)
        mean = total / count
        std = np.sqrt(np.maximum(0.0, squares / count - mean**2))
        return cls(mean, np.where(std > 0, std, 1.0))

    @classmethod
    def for_dataset(cls, dataset):
        """Use data/<name>/scaler.npz if it exists, otherwise fit on the training split."""
        path = dataset.root / "scaler.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                return cls(saved["mean"], saved["std"])
        return cls.fit(dataset.values("train"), dataset.norm_each_node)

    def _stats(self, reference):
        """Statistics as tensors that broadcast over [B, L, V, 1]."""
        import torch

        mean = torch.as_tensor(self.mean, device=reference.device, dtype=reference.dtype)
        std = torch.as_tensor(self.std, device=reference.device, dtype=reference.dtype)
        return mean.reshape(1, 1, -1, 1), std.reshape(1, 1, -1, 1)

    def inverse(self, prediction):
        """Map normalized predictions back to the original scale."""
        mean, std = self._stats(prediction)
        return prediction * std + mean

    def normalize(self, target):
        """Normalize target tensors."""
        mean, std = self._stats(target)
        return (target - mean) / std

    def transform(self, raw):
        """Normalize model inputs; missing inputs become 0."""
        normalized = (np.asarray(raw, dtype=np.float32) - self.mean) / self.std
        return np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32)


class WindowDataset:
    """Forecasting samples (X_s, Y_s) at the given starts."""

    def __init__(self, raw, stamps, starts, H, P, scaler):
        """Bind one split, its calendar features and the sample starts."""
        self.raw, self.stamps, self.starts = raw, stamps, starts
        self.H, self.P, self.scaler = H, P, scaler

    def __len__(self):
        """Number of samples."""
        return len(self.starts)

    def __getitem__(self, index):
        """Normalized history, raw targets and calendar features of one sample."""
        start = int(self.starts[index])
        stop = start + self.H
        x = self.scaler.transform(self.raw[start:stop])[..., None]
        y = np.array(self.raw[stop : stop + self.P], dtype=np.float32)[..., None]
        calendar = np.broadcast_to(self.stamps[start:stop, None, :], (self.H, x.shape[1], 2)).copy()
        return x, y, calendar
