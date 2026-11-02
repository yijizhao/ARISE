"""Convert a time-indexed table (CSV or pandas HDF5) into the array format of data/<name>/.

Example:
    python tools/prepare_data.py --input electricity.csv --time-column date \
        --steps-per-day 24 --split 7:1:2 --output data/Electricity
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path, time_column, key=None):
    """Read a table whose rows are time steps and whose columns are nodes."""
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        frame = pd.read_hdf(path, key=key)
    else:
        frame = pd.read_csv(path)
    if time_column in frame.columns:
        frame = frame.set_index(time_column)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame.sort_index()


def calendar_features(times, steps_per_day):
    """Time of day in [0, 1) and day of week / 7 (Monday = 0)."""
    seconds = times.hour * 3600 + times.minute * 60 + times.second
    time_of_day = seconds / 86400
    if not np.allclose(time_of_day * steps_per_day, np.rint(time_of_day * steps_per_day)):
        raise ValueError("Timestamps are not aligned with --steps-per-day")
    return np.column_stack([time_of_day, times.dayofweek / 7]).astype(np.float32)


def main():
    """Split chronologically and save {train,val,test}_{data,timestamps}.npy."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps-per-day", type=int, required=True)
    parser.add_argument("--split", default="6:2:2", help="train:val:test ratio, e.g. 7:1:2")
    parser.add_argument("--time-column", default="date")
    parser.add_argument("--hdf-key", help="key inside an HDF5 file")
    parser.add_argument(
        "--drop-empty-nodes",
        action="store_true",
        help="drop nodes without any finite value in the training split (AirQuality)",
    )
    args = parser.parse_args()

    frame = read_table(args.input, args.time_column, args.hdf_key)
    step = pd.Timedelta(days=1) / args.steps_per_day
    if (frame.index[1:] - frame.index[:-1] != step).any():
        raise ValueError("The time index has gaps or a different sampling interval")

    ratio = np.array([float(v) for v in args.split.split(":")])
    T = len(frame)
    train_end = int(T * ratio[0] / ratio.sum())
    val_end = int(T * (ratio[0] + ratio[1]) / ratio.sum())
    values = frame.to_numpy(dtype=np.float32)
    if args.drop_empty_nodes:
        keep = np.isfinite(values[:train_end]).any(axis=0)
        values = values[:, keep]
        print(f"Kept {keep.sum()} of {len(keep)} nodes")
    calendar = calendar_features(frame.index, args.steps_per_day)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split, lo, hi in [
        ("train", 0, train_end),
        ("val", train_end, val_end),
        ("test", val_end, T),
    ]:
        np.save(output / f"{split}_data.npy", values[lo:hi])
        np.save(output / f"{split}_timestamps.npy", calendar[lo:hi])
        print(f"{split}: {hi - lo} steps x {values.shape[1]} nodes")


if __name__ == "__main__":
    main()
