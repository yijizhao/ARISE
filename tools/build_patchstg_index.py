"""Build data/<name>/spatial_indices.npz for PatchSTG from sensor coordinates.

Example:
    python tools/build_patchstg_index.py --data-root data/SD --metadata data/SD/meta.csv --steps-per-day 96
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from downstream.data import StandardScaler  # noqa: E402


def daily_similarity(train, steps_per_day):
    """exp-normalized cosine similarity of the nodes' mean daily profiles (training split only)."""
    normalized = StandardScaler.fit(train).transform(train)
    days = len(train) // steps_per_day
    profile = normalized[: days * steps_per_day].reshape(days, steps_per_day, -1).mean(axis=0)
    similarity = cosine_similarity(profile.T, profile.T)
    return np.exp((similarity - similarity.mean()) / similarity.std())


def kd_tree(locations, depth, axis=0):
    """Split nodes into 2**depth groups by alternating latitude/longitude medians."""
    order = np.argsort(locations[axis])
    half = locations.shape[1] // 2
    left, right = np.sort(order[:half]), np.sort(order[half:])
    if depth == 1:
        return [left, right]
    parts = [left[p] for p in kd_tree(locations[:, left], depth - 1, axis ^ 1)]
    parts += [right[p] for p in kd_tree(locations[:, right], depth - 1, axis ^ 1)]
    return parts


def pad_with_similar(part, similarity, size):
    """Fill a group up to `size` nodes with the most similar nodes outside it."""
    if len(part) >= size:
        return part
    scores = similarity[part, :].copy()
    scores[:, part] = 0
    candidates = np.argsort(-scores.reshape(-1)) % similarity.shape[-1]
    extra = []
    for node in candidates:
        if node not in extra:
            extra.append(node)
        if len(extra) == size - len(part):
            break
    return np.concatenate([part, np.array(extra, dtype=int)])


def build_index(parts, similarity, size):
    """Original positions, positions after padding, and the padded node order."""
    ori, reo, padded = [], [], []
    for i, part in enumerate(parts):
        ori.append(part)
        reo.append(np.arange(len(part)) + size * i)
        padded.append(pad_with_similar(part, similarity, size))
    return np.concatenate(ori), np.concatenate(reo), np.concatenate(padded)


def main():
    """Partition the sensors, pad each group and save ori/reo/all index arrays."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--metadata", required=True, type=Path, help="CSV with Lat and Lng in node order"
    )
    parser.add_argument("--steps-per-day", required=True, type=int)
    parser.add_argument("--patch-size", default=2, type=int)
    parser.add_argument(
        "--depth", type=int, help="KD-tree depth (default: smallest depth that fits the patch size)"
    )
    parser.add_argument(
        "--adjacency", type=Path, help="optional precomputed node-by-node similarity (.npy)"
    )
    args = parser.parse_args()

    train = np.load(args.data_root / "train_data.npy")
    train = train[..., 0] if train.ndim == 3 else train
    coordinates = pd.read_csv(args.metadata)[["Lat", "Lng"]].to_numpy().T
    nodes = train.shape[1]
    if coordinates.shape[1] != nodes:
        raise ValueError(f"metadata has {coordinates.shape[1]} rows, data has {nodes} nodes")
    depth = args.depth or math.ceil(math.log2(nodes / args.patch_size))

    parts = kd_tree(coordinates, depth)
    if max(map(len, parts)) > args.patch_size:
        raise ValueError("A group is larger than --patch-size; increase --depth")
    if args.adjacency:
        similarity = np.load(args.adjacency)
    else:
        similarity = daily_similarity(train, args.steps_per_day)
    ori, reo, padded = build_index(parts, similarity, args.patch_size)
    np.savez(args.data_root / "spatial_indices.npz", ori=ori, reo=reo, all=padded)
    print(f"depth {depth}: {len(parts)} groups, {len(padded)} padded nodes")


if __name__ == "__main__":
    main()
