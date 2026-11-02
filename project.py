"""Paths, configuration files and dataset loading shared by both steps."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def project_path(value):
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_yaml(path):
    """Read a YAML mapping."""
    with Path(path).open(encoding="utf8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def merge(base, override):
    """Recursively merge two configuration mappings into a new one."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def dataset_config(name):
    """Load configs/datasets/<name>.yaml."""
    path = ROOT / "configs" / "datasets" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in path.parent.glob("*.yaml"))
        raise ValueError(f"Unknown dataset {name}; available: {available}")
    config = read_yaml(path)
    config["name"] = name
    return config


def load_values(root, split):
    """Load the value channel of one split as a [time, node] array."""
    path = project_path(root) / f"{split}_data.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/README.md")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Expected a numeric [time, node] array: {path}")
    return values


def file_sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    """Write a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf8")


def load_json(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf8"))
