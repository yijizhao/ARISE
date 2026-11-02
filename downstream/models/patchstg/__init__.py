"""PatchSTG backbone; needs data/<name>/spatial_indices.npz built from sensor coordinates."""

import numpy as np
import torch

from .model import PatchSTG


def build(params, dataset, H, P):
    """Create PatchSTG from the spatial index file of the dataset."""
    if not dataset.has_coordinates:
        raise ValueError(
            f"PatchSTG needs sensor coordinates and is run only on SD, GBA, GLA and CA; "
            f"{dataset.name} has none. Use --model STID."
        )
    if H != P:
        raise ValueError("PatchSTG requires history == horizon")
    path = dataset.root / "spatial_indices.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with tools/build_patchstg_index.py")
    indices = np.load(path, allow_pickle=False)
    ori, reo, all_nodes = indices["ori"], indices["reo"], indices["all"]

    params = dict(params)
    size = params.pop("spa_patchsize")
    if len(all_nodes) % size or (len(all_nodes) // size) % params["factors"]:
        raise ValueError("spa_patchsize and factors must divide the padded node count")
    params.update(
        tem_patchsize=H,
        tem_patchnum=1,
        node_num=dataset.values("train").shape[1],
        spa_patchsize=size,
        spa_patchnum=len(all_nodes) // size,
        tod=dataset.steps_per_day,
        dow=7,
        ori_parts_idx=ori,
        reo_parts_idx=reo,
        reo_all_idx=all_nodes,
    )
    return PatchSTG(**params)


def predict(model, x, calendar):
    """Convert calendar features to integer indices and predict normalized targets."""
    calendar = torch.round(calendar * calendar.new_tensor([model.tod, model.dow])).long()
    return model(x, calendar)
