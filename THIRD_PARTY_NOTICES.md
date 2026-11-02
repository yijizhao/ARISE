# Third-party code and data

## Code

| Path | Source | License | Changes |
| --- | --- | --- | --- |
| `downstream/models/stid/` | [STID](https://github.com/zezhishao/STID) (Shao et al., CIKM 2022) | Apache-2.0, [text](third_party_licenses/STID_LICENSE) | Unused switches removed, forward signature simplified, layers renamed |
| `downstream/models/patchstg/`, `tools/build_patchstg_index.py` | [PatchSTG](https://github.com/LMissher/PatchSTG) (Fang et al., KDD 2025) | MIT, Copyright (c) 2024 LMissher, [text](third_party_licenses/PatchSTG_LICENSE) | Model and partition code merged and reformatted |

PatchSTG imports `Attention` and `Mlp` from [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0), installed as a dependency.

## Data

| Path | Source | License |
| --- | --- | --- |
| `data/SD_demo/` | First 8 sensors and first 384/96/96 steps of the train/val/test splits of SD from [LargeST](https://github.com/liuxu77/LargeST) | CC BY-NC 4.0 |
| `data/Weather/` | Weather from [BasicTS](https://github.com/GestaltCogTeam/BasicTS) (originally from the [Autoformer](https://github.com/thuml/Autoformer) benchmark, Max Planck Institute for Biogeochemistry weather station) | see the sources |
