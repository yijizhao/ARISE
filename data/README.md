# Data

Each dataset is a folder `data/<name>/` with

```text
train_data.npy        [time, node] or [time, node, feature]; feature 0 is used
val_data.npy
test_data.npy
train_timestamps.npy  [time, >=2]; column 0 = time of day in [0, 1), column 1 = day of week / 7 (Monday = 0)
val_timestamps.npy
test_timestamps.npy
spatial_indices.npz   PatchSTG only (SD, GBA, GLA, CA)
```

The reader settings (folder, steps per day, missing-value rule, normalization) are in `configs/datasets/<name>.yaml`.

## Included

- `Weather/`: the full Weather dataset from BasicTS (21 variables, 10 min, 7:1:2 split; 36,887 / 5,269 / 10,540 steps).
- `SD_demo/`: the first 8 sensors and the first 384 / 96 / 96 steps of the SD train / val / test splits, with sensor metadata and the PatchSTG index. It is only for checking that the code runs.

## Download

**SD, GBA, GLA, CA, Electricity.** Download `datasets.zip` from BasicTS ([Google Drive](https://drive.google.com/file/d/1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1UcZCCKPCeS7mHSnCO4-COA?pwd=j9ev)) and copy the folders `SD`, `GBA`, `GLA`, `CA` and `Electricity` into `data/`. They already use the format above; the included `Weather/` comes from the same archive. The original sources are [LargeST](https://github.com/liuxu77/LargeST) (traffic, 2019) and [Autoformer](https://github.com/thuml/Autoformer) (Electricity).

**AirQuality.** Download the [TSL air-quality archive](https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/air/air_quality.zip). Convert its hourly PM2.5 table (437 stations) with a 6:2:2 split. Stations without any observation in the training split are dropped, which leaves 404:

```bash
python tools/prepare_data.py --input <pm25 table .h5 or .csv> --hdf-key <key> \
    --steps-per-day 24 --split 6:2:2 --drop-empty-nodes --output data/AirQuality
```

`tools/prepare_data.py` converts any CSV or pandas HDF5 table with a time index and one column per node.

## PatchSTG spatial index

PatchSTG groups sensors by coordinates. Take `Lat` and `Lng` for the dataset's sensors, in the same order as the data, from the LargeST metadata file (`ca_meta.csv`). Then build the index once:

```bash
python tools/build_patchstg_index.py --data-root data/SD --metadata data/SD/meta.csv --steps-per-day 96
```

By default the KD-tree depth is the smallest one that gives groups of at most two sensors: 9 for SD, 11 for GBA and GLA, 13 for CA. `--depth` overrides it.
