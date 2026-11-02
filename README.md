# ARISE


ARISE keeps a few contiguous segments of the raw training series under a raw-data budget `C = floor(beta * T)`. A forecasting model is then trained on these segments and tested on the unchanged test set.

The code has two independent steps:

| Step | Entry | Folder | What it does |
| --- | --- | --- | --- |
| 1. Sampling | `sample.py` | `sampling/` | Selects training samples with ARISE or Random and saves their start indices |
| 2. Downstream | `train.py` | `downstream/` | Trains STID or PatchSTG on the saved samples, selects the best epoch on validation, and tests once |

```text
ARISE/
├── sample.py                  step 1
├── train.py                   step 2
├── sampling/
│   ├── arise/
│   │   ├── probe.py           Phase I: rolling-mean residual losses (Eqs. 13-14)
│   │   ├── objective.py       quantized Wasserstein distance J (Eqs. 17-18)
│   │   ├── strata.py          Phase II: strata, G fallback, initialization
│   │   └── refine.py          Phase III: relocation, length transfer, splitting
│   └── random_sampling.py     Random baseline
├── downstream/
│   ├── data.py                data loading, normalization, windows
│   ├── trainer.py             training, validation and testing
│   ├── metrics.py
│   └── models/                stid/, patchstg/
├── configs/                   datasets/*.yaml, models/*.yaml
├── data/                      SD_demo/ and Weather/ included
└── tools/                     data preparation, PatchSTG spatial index
```

## Installation

Python 3.10 or newer. Install PyTorch for your platform first ([instructions](https://pytorch.org/get-started/locally/)), then:

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 and PyTorch 2.3 (the paper experiments used Python 3.11-3.13 and PyTorch 2.3-2.9).

## Quick start

The repository includes the full Weather dataset and an 8-sensor slice of SD (`SD_demo`), so these commands run without downloads.

```bash
# Step 1: ARISE keeps 10% of the raw Weather training steps
python sample.py --dataset Weather --method ARISE --budget 0.10

# Step 2: train STID on that view
python train.py --dataset Weather --model STID --selection outputs/selections/Weather_ARISE_b0.10_H12P12
```

A two-epoch check of both models on CPU (under a minute):

```bash
python sample.py --dataset SD_demo --budget 0.5 --strata 2
python train.py --dataset SD_demo --model STID --selection outputs/selections/SD_demo_ARISE_b0.5_H12P12_G2
python train.py --dataset SD_demo --model PatchSTG --selection outputs/selections/SD_demo_ARISE_b0.5_H12P12_G2
```

## Step 1: sampling

```bash
python sample.py --dataset <name> --method <ARISE|Random> --budget <ratio>
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--budget` | required | ARISE: raw-data budget `beta`, `C = floor(beta * T)`. Random: training-sample ratio `q`, `n = floor(q * N)` |
| `--history`, `--horizon` | 12, 12 | `H` and `P` |
| `--strata` | 16 | Number of temporal strata `G`, chosen by the user (ARISE only) |
| `--seed` | 0 | Sampling seed (Random only) |
| `--output` | `outputs/selections/<dataset>_<method>_b<budget>_H<H>P<P>` | Output folder |

If the chosen `G` strata cannot each hold one segment of length `H + P` (`C < G(H + P)`), ARISE keeps `C` and retries with `G/2, G/4, ..., 1`. It prints the `G` it used. If even `G = 1` is infeasible, it stops with an error stating the smallest feasible budget.

The baseline **Random** draws `n` training samples uniformly without replacement.

Each run writes two files:
- `window_starts.npy`: start indices `s` of the selected samples in the training split.
- `selection.json`: the method and budget, `n/N`, raw-data usage `|U(A)|/T`, the time taken, and a hash of `train_data.npy`. For ARISE it also records the segments, `G`, and `J`.

## Step 2: downstream training

```bash
python train.py --dataset <name> --model <STID|PatchSTG> --selection <folder from step 1>
python train.py --dataset <name> --model <STID|PatchSTG> --full    # full-data reference
```

`H` and `P` are read from the selection. With `--full` they are set by `--history` and `--horizon`.

Training settings are in `configs/models/STID.yaml` and `configs/models/PatchSTG.yaml` (Table 3 of the paper). `--epochs`, `--batch-size`, `--seed` and `--device` override them. The model is trained for a fixed number of epochs. The epoch with the lowest validation MAE is tested once on the complete test set.

To report MAE degradation against the full-data model, pass its result:

```bash
python train.py --dataset Weather --model STID --full
python train.py --dataset Weather --model STID --selection outputs/selections/Weather_ARISE_b0.10_H12P12 \
    --full-result outputs/runs/Weather_STID_Full_H12P12/result.json
```

Outputs go to `outputs/runs/<model>_<selection>/`:
- `result.json`: test MAE, RMSE and MAPE, the best epoch, the training-view statistics, the settings, and the degradation if `--full-result` was given.
- `history.jsonl`: one line per epoch.
- `best.pt`: the best model weights.

PatchSTG needs sensor coordinates, so it runs only on SD, GBA, GLA and CA. On other datasets it stops with an error.

GPU and CPU runs give slightly different numbers. A given device and seed is deterministic.

## Settings used in the paper

| Experiment | Step 1 | Step 2 |
| --- | --- | --- |
| Budget–accuracy trade-off (RQ1), 7 datasets, 12→12 | ARISE `--budget beta --strata G`; Random `--method Random --budget q` | `--model STID` |
| Cross-architecture (RQ2) | The same selections, on SD, GBA, GLA, CA | `--model PatchSTG`; reduced views on CA with `--batch-size 32 --epochs 35` |
| Long horizon (RQ3), 96→96 | `--history 96 --horizon 96 --strata G` | `--model STID`; CA full-data reference with `--epochs 50` |
| Number of strata | `--strata 1`, `2`, `4`, `8`, `16`, `32` | `--model STID` |
| Full-data references (Table 4) | none | `--full` |

`G` is set by `--strata` (default 16). Ratios: RQ1 uses {0.1, 0.2, 0.5, 1, 2, ..., 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 100}%; the other experiments use {10, 20, 30, 40, 50}%. 

```bash
G=16
for b in 0.10 0.20 0.30 0.40 0.50; do
    python sample.py --dataset SD --budget $b --strata $G
    python train.py --dataset SD --model STID --selection outputs/selections/SD_ARISE_b${b}_H12P12_G$G
done
```

## Data

`data/<name>/` holds `{train,val,test}_data.npy` (`[time, node]` or `[time, node, feature]`; the first feature is used) and `{train,val,test}_timestamps.npy` (time of day and day of week, both in `[0, 1)`). See [data/README.md](data/README.md) for downloads and preparation.

| Dataset | Nodes | Interval | Split | Download |
| --- | ---: | --- | --- | --- |
| SD, GBA, GLA, CA | 716 / 2,352 / 3,834 / 8,600 | 15 min | 6:2:2 | [BasicTS datasets](https://github.com/GestaltCogTeam/BasicTS/blob/master/docs/getting_started.md) |
| Electricity | 321 | 60 min | 7:1:2 | [BasicTS datasets](https://github.com/GestaltCogTeam/BasicTS/blob/master/docs/getting_started.md) |
| Weather | 21 | 10 min | 7:1:2 | included (from BasicTS) |
| AirQuality | 437 (404 kept) | 60 min | 6:2:2 | [TSL air quality](https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/air/air_quality.zip) |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## License

The code is released under the [MIT License](LICENSE). STID and PatchSTG code keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The SD slice in `data/SD_demo/` is from LargeST and is licensed under CC BY-NC 4.0.


