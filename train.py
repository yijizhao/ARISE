"""Step 2: train a downstream model on a saved training view and test it.

Examples:
    python train.py --dataset Weather --model STID --selection outputs/selections/Weather_ARISE_b0.10_H12P12
    python train.py --dataset Weather --model STID --full
"""

import argparse
import sys

import numpy as np

from downstream.data import Dataset
from downstream.models import MODELS
from downstream.trainer import Trainer
from project import (
    ROOT,
    dataset_config,
    file_sha256,
    load_json,
    merge,
    project_path,
    read_yaml,
    save_json,
)


def parse_args(argv=None):
    """Command-line options of the training step."""
    parser = argparse.ArgumentParser(description="Train and test a forecasting model.")
    parser.add_argument("--dataset", required=True, help="name in configs/datasets/")
    parser.add_argument("--model", default="STID", choices=list(MODELS))
    view = parser.add_mutually_exclusive_group(required=True)
    view.add_argument("--selection", help="folder written by sample.py")
    view.add_argument("--full", action="store_true", help="train on all training samples")
    parser.add_argument("--history", type=int, default=12, help="H, only with --full")
    parser.add_argument("--horizon", type=int, default=12, help="P, only with --full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, help="training seed")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    parser.add_argument(
        "--full-result", help="result.json of the full-data run, to report MAE degradation"
    )
    parser.add_argument("--output", help="output folder (default: outputs/runs/<name>)")
    return parser.parse_args(argv)


def model_settings(name, data, args):
    """Model and training settings: model YAML, then dataset overrides, then the command line."""
    config = read_yaml(ROOT / "configs" / "models" / f"{name}.yaml")
    overrides = data.get("overrides", {})
    config = merge(config, {"training": overrides.get("training", {})})
    config = merge(config, overrides.get(name, {}))
    for key in ["epochs", "batch_size", "seed"]:
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    return config


def load_selection(path, dataset):
    """Read sample starts saved by sample.py and check that they belong to this dataset."""
    path = project_path(path)
    info = load_json(path / "selection.json")
    if info["dataset"] != dataset.name:
        raise ValueError(f"Selection was made for {info['dataset']}, not {dataset.name}")
    if info["train_sha256"] != file_sha256(dataset.root / "train_data.npy"):
        raise ValueError("Selection was made from a different train_data.npy")
    starts = np.load(path / "window_starts.npy", allow_pickle=False)
    return starts, info


def run(args):
    """Load the view, train, select by validation MAE, test once and save result.json."""
    data = dataset_config(args.dataset)
    dataset = Dataset(data)

    if args.full:
        H, P = args.history, args.horizon
        starts = dataset.all_starts("train", H, P)
        view = {
            "method": "Full",
            "n": len(starts),
            "raw_data_usage": 1.0,
            "training_sample_ratio": 1.0,
        }
        name = f"{args.dataset}_{args.model}_Full_H{H}P{P}"
    else:
        starts, info = load_selection(args.selection, dataset)
        H, P = info["H"], info["P"]
        view = {
            key: info[key]
            for key in ["method", "budget", "n", "raw_data_usage", "training_sample_ratio"]
        }
        view["selection"] = str(args.selection)
        name = f"{args.model}_{project_path(args.selection).name}"

    settings = model_settings(args.model, data, args)
    trainer = Trainer(
        args.model, settings["model"], settings["training"], dataset, H, P, args.device
    )
    output = project_path(args.output or f"outputs/runs/{name}")
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.model} on {args.dataset}, H={H}, P={P}, {view['method']}: {len(starts)} training samples"
    )
    fit = trainer.fit(starts, output)

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "H": H,
        "P": P,
        "view": view,
        **fit,
        "settings": settings,
    }
    if args.full_result:
        full = load_json(project_path(args.full_result))
        if (full["dataset"], full["model"], full["H"], full["P"]) != (
            args.dataset,
            args.model,
            H,
            P,
        ):
            raise ValueError("--full-result must come from the same dataset, model, H and P")
        result["full_MAE"] = full["test"]["MAE"]
        result["degradation_percent"] = 100 * (fit["test"]["MAE"] / full["test"]["MAE"] - 1)
    save_json(output / "result.json", result)

    test = fit["test"]
    mape = "n/a" if test["MAPE"] is None else f"{test['MAPE']:.2f}%"
    print(
        f"Best epoch {fit['best_epoch']}. Test MAE {test['MAE']:.4f}, RMSE {test['RMSE']:.4f}, MAPE {mape}"
    )
    if "degradation_percent" in result:
        print(f"MAE degradation vs. full data: {result['degradation_percent']:.2f}%")
    print(f"Saved to {output}")


def main():
    """Run the training step and report configuration errors without a traceback."""
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
