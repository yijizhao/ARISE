"""Step 1: select a training view and save its sample starts.

Example:
    python sample.py --dataset Weather --method ARISE --budget 0.10
"""

import argparse
import sys
import time

import numpy as np

from project import dataset_config, file_sha256, load_values, project_path, save_json
from sampling import BUDGET_UNIT, METHODS, SamplingError, covered_steps


def parse_args(argv=None):
    """Command-line options of the sampling step."""
    parser = argparse.ArgumentParser(description="Select a training view.")
    parser.add_argument("--dataset", required=True, help="name in configs/datasets/")
    parser.add_argument("--method", default="ARISE", choices=list(METHODS))
    parser.add_argument("--budget", required=True, help="ratio in (0, 1], e.g. 0.10")
    parser.add_argument("--history", type=int, default=12, help="H")
    parser.add_argument("--horizon", type=int, default=12, help="P")
    parser.add_argument(
        "--strata", type=int, default=16, help="G for ARISE; reduced automatically if infeasible"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for Random")
    parser.add_argument("--output", help="output folder (default: outputs/selections/<name>)")
    return parser.parse_args(argv)


def default_name(args):
    """Folder name that identifies one selection."""
    name = f"{args.dataset}_{args.method}_b{args.budget}_H{args.history}P{args.horizon}"
    if args.method == "ARISE" and args.strata != 16:
        name += f"_G{args.strata}"
    if args.method == "Random" and args.seed != 0:
        name += f"_seed{args.seed}"
    return name


def run(args):
    """Select sample starts and write window_starts.npy and selection.json."""
    data = dataset_config(args.dataset)
    train = load_values(data["root"], "train")
    T = len(train)
    H, P = args.history, args.horizon
    W = H + P
    if H < 1 or P < 1 or T < W:
        raise SamplingError(f"Need H, P >= 1 and T >= H + P (T={T}).")
    N = T - W + 1

    began = time.perf_counter()
    if args.method == "ARISE":
        starts, extra = METHODS["ARISE"](train, args.budget, H, P, data["missing"], args.strata)
    else:
        starts, extra = METHODS["Random"](train, args.budget, H, P, args.seed)
    seconds = time.perf_counter() - began

    used = covered_steps(starts, W)
    selection = {
        "dataset": args.dataset,
        "method": args.method,
        "budget": str(args.budget),
        "budget_unit": BUDGET_UNIT[args.method],
        "H": H,
        "P": P,
        "T": T,
        "N": N,
        "n": int(len(starts)),
        "raw_steps_used": used,
        "raw_data_usage": used / T,
        "training_sample_ratio": len(starts) / N,
        **extra,
        "seconds": seconds,
        "train_sha256": file_sha256(project_path(data["root"]) / "train_data.npy"),
    }

    output = project_path(args.output or f"outputs/selections/{default_name(args)}")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "window_starts.npy", starts)
    save_json(output / "selection.json", selection)

    print(
        f"{args.method} on {args.dataset}: n/N = {len(starts)}/{N} "
        f"({100 * selection['training_sample_ratio']:.2f}%), "
        f"raw-data usage = {100 * selection['raw_data_usage']:.2f}%, {seconds:.1f} s"
    )
    if args.method == "ARISE" and extra.get("strata") is not None:
        if extra["strata"] != args.strata:
            print(f"G = {args.strata} is infeasible for this budget; used G = {extra['strata']}.")
        print(
            f"{len(extra['segments'])} segments, G = {extra['strata']}, J = {extra['objective_J']:.6f}"
        )
    print(f"Saved to {output}")
    return output


def main():
    """Run the sampling step and report infeasible budgets without a traceback."""
    try:
        run(parse_args())
    except (SamplingError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
