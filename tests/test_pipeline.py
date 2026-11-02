"""End-to-end runs of sample.py and train.py on the bundled SD slice."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    """Run a project script and return the completed process."""
    return subprocess.run([sys.executable, "-B", *args], cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("model", ["STID", "PatchSTG"])
def test_sample_then_train(model, tmp_path):
    """Both models train on an ARISE selection and write result.json."""
    selection = tmp_path / "selection"
    done = run(
        "sample.py",
        "--dataset",
        "SD_demo",
        "--budget",
        "0.5",
        "--strata",
        "2",
        "--output",
        str(selection),
    )
    assert done.returncode == 0, done.stderr
    output = tmp_path / model
    done = run(
        "train.py",
        "--dataset",
        "SD_demo",
        "--model",
        model,
        "--selection",
        str(selection),
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--output",
        str(output),
    )
    assert done.returncode == 0, done.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["view"]["method"] == "ARISE" and result["test"]["MAE"] > 0


def test_patchstg_rejects_datasets_without_coordinates(tmp_path):
    """PatchSTG stops with a clear message on Weather."""
    selection = tmp_path / "selection"
    assert (
        run(
            "sample.py",
            "--dataset",
            "Weather",
            "--method",
            "Random",
            "--budget",
            "0.01",
            "--output",
            str(selection),
        ).returncode
        == 0
    )
    done = run(
        "train.py", "--dataset", "Weather", "--model", "PatchSTG", "--selection", str(selection)
    )
    assert done.returncode != 0 and "SD, GBA, GLA and CA" in done.stderr


def test_infeasible_budget_has_no_traceback():
    """An infeasible ARISE budget ends with a one-line error."""
    done = run("sample.py", "--dataset", "SD_demo", "--budget", "0.01")
    assert done.returncode != 0 and "Traceback" not in done.stderr
