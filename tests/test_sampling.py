"""Tests for ARISE and the baseline samplers."""

import json
from pathlib import Path

import numpy as np
import pytest

from sampling import SamplingError, covered_steps
from sampling.arise import select
from sampling.arise.objective import exact_w1
from sampling.arise.strata import choose_strata
from sampling.random_sampling import select_random

CASES = json.loads((Path(__file__).parent / "arise_expected.json").read_text())


@pytest.mark.parametrize("case", CASES)
def test_arise_matches_saved_selection(case):
    """ARISE reproduces saved selections and objective values."""
    raw = np.random.default_rng(case["seed"]).normal(5, 2, (150, 3))
    raw[3, 1] = np.nan
    raw[8, 2] = 0
    starts, info = select(raw, "0.6", 4, 3, case["mode"], strata=2)
    np.testing.assert_array_equal(starts, case["starts"])
    assert info["objective_J"] == pytest.approx(case["score"], abs=1e-14)


def test_arise_budget_accounting():
    """Segments use exactly C steps and yield n = C - R(H+P-1) samples (Eq. 9)."""
    raw = np.random.default_rng(0).gamma(2, 3, (2000, 4))
    starts, info = select(raw, "0.2", 12, 12)
    segments = info["segments"]
    assert sum(b - a for a, b in segments) == info["C"] == 400
    assert len(starts) == info["C"] - len(segments) * 23
    assert covered_steps(starts, 24) == info["C"]
    assert all(b1 < a2 for (_, b1), (a2, _) in zip(segments, segments[1:]))


def test_strata_fallback_matches_paper_example():
    """AirQuality with H=P=96 uses G=2 at 10% and G=4 at 20% (Sect. 5.1)."""
    assert choose_strata(5256, 525, 192, 16) == 2
    assert choose_strata(5256, 1051, 192, 16) == 4
    assert choose_strata(5256, 100, 192, 16) is None


def test_arise_reports_fallback_and_infeasible_budget():
    """A too-large G is reduced; a budget below H+P raises SamplingError."""
    raw = np.random.default_rng(0).gamma(2, 3, (1000, 2))
    _, info = select(raw, "0.1", 12, 12, strata=16)
    assert info["requested_strata"] == 16 and info["strata"] == 4
    with pytest.raises(SamplingError):
        select(raw, "0.01", 12, 12)


def test_full_budget_keeps_all_samples():
    """beta = 1 keeps every training sample."""
    starts, _ = select(np.ones((100, 2)), "1", 12, 12)
    np.testing.assert_array_equal(starts, np.arange(77))


def test_random_is_seeded_and_without_replacement():
    """Random draws n = floor(q * N) distinct starts, reproducibly for a seed."""
    raw = np.zeros((1023, 2))
    a, _ = select_random(raw, "0.1", 12, 12, seed=0)
    b, _ = select_random(raw, "0.1", 12, 12, seed=0)
    c, _ = select_random(raw, "0.1", 12, 12, seed=1)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 100 and len(np.unique(a)) == 100 and not np.array_equal(a, c)
    full, _ = select_random(raw, "1", 12, 12)
    np.testing.assert_array_equal(full, np.arange(1000))


def test_exact_w1_of_shifted_sample():
    """W1 between a sample and its shift by 2 equals 2."""
    assert exact_w1(np.array([0.0, 1, 2]), np.array([2.0, 3, 4])) == 2
