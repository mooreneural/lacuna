"""Tests for the size-robust benchmark metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

# metrics.py lives in benchmarks/, which is not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import metrics  # noqa: E402


def test_jaccard_penalizes_volume_gaming():
    """A bigger pocket raises recall but must NOT raise Jaccard."""
    known = set(range(10, 20))  # 10 known residues
    small = set(range(10, 18))  # 8/10 known, no extras
    huge = set(range(10, 20)) | set(range(100, 200))  # all known + 100 junk

    # Recall rewards the huge pocket (it covers the whole site)...
    assert metrics.residue_recall(huge, known) > metrics.residue_recall(small, known)
    # ...but Jaccard punishes it for being oversized.
    assert metrics.jaccard(huge, known) < metrics.jaccard(small, known)


def test_jaccard_and_recall_bounds():
    known = {1, 2, 3}
    assert metrics.jaccard(set(), known) == 0.0
    assert metrics.jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert metrics.residue_recall(set(), known) == 0.0
    assert metrics.residue_recall(set(), set()) == 0.0  # no divide-by-zero


def test_centroid_distance_handles_missing():
    assert metrics.centroid_distance(None, (0, 0, 0)) == math.inf
    assert metrics.centroid_distance((0, 0, 0), None) == math.inf
    assert metrics.centroid_distance((0, 0, 0), (3, 4, 0)) == 5.0


def test_hotspot_core_hit():
    ca = [(0, 0, 0), (1, 0, 0), (50, 0, 0)]  # 2 near origin, 1 far
    assert metrics.hotspot_core_hit((0, 0, 0), ca, radius=8.0) == 2 / 3
    assert metrics.hotspot_core_hit(None, ca) == 0.0
    assert metrics.hotspot_core_hit((0, 0, 0), []) == 0.0


def test_headline_and_strict_thresholds():
    # Passes headline by centroid alone, but fails strict (needs Jaccard too).
    assert metrics.headline_hit(jac=0.0, centroid_dist=3.0)
    assert not metrics.strict_localized_hit(jac=0.0, centroid_dist=3.0)
    # Passes both when localized AND overlapping.
    assert metrics.strict_localized_hit(jac=0.30, centroid_dist=5.0)
    # Jaccard alone satisfies the headline even with a far centroid.
    assert metrics.headline_hit(jac=0.40, centroid_dist=99.0)


def test_topk_summary_monotone():
    # Hits can only accumulate as k grows, so the rate must be nondecreasing.
    per_k = {1: [False, False], 3: [True, False], 5: [True, True]}
    rates = metrics.summarize_topk(per_k)
    ks = sorted(rates)
    assert all(rates[a] <= rates[b] for a, b in zip(ks, ks[1:]))


def test_bootstrap_ci_brackets_mean():
    hits = [True] * 7 + [False] * 3
    mean, lo, hi = metrics.paired_bootstrap_ci(hits, n_boot=1000, seed=1)
    assert abs(mean - 0.7) < 1e-9
    assert lo <= mean <= hi
    assert metrics.paired_bootstrap_ci([], n_boot=10) == (0.0, 0.0, 0.0)
