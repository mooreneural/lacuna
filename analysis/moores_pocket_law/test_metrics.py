"""Tests for the accounting in dataio.

These cover the arithmetic the whole analysis rests on: what counts as a hit,
where the first qualifying candidate sits, and that R_k = C * V_k holds by
construction rather than by luck.

    python -m pytest analysis/moores_pocket_law/test_metrics.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import JACCARD_THRESHOLD as T
from dataio import first_qualifying_rank as rank_of


def test_threshold_matches_upstream():
    src = (dataio.REPO / "benchmarks/train_ranker.py").read_text(encoding="utf-8")
    m = re.search(r"^JACCARD_THRESHOLD\s*=\s*([0-9.]+)", src, re.M)
    assert m, "could not locate JACCARD_THRESHOLD upstream"
    assert float(m.group(1)) == T


def test_first_qualifying_rank_is_one_based():
    assert rank_of([0.9]) == 1
    assert rank_of([0.0, 0.0, 0.5]) == 3
    assert rank_of([]) is None
    assert rank_of([0.0, 0.1]) is None


def test_rank_uses_inclusive_threshold():
    assert rank_of([T]) == 1
    assert rank_of([T - 1e-12]) is None


def test_rank_finds_the_first_not_the_best():
    assert rank_of([0.0, 0.30, 0.0, 0.99]) == 2


def test_hit_at_k_is_monotone_in_k():
    jl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    r = rank_of(jl)
    hits = [int(r is not None and r <= k) for k in (1, 3, 5, 10, 20)]
    assert hits == sorted(hits)
    assert hits == [0, 0, 0, 1, 1]


def test_identity_recovery_equals_coverage_times_conversion():
    rng = np.random.default_rng(0)
    lists = [list(rng.uniform(0, 0.6, size=int(rng.integers(1, 30))))
             for _ in range(400)]
    for k in (1, 3, 5, 10):
        cov = [any(j >= T for j in jl) for jl in lists]
        hit = [any(j >= T for j in jl[:k]) for jl in lists]
        C = float(np.mean(cov))
        R = float(np.mean(hit))
        covered = [h for h, c in zip(hit, cov) if c]
        V = float(np.mean(covered)) if covered else 0.0
        assert R == pytest.approx(C * V, abs=1e-12)


def test_a_hit_implies_coverage():
    rng = np.random.default_rng(1)
    for _ in range(300):
        jl = list(rng.uniform(0, 0.6, size=int(rng.integers(1, 20))))
        r = rank_of(jl)
        if r is not None and r <= 5:
            assert any(j >= T for j in jl)


def test_pdb_id_handles_multi_character_chains():
    assert dataio.pdb_id_of("1a4uB") == "1a4u"
    assert dataio.pdb_id_of("3brkX") == "3brk"
    assert dataio.pdb_id_of("7qoqAAA") == "7qoq"


def test_bootstrap_is_seeded_and_reproducible():
    v = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    a = dataio.boot_ci(v, n_boot=2000, seed=7)
    b = dataio.boot_ci(v, n_boot=2000, seed=7)
    assert a == b
    m, lo, hi = a
    assert lo <= m <= hi


def test_bootstrap_ci_brackets_a_known_mean():
    v = [1] * 70 + [0] * 30
    m, lo, hi = dataio.boot_ci(v, n_boot=4000, seed=0)
    assert m == pytest.approx(0.70)
    assert lo < 0.70 < hi


def test_loaded_cohort_matches_the_published_pairing():
    cands = dataio.load_candidates()
    paired = {s: d for s, d in cands.items()
              if all(m in d for m in dataio.METHODS)}
    folds = dataio.fold_map()
    n_test = sum(1 for s in paired if dataio.split_of(s, folds) == "test")
    n_train = sum(1 for s in paired if dataio.split_of(s, folds) == "train")
    assert n_test == 178, "test cohort drifted from the published 178"
    assert n_train == 734, "train cohort drifted from the published 734"
