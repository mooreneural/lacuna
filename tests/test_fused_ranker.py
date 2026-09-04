"""The fused ranker ships, and the fused pool is not ranked by the wrong model.

Ranking candidates from both detectors with the alpha-only `learned` weights
costs top-five recovery, because a surface proposal is a wrong answer that model
scores highly. Selecting --detector surface-fusion therefore has to bring its
own ranker unless the user asked for a specific one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lacuna.pockets import clusterer


def test_fused_ranker_ships_with_the_package():
    assert (Path(clusterer.__file__).with_name("fused_ranker.npz")).exists()


def test_fused_ranker_is_a_strategy():
    assert "learned-fused" in clusterer.RANK_STRATEGIES


def test_fused_ranker_uses_the_geometry_feature_set():
    feats, mean, scale, coef, _b = clusterer._fused_ranker()
    assert list(feats) == list(clusterer._RANKER_FEATURES)
    assert len(mean) == len(scale) == len(coef) == len(feats)


def test_fused_ranker_is_packaged_in_the_wheel():
    text = (Path(clusterer.__file__).parents[2] / "pyproject.toml").read_text(
        encoding="utf-8")
    assert "lacuna/pockets/fused_ranker.npz" in text, (
        "the model must be force-included or an installed wheel cannot rank "
        "with it, while a source checkout can, which is the worst failure mode")


def test_scoring_differs_from_the_alpha_only_ranker():
    # Same cluster, two fitted models: identical scores would mean one of them
    # is not being read.
    from lacuna.models import PocketCluster

    c = PocketCluster(
        rank=1, centroid=(0.0, 0.0, 0.0), volume_a3=250.0, druggability=0.6,
        persistence=0.5, cryptic=True, lining_residues=["A12:A", "V29:A"],
        appears_in_conformers=[0, 1], volume_min_a3=100.0, volume_max_a3=400.0,
        max_druggability=0.7, apo_volume_a3=90.0, crypticity=0.4,
        member_pockets=[])
    assert clusterer.learned_fused_score(c) != pytest.approx(
        clusterer.learned_score(c))
