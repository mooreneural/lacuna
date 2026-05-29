"""Tests for pocket druggability scoring."""

from __future__ import annotations

import pytest

from lacuna.models import Pocket
from lacuna.pockets.scorer import score_pocket


def _make_pocket(**overrides) -> Pocket:
    defaults = dict(
        centroid=(0.0, 0.0, 0.0),
        volume_a3=300.0,
        enclosure=0.75,
        hydrophobic_fraction=0.6,
        aromatic_count=2,
        lining_residues=["ALA1:A", "PHE2:A"],
        conformer_idx=0,
    )
    defaults.update(overrides)
    return Pocket(**defaults)


class TestScorePocket:
    def test_composite_in_range(self):
        pocket = _make_pocket()
        score = score_pocket(pocket)
        assert 0.0 <= score.composite <= 1.0

    def test_optimal_volume_scores_highest(self):
        ideal = _make_pocket(volume_a3=300.0)
        too_small = _make_pocket(volume_a3=30.0)
        too_large = _make_pocket(volume_a3=2500.0)
        assert score_pocket(ideal).volume_score > score_pocket(too_small).volume_score
        assert score_pocket(ideal).volume_score > score_pocket(too_large).volume_score

    def test_higher_enclosure_scores_better(self):
        buried = _make_pocket(enclosure=0.9)
        open_ = _make_pocket(enclosure=0.1)
        assert score_pocket(buried).enclosure_score > score_pocket(open_).enclosure_score

    def test_hydrophobic_pocket_scores_better(self):
        hyd = _make_pocket(hydrophobic_fraction=0.8)
        polar = _make_pocket(hydrophobic_fraction=0.1)
        assert score_pocket(hyd).hydrophobic_score > score_pocket(polar).hydrophobic_score

    def test_aromatic_count_saturates(self):
        few = _make_pocket(aromatic_count=1)
        many = _make_pocket(aromatic_count=10)
        # Both above max should give the same capped score
        score_few = score_pocket(few).aromatic_score
        score_many = score_pocket(many).aromatic_score
        assert score_many >= score_few
        assert score_many <= 1.0

    def test_perfect_pocket_scores_high(self):
        """A pocket with ideal characteristics should score above 0.7."""
        perfect = _make_pocket(
            volume_a3=300.0,
            enclosure=0.9,
            hydrophobic_fraction=0.8,
            aromatic_count=5,
        )
        score = score_pocket(perfect)
        assert score.composite > 0.7, f"Perfect pocket scored only {score.composite}"

    def test_bad_pocket_scores_low(self):
        """A tiny, open, polar pocket should score below 0.3."""
        bad = _make_pocket(
            volume_a3=20.0,
            enclosure=0.05,
            hydrophobic_fraction=0.0,
            aromatic_count=0,
        )
        score = score_pocket(bad)
        assert score.composite < 0.3, f"Bad pocket scored {score.composite}"
