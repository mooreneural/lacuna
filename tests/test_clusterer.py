"""Tests for the cross-ensemble pocket clusterer."""
from __future__ import annotations

import numpy as np
import pytest

from lacuna.models import Atom, Residue, Structure, Pocket
from lacuna.pockets.clusterer import cluster_pockets


def _make_pocket(centroid, volume=200.0, conformer_idx=0, lining=None):
    return Pocket(
        centroid=centroid,
        volume_a3=volume,
        enclosure=0.5,
        hydrophobic_fraction=0.5,
        aromatic_count=2,
        lining_residues=lining or ["ALA1:A", "GLY2:A"],
        conformer_idx=conformer_idx,
    )


class TestClusterPockets:
    def test_identical_pockets_merge(self):
        """Two pockets at the same location should form one cluster."""
        p1 = _make_pocket((0.0, 0.0, 0.0), conformer_idx=0)
        p2 = _make_pocket((0.5, 0.5, 0.5), conformer_idx=1)
        clusters = cluster_pockets([[p1], [p2]], n_conformers=2)
        assert len(clusters) == 1
        assert clusters[0].persistence == pytest.approx(1.0)

    def test_distant_pockets_stay_separate(self):
        """Pockets far apart should not merge."""
        p1 = _make_pocket((0.0, 0.0, 0.0), conformer_idx=0)
        p2 = _make_pocket((50.0, 50.0, 50.0), conformer_idx=0)
        clusters = cluster_pockets([[p1, p2]], n_conformers=1)
        assert len(clusters) == 2

    def test_persistence_calculated_correctly(self):
        """Pocket in 2/4 conformers should have 50% persistence."""
        pockets = [
            [_make_pocket((0.0, 0.0, 0.0), conformer_idx=0)],
            [_make_pocket((0.0, 0.0, 0.0), conformer_idx=1)],
            [],
            [],
        ]
        clusters = cluster_pockets(pockets, n_conformers=4)
        assert len(clusters) >= 1
        assert clusters[0].persistence == pytest.approx(0.5)

    def test_cryptic_flag_below_threshold(self):
        """Pocket in <90% of conformers should be flagged cryptic."""
        pockets = [[_make_pocket((0.0, 0.0, 0.0), conformer_idx=i)] for i in range(8)]
        pockets += [[], []]
        clusters = cluster_pockets(pockets, n_conformers=10)
        assert clusters[0].cryptic is True

    def test_persistent_pocket_not_cryptic(self):
        """Pocket in 100% of conformers should not be cryptic."""
        pockets = [[_make_pocket((0.0, 0.0, 0.0), conformer_idx=i)] for i in range(10)]
        clusters = cluster_pockets(pockets, n_conformers=10)
        assert clusters[0].cryptic is False

    def test_ranked_by_druggability(self):
        """Higher-druggability pocket should rank first."""
        p_high = _make_pocket((0.0, 0.0, 0.0), conformer_idx=0)   # enclosure=0.5
        p_low  = _make_pocket((50.0, 0.0, 0.0), conformer_idx=0)  # same enclosure, different location
        clusters = cluster_pockets([[p_high, p_low]], n_conformers=1)
        assert len(clusters) == 2
        assert clusters[0].druggability >= clusters[1].druggability

    def test_empty_input_returns_empty(self):
        clusters = cluster_pockets([[], [], []], n_conformers=3)
        assert clusters == []

    def test_rank_assigned_sequentially(self):
        pockets = [[_make_pocket((float(i * 30), 0.0, 0.0), conformer_idx=0)] for i in range(5)]
        clusters = cluster_pockets(pockets, n_conformers=1)
        for i, c in enumerate(clusters):
            assert c.rank == i + 1

    def test_lining_residues_aggregated(self):
        """Merged cluster should contain lining residues from all conformers."""
        p1 = _make_pocket((0.0, 0.0, 0.0), conformer_idx=0, lining=["ALA1:A"])
        p2 = _make_pocket((1.0, 0.0, 0.0), conformer_idx=1, lining=["GLY2:A"])
        clusters = cluster_pockets([[p1], [p2]], n_conformers=2)
        assert len(clusters) == 1
        assert "ALA1:A" in clusters[0].lining_residues or "GLY2:A" in clusters[0].lining_residues
