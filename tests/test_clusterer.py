"""Tests for the cross-ensemble pocket clusterer."""
from __future__ import annotations

import numpy as np
import pytest

from lacuna.models import Atom, Residue, Structure, Pocket
from lacuna.pockets.clusterer import cluster_pockets, compute_crypticity


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


class TestComputeCrypticity:
    def test_absent_in_apo_is_maximally_cryptic(self):
        """A pocket absent in the apo state scores crypticity == druggability-when-open."""
        assert compute_crypticity(apo_volume=0.0, max_volume=300.0,
                                  max_druggability=0.8) == pytest.approx(0.8)

    def test_constitutive_pocket_is_not_cryptic(self):
        """A pocket the same size in apo and open states is not cryptic."""
        assert compute_crypticity(apo_volume=300.0, max_volume=300.0,
                                  max_druggability=0.9) == 0.0

    def test_partial_opening(self):
        """Half-opening yields half-weighted crypticity."""
        # opening = (300-150)/300 = 0.5 -> crypticity = 0.5 * 0.6 = 0.3
        assert compute_crypticity(apo_volume=150.0, max_volume=300.0,
                                  max_druggability=0.6) == pytest.approx(0.3)

    def test_zero_volume_is_safe(self):
        assert compute_crypticity(apo_volume=0.0, max_volume=0.0,
                                  max_druggability=0.5) == 0.0


class TestCrypticityIntegration:
    def _scenario(self):
        """Constitutive pocket C (always open) + cryptic pocket X (absent in apo)."""
        constitutive = [
            [_make_pocket((0.0, 0.0, 0.0), volume=300.0, conformer_idx=i)]
            for i in range(4)
        ]
        # X absent from conformer 0, appears in 1,2,3
        cryptic = [[],
                   [_make_pocket((40.0, 0.0, 0.0), volume=300.0, conformer_idx=1)],
                   [_make_pocket((40.0, 0.0, 0.0), volume=300.0, conformer_idx=2)],
                   [_make_pocket((40.0, 0.0, 0.0), volume=300.0, conformer_idx=3)]]
        pocket_lists = [constitutive[i] + cryptic[i] for i in range(4)]
        return pocket_lists

    def _by_centroid(self, clusters, x):
        return next(c for c in clusters if abs(c.centroid[0] - x) < 1.0)

    def test_cryptic_pocket_has_higher_crypticity(self):
        clusters = cluster_pockets(self._scenario(), n_conformers=4)
        C = self._by_centroid(clusters, 0.0)
        X = self._by_centroid(clusters, 40.0)
        assert X.crypticity > C.crypticity
        assert C.crypticity == 0.0  # constitutive
        assert X.apo_volume_a3 == 0.0  # absent in apo

    def test_crypticity_ranking_surfaces_cryptic_first(self):
        clusters = cluster_pockets(self._scenario(), n_conformers=4, rank_by="crypticity")
        assert abs(clusters[0].centroid[0] - 40.0) < 1.0  # X ranks first

    def test_balanced_ranking_favors_persistent(self):
        clusters = cluster_pockets(self._scenario(), n_conformers=4, rank_by="balanced")
        assert abs(clusters[0].centroid[0] - 0.0) < 1.0  # constitutive C ranks first

    def test_volume_dynamics_recorded(self):
        clusters = cluster_pockets(self._scenario(), n_conformers=4)
        for c in clusters:
            assert c.volume_min_a3 <= c.volume_a3 <= c.volume_max_a3
            assert c.max_druggability >= c.druggability - 1e-9

    def test_invalid_rank_by_raises(self):
        with pytest.raises(ValueError):
            cluster_pockets(self._scenario(), n_conformers=4, rank_by="nonsense")
