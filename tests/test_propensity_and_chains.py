# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Tests for the v1.4.0 report additions: residue propensity and chain shares.

Both are additive: they must not alter ranking, crypticity or any existing field.
The propensity in particular is scored externally against CryptoBench, so the
tests here pin the ARITHMETIC rather than the benchmark number, which lives in
docs/BENCHMARKS.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from lacuna.models import Pocket, PocketCluster
from lacuna.pockets.clusterer import (
    INTERFACE_MIN_RESIDUES,
    INTERFACE_MIN_SHARE,
    _chain_shares,
    residue_propensity,
)


def _pocket(conformer_idx, lining, volume=200.0, centroid=(0.0, 0.0, 0.0)):
    return Pocket(
        centroid=centroid,
        volume_a3=volume,
        enclosure=0.3,
        hydrophobic_fraction=0.5,
        aromatic_count=1,
        lining_residues=list(lining),
        conformer_idx=conformer_idx,
    )


def _cluster(rank, members, lining=None):
    confs = sorted({m.conformer_idx for m in members})
    return PocketCluster(
        rank=rank,
        centroid=(0.0, 0.0, 0.0),
        volume_a3=200.0,
        druggability=0.5,
        persistence=1.0,
        cryptic=False,
        lining_residues=list(lining if lining is not None else []),
        appears_in_conformers=confs,
        member_pockets=members,
    )


class TestChainShares:
    def test_single_chain_is_whole_share(self):
        assert _chain_shares(["LYS12:A", "ASP13:A"]) == {"A": 1.0}

    def test_two_chains_split_by_count(self):
        shares = _chain_shares(["LYS12:A", "ASP13:A", "GLU9:B"])
        assert shares["A"] == pytest.approx(2 / 3)
        assert shares["B"] == pytest.approx(1 / 3)

    def test_empty_lining_gives_empty_shares(self):
        assert _chain_shares([]) == {}

    def test_chain_id_takes_the_last_colon(self):
        # Residue names never contain a colon, but be explicit about the parse.
        assert _chain_shares(["LYS12:A"]) == {"A": 1.0}

    def test_shares_sum_to_one(self):
        labels = [f"ALA{i}:{'A' if i % 3 else 'B'}" for i in range(10)]
        assert sum(_chain_shares(labels).values()) == pytest.approx(1.0)


class TestPropensityArithmetic:
    def test_single_cluster_rank_one_gives_occupancy(self):
        # Residue X lines 3 of the cluster's 4 conformers -> 1/1 * 3/4.
        members = [_pocket(i, ["LYS12:A"]) for i in range(3)] + [_pocket(3, ["ASP99:A"])]
        scores = residue_propensity([_cluster(1, members)])
        assert scores["LYS12:A"] == pytest.approx(0.75)
        assert scores["ASP99:A"] == pytest.approx(0.25)

    def test_rank_two_is_halved(self):
        members = [_pocket(i, ["LYS12:A"]) for i in range(4)]
        assert residue_propensity([_cluster(2, members)])["LYS12:A"] == pytest.approx(0.5)

    def test_clusters_sum(self):
        a = _cluster(1, [_pocket(i, ["LYS12:A"]) for i in range(2)])
        b = _cluster(2, [_pocket(i, ["LYS12:A"]) for i in range(2)])
        assert residue_propensity([a, b])["LYS12:A"] == pytest.approx(1.5)

    def test_one_conformer_contributing_two_pockets_counts_once(self):
        """A conformer can put several member pockets in one cluster.

        Counting member pockets instead of distinct conformers would let occupancy
        exceed 1.0, which is the bug this test exists for.
        """
        members = [_pocket(0, ["LYS12:A"]), _pocket(0, ["LYS12:A"]), _pocket(1, ["LYS12:A"])]
        scores = residue_propensity([_cluster(1, members)])
        assert scores["LYS12:A"] == pytest.approx(1.0)
        assert scores["LYS12:A"] <= 1.0

    def test_unlined_residues_are_absent_not_zero(self):
        scores = residue_propensity([_cluster(1, [_pocket(0, ["LYS12:A"])])])
        assert "ASP99:A" not in scores

    def test_unranked_clusters_are_skipped(self):
        c = _cluster(0, [_pocket(0, ["LYS12:A"])])  # rank 0 = not yet sorted
        assert residue_propensity([c]) == {}

    def test_empty_input(self):
        assert residue_propensity([]) == {}


class TestInterfaceRule:
    def test_thresholds_are_the_documented_ones(self):
        assert INTERFACE_MIN_SHARE == 0.10
        assert INTERFACE_MIN_RESIDUES == 3

    def test_minor_chain_below_residue_floor_is_not_an_interface(self):
        # 2 residues from B out of 20 clears the 10% share but not the 3-residue
        # floor, so a pocket that merely grazes a neighbour is not flagged.
        labels = [f"ALA{i}:A" for i in range(18)] + ["GLY1:B", "GLY2:B"]
        shares = _chain_shares(labels)
        qualifying = sum(
            1 for s in shares.values()
            if s >= INTERFACE_MIN_SHARE and s * len(labels) >= INTERFACE_MIN_RESIDUES
        )
        assert qualifying == 1


class TestReportIntegration:
    def test_report_carries_the_new_fields(self, tmp_path):
        from lacuna.io.writers import write_report
        from lacuna.models import Atom, Residue, Structure

        members = [_pocket(i, ["LYS12:A", "GLU9:B"]) for i in range(2)]
        cluster = _cluster(1, members, lining=["LYS12:A", "GLU9:B"])
        cluster.chain_shares = _chain_shares(cluster.lining_residues)
        cluster.interface = True

        structure = Structure(
            path="x.pdb",
            atoms=[Atom(serial=0, name="CA", res_name="LYS", chain_id="A",
                        res_seq=12, coords=(0.0, 0.0, 0.0), element="C")],
            residues=[Residue(chain_id="A", seq_num=12, name="LYS")],
        )
        path = write_report([cluster], structure, 2, tmp_path)

        import json
        report = json.loads(path.read_text())
        assert "residue_propensity" in report
        assert report["residue_propensity"]["LYS12:A"] == pytest.approx(1.0)
        pocket = report["pockets"][0]
        assert pocket["interface"] is True
        assert pocket["chain_shares"] == {"A": 0.5, "B": 0.5}

    def test_existing_fields_are_untouched(self, tmp_path):
        """The additions must not perturb any field a 1.3.x consumer reads."""
        from lacuna.io.writers import write_report
        from lacuna.models import Structure

        cluster = _cluster(1, [_pocket(0, ["LYS12:A"])], lining=["LYS12:A"])
        structure = Structure(path="x.pdb", atoms=[], residues=[])
        import json
        report = json.loads(write_report([cluster], structure, 1, tmp_path).read_text())
        for key in ("rank", "centroid", "volume_A3", "volume_range_A3",
                    "apo_volume_A3", "druggability", "max_druggability",
                    "persistence", "cryptic", "crypticity", "lining_residues",
                    "appears_in_conformers"):
            assert key in report["pockets"][0], f"1.3.x field {key} disappeared"


class TestBoltzSuperposition:
    def test_rotated_copy_is_recovered(self):
        """A rigidly rotated frame must come back onto the reference.

        This is the Boltz-backend bug in miniature: it folds de novo in its own
        frame, and without superposition every pocket lands in its own cluster.
        """
        from lacuna.ensemble.boltz_backend import _superpose

        rng = np.random.default_rng(0)
        target = rng.normal(size=(40, 3))
        theta = 0.7
        rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
        mobile = target @ rot.T + np.array([10.0, -5.0, 3.0])
        out = _superpose(mobile, target, np.arange(40))
        assert np.allclose(out, target, atol=1e-8)

    def test_no_reflection_is_introduced(self):
        from lacuna.ensemble.boltz_backend import _superpose

        rng = np.random.default_rng(1)
        target = rng.normal(size=(30, 3))
        mirrored = target * np.array([1.0, 1.0, -1.0])
        out = _superpose(mirrored, target, np.arange(30))
        # A mirrored structure is NOT recoverable by a proper rotation, so the
        # result must stay far from the target rather than silently flipping.
        assert not np.allclose(out, target, atol=1e-3)

    def test_too_few_atoms_returns_unchanged(self):
        from lacuna.ensemble.boltz_backend import _superpose

        mobile = np.ones((2, 3))
        out = _superpose(mobile, np.zeros((2, 3)), np.arange(2))
        assert np.allclose(out, mobile)
