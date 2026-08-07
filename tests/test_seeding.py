"""Sequence-seeded pocket proposals.

The value of seeding depends on it staying within a fixed candidate budget: added
coverage converts to recovery only while the candidate set is small, so a seeding
rule that quietly proposes ten sites on some structures would defeat its own
purpose. Most of what is pinned here is therefore about restraint rather than
recall.
"""
from __future__ import annotations

import numpy as np
import pytest

from lacuna.models import Atom, Residue, Structure
from lacuna.pockets import seeding


def _structure(n_res: int = 30, spacing: float = 3.8) -> Structure:
    """A straight chain of single-atom residues, one per position."""
    atoms, residues = [], []
    for i in range(n_res):
        atoms.append(Atom(serial=i, name="CA", res_name="ALA", chain_id="A",
                          res_seq=i + 1, coords=(i * spacing, 0.0, 0.0),
                          element="C"))
        residues.append(Residue(chain_id="A", seq_num=i + 1, name="ALA",
                                atom_indices=[i]))
    return Structure(path="test.pdb", atoms=atoms, residues=residues)


class TestSeedCenters:
    def test_no_probabilities_proposes_nothing(self):
        assert seeding.seed_centers(_structure(), {}) == []

    def test_scattered_residues_propose_nothing(self):
        """Isolated high scorers are not a site.

        Every other residue on a 3.8 A chain is 7.6 A apart, which links, so the
        scatter has to be wider than LINK_A to be genuinely ungrouped.
        """
        s = _structure(n_res=60)
        probs = {i: 1.0 for i in range(1, 61, 10)}   # 38 A apart
        assert seeding.seed_centers(s, probs) == []

    def test_clustered_residues_propose_one_center(self):
        s = _structure()
        probs = {i: 1.0 for i in (10, 11, 12, 13)}
        centers = seeding.seed_centers(s, probs)
        assert len(centers) == 1
        # Centroid of residues 10-13, which sit at x = 9..12 times the spacing.
        assert centers[0][0] == pytest.approx(np.mean([9, 10, 11, 12]) * 3.8)

    def test_respects_the_seed_budget(self):
        """Candidate count is the quantity that decides whether seeding helps."""
        s = _structure(n_res=200)
        probs = {}
        for start in (1, 40, 80, 120, 160):        # five well-separated groups
            for k in range(4):
                probs[start + k] = 1.0
        centers = seeding.seed_centers(s, probs)
        assert len(centers) <= seeding.MAX_SEEDS

    def test_ranks_groups_by_summed_probability(self):
        s = _structure(n_res=120)
        probs = {1: 0.1, 2: 0.1, 3: 0.1, 60: 0.9, 61: 0.9, 62: 0.9}
        centers = seeding.seed_centers(s, probs, max_seeds=1)
        assert len(centers) == 1
        assert centers[0][0] == pytest.approx(np.mean([59, 60, 61]) * 3.8)

    def test_only_the_top_n_are_eligible(self):
        """A long tail of low-scoring residues must not form its own group."""
        s = _structure(n_res=120)
        probs = {i: 0.9 for i in (10, 11, 12)}
        probs.update({i: 0.01 for i in range(60, 100)})
        centers = seeding.seed_centers(s, probs, top_n=3)
        assert len(centers) == 1
        assert centers[0][0] == pytest.approx(np.mean([9, 10, 11]) * 3.8)

    def test_uses_the_conformer_coordinates_it_is_given(self):
        """A seed is a location in the frame being detected, not the reference.

        Passing reference coordinates for every conformer would place all proposals
        at the apo position and quietly undo the point of an ensemble.
        """
        s = _structure()
        probs = {i: 1.0 for i in (10, 11, 12, 13)}
        shifted = np.asarray([a.coords for a in s.atoms], dtype=float) + [0.0, 25.0, 0.0]
        base = seeding.seed_centers(s, probs)
        moved = seeding.seed_centers(s, probs, coords=shifted)
        assert moved[0][1] == pytest.approx(base[0][1] + 25.0)


class TestSeededPockets:
    def test_no_centers_yields_no_pockets(self):
        s = _structure()
        coords = np.asarray([a.coords for a in s.atoms], dtype=np.float32)
        assert seeding.seeded_pockets(coords, s, {}) == []
