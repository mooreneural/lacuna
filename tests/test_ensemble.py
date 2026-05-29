"""Tests for the ensemble generation backends."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from lacuna.ensemble.random_backend import RandomBackend


# Minimal valid PDB content — 10 alanine residues in a helix
_MINI_PDB = """\
ATOM      1  N   ALA A   1      -0.677   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   1.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   0.000   1.500  1.00  0.00           O
ATOM      5  N   ALA A   2       1.500   2.000   1.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       2.500   2.500   2.000  1.00  0.00           C
ATOM      7  C   ALA A   2       3.500   1.500   2.500  1.00  0.00           C
ATOM      8  O   ALA A   2       4.000   0.500   2.000  1.00  0.00           O
ATOM      9  N   ALA A   3       3.500   2.000   3.500  1.00  0.00           N
ATOM     10  CA  ALA A   3       4.500   2.500   4.500  1.00  0.00           C
END
"""


@pytest.fixture
def mini_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "mini.pdb"
    p.write_text(_MINI_PDB)
    return p


class TestRandomBackend:
    def test_generates_correct_count(self, mini_pdb):
        backend = RandomBackend(seed=0)
        conformers = backend.generate(mini_pdb, n_conformers=5)
        assert len(conformers) == 5

    def test_output_shape_matches_input(self, mini_pdb):
        from lacuna.io.structure import load_structure, coords_array
        structure = load_structure(mini_pdb)
        n_atoms = len(structure.atoms)

        backend = RandomBackend(seed=0)
        conformers = backend.generate(mini_pdb, n_conformers=3)
        for c in conformers:
            assert c.shape == (n_atoms, 3), f"Expected ({n_atoms}, 3), got {c.shape}"

    def test_conformers_are_diverse(self, mini_pdb):
        """Conformers should not all be identical to the input."""
        from lacuna.io.structure import load_structure, coords_array
        structure = load_structure(mini_pdb)
        base = coords_array(structure)

        backend = RandomBackend(seed=42)
        conformers = backend.generate(mini_pdb, n_conformers=5)

        rmsd_values = []
        for c in conformers:
            diff = c - base
            rmsd = float(np.sqrt((diff ** 2).mean()))
            rmsd_values.append(rmsd)

        assert max(rmsd_values) > 0.1, "Conformers should differ from input structure"

    def test_custom_noise_levels(self, mini_pdb):
        """Custom noise levels should produce correspondingly scaled displacement."""
        backend_low = RandomBackend(noise_levels=[0.1], seed=7)
        backend_high = RandomBackend(noise_levels=[2.0], seed=7)

        from lacuna.io.structure import load_structure, coords_array
        structure = load_structure(mini_pdb)
        base = coords_array(structure)

        low_conf = backend_low.generate(mini_pdb, n_conformers=1)[0]
        high_conf = backend_high.generate(mini_pdb, n_conformers=1)[0]

        rmsd_low = float(np.sqrt(((low_conf - base) ** 2).mean()))
        rmsd_high = float(np.sqrt(((high_conf - base) ** 2).mean()))

        assert rmsd_low < rmsd_high, (
            f"Low noise ({rmsd_low:.3f}) should produce smaller RMSD than "
            f"high noise ({rmsd_high:.3f})"
        )

    def test_seed_reproducibility(self, mini_pdb):
        """Same seed should give identical conformers."""
        b1 = RandomBackend(seed=123)
        b2 = RandomBackend(seed=123)
        c1 = b1.generate(mini_pdb, n_conformers=3)
        c2 = b2.generate(mini_pdb, n_conformers=3)
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a, b)
