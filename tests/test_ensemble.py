"""Tests for the ensemble generation backends."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from lacuna.ensemble.random_backend import RandomBackend
from lacuna.ensemble.nma_backend import NMABackend


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


# Larger mini PDB — enough Cα atoms for ENM to have non-trivial modes
_MINI_PDB_LARGE = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.520   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.100   1.200   0.000  1.00  0.00           C
ATOM      4  N   ALA A   2       3.620   1.200   0.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       4.200   2.400   0.000  1.00  0.00           C
ATOM      6  C   ALA A   2       5.720   2.400   0.000  1.00  0.00           C
ATOM      7  N   ALA A   3       6.300   3.600   0.000  1.00  0.00           N
ATOM      8  CA  ALA A   3       7.820   3.600   0.000  1.00  0.00           C
ATOM      9  C   ALA A   3       8.400   4.800   0.000  1.00  0.00           C
ATOM     10  N   ALA A   4       9.920   4.800   0.000  1.00  0.00           N
ATOM     11  CA  ALA A   4      10.500   6.000   0.000  1.00  0.00           C
ATOM     12  C   ALA A   4      12.020   6.000   0.000  1.00  0.00           C
ATOM     13  N   ALA A   5      12.600   7.200   0.000  1.00  0.00           N
ATOM     14  CA  ALA A   5      14.120   7.200   0.000  1.00  0.00           C
ATOM     15  C   ALA A   5      14.700   8.400   0.000  1.00  0.00           C
ATOM     16  N   ALA A   6      16.220   8.400   0.000  1.00  0.00           N
ATOM     17  CA  ALA A   6      16.800   9.600   0.000  1.00  0.00           C
ATOM     18  C   ALA A   6      18.320   9.600   0.000  1.00  0.00           C
ATOM     19  N   ALA A   7      18.900  10.800   0.000  1.00  0.00           N
ATOM     20  CA  ALA A   7      20.420  10.800   0.000  1.00  0.00           C
ATOM     21  C   ALA A   7      21.000  12.000   0.000  1.00  0.00           C
ATOM     22  N   ALA A   8      22.520  12.000   0.000  1.00  0.00           N
ATOM     23  CA  ALA A   8      23.100  13.200   0.000  1.00  0.00           C
ATOM     24  C   ALA A   8      24.620  13.200   0.000  1.00  0.00           C
ATOM     25  N   ALA A   9      25.200  14.400   0.000  1.00  0.00           N
ATOM     26  CA  ALA A   9      26.720  14.400   0.000  1.00  0.00           C
ATOM     27  C   ALA A   9      27.300  15.600   0.000  1.00  0.00           C
ATOM     28  N   ALA A  10      28.820  15.600   0.000  1.00  0.00           N
ATOM     29  CA  ALA A  10      29.400  16.800   0.000  1.00  0.00           C
ATOM     30  C   ALA A  10      30.920  16.800   0.000  1.00  0.00           C
END
"""


@pytest.fixture
def large_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "large.pdb"
    p.write_text(_MINI_PDB_LARGE)
    return p


class TestNMABackend:
    def test_generates_correct_count(self, large_pdb):
        backend = NMABackend()
        conformers = backend.generate(large_pdb, n_conformers=5)
        assert len(conformers) == 5

    def test_output_shape_matches_input(self, large_pdb):
        from lacuna.io.structure import load_structure, coords_array
        structure = load_structure(large_pdb)
        n_atoms = len(structure.atoms)

        backend = NMABackend()
        conformers = backend.generate(large_pdb, n_conformers=3)
        for c in conformers:
            assert c.shape == (n_atoms, 3)

    def test_conformers_are_diverse(self, large_pdb):
        from lacuna.io.structure import load_structure, coords_array
        base = coords_array(load_structure(large_pdb))

        backend = NMABackend()
        conformers = backend.generate(large_pdb, n_conformers=5)
        rmsds = [float(np.sqrt(((c - base) ** 2).mean())) for c in conformers]

        assert max(rmsds) > 0.1

    def test_amplitude_increases_with_conformer_index(self, large_pdb):
        """Later conformers should have larger RMSD (amplitude is monotonically scaled)."""
        from lacuna.io.structure import load_structure, coords_array
        base = coords_array(load_structure(large_pdb))

        backend = NMABackend(max_rmsd=2.0)
        conformers = backend.generate(large_pdb, n_conformers=6)
        rmsds = [float(np.sqrt(((c - base) ** 2).mean())) for c in conformers]

        # First conformer should have smaller RMSD than last
        assert rmsds[0] < rmsds[-1]

    def test_falls_back_on_tiny_structure(self, mini_pdb):
        """Structures with fewer than 6 Cα atoms should still return conformers."""
        backend = NMABackend()
        conformers = backend.generate(mini_pdb, n_conformers=4)
        assert len(conformers) == 4

    def test_uniform_gamma_matches_default(self, large_pdb):
        """gamma=None (default) must be numerically identical to an all-ones gamma."""
        from lacuna.io.structure import load_structure, coords_array
        base = coords_array(load_structure(large_pdb))
        backend = NMABackend()
        ca = base[[a.serial for a in load_structure(large_pdb).atoms if a.name == "CA"]]

        # An all-ones spring matrix must reproduce the uniform network. (Tolerance
        # is float32-scale: the default gamma=None path keeps strength as a Python
        # float, while an explicit ndarray promotes the Hessian dtype.)
        modes0, freqs0 = backend._compute_modes(ca)
        modes1, freqs1 = backend._compute_modes(ca, gamma=np.ones((len(ca), len(ca))))
        np.testing.assert_allclose(freqs0, freqs1, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(np.abs(modes0), np.abs(modes1), rtol=1e-3, atol=1e-3)

class TestOpenMMBackend:
    """OpenMM implicit-MD backend — skipped when openmm/pdbfixer are absent."""

    def _backend(self):
        pytest.importorskip("openmm")
        pytest.importorskip("pdbfixer")
        from lacuna.ensemble.openmm_backend import OpenMMBackend
        # Tiny, fast settings for the test.
        return OpenMMBackend(simulation_time_ps=2.0, equilibrate_ps=1.0)

    def test_generates_aligned_conformers(self, large_pdb):
        from lacuna.io.structure import load_structure
        backend = self._backend()
        s = load_structure(large_pdb)
        conformers = backend.generate(large_pdb, n_conformers=2)
        assert len(conformers) == 2
        for c in conformers:
            # Coords must align with the detection structure's heavy-atom order.
            assert c.shape == (len(s.atoms), 3)
            assert np.isfinite(c).all()

    def test_conformers_move_from_input(self, large_pdb):
        from lacuna.io.structure import load_structure, coords_array
        backend = self._backend()
        base = coords_array(load_structure(large_pdb))
        conformers = backend.generate(large_pdb, n_conformers=2)
        assert max(float(np.sqrt(((c - base) ** 2).mean())) for c in conformers) > 1e-3


class TestSpringSoftening:
    def test_softening_springs_changes_modes(self, large_pdb):
        """Per-pair gamma < 1 (the spring-perturbation hook) must change the modes."""
        from lacuna.io.structure import load_structure, coords_array
        structure = load_structure(large_pdb)
        ca = coords_array(structure)[[a.serial for a in structure.atoms if a.name == "CA"]]

        backend = NMABackend()
        gamma = np.ones((len(ca), len(ca)))
        gamma[0, 1] = gamma[1, 0] = 0.05  # soften one (symmetric) contact
        _, freqs_uniform = backend._compute_modes(ca)
        _, freqs_soft = backend._compute_modes(ca, gamma=gamma)
        assert not np.allclose(freqs_uniform, freqs_soft)
