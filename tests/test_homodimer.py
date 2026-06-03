"""Tests for biological assembly construction (--homodimer support)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lacuna.io.structure import (
    load_structure,
    coords_array,
    make_biological_assembly,
    _parse_biomt_pdb,
    _is_identity,
)


# Single-chain mini structure with REMARK 350 BIOMT for a C2 homodimer.
# BIOMT 1 = identity (keep chain A), BIOMT 2 = 180° rotation around z-axis.
_MONOMER_WITH_BIOMT = """\
REMARK 350 BIOMOLECULE: 1
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   2  0.000000 -1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  N   ALA A   2       4.000   5.000   6.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       5.000   6.000   7.000  1.00  0.00           C
END
"""

# Same structure but no BIOMT records
_MONOMER_NO_BIOMT = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  N   ALA A   2       4.000   5.000   6.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       5.000   6.000   7.000  1.00  0.00           C
END
"""


@pytest.fixture
def monomer_with_biomt(tmp_path: Path) -> Path:
    p = tmp_path / "monomer_biomt.pdb"
    p.write_text(_MONOMER_WITH_BIOMT)
    return p


@pytest.fixture
def monomer_no_biomt(tmp_path: Path) -> Path:
    p = tmp_path / "monomer_no_biomt.pdb"
    p.write_text(_MONOMER_NO_BIOMT)
    return p


class TestParseBiomtPdb:
    def test_reads_two_matrices(self, monomer_with_biomt):
        matrices = _parse_biomt_pdb(monomer_with_biomt)
        assert len(matrices) == 2

    def test_first_matrix_is_identity(self, monomer_with_biomt):
        matrices = _parse_biomt_pdb(monomer_with_biomt)
        R, t = matrices[0]
        assert _is_identity(R, t)

    def test_second_matrix_is_c2_rotation(self, monomer_with_biomt):
        matrices = _parse_biomt_pdb(monomer_with_biomt)
        R, t = matrices[1]
        expected_R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected_R, atol=1e-5)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-5)

    def test_no_biomt_returns_empty(self, monomer_no_biomt):
        matrices = _parse_biomt_pdb(monomer_no_biomt)
        assert matrices == []


class TestMakeBiologicalAssembly:
    def test_doubles_atom_count(self, monomer_with_biomt):
        structure = load_structure(monomer_with_biomt)
        n_orig = len(structure.atoms)
        assembly = make_biological_assembly(monomer_with_biomt, structure)
        assert len(assembly.atoms) == n_orig * 2

    def test_doubles_residue_count(self, monomer_with_biomt):
        structure = load_structure(monomer_with_biomt)
        n_orig = len(structure.residues)
        assembly = make_biological_assembly(monomer_with_biomt, structure)
        assert len(assembly.residues) == n_orig * 2

    def test_creates_second_chain(self, monomer_with_biomt):
        structure = load_structure(monomer_with_biomt)
        assembly = make_biological_assembly(monomer_with_biomt, structure)
        chain_ids = set(a.chain_id for a in assembly.atoms)
        assert len(chain_ids) == 2
        assert "A" in chain_ids

    def test_c2_rotation_applied_correctly(self, monomer_with_biomt):
        """Atoms in the second chain should be the C2-rotated version of chain A."""
        structure = load_structure(monomer_with_biomt)
        assembly = make_biological_assembly(monomer_with_biomt, structure)

        orig_atoms = [a for a in assembly.atoms if a.chain_id == "A"]
        new_chain_id = next(a.chain_id for a in assembly.atoms if a.chain_id != "A")
        mate_atoms = [a for a in assembly.atoms if a.chain_id == new_chain_id]

        assert len(orig_atoms) == len(mate_atoms)

        # C2 rotation: x → -x, y → -y, z unchanged
        for orig, mate in zip(orig_atoms, mate_atoms):
            ox, oy, oz = orig.coords
            mx, my, mz = mate.coords
            assert abs(mx - (-ox)) < 1e-3
            assert abs(my - (-oy)) < 1e-3
            assert abs(mz - oz) < 1e-3

    def test_no_biomt_returns_unchanged(self, monomer_no_biomt):
        structure = load_structure(monomer_no_biomt)
        n_orig = len(structure.atoms)
        assembly = make_biological_assembly(monomer_no_biomt, structure)
        assert len(assembly.atoms) == n_orig

    def test_atom_residue_indices_consistent(self, monomer_with_biomt):
        """All atom serials in residue.atom_indices should resolve to valid atoms."""
        structure = load_structure(monomer_with_biomt)
        assembly = make_biological_assembly(monomer_with_biomt, structure)

        all_serials = {a.serial for a in assembly.atoms}
        for res in assembly.residues:
            for serial in res.atom_indices:
                assert serial in all_serials, f"Orphan serial {serial} in residue {res.label}"
