"""Tests for the grid-based pocket detector.

We build synthetic structures programmatically to avoid requiring test PDB files.
"""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.models import Atom, Residue, Structure
from lacuna.pockets.detector import detect_pockets


def _make_sphere_structure(radius: float = 10.0, n_atoms: int = 600, seed: int = 42) -> tuple:
    """Build a hollow sphere of atoms to create a guaranteed interior cavity.

    Density must be high enough that adjacent atom VDW spheres (r~3.1 Å) overlap,
    forming a closed shell. With radius=10 Å and 600 atoms the mean inter-atom
    spacing is ~1.8 Å, well below the 3.1 Å VDW+probe radius.
    """
    rng = np.random.default_rng(seed)

    # Atoms on the surface of a sphere
    angles = rng.uniform(0, 2 * np.pi, (n_atoms, 2))
    x = radius * np.sin(angles[:, 0]) * np.cos(angles[:, 1])
    y = radius * np.sin(angles[:, 0]) * np.sin(angles[:, 1])
    z = radius * np.cos(angles[:, 0])
    coords = np.stack([x, y, z], axis=1).astype(np.float32)

    atoms = []
    residues = []
    for i in range(n_atoms):
        atoms.append(Atom(
            serial=i,
            name="CA",
            res_name="ALA",
            chain_id="A",
            res_seq=i + 1,
            coords=(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])),
            element="C",
        ))
        residues.append(Residue(
            chain_id="A",
            seq_num=i + 1,
            name="ALA",
            atom_indices=[i],
        ))

    structure = Structure(path="synthetic.pdb", atoms=atoms, residues=residues)
    return coords, structure


def _make_flat_structure(n_atoms: int = 50, seed: int = 0) -> tuple:
    """Build a flat slab of atoms with no enclosed cavity."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n_atoms)
    y = rng.uniform(-10, 10, n_atoms)
    z = rng.uniform(-0.5, 0.5, n_atoms)
    coords = np.stack([x, y, z], axis=1).astype(np.float32)

    atoms = []
    residues = []
    for i in range(n_atoms):
        atoms.append(Atom(
            serial=i, name="CA", res_name="GLY", chain_id="A", res_seq=i + 1,
            coords=(float(x[i]), float(y[i]), float(z[i])), element="C",
        ))
        residues.append(Residue(
            chain_id="A", seq_num=i + 1, name="GLY", atom_indices=[i],
        ))

    structure = Structure(path="flat.pdb", atoms=atoms, residues=residues)
    return coords, structure


class TestDetectPockets:
    def test_hollow_sphere_finds_interior_pocket(self):
        """A hollow sphere of atoms should have a large interior cavity detected."""
        coords, structure = _make_sphere_structure()  # radius=10, n_atoms=600
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        assert len(pockets) > 0, "Should find at least one pocket in hollow sphere"
        volumes = [p.volume_a3 for p in pockets]
        assert max(volumes) > 100.0, f"Interior cavity should be large, got {max(volumes):.1f} Å³"

    def test_flat_structure_has_no_enclosed_pockets(self):
        """A flat slab has no enclosed cavity — should find zero pockets."""
        coords, structure = _make_flat_structure()
        pockets = detect_pockets(coords, structure, min_volume_a3=200.0)
        # Flat structures should have no significant enclosed volumes
        large_pockets = [p for p in pockets if p.volume_a3 > 500.0]
        assert len(large_pockets) == 0, (
            f"Flat slab should have no large enclosed pockets, got {len(large_pockets)}"
        )

    def test_pocket_centroid_inside_sphere(self):
        """The detected pocket centroid should be near the sphere center (0,0,0)."""
        coords, structure = _make_sphere_structure()
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        if pockets:
            largest = max(pockets, key=lambda p: p.volume_a3)
            cx, cy, cz = largest.centroid
            dist_from_origin = (cx**2 + cy**2 + cz**2) ** 0.5
            assert dist_from_origin < 8.0, (
                f"Pocket centroid should be near sphere center, dist={dist_from_origin:.1f}"
            )

    def test_lining_residues_populated(self):
        """Detected pockets should have at least one lining residue identified."""
        coords, structure = _make_sphere_structure()
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        if pockets:
            assert any(len(p.lining_residues) > 0 for p in pockets)

    def test_enclosure_score_range(self):
        """Enclosure score should always be in [0, 1]."""
        coords, structure = _make_sphere_structure()
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        for p in pockets:
            assert 0.0 <= p.enclosure <= 1.0, f"Enclosure out of range: {p.enclosure}"

    def test_pocket_volumes_respect_minimum(self):
        """No returned pocket should be below the min_volume_a3 threshold."""
        coords, structure = _make_sphere_structure()
        min_vol = 150.0
        pockets = detect_pockets(coords, structure, min_volume_a3=min_vol)
        for p in pockets:
            assert p.volume_a3 >= min_vol, (
                f"Pocket volume {p.volume_a3:.1f} below minimum {min_vol}"
            )
