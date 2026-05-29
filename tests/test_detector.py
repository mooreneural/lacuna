"""Tests for the surface-concavity pocket detector.

The detector finds surface pockets (concavities) via a buriedness filter,
NOT interior voids. Synthetic structures are designed to match this semantic.
"""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.models import Atom, Residue, Structure
from lacuna.pockets.detector import detect_pockets


# ── synthetic structure helpers ───────────────────────────────────────────────

def _make_structure(coords: np.ndarray, res_name: str = "ALA") -> tuple[np.ndarray, Structure]:
    atoms, residues = [], []
    for i, (x, y, z) in enumerate(coords):
        atoms.append(Atom(
            serial=i, name="CA", res_name=res_name,
            chain_id="A", res_seq=i + 1,
            coords=(float(x), float(y), float(z)), element="C",
        ))
        residues.append(Residue(chain_id="A", seq_num=i + 1, name=res_name, atom_indices=[i]))
    return coords.astype(np.float32), Structure(path="syn.pdb", atoms=atoms, residues=residues)


def _box_pocket(inner: float = 6.0, atom_spacing: float = 1.5) -> tuple[np.ndarray, Structure]:
    """Rectangular box open at the top: 5-sided enclosure = deep surface pocket."""
    pts = []
    # Floor
    for x in np.arange(-inner, inner + atom_spacing, atom_spacing):
        for y in np.arange(-inner, inner + atom_spacing, atom_spacing):
            pts.append((x, y, -inner))
    # Four walls (z from -inner+step up to 0, open at z=0)
    for z in np.arange(-inner + atom_spacing, atom_spacing, atom_spacing):
        for x in np.arange(-inner, inner + atom_spacing, atom_spacing):
            pts.append((x, -inner, z))
            pts.append((x,  inner, z))
        for y in np.arange(-inner + atom_spacing, inner, atom_spacing):
            pts.append((-inner, y, z))
            pts.append(( inner, y, z))
    arr = np.array(pts)
    return _make_structure(arr)


def _flat_slab(n: int = 60, width: float = 12.0) -> tuple[np.ndarray, Structure]:
    """Single-layer 2D slab: essentially no enclosure on either flat face."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-width, width, n)
    y = rng.uniform(-width, width, n)
    z = np.zeros(n)
    return _make_structure(np.stack([x, y, z], axis=1))


# ── tests ─────────────────────────────────────────────────────────────────────

class TestDetectPockets:
    def test_box_pocket_is_found(self):
        """A 5-sided box open at the top is a textbook surface pocket."""
        coords, structure = _box_pocket(inner=6.0)
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        assert len(pockets) > 0, "Should find pocket inside the box"

    def test_box_pocket_centroid_inside(self):
        """The detected pocket centroid should be inside the box interior."""
        coords, structure = _box_pocket(inner=6.0)
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        if not pockets:
            pytest.skip("No pockets found — box may be too small for this grid spacing")
        largest = max(pockets, key=lambda p: p.volume_a3)
        cx, cy, cz = largest.centroid
        # Centroid should be within the box interior
        assert abs(cx) < 7.0 and abs(cy) < 7.0, f"Centroid outside box: ({cx:.1f},{cy:.1f},{cz:.1f})"
        assert cz < 0.5, f"Centroid should be inside box (z<0), got z={cz:.2f}"

    def test_flat_slab_no_large_buried_pocket(self):
        """A flat slab has low buriedness — no large pocket should survive the filter."""
        coords, structure = _flat_slab()
        # Use a high min_volume to avoid tiny spurious hits
        pockets = detect_pockets(coords, structure, min_volume_a3=500.0)
        # If any pockets are found, their enclosure should be low (open geometry)
        for p in pockets:
            assert p.enclosure < 0.85, (
                f"Flat slab pocket has suspiciously high enclosure: {p.enclosure:.3f}"
            )

    def test_pocket_enclosure_range(self):
        """Enclosure score should always be in [0, 1]."""
        coords, structure = _box_pocket()
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        for p in pockets:
            assert 0.0 <= p.enclosure <= 1.0, f"Enclosure out of range: {p.enclosure}"

    def test_volumes_respect_minimum(self):
        coords, structure = _box_pocket()
        min_vol = 100.0
        pockets = detect_pockets(coords, structure, min_volume_a3=min_vol)
        for p in pockets:
            assert p.volume_a3 >= min_vol, (
                f"Pocket volume {p.volume_a3:.1f} < minimum {min_vol}"
            )

    def test_lining_residues_populated(self):
        coords, structure = _box_pocket()
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        if pockets:
            assert any(len(p.lining_residues) > 0 for p in pockets)

    def test_hydrophobic_fraction_range(self):
        coords, structure = _box_pocket()
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        for p in pockets:
            assert 0.0 <= p.hydrophobic_fraction <= 1.0
