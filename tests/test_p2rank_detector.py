"""Tests for the P2Rank detector adapter and the shared pocket characterizer.

P2Rank itself is a JVM tool that is not installed in CI, so these tests cover the
parts that run without it: the predictions CSV parser (pure), availability
detection, and ``characterize_pocket`` (Lacuna's own geometry used to describe an
externally-proposed pocket location).
"""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.models import Atom, Residue, Structure
from lacuna.pockets.detector import detect_pockets, characterize_pocket
from lacuna.pockets import p2rank_detector as p2r


# ── synthetic cavity (mirrors tests/test_detector.py) ───────────────────────────

def _make_structure(coords: np.ndarray, res_name: str = "ALA"):
    atoms, residues = [], []
    for i, (x, y, z) in enumerate(coords):
        atoms.append(Atom(
            serial=i, name="CA", res_name=res_name,
            chain_id="A", res_seq=i + 1,
            coords=(float(x), float(y), float(z)), element="C",
        ))
        residues.append(Residue(chain_id="A", seq_num=i + 1, name=res_name, atom_indices=[i]))
    return coords.astype(np.float32), Structure(path="syn.pdb", atoms=atoms, residues=residues)


def _box_pocket(inner: float = 6.0, atom_spacing: float = 1.5):
    """Rectangular box open at the top: 5-sided enclosure = deep surface pocket."""
    pts = []
    for x in np.arange(-inner, inner + atom_spacing, atom_spacing):
        for y in np.arange(-inner, inner + atom_spacing, atom_spacing):
            pts.append((x, y, -inner))
    for z in np.arange(-inner + atom_spacing, atom_spacing, atom_spacing):
        for x in np.arange(-inner, inner + atom_spacing, atom_spacing):
            pts.append((x, -inner, z))
            pts.append((x, inner, z))
        for y in np.arange(-inner + atom_spacing, inner, atom_spacing):
            pts.append((-inner, y, z))
            pts.append((inner, y, z))
    return _make_structure(np.array(pts))


# ── fixtures: a realistic P2Rank predictions.csv ────────────────────────────────

_PRED_CSV = (
    "name,rank,score,probability,sas_points,surf_atoms,"
    "center_x,center_y,center_z,residue_ids,surf_atom_ids\n"
    "input.pdb,1,28.34,0.812,52,41,  11.20,  5.88, 20.11, A_10 A_11 A_57,  120 121 122\n"
    "input.pdb,2,9.11,0.341,18,15,  -3.40, 12.70, -1.05, A_30 A_31,  201 202\n"
)


class TestParsePredictions:
    def test_basic_parse(self):
        rows = p2r.parse_p2rank_predictions(_PRED_CSV)
        assert len(rows) == 2
        assert rows[0]["rank"] == 1
        assert rows[0]["probability"] == pytest.approx(0.812)
        assert rows[0]["center"] == pytest.approx((11.20, 5.88, 20.11))
        assert rows[0]["residue_ids"] == ["A_10", "A_11", "A_57"]
        # Without a structure, labels fall back to seq:chain form.
        assert rows[0]["residues"] == ["10:A", "11:A", "57:A"]

    def test_residue_names_resolved_from_structure(self):
        residues = [
            Residue(chain_id="A", seq_num=10, name="ALA"),
            Residue(chain_id="A", seq_num=11, name="PHE"),
            Residue(chain_id="A", seq_num=57, name="TRP"),
        ]
        struct = Structure(path="x", atoms=[], residues=residues)
        rows = p2r.parse_p2rank_predictions(_PRED_CSV, struct)
        assert rows[0]["residues"] == ["ALA10:A", "PHE11:A", "TRP57:A"]

    def test_ranks_and_ordering(self):
        rows = p2r.parse_p2rank_predictions(_PRED_CSV)
        assert [r["rank"] for r in rows] == [1, 2]

    def test_empty_and_malformed_lines_ignored(self):
        text = _PRED_CSV + "\n" + "garbage,line,without,numbers\n"
        rows = p2r.parse_p2rank_predictions(text)
        assert len(rows) == 2  # the malformed trailing line is skipped

    def test_metrics_can_parse_labels(self):
        # The size-robust metric parses seq numbers from the labels; ensure ours work.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from metrics import found_resnums
        rows = p2r.parse_p2rank_predictions(_PRED_CSV)
        assert found_resnums(rows[0]["residues"]) == {10, 11, 57}


class TestAvailability:
    def test_executable_lookup_returns_none_or_path(self):
        exe = p2r.p2rank_executable()
        assert exe is None or isinstance(exe, str)
        assert p2r.p2rank_available() == (exe is not None)

    def test_run_p2rank_raises_when_unavailable(self, tmp_path):
        if p2r.p2rank_available():
            pytest.skip("P2Rank is installed; cannot test the unavailable path")
        pdb = tmp_path / "x.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
        with pytest.raises(RuntimeError, match="P2Rank not found"):
            p2r.run_p2rank(pdb, chain="A", executable=None)


class TestCharacterizePocket:
    def test_characterize_at_alpha_pocket_center_matches(self):
        """Characterizing the alpha detector's own pocket center reproduces a
        comparable pocket - proving external detector locations get features on
        the same scale as the built-in detector."""
        coords, structure = _box_pocket(inner=6.0)
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        assert pockets, "box pocket should be detected"
        ref = max(pockets, key=lambda p: p.volume_a3)

        got = characterize_pocket(coords, structure, ref.centroid)
        assert got is not None
        assert got.volume_a3 > 0.0
        assert 0.0 <= got.enclosure <= 1.0
        assert 0.0 <= got.hydrophobic_fraction <= 1.0
        assert len(got.lining_residues) > 0
        # Centroid should land near the reference pocket (same cavity).
        d = float(np.linalg.norm(np.array(got.centroid) - np.array(ref.centroid)))
        assert d < 5.0, f"characterized centroid {d:.1f} A from alpha pocket"

    def test_characterize_far_from_protein_returns_none(self):
        coords, structure = _box_pocket(inner=6.0)
        # A point far outside the padded grid has no void voxels -> None.
        got = characterize_pocket(coords, structure, (1000.0, 1000.0, 1000.0))
        assert got is None

    def test_characterized_pocket_scores(self):
        """A characterized pocket must flow through the druggability scorer."""
        from lacuna.pockets.scorer import score_pocket
        coords, structure = _box_pocket(inner=6.0)
        pockets = detect_pockets(coords, structure, min_volume_a3=50.0)
        ref = max(pockets, key=lambda p: p.volume_a3)
        got = characterize_pocket(coords, structure, ref.centroid)
        s = score_pocket(got)
        assert 0.0 <= s.composite <= 1.0
