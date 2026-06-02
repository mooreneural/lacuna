"""Tests for output writers."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lacuna.models import Atom, Residue, Structure, PocketCluster


def _make_cluster(rank=1, centroid=(1.0, 2.0, 3.0), volume=250.0,
                  druggability=0.75, persistence=0.6, cryptic=True,
                  lining=None):
    return PocketCluster(
        rank=rank,
        centroid=centroid,
        volume_a3=volume,
        druggability=druggability,
        persistence=persistence,
        cryptic=cryptic,
        lining_residues=lining or ["ALA100:A", "PHE200:A", "TRP336:A"],
        appears_in_conformers=[0, 1, 2],
    )


def _make_structure():
    atoms = [Atom(serial=1, name="CA", res_name="ALA", chain_id="A",
                  res_seq=1, coords=(0.0, 0.0, 0.0), element="C")]
    residues = [Residue(chain_id="A", seq_num=1, name="ALA", atom_indices=[0])]
    return Structure(
        path="test.pdb",
        atoms=atoms,
        residues=residues,
        sequence={"A": "A"},
    )


class TestWriteReport:
    def test_creates_json_file(self):
        from lacuna.io.writers import write_report
        clusters = [_make_cluster(rank=1), _make_cluster(rank=2)]
        structure = _make_structure()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_report(clusters, structure, n_conformers=10,
                               output_dir=Path(tmpdir))
            assert out.exists()
            data = json.loads(out.read_text())
            assert data["n_pockets_found"] == 2
            assert data["n_conformers"] == 10

    def test_report_pocket_fields(self):
        from lacuna.io.writers import write_report
        c = _make_cluster(druggability=0.85, persistence=0.4, cryptic=True)
        structure = _make_structure()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_report([c], structure, n_conformers=5,
                               output_dir=Path(tmpdir))
            data = json.loads(out.read_text())
            p = data["pockets"][0]
            assert p["druggability"] == pytest.approx(0.85, abs=0.01)
            assert p["cryptic"] is True
            assert p["rank"] == 1


class TestWriteVinaBox:
    def test_creates_conf_file(self):
        from lacuna.io.writers import write_vina_box
        c = _make_cluster(centroid=(5.0, 10.0, -3.0), volume=300.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_vina_box(c, Path(tmpdir), index=0)
            assert out.exists()
            text = out.read_text()
            assert "center_x" in text
            assert "5.000" in text

    def test_box_size_scales_with_volume(self):
        from lacuna.io.writers import write_vina_box
        small = _make_cluster(volume=100.0)
        large = _make_cluster(volume=800.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_s = write_vina_box(small, Path(tmpdir), index=0)
            out_l = write_vina_box(large, Path(tmpdir), index=1)

            def get_size(path):
                for line in path.read_text().splitlines():
                    if line.startswith("size_x"):
                        return float(line.split("=")[1].strip())
            assert get_size(out_l) > get_size(out_s)


class TestWriteBoltzConstraint:
    def test_creates_yaml_file(self):
        from lacuna.io.writers import write_boltz_constraint
        c = _make_cluster(lining=["ALA100:A", "TRP336:A"])
        structure = _make_structure()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_boltz_constraint(c, structure, Path(tmpdir), index=0)
            assert out.exists()
            text = out.read_text()
            assert "sequences:" in text
            assert "constraints:" in text

    def test_contains_smiles_placeholder(self):
        from lacuna.io.writers import write_boltz_constraint
        c = _make_cluster()
        structure = _make_structure()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_boltz_constraint(c, structure, Path(tmpdir), index=0)
            assert "SMILES_HERE" in out.read_text()
