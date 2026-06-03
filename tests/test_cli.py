"""End-to-end tests for the lacuna CLI commands.

Uses click's CliRunner so tests run in-process — no real subprocess, no TTY.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from lacuna.cli import main


# ---------------------------------------------------------------------------
# PDB fixtures
# ---------------------------------------------------------------------------

def _write_box_pdb(path: Path, inner: float = 4.0, spacing: float = 2.0) -> None:
    """Write a box-shaped cavity structure with a real detectable pocket."""
    pts: list[tuple[float, float, float]] = []
    xs = list(np.arange(-inner, inner + spacing, spacing))
    ys = list(np.arange(-inner, inner + spacing, spacing))

    for x in xs:
        for y in ys:
            pts.append((x, y, -inner))

    for z in np.arange(-inner + spacing, spacing, spacing):
        for x in xs:
            pts.append((x, -inner, z))
            pts.append((x,  inner, z))
        for y in np.arange(-inner + spacing, inner, spacing):
            pts.append((-inner, y, z))
            pts.append(( inner, y, z))

    lines: list[str] = []
    for i, (x, y, z) in enumerate(pts, 1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


_MINI_PDB = """\
ATOM      1  N   ALA A   1      -0.677   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   1.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      4  N   ALA A   2       1.500   2.000   1.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       2.500   2.500   2.000  1.00  0.00           C
ATOM      6  N   ALA A   3       3.500   2.000   3.500  1.00  0.00           N
ATOM      7  CA  ALA A   3       4.500   2.500   4.500  1.00  0.00           C
END
"""

_MINI_PDB_WITH_BIOMT = """\
REMARK 350 BIOMOLECULE: 1
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   2  0.000000 -1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000
ATOM      1  N   ALA A   1      -0.677   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   1.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.000   1.000   1.000  1.00  0.00           C
ATOM      4  N   ALA A   2       1.500   2.000   1.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       2.500   2.500   2.000  1.00  0.00           C
ATOM      6  N   ALA A   3       3.500   2.000   3.500  1.00  0.00           N
ATOM      7  CA  ALA A   3       4.500   2.500   4.500  1.00  0.00           C
END
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mini_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "mini.pdb"
    p.write_text(_MINI_PDB)
    return p


@pytest.fixture
def mini_pdb_with_biomt(tmp_path: Path) -> Path:
    p = tmp_path / "mini_biomt.pdb"
    p.write_text(_MINI_PDB_WITH_BIOMT)
    return p


@pytest.fixture
def pocket_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "box.pdb"
    _write_box_pdb(p)
    return p


# ---------------------------------------------------------------------------
# lacuna discover
# ---------------------------------------------------------------------------

class TestDiscoverCommand:
    def test_exits_zero_on_valid_input(self, runner, mini_pdb, tmp_path):
        result = runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "random", "--conformers", "3",
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output

    def test_creates_output_directory(self, runner, mini_pdb, tmp_path):
        out = tmp_path / "lacuna_output"
        runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "random", "--conformers", "3",
            "--output", str(out),
        ])
        assert out.is_dir()

    def test_writes_json_report(self, runner, mini_pdb, tmp_path):
        out = tmp_path / "out"
        runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "random", "--conformers", "3",
            "--output", str(out),
        ])
        report = out / "pocket_report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert "pockets" in data
        assert "n_conformers" in data
        assert data["n_conformers"] == 4  # 3 generated + input structure

    def test_quiet_suppresses_stdout(self, runner, mini_pdb, tmp_path):
        result = runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "random", "--conformers", "3",
            "--quiet", "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0
        assert "Lacuna" not in result.output

    def test_nma_backend(self, runner, mini_pdb, tmp_path):
        result = runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "nma", "--conformers", "3",
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output

    def test_random_backend(self, runner, mini_pdb, tmp_path):
        result = runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "random", "--conformers", "3",
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output

    def test_emit_pocket_pdbs(self, runner, pocket_pdb, tmp_path):
        out = tmp_path / "out"
        runner.invoke(main, [
            "discover", str(pocket_pdb),
            "--backend", "random", "--conformers", "3",
            "--emit-pocket-pdbs", "--output", str(out),
        ])
        assert len(list(out.glob("pocket_*_site.pdb"))) > 0

    def test_emit_vina_boxes(self, runner, pocket_pdb, tmp_path):
        out = tmp_path / "out"
        runner.invoke(main, [
            "discover", str(pocket_pdb),
            "--backend", "random", "--conformers", "3",
            "--emit-vina-boxes", "--output", str(out),
        ])
        assert len(list(out.glob("pocket_*_vina.conf"))) > 0

    def test_emit_boltz_constraints(self, runner, pocket_pdb, tmp_path):
        out = tmp_path / "out"
        runner.invoke(main, [
            "discover", str(pocket_pdb),
            "--backend", "random", "--conformers", "3",
            "--emit-boltz-constraints", "--output", str(out),
        ])
        assert len(list(out.glob("pocket_*_constraint.yaml"))) > 0

    def test_top_limits_reported_pockets(self, runner, pocket_pdb, tmp_path):
        out = tmp_path / "out"
        runner.invoke(main, [
            "discover", str(pocket_pdb),
            "--backend", "random", "--conformers", "3",
            "--top", "2", "--output", str(out),
        ])
        report = json.loads((out / "pocket_report.json").read_text())
        assert len(report["pockets"]) <= 2

    def test_min_druggability_filter(self, runner, mini_pdb, tmp_path):
        out = tmp_path / "out"
        runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "random", "--conformers", "3",
            "--min-druggability", "0.99", "--output", str(out),
        ])
        report = json.loads((out / "pocket_report.json").read_text())
        for p in report["pockets"]:
            assert p["druggability"] >= 0.99

    def test_homodimer_no_biomt_exits_zero(self, runner, mini_pdb, tmp_path):
        """--homodimer on a structure with no BIOMT records should warn but succeed."""
        result = runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "random", "--conformers", "3",
            "--homodimer", "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output

    def test_homodimer_with_biomt_doubles_chains(self, runner, mini_pdb_with_biomt, tmp_path):
        """--homodimer with BIOMT records should report the combined assembly."""
        result = runner.invoke(main, [
            "discover", str(mini_pdb_with_biomt),
            "--backend", "random", "--conformers", "3",
            "--homodimer", "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output
        assert "2 chains" in result.output

    def test_invalid_backend_exits_nonzero(self, runner, mini_pdb, tmp_path):
        result = runner.invoke(main, [
            "discover", str(mini_pdb),
            "--backend", "notabackend",
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_version_flag(self, runner):
        from lacuna import __version__
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


# ---------------------------------------------------------------------------
# lacuna dock-prep
# ---------------------------------------------------------------------------

class TestDockPrepCommand:
    @pytest.fixture
    def discover_report(self, runner, pocket_pdb, tmp_path) -> tuple[Path, Path]:
        """Run discover on a pocket structure; return (report_path, pdb_path)."""
        out = tmp_path / "discover_out"
        runner.invoke(main, [
            "discover", str(pocket_pdb),
            "--backend", "random", "--conformers", "3",
            "--output", str(out),
        ])
        return out / "pocket_report.json", pocket_pdb

    def test_exits_zero(self, runner, discover_report, tmp_path):
        report, pdb = discover_report
        result = runner.invoke(main, [
            "dock-prep", str(report), str(pdb),
            "--output", str(tmp_path / "dock"),
        ])
        assert result.exit_code == 0, result.output

    def test_vina_format(self, runner, discover_report, tmp_path):
        report, pdb = discover_report
        dock_out = tmp_path / "dock"
        runner.invoke(main, [
            "dock-prep", str(report), str(pdb),
            "--format", "vina", "--output", str(dock_out),
        ])
        assert len(list(dock_out.glob("pocket_*_vina.conf"))) > 0

    def test_boltz_format(self, runner, discover_report, tmp_path):
        report, pdb = discover_report
        dock_out = tmp_path / "dock"
        runner.invoke(main, [
            "dock-prep", str(report), str(pdb),
            "--format", "boltz", "--output", str(dock_out),
        ])
        assert len(list(dock_out.glob("pocket_*_constraint.yaml"))) > 0

    def test_top_limits_files(self, runner, discover_report, tmp_path):
        report, pdb = discover_report
        n_pockets = len(json.loads(report.read_text())["pockets"])
        if n_pockets < 2:
            pytest.skip("need at least 2 pockets to test --top")
        dock_out = tmp_path / "dock"
        runner.invoke(main, [
            "dock-prep", str(report), str(pdb),
            "--format", "vina", "--top", "1", "--output", str(dock_out),
        ])
        assert len(list(dock_out.glob("pocket_*_vina.conf"))) == 1
