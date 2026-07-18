# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""P2Rank detector adapter - a second pocket detector for detector fusion.

P2Rank (Krivak & Hoksza 2018, https://github.com/rdk/p2rank) is a fast,
template-free machine-learning pocket predictor that scores points on the
solvent-accessible surface. It is complementary to Lacuna's geometric
alpha-sphere detector: where the alpha detector reasons purely about concavity
geometry, P2Rank's learned surface model can rank shallow or unusual sites that
the geometric filter discards or ranks low.

This module runs P2Rank per conformer and returns Lacuna ``Pocket`` objects so
its proposals can be pooled with the alpha detector's and fed into the existing
ensemble clusterer (``lacuna.pockets.clusterer.cluster_pockets``) - i.e. true
detector fusion, not a separate downstream comparison.

Design:
  * P2Rank supplies the *localization* (a pocket center - its learned strength).
  * Pocket features (volume, enclosure, lining residues, hydrophobicity,
    aromaticity) are recomputed at that center by Lacuna's own geometry via
    ``detector.characterize_pocket``, so fused pockets share one identical scale
    and the crypticity/druggability math stays coherent across detectors.

P2Rank is a JVM tool. It is NOT a Python dependency of Lacuna; install it
separately (needs Java 11+) and either put ``prank`` on PATH or point the
``LACUNA_P2RANK`` environment variable at the launcher. When P2Rank is
unavailable the fusion path degrades to the alpha detector alone.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from lacuna.models import Pocket, Structure
from lacuna.io.structure import load_structure
from lacuna.pockets.detector import characterize_pockets

_P2RANK_ENV = "LACUNA_P2RANK"
_EXE_NAMES = ("prank", "p2rank", "prank.sh", "prank.bat")


# ── availability ────────────────────────────────────────────────────────────────

def p2rank_executable() -> str | None:
    """Return the P2Rank launcher path, or None if it cannot be located.

    Checks ``$LACUNA_P2RANK`` first (a direct path to the launcher), then common
    launcher names on PATH.
    """
    env = os.environ.get(_P2RANK_ENV)
    if env:
        p = Path(env)
        if p.exists():
            return str(p)
        found = shutil.which(env)
        if found:
            return found
    for name in _EXE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def p2rank_available() -> bool:
    """True if a P2Rank launcher can be located."""
    return p2rank_executable() is not None


# ── output parsing (pure - unit-testable without a JVM) ──────────────────────────

def _p2rank_residue_labels(res_ids: list[str], structure: Structure | None) -> list[str]:
    """Convert P2Rank ``chain_resnum`` residue ids into Lacuna ``NAME+seq:chain``
    labels, resolving residue names from ``structure`` when available.

    P2Rank emits ids like ``A_123``. Lacuna's downstream metrics only need the
    sequence number and chain, but including the residue name keeps the label
    format identical to the alpha detector's for consistent reporting.
    """
    lookup: dict[tuple[str, int], str] = {}
    if structure is not None:
        for r in structure.residues:
            lookup[(r.chain_id, r.seq_num)] = r.name

    out: list[str] = []
    for rid in res_ids:
        chain, sep, num = rid.partition("_")
        if not sep:  # no underscore - treat whole token as the number, unknown chain
            chain, num = "", rid
        digits = "".join(c for c in num if c.isdigit() or c == "-")
        if not digits:
            continue
        try:
            seq = int(digits)
        except ValueError:
            continue
        name = lookup.get((chain, seq))
        out.append(f"{name}{seq}:{chain}" if name else f"{seq}:{chain}")
    return out


def parse_p2rank_predictions(
    csv_text: str,
    structure: Structure | None = None,
) -> list[dict]:
    """Parse a P2Rank ``*_predictions.csv`` into ranked pocket dicts.

    P2Rank's CSV columns are whitespace-padded:
        name, rank, score, probability, sas_points, surf_atoms,
        center_x, center_y, center_z, residue_ids, surf_atom_ids

    Returns a list of ``{rank, score, probability, center, residue_ids,
    residues}`` dicts, ordered as in the file (rank 1 first).
    """
    rows: list[dict] = []
    reader = csv.reader(io.StringIO(csv_text))
    header: list[str] | None = None
    idx: dict[str, int] = {}
    for raw in reader:
        if not raw:
            continue
        cols = [c.strip() for c in raw]
        if header is None:
            header = cols
            idx = {h: i for i, h in enumerate(header)}
            continue

        def col(key: str) -> str:
            i = idx.get(key)
            return cols[i] if (i is not None and i < len(cols)) else ""

        try:
            center = (float(col("center_x")), float(col("center_y")), float(col("center_z")))
        except ValueError:
            continue  # header echo / malformed line

        rank_s = col("rank")
        rank = int(float(rank_s)) if rank_s else len(rows) + 1
        score = float(col("score")) if col("score") else 0.0
        prob = float(col("probability")) if col("probability") else 0.0
        res_ids = [t for t in col("residue_ids").split() if t]

        rows.append({
            "rank": rank,
            "score": score,
            "probability": prob,
            "center": center,
            "residue_ids": res_ids,
            "residues": _p2rank_residue_labels(res_ids, structure),
        })
    return rows


# ── running P2Rank ───────────────────────────────────────────────────────────────

def run_p2rank(
    pdb_path: Path,
    chain: str | None = None,
    executable: str | None = None,
    extra_args: list[str] | None = None,
) -> list[dict]:
    """Run P2Rank on a structure and return parsed pocket dicts.

    The input is normalized to a clean, single-chain PDB via
    ``load_structure``/``write_structure_pdb`` before invocation - the same
    chain-filtering discipline the fpocket comparison uses - so a dimeric input
    cannot leak pockets from the wrong physical chain.

    Raises ``RuntimeError`` if P2Rank is not installed or the run fails.
    """
    exe = executable or p2rank_executable()
    if exe is None:
        raise RuntimeError(
            "P2Rank not found. Install it (needs Java 11+) and put 'prank' on PATH "
            f"or set ${_P2RANK_ENV} to the launcher path."
        )

    from lacuna.io.writers import write_structure_pdb

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_pdb = tmp_dir / "input.pdb"
        out_dir = tmp_dir / "out"
        structure = load_structure(pdb_path, chain=chain)
        write_structure_pdb(structure, in_pdb)

        cmd = [exe, "predict", "-f", str(in_pdb), "-o", str(out_dir)]
        if extra_args:
            cmd += extra_args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"P2Rank failed (exit {result.returncode}): {result.stderr[-400:]}")

        csv_path = out_dir / "input.pdb_predictions.csv"
        if not csv_path.exists():
            candidates = sorted(out_dir.glob("*_predictions.csv"))
            if not candidates:
                return []
            csv_path = candidates[0]
        return parse_p2rank_predictions(csv_path.read_text(), structure)


def detect_pockets_p2rank(
    coords: np.ndarray,
    structure: Structure,
    executable: str | None = None,
) -> list[Pocket]:
    """Detect pockets in one conformer with P2Rank, returned as Lacuna Pockets.

    Mirrors ``detector.detect_pockets``'s signature so it is a drop-in second
    detector in the ensemble loop. P2Rank supplies each pocket's location; the
    Pocket's features are recomputed by ``characterize_pocket`` so they match the
    alpha detector's scale. Centers with no Lacuna-recognizable cavity in this
    conformer are dropped (see ``characterize_pocket``).
    """
    # Write this conformer's coordinates (structure metadata + conformer coords).
    with tempfile.TemporaryDirectory() as tmp:
        from lacuna.io.writers import write_structure_pdb

        conf_pdb = Path(tmp) / "conf.pdb"
        atoms2 = [
            replace(a, coords=(float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            for i, a in enumerate(structure.atoms)
        ]
        struct2 = replace(structure, atoms=atoms2)
        write_structure_pdb(struct2, conf_pdb)
        preds = run_p2rank(conf_pdb, chain=None, executable=executable)

    if not preds:
        return []

    # Characterize all P2Rank centers against one shared grid (built once).
    characterized = characterize_pockets(coords, structure, [pr["center"] for pr in preds])

    pockets: list[Pocket] = []
    for pr, pocket in zip(preds, characterized):
        if pocket is None:
            continue
        pocket.source = "p2rank"
        pocket.score = pr.get("probability")
        pockets.append(pocket)
    return pockets
