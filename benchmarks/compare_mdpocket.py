# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Lacuna vs MDpocket: the ensemble-to-ensemble comparison.

MDpocket (Schmidtke et al., Bioinformatics 2011) is the closest prior art to
Lacuna: it runs fpocket's cavity detection over a conformational ensemble or MD
trajectory and reports where pockets recur, explicitly to find transiently formed
sites. It is the fair ensemble baseline, where plain fpocket is a single-structure
one.

Both tools are given the SAME NMA ensemble (same seed, same conformer count), so
this isolates the analysis pipeline: detection, aggregation and ranking. Anything
either tool does to generate conformers is held constant.

Scoring is the repo-standard size-robust criterion (top-5, residue Jaccard >= 0.25)
with lining residues defined identically for both tools (any residue with an atom
within 5 A of a pocket voxel), so neither is favoured by a different convention.

Adapting MDpocket for automated benchmarking
--------------------------------------------
MDpocket outputs a frequency grid, not a ranked pocket list: its intended workflow
is to load that grid in VMD/PyMOL, choose an isovalue by eye, and inspect the
pocket you care about. A benchmark needs a ranked shortlist, so this script
thresholds the grid, groups the surviving voxels into connected pockets, and
ranks them. That adaptation is mine, not MDpocket's, so to avoid handicapping it
the isovalue and ranking rule are swept and its BEST configuration is reported.

Requires mdpocket on PATH (ships with fpocket; WSL/Linux).

Usage:
    python benchmarks/compare_mdpocket.py --folds test --limit 180
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import run_lacuna  # noqa: E402
from cryptobench_benchmark import _fetch, download_cif, main_pocket, MAX_RESIDUES  # noqa: E402
from metrics import jaccard, found_resnums, JACCARD_THRESHOLD  # noqa: E402
from lacuna.io.structure import load_structure, coords_array  # noqa: E402
from lacuna.io.writers import write_structure_pdb  # noqa: E402
from lacuna.ensemble.nma_backend import NMABackend  # noqa: E402

LINING_A = 5.0          # identical to the detector's lining definition
ISOVALUES = (0.2, 0.3, 0.4, 0.5, 0.7)
RANKINGS = ("volume", "freqmass")
MIN_POCKET_VOXELS = 30  # a handful of stray voxels is not a pocket


def parse_dx(path: Path):
    """Read an OpenDX scalar grid. Returns (values[nx,ny,nz], origin, deltas)."""
    counts = origin = None
    deltas = []
    values: list[float] = []
    reading = False
    with open(path) as fh:
        for line in fh:
            if not reading:
                s = line.strip()
                if s.startswith("object 1"):
                    counts = tuple(int(x) for x in s.split()[-3:])
                elif s.startswith("origin"):
                    origin = np.array([float(x) for x in s.split()[1:4]])
                elif s.startswith("delta"):
                    deltas.append([float(x) for x in s.split()[1:4]])
                elif "data follows" in s:
                    reading = True
                continue
            for tok in line.split():
                try:
                    values.append(float(tok))
                except ValueError:
                    pass
    if counts is None or origin is None or len(deltas) < 3:
        raise ValueError(f"malformed dx: {path}")
    arr = np.array(values, dtype=np.float32)
    n = counts[0] * counts[1] * counts[2]
    if arr.size < n:
        raise ValueError(f"dx truncated: {arr.size} < {n}")
    spacing = np.array([deltas[0][0], deltas[1][1], deltas[2][2]])
    return arr[:n].reshape(counts), origin, spacing


def grid_pockets(vals, origin, spacing, isovalue):
    """Threshold the grid and group surviving voxels into connected pockets.

    Returns a list of (centroid, voxel_world_coords, mean_frequency), largest
    first, which the caller then ranks.
    """
    from scipy import ndimage

    mask = vals >= isovalue
    if not mask.any():
        return []
    labels, n = ndimage.label(mask)
    out = []
    for lab in range(1, n + 1):
        idx = np.argwhere(labels == lab)
        if len(idx) < MIN_POCKET_VOXELS:
            continue
        world = origin + idx * spacing
        freq = float(vals[labels == lab].mean())
        out.append((world.mean(axis=0), world, freq))
    return out


def lining_index(structure):
    """Prebuild the atom KD-tree and atom->residue map for a structure.

    Built once per structure rather than per pocket: the isovalue/ranking sweep
    asks for lining residues many times over, and rebuilding the tree each call
    dominated the runtime.
    """
    from scipy.spatial import cKDTree

    coords = np.array([a.coords for a in structure.atoms], dtype=float)
    atom_res: dict[int, int] = {}
    for r in structure.residues:
        for i in r.atom_indices:
            atom_res[i] = r.seq_num
    return cKDTree(coords), atom_res


def lining_residues(index, voxels) -> set[int]:
    """Residues with any atom within LINING_A of a pocket voxel.

    Same rule the detector uses for its own pockets, so the two tools are scored
    on identical footing.
    """
    tree, atom_res = index
    hit_atoms: set[int] = set()
    for ai in tree.query_ball_point(voxels, r=LINING_A):
        hit_atoms.update(ai)
    return {atom_res[i] for i in hit_atoms if i in atom_res}


def run_mdpocket(structure, ensemble_coords, workdir: Path):
    """Write the ensemble as PDBs and run mdpocket over it."""
    names = []
    for i, c in enumerate(ensemble_coords):
        atoms = [replace(a, coords=(float(c[j][0]), float(c[j][1]), float(c[j][2])))
                 for j, a in enumerate(structure.atoms)]
        p = workdir / f"conf{i:03d}.pdb"
        write_structure_pdb(replace(structure, atoms=atoms), p)
        names.append(str(p))
    lst = workdir / "list.txt"
    lst.write_text("\n".join(names) + "\n")
    res = subprocess.run(
        ["mdpocket", "--pdb_list", str(lst), "-o", str(workdir / "mdpout")],
        capture_output=True, text=True, cwd=workdir,
    )
    dx = workdir / "mdpout_freq.dx"
    if res.returncode != 0 or not dx.exists():
        raise RuntimeError(f"mdpocket failed: {res.stderr[-200:]}")
    return dx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="test")
    ap.add_argument("--limit", type=int, default=180)
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if shutil.which("mdpocket") is None:
        print("mdpocket not on PATH. It ships with fpocket; build it and retry "
              "(Linux/WSL only).")
        return

    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    ids = [p for f in args.folds.split(",") for p in folds[f.strip()]]

    rows, t0 = [], time.perf_counter()
    for apo in ids:
        if len(rows) >= args.limit:
            break
        assocs = dataset.get(apo)
        if not assocs:
            continue
        chain, known = main_pocket(assocs)
        if not known:
            continue
        try:
            cif = download_cif(apo)
            s = load_structure(cif, chain=chain)
            if not (10 <= len(s.residues) <= MAX_RESIDUES):
                continue
        except Exception:
            continue

        row = {"id": f"{apo}{chain}", "n_known": len(known)}
        # Lacuna, its shipped configuration.
        try:
            clusters, _ = run_lacuna(cif, args.conformers, chain=chain,
                                     backend_name="nma", rank_by="learned")
            row["lacuna"] = round(max(
                (jaccard(found_resnums(c.lining_residues), known)
                 for c in clusters[:5]), default=0.0), 3)
        except Exception:
            continue

        # MDpocket on the identical ensemble.
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wd = Path(tmp)
                ref = wd / "ref.pdb"
                write_structure_pdb(s, ref)
                sets = NMABackend(seed=42).generate(ref, n_conformers=args.conformers,
                                                    chain=None)
                dx = run_mdpocket(s, [coords_array(s)] + sets, wd)
                vals, origin, spacing = parse_dx(dx)
                index = lining_index(s)
                for iso in ISOVALUES:
                    pockets = grid_pockets(vals, origin, spacing, iso)
                    for rank_by in RANKINGS:
                        key = (lambda p: len(p[1])) if rank_by == "volume" \
                            else (lambda p: len(p[1]) * p[2])
                        top = sorted(pockets, key=key, reverse=True)[:5]
                        best = max((jaccard(lining_residues(index, p[1]), known)
                                    for p in top), default=0.0)
                        row[f"md_{iso}_{rank_by}"] = round(best, 3)
                    row[f"md_{iso}_n"] = len(pockets)
        except Exception as e:
            row["md_error"] = f"{type(e).__name__}: {str(e)[:60]}"

        rows.append(row)
        if len(rows) % 20 == 0:
            lac = sum(1 for r in rows if r.get("lacuna", 0) >= JACCARD_THRESHOLD)
            print(f"  [{len(rows)}/{args.limit}] lacuna={lac}/{len(rows)}  "
                  f"({(time.perf_counter()-t0)/60:.1f}m)", flush=True)

    out = Path(args.out) if args.out else Path(__file__).parent / "mdpocket_comparison.json"
    out.write_text(json.dumps(rows, indent=2))
    _summarize(rows, out)


def _summarize(rows, out):
    import random
    n = len(rows)
    scored = [r for r in rows if "md_error" not in r]
    print(f"\nn={n} structures ({n - len(scored)} mdpocket errors)\n")

    lac = [r.get("lacuna", 0.0) >= JACCARD_THRESHOLD for r in scored]
    print(f"  {'configuration':<28}{'top-5 recovery'}")
    print(f"  {'Lacuna (shipped)':<28}{sum(lac)}/{len(scored)} ({sum(lac)/len(scored):.1%})")

    best_key, best_hits = None, -1
    for iso in ISOVALUES:
        for rank_by in RANKINGS:
            k = f"md_{iso}_{rank_by}"
            hits = [r.get(k, 0.0) >= JACCARD_THRESHOLD for r in scored]
            print(f"  {'MDpocket iso=' + str(iso) + ' ' + rank_by:<28}"
                  f"{sum(hits)}/{len(scored)} ({sum(hits)/len(scored):.1%})")
            if sum(hits) > best_hits:
                best_key, best_hits = k, sum(hits)

    md = [r.get(best_key, 0.0) >= JACCARD_THRESHOLD for r in scored]
    rng = random.Random(0)
    diffs = []
    for _ in range(10000):
        idx = [rng.randrange(len(scored)) for _ in scored]
        diffs.append((sum(lac[i] for i in idx) - sum(md[i] for i in idx)) / len(scored))
    diffs.sort()
    d = (sum(lac) - sum(md)) / len(scored)
    sig = "EXCLUDES 0" if (diffs[250] > 0 or diffs[9750] < 0) else "includes 0"
    print(f"\n  MDpocket best configuration: {best_key}")
    print(f"  Lacuna - MDpocket(best): {d:+.1%}  CI[{diffs[250]:+.1%},{diffs[9750]:+.1%}]  {sig}")
    both = sum(1 for a, b in zip(lac, md) if a and b)
    print(f"  both {both}   Lacuna-only {sum(1 for a, b in zip(lac, md) if a and not b)}"
          f"   MDpocket-only {sum(1 for a, b in zip(lac, md) if b and not a)}")
    print(f"\n  full results -> {out}")


if __name__ == "__main__":
    main()
