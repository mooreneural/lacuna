# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""COACH420 general binding-site benchmark.

Read this before interpreting the numbers
-----------------------------------------
COACH420 is a **general** ligand binding-site set, not a cryptic one, and its
structures are **holo**: the ligand is present and the pocket is already open. It
measures a different task from the rest of this suite. Lacuna is built to find
sites that are shut in the input and open only across an ensemble, so here:

  * ensemble sampling contributes little, because nothing needs to open, and
  * the default ``learned`` ranker was fitted on cryptic sites, while
    ``crypticity`` actively demotes pockets that are already open.

The repository already documents the default backend as a relatively weak
general-purpose finder (orthosteric controls recover 4/6). This benchmark exists
to put a number on that rather than leave it as a caveat, and to answer the
reasonable objection that Lacuna has only been measured on CryptoBench.

Expect Lacuna to trail dedicated general-purpose predictors here. A specialised
tool losing on a task it was not built for is specialisation evidence, not a
defect, but it is only honest to say so with the number attached.

Ground truth follows P2Rank's own evaluation: relevant ligands come from the
``coach420(mlig).ds`` annotation (MOAD 2013 codes), so ions and buffers do not
count as binding sites.

    python benchmarks/coach420_benchmark.py --limit 100
    python benchmarks/coach420_benchmark.py --limit 100 --rank-by druggability
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import HOLO_CUTOFF, run_lacuna  # noqa: E402
from metrics import (  # noqa: E402
    CENTROID_THRESHOLD,
    JACCARD_THRESHOLD,
    OVERLAP_THRESHOLD,
    centroid_distance,
    found_resnums,
    jaccard,
    residue_recall,
)

from lacuna.io.structure import load_structure  # noqa: E402
from lacuna.pockets.clusterer import DEFAULT_RANK_BY  # noqa: E402

DATA_DIR = Path(__file__).parent / "coach420_data"
MLIG_URL = "https://raw.githubusercontent.com/rdk/p2rank-datasets/master/coach420(mlig).ds"
MAX_RESIDUES = 700


def fetch_mlig() -> list[tuple[str, str, set[str]]]:
    """Return [(pdb_id, chain, {ligand codes})] from P2Rank's mlig annotation."""
    DATA_DIR.mkdir(exist_ok=True)
    local = DATA_DIR / "coach420_mlig.ds"
    if not local.exists():
        print("  Downloading COACH420 ligand annotation ...", flush=True)
        urllib.request.urlretrieve(MLIG_URL, local)
    out = []
    for line in local.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("HEADER"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        stem = Path(parts[0]).stem          # e.g. "1a26A"
        codes = {c.strip() for p in parts[1:] for c in p.split(",") if c.strip()}
        if len(stem) < 5:
            continue
        out.append((stem[:4], stem[4:], codes))
    return out


def download_pdb(pdb_id: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    dst = DATA_DIR / f"{pdb_id.upper()}.pdb"
    if not dst.exists():
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", dst)
    return dst


def binding_site(pdb_path: Path, chain: str, codes: set[str]):
    """Protein residues within HOLO_CUTOFF of the named ligands, and their centroid.

    Parsed from the raw file rather than through load_structure, which drops
    HETATM records: the ligand is exactly what defines the answer here.
    """
    lig_atoms, prot = [], []
    for line in pdb_path.read_text(errors="replace").splitlines():
        rec = line[:6]
        if rec == "HETATM" and line[17:20].strip().upper() in codes:
            try:
                lig_atoms.append((float(line[30:38]), float(line[38:46]),
                                  float(line[46:54])))
            except ValueError:
                continue
        elif rec == "ATOM  " and line[21] == chain:
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            num = line[22:26].strip()
            if num.lstrip("-").isdigit():
                prot.append((int(num), xyz))
    if not lig_atoms or not prot:
        return set(), None
    lig = np.asarray(lig_atoms)
    site, coords = set(), []
    for num, xyz in prot:
        d = np.linalg.norm(lig - np.asarray(xyz), axis=1).min()
        if d <= HOLO_CUTOFF:
            site.add(num)
            coords.append(xyz)
    centroid = tuple(np.mean(coords, axis=0).tolist()) if coords else None
    return site, centroid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only first N (0=all 420)")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--rank-by", dest="rank_by", default=DEFAULT_RANK_BY)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    entries = fetch_mlig()
    if args.limit:
        entries = entries[:args.limit]
    print("=" * 70)
    print(f"  COACH420 (general binding sites, holo)  n={len(entries)}  "
          f"rank_by={args.rank_by}")
    print("  NOTE: not a cryptic benchmark; see this script's docstring.")
    print("=" * 70, flush=True)

    rows, n_skip, t0 = [], 0, time.perf_counter()
    for i, (pdb_id, chain, codes) in enumerate(entries, 1):
        try:
            path = download_pdb(pdb_id)
            known, ref = binding_site(path, chain, codes)
            if not known or ref is None:
                n_skip += 1
                continue
            s = load_structure(path, chain=chain)
            if not (10 <= len(s.residues) <= MAX_RESIDUES):
                n_skip += 1
                continue
            clusters, _ = run_lacuna(path, args.conformers, chain=chain,
                                     backend_name="nma", rank_by=args.rank_by)
        except Exception as e:
            n_skip += 1
            print(f"  [{i}/{len(entries)}] [skip] {pdb_id}{chain}: "
                  f"{type(e).__name__}", flush=True)
            continue
        if not clusters:
            n_skip += 1
            continue

        top = clusters[:5]
        jac = max((jaccard(found_resnums(c.lining_residues), known) for c in top),
                  default=0.0)
        rec = max((residue_recall(found_resnums(c.lining_residues), known) for c in top),
                  default=0.0)
        dist = min((centroid_distance(c.centroid, ref) for c in top),
                   default=float("inf"))
        robust = jac >= JACCARD_THRESHOLD or dist <= CENTROID_THRESHOLD
        rows.append({"id": f"{pdb_id}{chain}", "ligands": sorted(codes),
                     "n_known": len(known), "jaccard": round(jac, 3),
                     "recall": round(rec, 3),
                     "centroid_A": None if dist == float("inf") else round(dist, 2),
                     "robust": robust})
        n = len(rows)
        n_hit = sum(1 for r in rows if r["robust"])
        print(f"  [{i}/{len(entries)}] {'PASS' if robust else 'miss'} {pdb_id}{chain} "
              f"jac={jac:.0%} dist={'n/a' if dist==float('inf') else f'{dist:.1f}A'}  "
              f"running={n_hit}/{n} ({n_hit/n:.0%})", flush=True)

    n = len(rows)
    if not n:
        print("no structures scored")
        return
    robust = sum(1 for r in rows if r["robust"])
    legacy = sum(1 for r in rows if r["recall"] >= OVERLAP_THRESHOLD
                 or (r["centroid_A"] is not None
                     and r["centroid_A"] <= CENTROID_THRESHOLD))
    cen = sum(1 for r in rows if r["centroid_A"] is not None
              and r["centroid_A"] <= CENTROID_THRESHOLD)
    print("-" * 70)
    print(f"  n = {n} scored ({n_skip} skipped), {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  size-robust (jac>={JACCARD_THRESHOLD:.0%} or cen<={CENTROID_THRESHOLD:.0f}A): "
          f"{robust}/{n} ({robust/n:.0%})")
    print(f"  legacy recall-based:  {legacy}/{n} ({legacy/n:.0%})")
    print(f"  centroid alone:       {cen}/{n} ({cen/n:.0%})")
    out = Path(args.out) if args.out else Path(__file__).parent / \
        f"coach420_{args.rank_by}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"  full results -> {out}")


if __name__ == "__main__":
    main()
