# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""CryptoBench cryptic-binding-site benchmark (Vavra et al. 2024, Bioinformatics).

CryptoBench (https://osf.io/pz4a9/) is the largest cryptic-pocket dataset: 1107
apo structures with holo-derived pocket residue annotations by auth_seq_id. This
script evaluates Lacuna on the held-out **test fold** (222 apo structures).

For each apo structure the main (highest-pRMSD) cryptic pocket is the target; the
apo residues are taken directly from ``apo_pocket_selection`` (e.g. ``"B_12"`` =
chain B, residue 12 — no order-based mapping needed). Lacuna runs with its default
configuration (NMA backend, crypticity ranking); a top-5 pocket counts as a hit if
its lining residues overlap >=30% of the pocket residues or it centres within 4 A
of their Ca centroid.

    python benchmarks/cryptobench_benchmark.py            # full test fold (~1-2 h)
    python benchmarks/cryptobench_benchmark.py --limit 30 # quick subset

Dataset files (dataset.json, folds.json) are auto-downloaded from OSF.
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
from cryptic_benchmark import (  # noqa: E402
    run_lacuna, residue_overlap, pocket_min_centroid_dist,
    CENTROID_THRESHOLD, OVERLAP_THRESHOLD,
)
from lacuna.io.structure import load_structure  # noqa: E402

CB_DIR = Path(__file__).parent / "cb_data"
CIF_DIR = CB_DIR / "cif"
MAX_RESIDUES = 700
_OSF = {
    "dataset.json": "https://osf.io/download/ta2ju/",
    "folds.json": "https://osf.io/download/5s93p/",
}


def _fetch(name: str) -> Path:
    CB_DIR.mkdir(exist_ok=True)
    dst = CB_DIR / name
    if not dst.exists():
        print(f"  Downloading CryptoBench {name} ...", flush=True)
        urllib.request.urlretrieve(_OSF[name], dst)
    return dst


def download_cif(pdb: str) -> Path:
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    out = CIF_DIR / f"{pdb.upper()}.cif"
    if not out.exists():
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb.upper()}.cif", out)
    return out


def main_pocket(assocs: list) -> tuple[str, set[int]]:
    """Return (apo_chain, residue_set) for the highest-pRMSD (main) cryptic pocket."""
    main = next((a for a in assocs if a.get("is_main_holo_structure")), None)
    if main is None:
        main = max(assocs, key=lambda a: a.get("pRMSD", 0.0))
    chain = main["apo_chain"]
    res = {int(s.split("_")[1]) for s in main["apo_pocket_selection"]
           if s.split("_")[0] == chain}
    return chain, res


def known_centroid(structure, chain, known):
    ca = [a.coords for a in structure.atoms
          if a.name == "CA" and a.chain_id == chain and a.res_seq in known]
    return tuple(np.mean(ca, axis=0).tolist()) if ca else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only run first N (0=all)")
    ap.add_argument("--conformers", type=int, default=20)
    args = ap.parse_args()

    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    test_ids = folds["test"]
    if args.limit:
        test_ids = test_ids[:args.limit]

    print("=" * 70)
    print(f"  CRYPTOBENCH TEST FOLD  ({len(test_ids)} apo structures, NMA + crypticity)")
    print("=" * 70, flush=True)

    n_pass = n_run = n_skip = 0
    t_start = time.perf_counter()
    for i, apo in enumerate(test_ids, 1):
        assocs = dataset.get(apo)
        if not assocs:
            n_skip += 1
            continue
        chain, known = main_pocket(assocs)
        if not known:
            n_skip += 1
            continue
        tag = f"{apo}{chain}"
        try:
            cif = download_cif(apo)
            s = load_structure(cif, chain=chain)
            if len(s.residues) > MAX_RESIDUES or len(s.residues) < 10:
                n_skip += 1
                print(f"  [skip] {tag}: {len(s.residues)} residues", flush=True)
                continue
            ref = known_centroid(s, chain, known)
            clusters, elapsed = run_lacuna(cif, args.conformers, chain=chain,
                                           rank_by="crypticity")
        except Exception as e:
            n_skip += 1
            print(f"  [skip] {tag}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue

        best_ov = max((residue_overlap(c.lining_residues, known) for c in clusters[:5]),
                      default=0.0)
        best_dist, _ = pocket_min_centroid_dist(clusters, ref, top_n=5)
        found = best_ov >= OVERLAP_THRESHOLD or best_dist <= CENTROID_THRESHOLD
        n_run += 1
        n_pass += int(found)
        mark = "PASS" if found else "miss"
        dist_s = f"{best_dist:.1f}A" if best_dist < float("inf") else "n/a"
        rate = n_pass / max(n_run, 1)
        print(f"  [{i}/{len(test_ids)}] {mark} {tag}  ov={best_ov:.0%} dist={dist_s} "
              f"({len(known)} res, {elapsed:.1f}s)  running={n_pass}/{n_run} ({rate:.0%})",
              flush=True)

    dt = time.perf_counter() - t_start
    print("-" * 70)
    print(f"  CRYPTOBENCH TEST: {n_pass}/{n_run} ({n_pass / max(n_run, 1):.0%})  "
          f"[{n_skip} skipped, {dt/60:.1f} min]")


if __name__ == "__main__":
    main()
