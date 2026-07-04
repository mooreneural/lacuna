# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Extract per-pocket-cluster features + labels for the learned re-ranker.

Runs Lacuna (NMA, crypticity ranking, 20 conformers) over CryptoBench fold
structures. For every detected pocket cluster it writes one row of POCKET-INTRINSIC
features (nothing derived from the known answer) plus a binary label: 1 if the
cluster meets the benchmark hit criterion (>=30% residue overlap with the known
cryptic pocket OR centre within 4 A of the site centroid), else 0.

Resumable: rows are appended per protein to a CSV, and proteins already present are
skipped on re-run.

    python benchmarks/extract_features.py --folds test
    python benchmarks/extract_features.py --folds train-0,train-1,train-2,train-3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import (  # noqa: E402
    run_lacuna, residue_overlap, pocket_min_centroid_dist,
    CENTROID_THRESHOLD, OVERLAP_THRESHOLD,
)
from cryptobench_benchmark import (  # noqa: E402
    _fetch, download_cif, main_pocket, known_centroid, MAX_RESIDUES,
)
from lacuna.io.structure import load_structure  # noqa: E402

OUT_DIR = Path(__file__).parent / "cb_data"

FEATURES = [
    "crypticity", "persistence", "volume_mean", "volume_min", "volume_max",
    "apo_volume", "volume_range", "volume_delta_apo", "druggability",
    "max_druggability", "n_members", "n_lining", "enclosure_mean", "enclosure_max",
    "hydrophobic_mean", "aromatic_mean", "aromatic_max", "phys_rank", "phys_rank_frac",
]


def cluster_features(c, phys_rank: int, n_clusters: int) -> dict:
    members = c.member_pockets or []
    enc = [p.enclosure for p in members] or [0.0]
    hyd = [p.hydrophobic_fraction for p in members] or [0.0]
    aro = [p.aromatic_count for p in members] or [0]
    return {
        "crypticity": c.crypticity,
        "persistence": c.persistence,
        "volume_mean": c.volume_a3,
        "volume_min": c.volume_min_a3,
        "volume_max": c.volume_max_a3,
        "apo_volume": c.apo_volume_a3,
        "volume_range": c.volume_max_a3 - c.volume_min_a3,
        "volume_delta_apo": c.volume_max_a3 - c.apo_volume_a3,
        "druggability": c.druggability,
        "max_druggability": c.max_druggability,
        "n_members": len(members),
        "n_lining": len(c.lining_residues),
        "enclosure_mean": float(np.mean(enc)),
        "enclosure_max": float(np.max(enc)),
        "hydrophobic_mean": float(np.mean(hyd)),
        "aromatic_mean": float(np.mean(aro)),
        "aromatic_max": float(np.max(aro)),
        "phys_rank": phys_rank,
        "phys_rank_frac": phys_rank / max(n_clusters, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, help="comma-separated fold names")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="only first N (smoke test)")
    args = ap.parse_args()

    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    want = []
    for f in args.folds.split(","):
        want += folds[f.strip()]
    if args.limit:
        want = want[:args.limit]

    out = OUT_DIR / f"features_{args.folds.replace(',', '+').replace('train-', 't')}.csv"
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {r["protein"] for r in csv.DictReader(fh)}
    new = out.exists()
    fh = open(out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=["protein", "label"] + FEATURES)
    if not new:
        writer.writeheader()

    print(f"folds={args.folds}: {len(want)} structures, {len(done)} already done -> {out.name}")
    n_ok = 0
    for i, apo in enumerate(want, 1):
        if apo in done:
            continue
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
            ref = known_centroid(s, chain, known)
            clusters, _ = run_lacuna(cif, args.conformers, chain=chain, rank_by="crypticity")
        except Exception as e:
            print(f"  [skip] {apo}{chain}: {type(e).__name__}", flush=True)
            continue

        n = len(clusters)
        rows = []
        for rank, c in enumerate(clusters, 1):
            ov = residue_overlap(c.lining_residues, known)
            dist, _ = pocket_min_centroid_dist([c], ref, top_n=1)
            label = int(ov >= OVERLAP_THRESHOLD or dist <= CENTROID_THRESHOLD)
            feats = cluster_features(c, rank, n)
            rows.append({"protein": apo, "label": label, **feats})
        for r in rows:
            writer.writerow(r)
        fh.flush()
        n_ok += 1
        pos = sum(r["label"] for r in rows)
        if i % 10 == 0 or pos:
            print(f"  [{i}/{len(want)}] {apo}{chain}: {n} clusters, {pos} positive", flush=True)

    fh.close()
    print(f"done: {n_ok} new proteins written to {out}")


if __name__ == "__main__":
    main()
