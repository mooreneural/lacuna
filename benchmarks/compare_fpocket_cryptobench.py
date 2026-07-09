# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Lacuna vs fpocket head-to-head on CryptoBench (larger sample than the 22-pair
curated set, for tighter confidence intervals on the size-robust comparison).

Reuses the CryptoBench loading/scoring machinery from cryptobench_benchmark.py
(same dataset, same known-site resolution, same Lacuna run configuration) and
the fpocket runner from compare_fpocket.py, so both benchmarks are directly
comparable in methodology.

fpocket must be on PATH (WSL/Linux; see compare_fpocket.py for build notes).
fpocket 4.x accepts mmCIF input directly.

Usage:
    python benchmarks/compare_fpocket_cryptobench.py --limit 60 --shuffle
    python benchmarks/compare_fpocket_cryptobench.py --folds test   # full 222 (slow)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import (  # noqa: E402
    run_lacuna, OVERLAP_THRESHOLD, JACCARD_THRESHOLD,
)
from cryptobench_benchmark import (  # noqa: E402
    _fetch, download_cif, main_pocket, MAX_RESIDUES,
)
from compare_fpocket import run_fpocket, residue_overlap, residue_jaccard  # noqa: E402
from lacuna.io.structure import load_structure  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="only run first N (0=all)")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--folds", default="test")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle id order (seed 0) so --limit is representative")
    args = ap.parse_args()

    import shutil
    has_fpocket = shutil.which("fpocket") is not None
    if not has_fpocket:
        print("WARNING: fpocket not found on PATH. See compare_fpocket.py for build notes.")
        return

    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    ids = [pid for f in args.folds.split(",") for pid in folds[f.strip()]]
    if args.shuffle:
        import random
        random.Random(0).shuffle(ids)
    if args.limit:
        ids = ids[:args.limit]

    print("=" * 70)
    print(f"  LACUNA vs FPOCKET on CryptoBench [{args.folds}]  ({len(ids)} candidates)")
    print("=" * 70, flush=True)

    rows = []
    n_skip = 0
    t0 = time.perf_counter()

    for i, apo in enumerate(ids, 1):
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
                continue
        except Exception as e:
            n_skip += 1
            print(f"  [{i}/{len(ids)}] [skip] {tag}: {type(e).__name__}", flush=True)
            continue

        # fpocket (mmCIF input supported natively by fpocket 4.x)
        try:
            fp_pockets = run_fpocket(cif, chain)
        except Exception:
            fp_pockets = []
        fp_ov = max((residue_overlap(p["residues"], known) for p in fp_pockets[:5]), default=0.0)
        fp_jac = max((residue_jaccard(p["residues"], known) for p in fp_pockets[:5]), default=0.0)

        # Lacuna, same config as cryptobench_benchmark.py
        try:
            clusters, elapsed = run_lacuna(cif, args.conformers, chain=chain,
                                           backend_name="nma", rank_by="crypticity")
        except Exception as e:
            n_skip += 1
            print(f"  [{i}/{len(ids)}] [skip] {tag}: lacuna {type(e).__name__}", flush=True)
            continue
        lac_ov = max((residue_overlap(c.lining_residues, known) for c in clusters[:5]), default=0.0)
        lac_jac = max((residue_jaccard(c.lining_residues, known) for c in clusters[:5]), default=0.0)

        rows.append({
            "id": tag, "n_known": len(known),
            "fpocket": {"recall": round(fp_ov, 3), "jaccard": round(fp_jac, 3)},
            "lacuna": {"recall": round(lac_ov, 3), "jaccard": round(lac_jac, 3)},
        })
        fp_pass = fp_jac >= JACCARD_THRESHOLD
        lac_pass = lac_jac >= JACCARD_THRESHOLD
        n = len(rows)
        fp_n = sum(1 for r in rows if r["fpocket"]["jaccard"] >= JACCARD_THRESHOLD)
        lac_n = sum(1 for r in rows if r["lacuna"]["jaccard"] >= JACCARD_THRESHOLD)
        print(f"  [{i}/{len(ids)}] {tag}  fp_jac={fp_jac:.0%}{'*' if fp_pass else ' '}  "
              f"lac_jac={lac_jac:.0%}{'*' if lac_pass else ' '}  "
              f"running fp={fp_n}/{n} lac={lac_n}/{n}  ({elapsed:.1f}s)", flush=True)

    dt = time.perf_counter() - t0
    n = len(rows)
    fp_legacy = sum(1 for r in rows if r["fpocket"]["recall"] >= OVERLAP_THRESHOLD)
    lac_legacy = sum(1 for r in rows if r["lacuna"]["recall"] >= OVERLAP_THRESHOLD)
    fp_robust = sum(1 for r in rows if r["fpocket"]["jaccard"] >= JACCARD_THRESHOLD)
    lac_robust = sum(1 for r in rows if r["lacuna"]["jaccard"] >= JACCARD_THRESHOLD)

    print("-" * 70)
    print(f"  n = {n} scored ({n_skip} skipped), {dt/60:.1f} min")
    print(f"  Legacy recall (>={OVERLAP_THRESHOLD:.0%}):      fpocket {fp_legacy}/{n} "
          f"({fp_legacy/max(n,1):.0%})   Lacuna {lac_legacy}/{n} ({lac_legacy/max(n,1):.0%})")
    print(f"  Size-robust (Jaccard>={JACCARD_THRESHOLD:.0%}): fpocket {fp_robust}/{n} "
          f"({fp_robust/max(n,1):.0%})   Lacuna {lac_robust}/{n} ({lac_robust/max(n,1):.0%})")

    out = Path(__file__).parent / "fpocket_cryptobench_comparison.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Full results -> {out}")


if __name__ == "__main__":
    main()
