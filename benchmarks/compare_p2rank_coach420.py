# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""P2Rank on COACH420, scored identically to Lacuna.

Lacuna scores well on COACH420, but that number means nothing on its own: these
are holo structures whose pocket is already open, so the task is easier than
cryptic detection, and published P2Rank figures for this dataset use a different
success criterion (DCA/DCC against ligand atoms) and the full 420 structures.

Comparing to those published numbers would be comparing across criteria. This
runs P2Rank over the same subset, with the same MOAD ligand ground truth and the
same size-robust criterion Lacuna is held to, so the two are actually
commensurable.

Needs P2Rank on PATH or LACUNA_P2RANK set (Java 11+; WSL/Linux here).

    python benchmarks/compare_p2rank_coach420.py --limit 150
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
from coach420_benchmark import MAX_RESIDUES, binding_site, download_pdb, fetch_mlig  # noqa: E402
from metrics import (  # noqa: E402
    CENTROID_THRESHOLD,
    JACCARD_THRESHOLD,
    OVERLAP_THRESHOLD,
    found_resnums,
    jaccard,
    residue_recall,
)

from lacuna.io.structure import load_structure  # noqa: E402
from lacuna.pockets.p2rank_detector import (  # noqa: E402
    p2rank_available,
    p2rank_executable,
    run_p2rank,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not p2rank_available():
        print("P2Rank not found. Install Java 11+ and put 'prank' on PATH, "
              "or set LACUNA_P2RANK to the launcher.")
        return
    print(f"Using P2Rank: {p2rank_executable()}")

    entries = fetch_mlig()[:args.limit] if args.limit else fetch_mlig()
    print("=" * 70)
    print(f"  P2RANK on COACH420  n={len(entries)}  (same criterion as Lacuna)")
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
            preds = run_p2rank(path, chain)
        except Exception as e:
            n_skip += 1
            print(f"  [{i}/{len(entries)}] [skip] {pdb_id}{chain}: "
                  f"{type(e).__name__}", flush=True)
            continue
        if not preds:
            n_skip += 1
            continue

        top = preds[:5]
        jac = max((jaccard(found_resnums(p["residues"]), known) for p in top),
                  default=0.0)
        rec = max((residue_recall(found_resnums(p["residues"]), known) for p in top),
                  default=0.0)
        # P2Rank reports a pocket centre; the size-robust criterion allows either
        # a Jaccard hit or a close centre, exactly as for Lacuna.
        dist = float("inf")
        for p in top:
            c = p.get("center")
            if c:
                d = sum((a - b) ** 2 for a, b in zip(c, ref)) ** 0.5
                dist = min(dist, d)
        robust = jac >= JACCARD_THRESHOLD or dist <= CENTROID_THRESHOLD
        rows.append({"id": f"{pdb_id}{chain}", "n_known": len(known),
                     "jaccard": round(jac, 3), "recall": round(rec, 3),
                     "centroid_A": None if dist == float("inf") else round(dist, 2),
                     "robust": robust})
        n = len(rows)
        hit = sum(1 for r in rows if r["robust"])
        print(f"  [{i}/{len(entries)}] {'PASS' if robust else 'miss'} {pdb_id}{chain} "
              f"jac={jac:.0%}  running={hit}/{n} ({hit/n:.0%})", flush=True)

    n = len(rows)
    if not n:
        print("no structures scored")
        return
    robust = sum(1 for r in rows if r["robust"])
    legacy = sum(1 for r in rows if r["recall"] >= OVERLAP_THRESHOLD
                 or (r["centroid_A"] is not None
                     and r["centroid_A"] <= CENTROID_THRESHOLD))
    print("-" * 70)
    print(f"  n = {n} scored ({n_skip} skipped), {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  size-robust: {robust}/{n} ({robust/n:.0%})")
    print(f"  legacy:      {legacy}/{n} ({legacy/n:.0%})")
    out = Path(args.out) if args.out else Path(__file__).parent / "coach420_p2rank.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"  full results -> {out}")


if __name__ == "__main__":
    main()
