# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Lacuna + P2Rank detector-fusion benchmark.

Measures whether adding P2Rank (a fast, template-free ML surface detector) to
Lacuna's ensemble improves cryptic-pocket recovery, on the curated 22-target
cryptic set (benchmarks/cryptic_benchmark.py DATASET, category="cryptic").

Both criteria used throughout the repo are reported, top-5:
    legacy recall     >= 30% residue overlap with the known site (size-gameable)
    size-robust (IoU) >= 25% Jaccard overlap (penalizes oversized pockets)

Three columns are scored so the fusion question is answered honestly:
    Lacuna   ensemble (NMA, crypticity ranking), alpha detector - the baseline
    P2Rank   run on the apo structure (its standard single-structure usage), the
             fair complement, exactly as fpocket is run in compare_fpocket.py
    Union    passes if EITHER Lacuna OR P2Rank passes - the complementarity ceiling

Reporting Lacuna-only / P2Rank-only / both / union with target-level bootstrap
CIs mirrors the fpocket head-to-head so the two comparisons are directly
readable side by side.

P2Rank is a JVM tool (needs Java 11+). Install it and put 'prank' on PATH or set
LACUNA_P2RANK to the launcher path. If missing, the script prints instructions
and runs Lacuna-only.

Usage:
    python benchmarks/compare_p2rank.py               # all 22 cryptic targets
    python benchmarks/compare_p2rank.py --limit 8      # quick subset
    python benchmarks/compare_p2rank.py --only IL2,MDM2
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
    DATASET, download_pdb, extract_binding_site, run_lacuna,
    OVERLAP_THRESHOLD, JACCARD_THRESHOLD,
)
# Reuse the residue-overlap helpers already used for the fpocket comparison so the
# two head-to-heads score identically.
from compare_fpocket import residue_overlap, residue_jaccard  # noqa: E402
from metrics import paired_bootstrap_ci  # noqa: E402
from lacuna.io.structure import load_structure  # noqa: E402
from lacuna.pockets.p2rank_detector import (  # noqa: E402
    run_p2rank, p2rank_available, p2rank_executable,
)

DEFAULT_CONFORMERS = 20


def _best_over_topk(items, residues_of, known, k=5):
    """Best legacy overlap and size-robust Jaccard over the top-k proposals.

    ``items`` is a ranked list; ``residues_of(item)`` yields its residue labels.
    Returns (best_overlap, best_overlap_rank, best_jaccard, best_jaccard_rank).
    """
    best_ov, best_ov_rank = 0.0, None
    best_jac, best_jac_rank = 0.0, None
    for rank, item in enumerate(items[:k], 1):
        residues = residues_of(item)
        ov = residue_overlap(residues, known)
        if ov > best_ov:
            best_ov, best_ov_rank = ov, rank
        jac = residue_jaccard(residues, known)
        if jac > best_jac:
            best_jac, best_jac_rank = jac, rank
    return best_ov, best_ov_rank, best_jac, best_jac_rank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="only first N cryptic targets")
    ap.add_argument("--only", default=None, help="comma-separated DATASET ids")
    ap.add_argument("--conformers", type=int, default=DEFAULT_CONFORMERS)
    args = ap.parse_args()

    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)

    has_p2rank = p2rank_available()
    if not has_p2rank:
        print("WARNING: P2Rank not found.")
        print("  Install: https://github.com/rdk/p2rank (needs Java 11+); put 'prank'")
        print("  on PATH or set LACUNA_P2RANK to the launcher. Continuing Lacuna-only.\n")
    else:
        print(f"Using P2Rank: {p2rank_executable()}\n")

    entries = [e for e in DATASET if e["category"] == "cryptic"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        entries = [e for e in entries if e["id"] in wanted]
    if args.limit:
        entries = entries[:args.limit]

    print("=" * 70)
    print(f"  LACUNA + P2RANK - DETECTOR FUSION ({len(entries)} curated cryptic targets)")
    if not has_p2rank:
        print("  (P2Rank unavailable - Lacuna only)")
    print("=" * 70)

    rows = []

    for entry in entries:
        pdb_id = entry["apo_pdb"]
        chain = entry.get("apo_chain", "A")
        print(f"\n{'─'*70}")
        print(f"  {entry['id']}  ({pdb_id})  {entry['name']}")
        print(f"{'─'*70}")

        try:
            pdb_path = download_pdb(pdb_id, pdb_dir)
        except Exception as e:
            print(f"  [SKIP] apo download failed: {e}")
            continue

        if "known_residues" in entry:
            known = entry["known_residues"]
        else:
            try:
                holo_path = download_pdb(entry["holo_pdb"], pdb_dir)
            except Exception as e:
                print(f"  [SKIP] holo download failed: {e}")
                continue
            known, _ = extract_binding_site(
                holo_path, entry.get("holo_chain", "A"), entry.get("extra_exclude", frozenset()))
        if not known:
            print("  [SKIP] no known binding site resolved")
            continue

        max_res = entry.get("max_residues", 600)
        try:
            s = load_structure(pdb_path, chain=chain)
            if len(s.residues) > max_res:
                print(f"  [SKIP] {len(s.residues)} residues > max_residues={max_res}")
                continue
        except Exception as e:
            print(f"  [SKIP] structure load failed: {e}")
            continue

        # ── P2Rank (on the apo structure - standard single-structure usage) ──────
        p2r_result = {"status": "n/a", "status_robust": "n/a"}
        if has_p2rank:
            print("  Running P2Rank...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                preds = run_p2rank(pdb_path, chain)
            except Exception as e:
                print(f"error: {e}")
                preds = []
            p2r_elapsed = time.perf_counter() - t0
            print(f"{len(preds)} pockets in {p2r_elapsed:.1f}s")

            ov, ov_rank, jac, jac_rank = _best_over_topk(
                preds, lambda p: p["residues"], known)
            p2r_found = ov >= OVERLAP_THRESHOLD
            p2r_found_r = jac >= JACCARD_THRESHOLD
            p2r_result = {
                "status": "PASS" if p2r_found else "MISS",
                "overlap": round(ov, 3), "rank": ov_rank,
                "status_robust": "PASS" if p2r_found_r else "MISS",
                "jaccard": round(jac, 3), "jaccard_rank": jac_rank,
                "n_pockets": len(preds), "elapsed_s": round(p2r_elapsed, 2),
            }
            print(f"  P2Rank:  {'PASS' if p2r_found else 'miss'} recall={ov:.0%}@{ov_rank}  "
                  f"{'PASS' if p2r_found_r else 'miss'} jaccard={jac:.0%}@{jac_rank}")

        # ── Lacuna (ensemble, same config as the rest of the suite) ─────────────
        print(f"  Running Lacuna (NMA, {args.conformers} conformers, crypticity)...",
              end=" ", flush=True)
        try:
            clusters, lac_elapsed = run_lacuna(
                pdb_path, args.conformers, chain=chain,
                backend_name="nma", rank_by="crypticity")
        except Exception as e:
            print(f"error: {e}")
            continue
        print(f"{len(clusters)} clusters in {lac_elapsed:.1f}s")

        ov, ov_rank, jac, jac_rank = _best_over_topk(
            clusters, lambda c: c.lining_residues, known)
        lac_found = ov >= OVERLAP_THRESHOLD
        lac_found_r = jac >= JACCARD_THRESHOLD
        lac_result = {
            "status": "PASS" if lac_found else "MISS",
            "overlap": round(ov, 3), "rank": ov_rank,
            "status_robust": "PASS" if lac_found_r else "MISS",
            "jaccard": round(jac, 3), "jaccard_rank": jac_rank,
            "n_clusters": len(clusters), "elapsed_s": round(lac_elapsed, 2),
        }
        print(f"  Lacuna:  {'PASS' if lac_found else 'miss'} recall={ov:.0%}@{ov_rank}  "
              f"{'PASS' if lac_found_r else 'miss'} jaccard={jac:.0%}@{jac_rank}")

        rows.append({
            "id": entry["id"], "pdb": pdb_id, "name": entry["name"],
            "p2rank": p2r_result, "lacuna": lac_result,
        })

    _summarize(rows, has_p2rank)

    out_path = Path(__file__).parent / "p2rank_comparison.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results -> {out_path}")


def _summarize(rows, has_p2rank):
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    header = f"{'ID':<18} {'P2Rank recall/jaccard':<26} {'Lacuna recall/jaccard'}"
    print(header)
    print("─" * len(header))

    for r in rows:
        p2r, lac = r["p2rank"], r["lacuna"]
        if p2r["status"] != "n/a":
            p2r_col = f"{p2r['overlap']:.0%}@{p2r['rank']} / {p2r['jaccard']:.0%}@{p2r['jaccard_rank']}"
        else:
            p2r_col = "n/a"
        lac_col = f"{lac['overlap']:.0%}@{lac['rank']} / {lac['jaccard']:.0%}@{lac['jaccard_rank']}"
        print(f"{r['id']:<18} {p2r_col:<26} {lac_col}")

    print("─" * len(header))
    n = len(rows)
    if not n:
        print("No targets scored.")
        return

    # Size-robust (Jaccard) hit vectors for CIs and the complementarity breakdown.
    lac_hits = [r["lacuna"]["status_robust"] == "PASS" for r in rows]
    lac_mean, lac_lo, lac_hi = paired_bootstrap_ci(lac_hits)

    if not has_p2rank:
        print(f"n = {n} targets (Lacuna only)")
        print(f"Lacuna  size-robust: {sum(lac_hits)}/{n} "
              f"({lac_mean:.0%}, CI[{lac_lo:.0%},{lac_hi:.0%}])")
        print(f"Lacuna  legacy:      "
              f"{sum(1 for r in rows if r['lacuna']['status']=='PASS')}/{n}")
        return

    p2r_hits = [r["p2rank"]["status_robust"] == "PASS" for r in rows]
    union_hits = [a or b for a, b in zip(lac_hits, p2r_hits)]
    both = [a and b for a, b in zip(lac_hits, p2r_hits)]
    lac_only = [a and not b for a, b in zip(lac_hits, p2r_hits)]
    p2r_only = [b and not a for a, b in zip(lac_hits, p2r_hits)]

    p2r_mean, p2r_lo, p2r_hi = paired_bootstrap_ci(p2r_hits)
    uni_mean, uni_lo, uni_hi = paired_bootstrap_ci(union_hits)

    lac_leg = sum(1 for r in rows if r["lacuna"]["status"] == "PASS")
    p2r_leg = sum(1 for r in rows if r["p2rank"]["status"] == "PASS")

    print(f"n = {n} targets")
    print(f"Legacy recall (>={OVERLAP_THRESHOLD:.0%}):      "
          f"P2Rank {p2r_leg}/{n}   Lacuna {lac_leg}/{n}")
    print(f"Size-robust (Jaccard>={JACCARD_THRESHOLD:.0%}): "
          f"P2Rank {sum(p2r_hits)}/{n} (CI[{p2r_lo:.0%},{p2r_hi:.0%}])   "
          f"Lacuna {sum(lac_hits)}/{n} (CI[{lac_lo:.0%},{lac_hi:.0%}])")
    print(f"\nComplementarity (size-robust):")
    print(f"  both pass     : {sum(both)}/{n}")
    print(f"  Lacuna only   : {sum(lac_only)}/{n}")
    print(f"  P2Rank only   : {sum(p2r_only)}/{n}")
    print(f"  union (either): {sum(union_hits)}/{n} "
          f"({uni_mean:.0%}, CI[{uni_lo:.0%},{uni_hi:.0%}])")
    if sum(p2r_only) > 0 or sum(lac_only) > 0:
        print(f"  -> fusion ceiling {sum(union_hits)}/{n} vs best single "
              f"{max(sum(lac_hits), sum(p2r_hits))}/{n}")


if __name__ == "__main__":
    main()
