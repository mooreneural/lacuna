# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Lacuna vs fpocket head-to-head benchmark.

Runs both tools on the same apo structures, from the curated 22-target cryptic
set used throughout the project (benchmarks/cryptic_benchmark.py DATASET,
category="cryptic"), and reports pocket recovery under both criteria used
elsewhere in this repo:

  legacy recall     >= 30% residue overlap with the known binding site, top-5
  size-robust (IoU) >= 25% Jaccard overlap, top-5 (penalizes oversized pockets)

Reusing the same DATASET, known-site resolution, and Lacuna run configuration
(NMA backend, 20 conformers, crypticity ranking) as the rest of the benchmark
suite keeps this an apples-to-apples, single-chain comparison for both tools.

fpocket must be on PATH (built from github.com/Discngine/fpocket or via a
package manager; not available as a Windows-native binary, use WSL or Linux).
If missing, the script prints instructions and runs Lacuna-only.

Usage:
    python benchmarks/compare_fpocket.py             # all 22 cryptic targets
    python benchmarks/compare_fpocket.py --limit 8    # quick subset
    python benchmarks/compare_fpocket.py --only T4L_L99A,4OBE
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import (  # noqa: E402
    DATASET, download_pdb, extract_binding_site, run_lacuna,
    OVERLAP_THRESHOLD, JACCARD_THRESHOLD,
)
from lacuna.io.structure import load_structure  # noqa: E402
from lacuna.io.writers import write_structure_pdb  # noqa: E402

DEFAULT_CONFORMERS = 20


# ── shared residue-overlap helpers ──────────────────────────────────────────────

def _parse_resnums(residue_labels: list[str]) -> set[int]:
    out: set[int] = set()
    for label in residue_labels:
        try:
            out.add(int("".join(c for c in label.split(":")[0] if c.isdigit())))
        except (ValueError, IndexError):
            pass
    return out


def residue_overlap(residue_labels: list[str], known: set[int]) -> float:
    """Legacy recall: |found ∩ known| / |known|. Size-gameable, kept for
    backward comparison; prefer residue_jaccard for the headline criterion."""
    found = _parse_resnums(residue_labels)
    return len(found & known) / len(known) if known else 0.0


def residue_jaccard(residue_labels: list[str], known: set[int]) -> float:
    """Size-robust IoU: |found ∩ known| / |found ∪ known|."""
    found = _parse_resnums(residue_labels)
    union = found | known
    return len(found & known) / len(union) if union else 0.0


# ── fpocket runner ─────────────────────────────────────────────────────────────

def parse_fpocket_atoms(pocket_pdb: Path, chain: str) -> list[str]:
    """Extract residue IDs from an fpocket pocket PDB file."""
    residues = set()
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                res_chain = line[21].strip()
                res_num = line[22:26].strip()
                if res_num.isdigit():
                    residues.add(f"{res_num}:{res_chain}")
    return list(residues)


def run_fpocket(pdb_path: Path, chain: str) -> list[dict]:
    """Run fpocket on a structure (PDB or mmCIF); return list of {rank, residues}.

    fpocket accepts mmCIF input directly, but then names its per-pocket output
    files pocketN_atm.cif instead of .pdb, silently producing zero results
    against a PDB-only glob (and mmCIF's whitespace-token atom format is not
    parsed by the fixed-column PDB reader below anyway). To keep exactly one
    tested parsing path, always normalize the input to a clean, single-chain
    PDB via load_structure/write_structure_pdb before invoking fpocket,
    regardless of the source format.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdb = Path(tmp) / "input.pdb"
        structure = load_structure(pdb_path, chain=chain)
        write_structure_pdb(structure, tmp_pdb)

        result = subprocess.run(
            ["fpocket", "-f", str(tmp_pdb)],
            capture_output=True, text=True, cwd=tmp,
        )
        if result.returncode != 0:
            print(f"    fpocket error: {result.stderr[:200]}")
            return []

        out_dir = Path(tmp) / f"{tmp_pdb.stem}_out"
        if not out_dir.exists():
            return []

        pocket_files = sorted(out_dir.glob("pockets/pocket*_atm.pdb"))
        pockets = []
        for i, pf in enumerate(pocket_files, 1):
            residues = parse_fpocket_atoms(pf, chain)
            pockets.append({"rank": i, "residues": residues})
        return pockets


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="only first N cryptic targets")
    ap.add_argument("--only", default=None, help="comma-separated DATASET ids")
    ap.add_argument("--conformers", type=int, default=DEFAULT_CONFORMERS)
    args = ap.parse_args()

    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)

    has_fpocket = shutil.which("fpocket") is not None
    if not has_fpocket:
        print("WARNING: fpocket not found on PATH.")
        print("  Install: https://github.com/Discngine/fpocket (Linux/WSL; build from source")
        print("  with libnetcdf-dev, `make`, `sudo make install`). Continuing Lacuna-only.\n")

    entries = [e for e in DATASET if e["category"] == "cryptic"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        entries = [e for e in entries if e["id"] in wanted]
    if args.limit:
        entries = entries[:args.limit]

    print("=" * 70)
    print(f"  LACUNA vs FPOCKET - HEAD-TO-HEAD ({len(entries)} curated cryptic targets)")
    if not has_fpocket:
        print("  (fpocket unavailable - Lacuna only)")
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

        # ── fpocket ──────────────────────────────────────────────────────────
        fp_result = {"status": "n/a", "overlap": None, "rank": None}
        if has_fpocket:
            print("  Running fpocket...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                fp_pockets = run_fpocket(pdb_path, chain)
            except Exception as e:
                print(f"error: {e}")
                fp_pockets = []
            fp_elapsed = time.perf_counter() - t0
            print(f"{len(fp_pockets)} pockets in {fp_elapsed:.1f}s")

            best_ov, best_rank = 0.0, None
            best_jac, best_jac_rank = 0.0, None
            for p in fp_pockets[:5]:
                ov = residue_overlap(p["residues"], known)
                if ov > best_ov:
                    best_ov, best_rank = ov, p["rank"]
                jac = residue_jaccard(p["residues"], known)
                if jac > best_jac:
                    best_jac, best_jac_rank = jac, p["rank"]

            found = best_ov >= OVERLAP_THRESHOLD
            found_robust = best_jac >= JACCARD_THRESHOLD
            fp_result = {
                "status": "PASS" if found else "MISS",
                "overlap": round(best_ov, 3), "rank": best_rank,
                "status_robust": "PASS" if found_robust else "MISS",
                "jaccard": round(best_jac, 3), "jaccard_rank": best_jac_rank,
                "n_pockets": len(fp_pockets), "elapsed_s": round(fp_elapsed, 2),
            }
            marker = "PASS" if found else "miss"
            rmarker = "PASS" if found_robust else "miss"
            print(f"  fpocket: {marker} recall={best_ov:.0%}@{best_rank}  "
                  f"{rmarker} jaccard={best_jac:.0%}@{best_jac_rank}")

        # ── Lacuna (same config as the rest of the benchmark suite) ─────────────
        print(f"  Running Lacuna (NMA, {args.conformers} conformers, crypticity)...",
              end=" ", flush=True)
        try:
            clusters, lac_elapsed = run_lacuna(
                pdb_path, args.conformers, chain=chain,
                backend_name="nma", rank_by="crypticity")
        except Exception as e:
            print(f"error: {e}")
            continue
        print(f"{len(clusters)} pocket clusters in {lac_elapsed:.1f}s")

        best_ov, best_rank = 0.0, None
        best_jac, best_jac_rank = 0.0, None
        for c in clusters[:5]:
            ov = residue_overlap(c.lining_residues, known)
            if ov > best_ov:
                best_ov, best_rank = ov, c.rank
            jac = residue_jaccard(c.lining_residues, known)
            if jac > best_jac:
                best_jac, best_jac_rank = jac, c.rank

        lac_found = best_ov >= OVERLAP_THRESHOLD
        lac_found_robust = best_jac >= JACCARD_THRESHOLD
        lac_result = {
            "status": "PASS" if lac_found else "MISS",
            "overlap": round(best_ov, 3), "rank": best_rank,
            "status_robust": "PASS" if lac_found_robust else "MISS",
            "jaccard": round(best_jac, 3), "jaccard_rank": best_jac_rank,
            "n_clusters": len(clusters), "elapsed_s": round(lac_elapsed, 2),
        }
        marker = "PASS" if lac_found else "miss"
        rmarker = "PASS" if lac_found_robust else "miss"
        print(f"  Lacuna:  {marker} recall={best_ov:.0%}@{best_rank}  "
              f"{rmarker} jaccard={best_jac:.0%}@{best_jac_rank}  {lac_elapsed:.1f}s")

        rows.append({
            "id": entry["id"], "pdb": pdb_id, "name": entry["name"],
            "fpocket": fp_result, "lacuna": lac_result,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    header = f"{'ID':<18} {'fpocket recall/jaccard':<26} {'Lacuna recall/jaccard'}"
    print(header)
    print("─" * len(header))

    fp_pass = lac_pass = fp_pass_r = lac_pass_r = fp_total = 0
    for r in rows:
        fp, lac = r["fpocket"], r["lacuna"]
        if fp["status"] != "n/a":
            fp_total += 1
            fp_col = f"{fp['overlap']:.0%}@{fp['rank']} / {fp['jaccard']:.0%}@{fp['jaccard_rank']}"
            fp_pass += fp["status"] == "PASS"
            fp_pass_r += fp["status_robust"] == "PASS"
        else:
            fp_col = "n/a"
        lac_col = f"{lac['overlap']:.0%}@{lac['rank']} / {lac['jaccard']:.0%}@{lac['jaccard_rank']}"
        lac_pass += lac["status"] == "PASS"
        lac_pass_r += lac["status_robust"] == "PASS"
        print(f"{r['id']:<18} {fp_col:<26} {lac_col}")

    print("─" * len(header))
    print(f"n = {len(rows)} targets ({fp_total} scored by fpocket)")
    if fp_total:
        print(f"Legacy recall (>={OVERLAP_THRESHOLD:.0%}) score:      "
              f"fpocket {fp_pass}/{fp_total}   Lacuna {lac_pass}/{len(rows)}")
        print(f"Size-robust (Jaccard>={JACCARD_THRESHOLD:.0%}) score: "
              f"fpocket {fp_pass_r}/{fp_total}   Lacuna {lac_pass_r}/{len(rows)}")
    else:
        print(f"Lacuna only: legacy {lac_pass}/{len(rows)}, size-robust {lac_pass_r}/{len(rows)}")

    out_path = Path(__file__).parent / "fpocket_comparison.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results -> {out_path}")


if __name__ == "__main__":
    main()
