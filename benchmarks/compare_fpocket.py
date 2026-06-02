"""Lacuna vs fpocket head-to-head benchmark.

Runs both tools on the same apo PDB structures and reports pocket recovery
using the same criterion: ≥30% residue overlap with known binding site in top-5.

fpocket must be on PATH.  If missing, the script prints instructions and exits.

Usage:
    python benchmarks/compare_fpocket.py

This requires the PDB files in benchmarks/pdb_cache/ (run run_benchmarks.py
first or download manually from RCSB).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# ── benchmark definitions ──────────────────────────────────────────────────────

BENCHMARKS = {
    "1HEL": {
        "name": "Hen lysozyme (orthosteric)",
        "chain": "A",
        "known_residues": {35, 52, 101, 102, 103, 104, 107, 108},
        "pocket_type": "Orthosteric (always open)",
    },
    "1L90": {
        "name": "T4 Lysozyme L99A (cryptic hydrophobic cavity)",
        "chain": "A",
        "known_residues": {99, 102, 106, 111, 118, 121, 133, 153},
        "pocket_type": "Cryptic (buried cavity)",
    },
    "4OBE": {
        "name": "K-Ras WT apo (switch-II cryptic pocket)",
        "chain": "A",
        "known_residues": {12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
        "pocket_type": "Cryptic (switch-II closed in GDP form)",
    },
    "1HPV": {
        "name": "HIV-1 protease apo (active site / flap region)",
        "chain": "A",
        "known_residues": {25, 26, 27, 28, 29, 30, 49, 50, 51, 52, 53},
        "pocket_type": "Active site (open)",
    },
}

DEFAULT_CONFORMERS = 20


# ── helpers ────────────────────────────────────────────────────────────────────

def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    out = dest_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id}...", end=" ", flush=True)
    urllib.request.urlretrieve(url, out)
    print("done")
    return out


def residue_overlap(residue_labels: list[str], known: set[int]) -> float:
    found: set[int] = set()
    for label in residue_labels:
        try:
            num = int("".join(c for c in label.split(":")[0] if c.isdigit()))
            found.add(num)
        except (ValueError, IndexError):
            pass
    return len(found & known) / len(known) if known else 0.0


# ── fpocket runner ─────────────────────────────────────────────────────────────

def parse_fpocket_pockets(out_dir: Path, chain: str) -> list[dict]:
    """Parse fpocket _info.txt summary into a ranked list of pocket dicts."""
    pockets = []
    # fpocket writes one pocketN_atm.pdb per pocket; parse info file for ranking
    info_files = sorted(out_dir.glob("*_info.txt"))
    if not info_files:
        return pockets

    with open(info_files[0]) as f:
        text = f.read()

    # Each pocket block starts with "Pocket N :"
    blocks = re.split(r"Pocket\s+(\d+)\s+:", text)
    # blocks: ['', rank, body, rank, body, ...]
    for i in range(1, len(blocks) - 1, 2):
        rank = int(blocks[i])
        body = blocks[i + 1]

        residues: list[str] = []
        for line in body.splitlines():
            m = re.search(r"Residue IDs\s*:\s*(.+)", line)
            if m:
                residues = [r.strip() for r in m.group(1).split() if r.strip()]
        pockets.append({"rank": rank, "residues": residues})

    return pockets


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
    """Run fpocket on a PDB file; return list of {rank, residues}."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdb = Path(tmp) / pdb_path.name
        shutil.copy(pdb_path, tmp_pdb)

        result = subprocess.run(
            ["fpocket", "-f", str(tmp_pdb)],
            capture_output=True, text=True, cwd=tmp,
        )
        if result.returncode != 0:
            print(f"    fpocket error: {result.stderr[:200]}")
            return []

        # fpocket creates <stem>_out/ directory
        out_dir = Path(tmp) / f"{tmp_pdb.stem}_out"
        if not out_dir.exists():
            return []

        # Parse per-pocket PDB files directly for residue lists
        pocket_files = sorted(out_dir.glob("pockets/pocket*_atm.pdb"))
        pockets = []
        for i, pf in enumerate(pocket_files, 1):
            residues = parse_fpocket_atoms(pf, chain)
            pockets.append({"rank": i, "residues": residues})
        return pockets


# ── lacuna runner ──────────────────────────────────────────────────────────────

def run_lacuna(pdb_path: Path) -> tuple[list, float]:
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.ensemble.random_backend import RandomBackend
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets

    structure = load_structure(pdb_path)
    backend = RandomBackend(seed=42)

    t0 = time.perf_counter()
    coord_sets = backend.generate(pdb_path, n_conformers=DEFAULT_CONFORMERS)
    base = coords_array(structure)
    all_coords = [base] + coord_sets

    pocket_lists = []
    for ci, coords in enumerate(all_coords):
        pockets = detect_pockets(coords, structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)

    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords))
    elapsed = time.perf_counter() - t0
    return clusters, elapsed


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)

    # Check fpocket availability
    has_fpocket = shutil.which("fpocket") is not None
    if not has_fpocket:
        print("WARNING: fpocket not found on PATH.")
        print("  Install: https://github.com/Discngine/fpocket or `sudo apt install fpocket`")
        print("  Continuing with Lacuna-only benchmarks.\n")

    print("=" * 70)
    print("  LACUNA vs FPOCKET — HEAD-TO-HEAD BENCHMARK")
    if not has_fpocket:
        print("  (fpocket unavailable — Lacuna only)")
    print("=" * 70)

    rows = []

    for pdb_id, spec in BENCHMARKS.items():
        print(f"\n{'─'*70}")
        print(f"  {pdb_id}  {spec['name']}")
        print(f"{'─'*70}")

        pdb_path = download_pdb(pdb_id, pdb_dir)
        known = spec["known_residues"]

        # ── fpocket ──────────────────────────────────────────────────────────
        fp_result = {"status": "n/a", "overlap": None, "rank": None}
        if has_fpocket:
            print("  Running fpocket...", end=" ", flush=True)
            t0 = time.perf_counter()
            fp_pockets = run_fpocket(pdb_path, spec["chain"])
            fp_elapsed = time.perf_counter() - t0
            print(f"{len(fp_pockets)} pockets in {fp_elapsed:.1f}s")

            best_ov, best_rank = 0.0, None
            for p in fp_pockets[:5]:
                ov = residue_overlap(p["residues"], known)
                if ov > best_ov:
                    best_ov = ov
                    best_rank = p["rank"]

            found = best_ov >= 0.30
            fp_result = {
                "status": "PASS" if found else "MISS",
                "overlap": round(best_ov, 3),
                "rank": best_rank,
                "n_pockets": len(fp_pockets),
                "elapsed_s": round(fp_elapsed, 2),
            }
            marker = "✅" if found else "❌"
            print(f"  fpocket: {marker} overlap={best_ov:.0%}  rank={best_rank}")

        # ── Lacuna ───────────────────────────────────────────────────────────
        print(f"  Running Lacuna (RandomBackend, {DEFAULT_CONFORMERS} conformers)...", end=" ", flush=True)
        clusters, lac_elapsed = run_lacuna(pdb_path)
        print(f"{len(clusters)} pocket clusters in {lac_elapsed:.1f}s")

        best_ov, best_rank = 0.0, None
        for c in clusters[:5]:
            ov = residue_overlap(c.lining_residues, known)
            if ov > best_ov:
                best_ov = ov
                best_rank = c.rank

        lac_found = best_ov >= 0.30
        lac_result = {
            "status": "PASS" if lac_found else "MISS",
            "overlap": round(best_ov, 3),
            "rank": best_rank,
            "n_clusters": len(clusters),
            "elapsed_s": round(lac_elapsed, 2),
        }
        marker = "✅" if lac_found else "❌"
        print(f"  Lacuna:  {marker} overlap={best_ov:.0%}  rank={best_rank}  {lac_elapsed:.1f}s")

        rows.append({
            "pdb": pdb_id,
            "name": spec["name"],
            "pocket_type": spec["pocket_type"],
            "fpocket": fp_result,
            "lacuna": lac_result,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    header = f"{'Target':<20} {'Pocket type':<30} {'fpocket':^12} {'Lacuna':^16}"
    print(header)
    print("─" * len(header))

    fp_pass = lac_pass = fp_total = 0
    for r in rows:
        fp = r["fpocket"]
        lac = r["lacuna"]

        if fp["status"] != "n/a":
            fp_total += 1
            fp_icon = "✅" if fp["status"] == "PASS" else "❌"
            fp_col = f"{fp_icon} {fp['overlap']:.0%} @{fp['rank']}"
        else:
            fp_col = "n/a"

        lac_icon = "✅" if lac["status"] == "PASS" else "❌"
        lac_col = f"{lac_icon} {lac['overlap']:.0%} @{lac['rank']} ({lac['elapsed_s']:.1f}s)"

        if fp["status"] == "PASS":
            fp_pass += 1
        if lac["status"] == "PASS":
            lac_pass += 1

        print(f"{r['pdb']:<8} {r['pocket_type']:<30}  {fp_col:<16}  {lac_col}")

    print("─" * len(header))
    fp_score = f"{fp_pass}/{fp_total}" if fp_total else "n/a"
    print(f"{'Score':<38} {fp_score:^12}   {lac_pass}/{len(rows)}")

    # ── JSON output ───────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "fpocket_comparison.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
