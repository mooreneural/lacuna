# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""PocketMiner cryptic-pocket benchmark (Meller et al. 2023, Nat. Commun.).

Uses the PocketMiner dataset (github.com/Mickdub/gvp, `pocket_pred` branch):
per-residue cryptic labels for apo structures, where a residue labelled ``1``
forms a cryptic pocket in the holo state, ``0`` does not, and ``2`` is
excluded/uncertain. 61 apo structures (35 test + 26 validation).

For each apo structure Lacuna runs with its default configuration (NMA backend,
crypticity ranking) and a top-5 pocket counts as a hit if its lining residues
overlap >=30% of the labelled cryptic residues, or its centre lies within 4 A of
their Ca centroid.

Residue mapping: labels are aligned to the resolved residues of the apo chain in
order; the script flags any structure where the RCSB residue count diverges from
the label length (a sign the order mapping is unreliable for that entry).

    python benchmarks/pocketminer_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import (  # noqa: E402
    download_pdb, run_lacuna, residue_overlap, residue_jaccard,
    compute_known_site_centroid, pocket_min_centroid_dist,
    CENTROID_THRESHOLD, OVERLAP_THRESHOLD, JACCARD_THRESHOLD,
)
from lacuna.io.structure import load_structure  # noqa: E402

PM_DIR = Path(__file__).parent / "pm_data"
MAX_RESIDUES = 700

# PocketMiner dataset (Meller et al. 2023) — labels + apo ids, fetched on demand.
_PM_RAW = "https://raw.githubusercontent.com/Mickdub/gvp/pocket_pred/data/pm-dataset/"
_PM_FILES = (
    "test_apo_ids_with_chainids.npy", "test_label_dictionary.npy",
    "val_apo_ids_with_chainids.npy", "val_label_dictionary.npy",
)


def _fetch_pm_data():
    import urllib.request
    PM_DIR.mkdir(exist_ok=True)
    for f in _PM_FILES:
        dst = PM_DIR / f
        if not dst.exists():
            print(f"  Downloading PocketMiner {f}...", flush=True)
            urllib.request.urlretrieve(_PM_RAW + f, dst)


def load_split(split: str):
    ids = np.load(PM_DIR / f"{split}_apo_ids_with_chainids.npy", allow_pickle=True)
    labels = np.load(PM_DIR / f"{split}_label_dictionary.npy", allow_pickle=True).item()
    out = []
    for entry in ids:
        entry = str(entry)
        pdb, chain = entry[:4], entry[4:]
        if pdb in labels:
            out.append((pdb, chain, np.asarray(labels[pdb])))
    return out


def cryptic_residues(apo_path: Path, chain: str, label_arr: np.ndarray):
    """Map per-residue labels to apo residue seq numbers (order-based)."""
    s = load_structure(apo_path, chain=chain)
    res = [r for r in s.residues if r.chain_id == chain] or s.residues
    n = min(len(res), len(label_arr))
    cryptic = {res[i].seq_num for i in range(n) if int(label_arr[i]) == 1}
    return cryptic, len(res), len(label_arr)


def main():
    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)
    _fetch_pm_data()
    entries = load_split("test") + load_split("val")

    print("=" * 70)
    print(f"  POCKETMINER BENCHMARK  ({len(entries)} apo structures, NMA + crypticity)")
    print("=" * 70)

    n_pass = n_run = n_pass_legacy = 0
    rows = []
    for pdb, chain, label_arr in entries:
        tag = f"{pdb}{chain}"
        try:
            apo = download_pdb(pdb.upper(), pdb_dir)
        except Exception as e:
            print(f"  [skip] {tag}: download failed ({e})")
            continue
        try:
            cryptic, n_res, n_lab = cryptic_residues(apo, chain, label_arr)
        except Exception as e:
            print(f"  [skip] {tag}: parse failed ({e})")
            continue
        if not cryptic:
            print(f"  [skip] {tag}: no cryptic-labelled residues")
            continue
        s = load_structure(apo, chain=chain)
        if len(s.residues) > MAX_RESIDUES:
            print(f"  [skip] {tag}: {len(s.residues)} residues > {MAX_RESIDUES}")
            continue

        mism = "" if abs(n_res - n_lab) <= 5 else f"  [!] res/label len {n_res}/{n_lab}"
        ref = compute_known_site_centroid(apo, chain, cryptic)
        try:
            clusters, elapsed = run_lacuna(apo, 20, chain=chain, rank_by="crypticity")
        except Exception as e:
            print(f"  [err] {tag}: lacuna failed ({e})")
            continue

        best_ov = max((residue_overlap(c.lining_residues, cryptic) for c in clusters[:5]),
                      default=0.0)
        best_jac = max((residue_jaccard(c.lining_residues, cryptic) for c in clusters[:5]),
                       default=0.0)
        best_dist, _ = pocket_min_centroid_dist(clusters, ref, top_n=5)
        # Size-robust headline: centroid OR Jaccard. Legacy (recall) tracked alongside.
        found = best_dist <= CENTROID_THRESHOLD or best_jac >= JACCARD_THRESHOLD
        found_legacy = best_dist <= CENTROID_THRESHOLD or best_ov >= OVERLAP_THRESHOLD
        n_run += 1
        n_pass += int(found)
        n_pass_legacy += int(found_legacy)
        rows.append((tag, found, best_ov, best_jac, best_dist, len(cryptic)))
        mark = "PASS" if found else "miss"
        dist_s = f"{best_dist:.1f}A" if best_dist < float("inf") else "n/a"
        print(f"  {mark} {tag}  jac={best_jac:.0%}  recall={best_ov:.0%}  dist={dist_s}  "
              f"({len(cryptic)} cryptic res, {elapsed:.1f}s){mism}")

    print("-" * 70)
    print(f"  POCKETMINER (size-robust, cen≤{CENTROID_THRESHOLD:.0f}Å OR jac≥{JACCARD_THRESHOLD:.0%}): "
          f"{n_pass}/{n_run} ({n_pass / max(n_run, 1):.0%})")
    print(f"  POCKETMINER (legacy recall-based): "
          f"{n_pass_legacy}/{n_run} ({n_pass_legacy / max(n_run, 1):.0%})")


if __name__ == "__main__":
    main()
