"""Sensitivity analysis: how many Boltz conformers does Lacuna need?

Uses cached 5-HT2A Boltz-2 conformers (30 available in 5ht2a_boltz_lacuna/).
Re-runs pocket detection with 5/10/15/20/25/30 conformer subsets.
Reports: n_clusters, top druggability, known-site recovery, wall time.
"""
import sys, time
sys.path.insert(0, r"C:\Users\clayt\Documents\GitHub\lacuna")

from pathlib import Path
import numpy as np
from lacuna.io.structure import load_structure, coords_array
from lacuna.pockets.detector import detect_pockets
from lacuna.pockets.clusterer import cluster_pockets

pdb_path = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\benchmarks\pdb_cache\5HT2A_AF.pdb")
cache    = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\5ht2a_boltz_lacuna\coord_sets.npy")

structure = load_structure(str(pdb_path))
all_conformers = [np.array(c, dtype=np.float32)
                  for c in np.load(str(cache), allow_pickle=True)]
base_coords = coords_array(structure)

known_sites = {
    "orthosteric": {155, 156, 159, 239, 242, 243, 336, 340},
    "Na_allosteric": {95, 96, 97, 98},
    "Gq_face": {143, 144, 145, 146, 147, 228, 310, 315, 316, 317},
    "ECL2": {175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185},
}

print("=" * 70)
print("  5-HT2A SENSITIVITY: conformer count vs pocket recovery (Boltz-2)")
print("=" * 70)
print(f"{'N':>4}  {'Clust':>6}  {'TopDrg':>7}  {'Time':>6}  Recovery (ortho / Na+ / Gq / ECL2)")
print("-" * 70)

for n in [5, 10, 15, 20, 25, 30]:
    subset = all_conformers[:n]
    all_coords = [base_coords] + list(subset)
    t0 = time.perf_counter()

    pocket_lists = []
    for ci, coords in enumerate(all_coords):
        pockets = detect_pockets(coords, structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)

    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords))
    elapsed = time.perf_counter() - t0
    top_drg = clusters[0].druggability if clusters else 0.0

    results = []
    for site_name, ref_res in known_sites.items():
        best = 0.0
        for c in clusters[:20]:
            try:
                nums = {int("".join(filter(str.isdigit, r.split(":")[0]))) for r in c.lining_residues}
            except Exception:
                continue
            ov = len(nums & ref_res) / len(ref_res) if ref_res else 0
            best = max(best, ov)
        results.append(f"{best:.0%}")

    print(f"{n:>4}  {len(clusters):>6}  {top_drg:>7.3f}  {elapsed:>5.1f}s  {' / '.join(results)}")

print()
print("Interpretation: plateau = minimum conformers needed for reliable discovery")
