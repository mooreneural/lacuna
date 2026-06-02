"""KRAS G12D Boltz-2 screen -- comparison to WT K-Ras (4OBE, 93% switch-II overlap).

6OIM: KRAS G12D with GDP, single chain, residues 0-169.
G12D is the most common KRAS mutation (pancreatic, colorectal, NSCLC).
No approved therapy -- G12C covalent strategy does not work for G12D.
"""
import sys, time
sys.path.insert(0, r"C:\Users\clayt\Documents\GitHub\lacuna")

from pathlib import Path
import numpy as np
from lacuna.io.structure import load_structure, coords_array
from lacuna.ensemble.boltz_backend import BoltzBackend
from lacuna.pockets.detector import detect_pockets
from lacuna.pockets.clusterer import cluster_pockets
from lacuna.io.writers import write_report, write_boltz_constraint, write_vina_box

pdb_path = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\benchmarks\pdb_cache\KRAS_G12D.pdb")
out_dir  = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\kras_g12d_boltz_lacuna")
out_dir.mkdir(exist_ok=True)

structure = load_structure(str(pdb_path))
n_res = sum(len(v) for v in structure.sequence.values())
print("=" * 60)
print("  KRAS G12D -- vs WT K-Ras benchmark")
print("=" * 60)
print(f"Structure: PDB 6OIM (KRAS G12D + GDP)")
print(f"Residues: {n_res}  Atoms: {len(structure.atoms)}")
print(f"Mutation: Gly12Asp -- most common KRAS mutation, NO approved therapy")
print()

t_start = time.perf_counter()
cache = out_dir / "coord_sets.npy"

if cache.exists():
    print("Loading cached conformers...")
    coord_sets = [np.array(c, dtype=np.float32) for c in np.load(str(cache), allow_pickle=True)]
    print(f"Loaded {len(coord_sets)} conformers")
else:
    print("Generating 30 Boltz-2 conformers (step_scale=1.3, RTX 5080)...")
    sys.stdout.flush()
    backend = BoltzBackend(step_scale=1.3, sampling_steps=200, accelerator="gpu")
    coord_sets = backend.generate(pdb_path, n_conformers=30)
    np.save(str(cache), np.array(coord_sets, dtype=object), allow_pickle=True)
    t = time.perf_counter() - t_start
    print(f"Boltz done: {len(coord_sets)} conformers in {t:.0f}s ({t/max(len(coord_sets),1):.1f}s/conf)")

sys.stdout.flush()

all_coords = [coords_array(structure)] + coord_sets
pocket_lists = []
for ci, coords in enumerate(all_coords):
    pockets = detect_pockets(coords, structure)
    for p in pockets:
        p.conformer_idx = ci
    pocket_lists.append(pockets)
    if ci % 5 == 0:
        print(f"  conformer {ci}/{len(all_coords)-1}: {len(pockets)} pockets", flush=True)

clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords))
print(f"\nFound {len(clusters)} pocket clusters across {len(all_coords)} conformers\n")

write_report(clusters, structure, len(all_coords), out_dir)
for i, c in enumerate(clusters[:5]):
    write_boltz_constraint(c, structure, out_dir, i)
    write_vina_box(c, out_dir, i)

print(f"{'Rank':<5} {'Drg':>6} {'Persist':>8} {'Vol A3':>8} {'Crypt':>6}  Centroid")
for c in clusters[:10]:
    cx, cy, cz = c.centroid
    crypt = "*" if c.cryptic else ""
    print(f"{c.rank:<5} {c.druggability:>6.3f} {c.persistence:>7.0%} {c.volume_a3:>8.0f} {crypt:>6}  ({cx:.1f},{cy:.1f},{cz:.1f})")

print("\nKRAS G12D pharmacological sites:")
known_sites = {
    "switch-II pocket (SIIP)": {58, 59, 60, 61, 62, 63, 71, 72, 73, 74, 95, 96},
    "switch-I region":         {25, 26, 27, 28, 29, 30, 32, 35},
    "P-loop / G12D site":      {10, 11, 12, 13, 14, 15, 16},
    "nucleotide binding":      {10, 15, 57, 116, 117, 119},
}
for site_name, ref_res in known_sites.items():
    best_ov, best_rank, best_persist, best_drg = 0, None, 0, 0
    for c in clusters[:20]:
        try:
            nums = {int("".join(filter(str.isdigit, r.split(":")[0]))) for r in c.lining_residues}
        except Exception:
            continue
        ov = len(nums & ref_res) / len(ref_res) if ref_res else 0
        if ov > best_ov:
            best_ov, best_rank, best_persist, best_drg = ov, c.rank, c.persistence, c.druggability
    status = "PASS" if best_ov >= 0.30 else "MISS"
    print(f"  {status}  {site_name:30s}  overlap={best_ov:.0%}  rank={best_rank}  persist={best_persist:.0%}  drg={best_drg:.3f}")

print(f"\nWT K-Ras reference (4OBE, RandomBackend): switch-II 93% rank 4")
print(f"Total wall time: {time.perf_counter()-t_start:.0f}s")
