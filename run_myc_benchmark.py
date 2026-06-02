"""c-MYC bHLH-LZ benchmark -- the most famous undruggable oncogene.

6G6K: MYC/MAX bHLH-LZ complex, 4 chains, residues 205-984.
Amplified/overexpressed in ~20% of all cancers. No approved drugs after 40 years.
Uses RandomBackend (fast) for first-pass screen.
"""
import sys, time
sys.path.insert(0, r"C:\Users\clayt\Documents\GitHub\lacuna")

from pathlib import Path
from lacuna.io.structure import load_structure, coords_array
from lacuna.ensemble.random_backend import RandomBackend
from lacuna.pockets.detector import detect_pockets
from lacuna.pockets.clusterer import cluster_pockets
from lacuna.io.writers import write_report, write_boltz_constraint, write_vina_box

pdb_path = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\benchmarks\pdb_cache\MYC.pdb")
out_dir  = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\myc_lacuna")
out_dir.mkdir(exist_ok=True)

structure = load_structure(str(pdb_path))
n_res = sum(len(v) for v in structure.sequence.values())
chains = list(structure.sequence.keys())
print("=" * 60)
print("  c-MYC bHLH-LZ -- UNDRUGGABLE ONCOGENE SCREEN")
print("=" * 60)
print(f"Structure: PDB 6G6K (MYC/MAX bHLH-LZ)")
print(f"Chains: {chains}  Residues: {n_res}  Atoms: {len(structure.atoms)}")
print(f"Status: NO approved drugs -- amplified in ~20% of all cancers")
print()

t0 = time.perf_counter()
print("Running RandomBackend (30 conformers)...")
backend = RandomBackend(seed=42)
coord_sets = backend.generate(pdb_path, n_conformers=30)
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
elapsed = time.perf_counter() - t0
print(f"\nFound {len(clusters)} pocket clusters in {elapsed:.1f}s\n")

write_report(clusters, structure, len(all_coords), out_dir)
for i, c in enumerate(clusters[:5]):
    write_boltz_constraint(c, structure, out_dir, i)
    write_vina_box(c, out_dir, i)

print(f"{'Rank':<5} {'Drg':>6} {'Persist':>8} {'Vol A3':>8} {'Crypt':>6}  Top lining residues")
for c in clusters[:12]:
    cx, cy, cz = c.centroid
    crypt = "*" if c.cryptic else ""
    lining = ", ".join(c.lining_residues[:4])
    print(f"{c.rank:<5} {c.druggability:>6.3f} {c.persistence:>7.0%} {c.volume_a3:>8.0f} {crypt:>6}  {lining}")

print("\nKnown MYC pharmacological interfaces (P01106 numbering):")
known_sites = {
    "MYC-MAX bHLH interface":  {366, 369, 370, 373, 375, 377, 378, 381, 382, 385, 387, 389},
    "MYC leucine zipper":      {409, 416, 423, 430, 437},
    "Omomyc binding surface":  {366, 370, 374, 378, 382, 386},
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
    status = "PASS" if best_ov >= 0.25 else "MISS"
    print(f"  {status}  {site_name:30s}  overlap={best_ov:.0%}  rank={best_rank}  drg={best_drg:.3f}")

print(f"\nTotal: {elapsed:.1f}s")
