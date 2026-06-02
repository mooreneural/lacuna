"""beta-catenin (CTNNB1) Boltz-2 undruggable protein screen.

PDB 2Z6H: armadillo repeat domain (residues 149-691, 533 res, 4030 atoms)
The TCF/LEF binding groove spans ~3000 A^2 -- too large for conventional small
molecules. No approved drugs target this surface after 40 years of attempts.

We test whether Lacuna finds cryptic sub-pockets within or adjacent to known
pharmacological interfaces using Boltz-2 conformational sampling.

Known interfaces (P35222 numbering):
  TCF/LEF groove      -- the primary undruggable surface (~K312, K345, R469, K508...)
  APC/axin binding    -- overlapping groove (~N387, S393, L430, K435...)
  E-cadherin (N-ARM)  -- N-terminal contacts (~D149, E171, R202...)
  Novel cryptic sites -- anywhere else
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

pdb_path = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\benchmarks\pdb_cache\CTNNB1.pdb")
out_dir  = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\ctnnb1_boltz_lacuna")
out_dir.mkdir(exist_ok=True)

structure = load_structure(str(pdb_path))
chain_key = list(structure.sequence.keys())[0]
n_res   = len(structure.sequence[chain_key])
n_atoms = len(structure.atoms)
print("=" * 60)
print("  beta-catenin CTNNB1 -- UNDRUGGABLE PROTEIN SCREEN")
print("=" * 60)
print(f"Structure: PDB 2Z6H (armadillo repeat domain, apo)")
print(f"Residues: {n_res} (149-691)  |  Atoms: {n_atoms}")
print(f"Status: NO approved small molecule drugs after 40 years")
print()

t_start = time.perf_counter()
cache = out_dir / "coord_sets.npy"

if cache.exists():
    print("Loading cached Boltz conformers...")
    coord_sets = [np.array(c, dtype=np.float32) for c in np.load(str(cache), allow_pickle=True)]
    print(f"Loaded {len(coord_sets)} conformers")
else:
    print("Generating 30 Boltz-2 conformers (step_scale=1.3, RTX 5080)...")
    sys.stdout.flush()
    backend = BoltzBackend(step_scale=1.3, sampling_steps=200, accelerator="gpu")
    coord_sets = backend.generate(pdb_path, n_conformers=30)
    np.save(str(cache), np.array(coord_sets, dtype=object), allow_pickle=True)
    boltz_time = time.perf_counter() - t_start
    print(f"Boltz done: {len(coord_sets)} conformers in {boltz_time:.0f}s ({boltz_time/max(len(coord_sets),1):.1f}s/conf)")

sys.stdout.flush()

# Pocket detection
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

# Write outputs
write_report(clusters, structure, len(all_coords), out_dir)
for i, c in enumerate(clusters[:5]):
    write_boltz_constraint(c, structure, out_dir, i)
    write_vina_box(c, out_dir, i)

# Ranked table
print(f"{'Rank':<5} {'Drg':>6} {'Persist':>8} {'Vol A3':>8} {'Crypt':>6}  Centroid")
for c in clusters[:12]:
    cx, cy, cz = c.centroid
    crypt = "*" if c.cryptic else ""
    print(f"{c.rank:<5} {c.druggability:>6.3f} {c.persistence:>7.0%} {c.volume_a3:>8.0f} {crypt:>6}  ({cx:.1f},{cy:.1f},{cz:.1f})")

# Known interface recovery
print("\nKnown beta-catenin pharmacological interfaces:")
known_sites = {
    "TCF/LEF binding groove":   {312, 345, 349, 354, 393, 435, 469, 472, 473, 508, 515, 551, 562},
    "APC/axin binding site":    {382, 383, 384, 386, 387, 388, 390, 391, 392, 393, 394, 395},
    "E-cadherin interface":     {174, 175, 176, 194, 195, 216, 220, 224, 262, 292, 293},
}
for site_name, ref_res in known_sites.items():
    best_ov, best_rank, best_persist, best_drg, best_vol = 0, None, 0, 0, 0
    for c in clusters[:25]:
        try:
            nums = {int("".join(filter(str.isdigit, r.split(":")[0]))) for r in c.lining_residues}
        except Exception:
            continue
        ov = len(nums & ref_res) / len(ref_res) if ref_res else 0
        if ov > best_ov:
            best_ov, best_rank, best_persist, best_drg, best_vol = ov, c.rank, c.persistence, c.druggability, c.volume_a3
    status = "PASS" if best_ov >= 0.25 else "MISS"  # lower threshold for large flat interfaces
    print(f"  {status}  {site_name:30s}  overlap={best_ov:.0%}  rank={best_rank}  persist={best_persist:.0%}  drg={best_drg:.3f}")

# Highlight any novel high-druggability cryptic pockets not at known interfaces
print("\nNovel cryptic pockets (druggability > 0.75, not in top known site):")
all_known = set().union(*[s for s in known_sites.values()])
novel_count = 0
for c in clusters[:20]:
    if not c.cryptic:
        continue
    if c.druggability < 0.75:
        continue
    try:
        nums = {int("".join(filter(str.isdigit, r.split(":")[0]))) for r in c.lining_residues}
    except Exception:
        nums = set()
    interface_overlap = len(nums & all_known) / max(len(all_known), 1)
    if interface_overlap < 0.15:  # mostly novel residues
        cx, cy, cz = c.centroid
        print(f"  Rank {c.rank}: drg={c.druggability:.3f} persist={c.persistence:.0%} "
              f"vol={c.volume_a3:.0f}A3  centroid=({cx:.1f},{cy:.1f},{cz:.1f})")
        if c.lining_residues:
            print(f"    Lining: {', '.join(c.lining_residues[:8])}")
        novel_count += 1

if novel_count == 0:
    print("  (none above threshold)")

total = time.perf_counter() - t_start
print(f"\nTotal wall time: {total:.0f}s")
