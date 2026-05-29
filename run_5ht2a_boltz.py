"""5-HT2A Boltz-2 ensemble screen -- 30 diffusion samples."""
import sys, time
sys.path.insert(0, r"C:\Users\clayt\Documents\GitHub\lacuna")

from pathlib import Path
import numpy as np
from lacuna.io.structure import load_structure, coords_array
from lacuna.ensemble.boltz_backend import BoltzBackend
from lacuna.pockets.detector import detect_pockets
from lacuna.pockets.clusterer import cluster_pockets
from lacuna.io.writers import write_report, write_boltz_constraint, write_vina_box

pdb_path = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\benchmarks\pdb_cache\5HT2A_AF.pdb")
out_dir  = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\5ht2a_boltz_lacuna")
out_dir.mkdir(exist_ok=True)

structure = load_structure(str(pdb_path))
chain_key = list(structure.sequence.keys())[0]
n_res   = len(structure.sequence[chain_key])
n_atoms = len(structure.atoms)
print(f"5-HT2A (P28223): {n_res} residues, {n_atoms} atoms")

t_start = time.perf_counter()
coord_sets_cache = out_dir / "coord_sets.npy"

if coord_sets_cache.exists():
    print("Loading cached Boltz conformers...")
    raw = np.load(str(coord_sets_cache), allow_pickle=True)
    coord_sets = [np.array(c, dtype=np.float32) for c in raw]
    boltz_time = 0
else:
    print("Generating 30 Boltz-2 conformers (step_scale=1.3, RTX 5080)...")
    sys.stdout.flush()
    backend = BoltzBackend(step_scale=1.3, sampling_steps=200, accelerator='gpu')
    coord_sets = backend.generate(pdb_path, n_conformers=30)
    boltz_time = time.perf_counter() - t_start
    np.save(str(coord_sets_cache), np.array(coord_sets, dtype=object), allow_pickle=True)
    print(f"Boltz done: {len(coord_sets)} conformers in {boltz_time:.0f}s ({boltz_time/max(len(coord_sets),1):.1f}s/conf)")

print(f"Using {len(coord_sets)} cached conformers")
sys.stdout.flush()

all_coords = [coords_array(structure)] + coord_sets
pocket_lists = []
for ci, coords in enumerate(all_coords):
    pockets = detect_pockets(coords, structure)
    for p in pockets:
        p.conformer_idx = ci
    pocket_lists.append(pockets)
    if ci % 5 == 0:
        print(f"  conformer {ci}/{len(all_coords)-1}: {len(pockets)} pockets")
        sys.stdout.flush()

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

print("\nKnown 5-HT2A pharmacological sites:")
known_sites = {
    'orthosteric_core':  {155,156,159,239,242,243,336,340},
    'extended_binding':  {120,151,229,233,370},
    'intracellular_Gq':  {143,144,145,146,147,228,310,315,316,317},
    'na_allosteric':     {95,96,97,98},
    'ecl2_vestibule':    {175,176,177,178,179,180,181,182,183,184,185},
}
for site_name, ref_res in known_sites.items():
    best_overlap, best_rank, best_persist = 0, None, 0
    for c in clusters[:20]:
        try:
            nums = {int(''.join(filter(str.isdigit, r))) for r in c.lining_residues}
        except Exception:
            continue
        ov = len(nums & ref_res) / len(ref_res) if ref_res else 0
        if ov > best_overlap:
            best_overlap, best_rank, best_persist = ov, c.rank, c.persistence
    status = "PASS" if best_overlap >= 0.30 else "MISS"
    print(f"  {status}  {site_name:25s}  overlap={best_overlap:.0%}  rank={best_rank}  persist={best_persist:.0%}")

total_time = time.perf_counter() - t_start
print(f"\nTotal wall time: {total_time:.0f}s")
