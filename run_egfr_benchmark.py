"""EGFR kinase domain benchmark — 2GS7 (apo, two-chain asymmetric dimer).

Tests Lacuna against four known pharmacological sites:
  1. ATP/inhibitor orthosteric site (erlotinib, gefitinib, osimertinib all bind here)
  2. Covalent C797 pocket (osimertinib, afatinib — 3rd/2nd gen EGFR inhibitors)
  3. Allosteric alpha-C helix back pocket (cryptic in active state)
  4. Asymmetric dimer interface C-lobe (activating interface, novel allosteric target)

Residue numbers follow P00533 (full-length EGFR) — PDB 2GS7 preserves this numbering.
"""
import sys, time
sys.path.insert(0, r"C:\Users\clayt\Documents\GitHub\lacuna")

from pathlib import Path
from lacuna.io.structure import load_structure, coords_array
from lacuna.ensemble.random_backend import RandomBackend
from lacuna.pockets.detector import detect_pockets
from lacuna.pockets.clusterer import cluster_pockets

pdb_path = Path(r"C:\Users\clayt\Documents\GitHub\lacuna\benchmarks\pdb_cache\EGFR.pdb")

structure = load_structure(str(pdb_path))
chains = list(structure.sequence.keys())
total_res = sum(len(s) for s in structure.sequence.values())
print("=" * 60)
print("  EGFR KINASE DOMAIN BENCHMARK")
print("=" * 60)
print(f"Structure: PDB 2GS7 (apo, no inhibitor)")
print(f"Chains: {chains}  |  Residues: {total_res}  |  Atoms: {len(structure.atoms)}")
print(f"Residue range: 677-992 (EGFR kinase domain, P00533 numbering)")
print()

# Known pharmacological sites (P00533 numbering, all within 677-992)
known_sites = {
    "ATP/orthosteric site": {
        "residues": {745, 762, 769, 790, 797, 854, 855, 856},
        "note": "Lys745, Glu762 (alphaC), Met769 (hinge), Thr790 (gatekeeper), Cys797, DFG motif -- all approved EGFR drugs bind here"
    },
    "Covalent C797 pocket": {
        "residues": {790, 792, 793, 797, 854, 855},
        "note": "Osimertinib/afatinib covalent site — T790M resistance mutant pocket"
    },
    "alphaC-helix allosteric back pocket": {
        "residues": {773, 776, 777, 779, 780, 793, 841},
        "note": "Cryptic in DFG-in active state. Opened in inactive/DFG-out conformations."
    },
    "Asymmetric dimer interface": {
        "residues": {834, 835, 836, 838, 956, 960, 963},
        "note": "C-lobe activating interface — novel allosteric target for EGFR dimers"
    },
}

print("Running RandomBackend (20 conformers)...")
t0 = time.perf_counter()
backend = RandomBackend(seed=42)
coord_sets = backend.generate(pdb_path, n_conformers=20)
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
n_raw = sum(len(pl) for pl in pocket_lists)
print(f"\n{len(clusters)} pocket clusters ({n_raw} raw) in {elapsed:.1f}s\n")

# Ranked table
print(f"{'Rank':<5} {'Drg':>6} {'Persist':>8} {'Vol A3':>8} {'Crypt':>6}  Lining residues (first 5)")
for c in clusters[:12]:
    cx, cy, cz = c.centroid
    crypt = "*" if c.cryptic else ""
    lining_short = ", ".join(c.lining_residues[:5])
    print(f"{c.rank:<5} {c.druggability:>6.3f} {c.persistence:>7.0%} {c.volume_a3:>8.0f} {crypt:>6}  {lining_short}")

# Known-site recovery
print(f"\n{'='*60}")
print("  KNOWN SITE RECOVERY")
print(f"{'='*60}")
passed = 0
for site_name, info in known_sites.items():
    ref_res = info["residues"]
    best_overlap, best_rank, best_persist, best_drg = 0, None, 0, 0
    for c in clusters[:20]:
        try:
            nums = {int("".join(filter(str.isdigit, r.split(":")[0]))) for r in c.lining_residues}
        except Exception:
            continue
        ov = len(nums & ref_res) / len(ref_res) if ref_res else 0
        if ov > best_overlap:
            best_overlap, best_rank, best_persist, best_drg = ov, c.rank, c.persistence, c.druggability
    status = "PASS" if best_overlap >= 0.30 else "MISS"
    if status == "PASS":
        passed += 1
    print(f"\n  {status}  {site_name}")
    print(f"       overlap={best_overlap:.0%}  rank={best_rank}  persist={best_persist:.0%}  drg={best_drg:.3f}")
    print(f"       {info['note']}")

print(f"\n{'='*60}")
print(f"  RESULT: {passed}/{len(known_sites)} known sites found (>=30% residue overlap in top-20)")
print(f"  Total time: {elapsed:.1f}s")
print(f"{'='*60}")
