"""Re-run IL-2 and Src near-misses with the Boltz-2 backend.

Both proteins scored 21-28% with RandomBackend (30% threshold).
Boltz-2 partial diffusion samples larger conformational changes
and may expose the buried pockets enough to flip them to PASS.

Usage:
    python benchmarks/boltz_nearmiss.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

CONFORMERS = 30
OVERLAP_THRESHOLD = 0.30
HOLO_CUTOFF = 4.5

SOLVENT_CODES = frozenset({
    "HOH","WAT","DOD","SO4","SUL","PO4","HPO","EDO","EGL","GOL","PGO",
    "ACT","ACE","ACY","FMT","CIT","TLA","TAR","MES","HEP","TRS",
    "DMF","DMS","DIO","IPA","CL","NA","MG","ZN","CA","K","FE","MN",
    "CO","CU","NI","CD","HG","PE3","PE4","PE5","PE6","PE7","PE8","PEG",
})

PROTEINS = [
    {
        "id": "IL2",
        "name": "Interleukin-2 (cryptic helix-α1 site)",
        "apo_pdb": "1M47",
        "holo_pdb": "1M49",
        "extra_exclude": frozenset(),
    },
    {
        "id": "SRC",
        "name": "Src kinase myristate/SH2-linker pocket",
        "apo_pdb": "2SRC",
        "holo_pdb": "3EL8",
        "extra_exclude": frozenset({"MYR", "ADP", "ATP", "ANP"}),
    },
]


def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    out = dest_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id}...", end=" ", flush=True)
    urllib.request.urlretrieve(url, out)
    print("done")
    return out


def extract_binding_site(holo_path: Path, extra_exclude: frozenset) -> set[int]:
    from collections import defaultdict
    exclude = SOLVENT_CODES | extra_exclude
    atoms = []
    with open(holo_path, errors="replace") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                atoms.append({
                    "record": rec,
                    "resname": line[17:20].strip(),
                    "resseq": int(line[22:26].strip()),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                pass

    lig_groups: dict[tuple, list] = defaultdict(list)
    for a in atoms:
        if a["record"] == "HETATM" and a["resname"] not in exclude:
            key = (a["resname"], a["resseq"])
            lig_groups[key].append(a)

    if not lig_groups:
        return set()

    principal = max(lig_groups, key=lambda k: len(lig_groups[k]))
    lig_coords = [(a["x"], a["y"], a["z"]) for a in lig_groups[principal]]
    print(f"    Ligand: {principal[0]} ({len(lig_coords)} atoms)")

    cutoff2 = HOLO_CUTOFF ** 2
    binding: set[int] = set()
    for a in atoms:
        if a["record"] == "ATOM":
            if any((a["x"]-lx)**2 + (a["y"]-ly)**2 + (a["z"]-lz)**2 <= cutoff2
                   for lx, ly, lz in lig_coords):
                binding.add(a["resseq"])
    return binding


def residue_overlap(cluster_residues: list[str], known: set[int]) -> float:
    found: set[int] = set()
    for label in cluster_residues:
        try:
            found.add(int("".join(c for c in label.split(":")[0] if c.isdigit())))
        except (ValueError, IndexError):
            pass
    return len(found & known) / len(known) if known else 0.0


def run_boltz(pdb_path: Path, cache_path: Path) -> list:
    from lacuna.ensemble.boltz_backend import BoltzBackend
    if cache_path.exists():
        print(f"    Loading cached conformers from {cache_path.name}...")
        return [np.array(c, dtype=np.float32)
                for c in np.load(str(cache_path), allow_pickle=True)]
    print(f"    Running Boltz-2 ({CONFORMERS} conformers, GPU)...")
    backend = BoltzBackend(step_scale=1.3, sampling_steps=200, accelerator="gpu")
    coord_sets = backend.generate(pdb_path, n_conformers=CONFORMERS)
    np.save(str(cache_path), np.array(coord_sets, dtype=object), allow_pickle=True)
    return coord_sets


def score_protein(entry: dict, pdb_dir: Path) -> dict:
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets

    print(f"\n{'='*60}")
    print(f"  {entry['id']}  —  {entry['name']}")
    print(f"{'='*60}")

    holo_path = download_pdb(entry["holo_pdb"], pdb_dir)
    known = extract_binding_site(holo_path, entry["extra_exclude"])
    print(f"  Binding site: {len(known)} residues at {HOLO_CUTOFF}Å")

    apo_path = download_pdb(entry["apo_pdb"], pdb_dir)
    cache = pdb_dir / f"{entry['id']}_boltz_coords.npy"

    t0 = time.perf_counter()
    coord_sets = run_boltz(apo_path, cache)
    boltz_time = time.perf_counter() - t0
    print(f"    {len(coord_sets)} conformers ready in {boltz_time:.1f}s")

    structure = load_structure(apo_path)
    all_coords = [coords_array(structure)] + coord_sets

    t1 = time.perf_counter()
    pocket_lists = []
    for ci, coords in enumerate(all_coords):
        pockets = detect_pockets(coords, structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)
    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords))
    detect_time = time.perf_counter() - t1

    best_ov, best_rank = 0.0, None
    for c in clusters[:5]:
        ov = residue_overlap(c.lining_residues, known)
        if ov > best_ov:
            best_ov, best_rank = ov, c.rank

    found = best_ov >= OVERLAP_THRESHOLD
    status = "PASS" if found else "MISS"
    marker = "PASS" if found else "MISS"
    print(f"\n  RandomBackend result (previous): MISS")
    print(f"  Boltz-2 result:  [{marker}]  overlap={best_ov:.0%}  rank={best_rank}  "
          f"clusters={len(clusters)}  detect={detect_time:.1f}s")

    return {
        "id": entry["id"],
        "name": entry["name"],
        "status": status,
        "overlap": round(best_ov, 3),
        "rank": best_rank,
        "n_clusters": len(clusters),
        "boltz_conformers": len(coord_sets),
    }


def main():
    print("=" * 60)
    print("  BOLTZ-2 NEAR-MISS RE-EVALUATION")
    print("  IL-2 (1M47) and Src (2SRC) — previously 21% and 28%")
    print("=" * 60)

    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)

    results = []
    for entry in PROTEINS:
        r = score_protein(entry, pdb_dir)
        results.append(r)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        marker = "PASS" if r["status"] == "PASS" else "MISS"
        print(f"  [{marker}]  {r['id']:6s}  overlap={r['overlap']:.0%}  rank={r['rank']}  {r['name']}")

    import json
    out = Path(__file__).parent / "boltz_nearmiss_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {out}")


if __name__ == "__main__":
    main()
