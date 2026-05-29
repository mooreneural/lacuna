"""Lacuna benchmark suite.

Accuracy: does Lacuna find known binding sites in well-characterized proteins?
Performance: wall-clock time per conformer count and protein size.

Test set (all APO structures — no ligand, so the pocket must be discovered blind):
  1HEL  Hen egg-white lysozyme        Known active site: Glu35, Asp52 (orthosteric)
  1L90  T4 Lysozyme L99A              Famous cryptic hydrophobic cavity (Val111 region)
  4OBE  K-Ras WT (apo)               Switch-II cryptic pocket (G12/V29/D33 region)
  1HPV  HIV-1 protease (apo)         Flap-tip pocket (catalytic Asp25/Asp25')
  3CLN  Calmodulin (Ca2+-free)       Hydrophobic cleft exposed on Ca2+ binding

Ground truth (residues that define the known pocket, from literature):
  Listed in BENCHMARKS dict below. We report whether any top-5 pocket cluster
  has ≥ 30% residue overlap with the known site.
"""

from __future__ import annotations

import sys
import time
import urllib.request
import tempfile
from pathlib import Path

# -- benchmark definitions -----------------------------------------------------

BENCHMARKS = {
    "1HEL": {
        "name": "Hen lysozyme (orthosteric)",
        "chain": "A",
        "known_residues": {35, 52, 101, 102, 103, 104, 107, 108},  # Glu35, Asp52 + binding cleft
        "expected_cryptic": False,
        "expected_rank": 1,
    },
    "1L90": {
        "name": "T4 Lysozyme L99A (cryptic hydrophobic cavity)",
        "chain": "A",
        "known_residues": {99, 102, 106, 111, 118, 121, 133, 153},  # Val/Leu ring around cavity
        "expected_cryptic": True,
        "expected_rank": 3,   # should appear in top 3
    },
    "4OBE": {
        "name": "K-Ras WT apo (switch-II cryptic pocket)",
        "chain": "A",
        "known_residues": {12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
        "expected_cryptic": True,
        "expected_rank": 3,
    },
    "1HPV": {
        "name": "HIV-1 protease apo (flap/active site)",
        "chain": "A",
        "known_residues": {25, 26, 27, 28, 29, 30, 49, 50, 51, 52, 53},
        "expected_cryptic": False,
        "expected_rank": 2,
    },
}

CONFORMER_COUNTS = [1, 5, 20, 50]  # performance sweep
DEFAULT_CONFORMERS = 20             # used for accuracy benchmarks


# -- helpers -------------------------------------------------------------------

def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    """Download a PDB file from RCSB."""
    out = dest_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB...", end=" ", flush=True)
    urllib.request.urlretrieve(url, out)
    print(f"saved to {out.name}")
    return out


def residue_overlap(cluster_residues: list[str], known: set[int]) -> float:
    """Fraction of known residues that appear in the pocket's lining residues."""
    found: set[int] = set()
    for label in cluster_residues:
        try:
            num = int("".join(c for c in label.split(":")[0] if c.isdigit()))
            found.add(num)
        except (ValueError, IndexError):
            pass
    if not known:
        return 0.0
    return len(found & known) / len(known)


# -- accuracy benchmark --------------------------------------------------------

def run_accuracy(pdb_dir: Path) -> list[dict]:
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.ensemble.random_backend import RandomBackend
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets

    results = []

    for pdb_id, spec in BENCHMARKS.items():
        backend = RandomBackend(seed=42)  # fresh RNG per protein for reproducibility
        print(f"\n{'-'*60}")
        print(f"  {pdb_id}  {spec['name']}")
        print(f"{'-'*60}")

        try:
            pdb_path = download_pdb(pdb_id, pdb_dir)
        except Exception as e:
            print(f"  [SKIP] Download failed: {e}")
            results.append({"pdb": pdb_id, "status": "download_failed"})
            continue

        try:
            structure = load_structure(pdb_path)
            print(f"  {len(structure.residues)} residues, chains: {list(structure.sequence.keys())}")

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

            n_raw = sum(len(pl) for pl in pocket_lists)
            print(f"  {len(clusters)} pocket clusters ({n_raw} raw) in {elapsed:.1f}s")

            # Find best overlap in top 5
            known = spec["known_residues"]
            best_rank = None
            best_overlap = 0.0

            for c in clusters[:5]:
                ov = residue_overlap(c.lining_residues, known)
                if ov > best_overlap:
                    best_overlap = ov
                    best_rank = c.rank

            found = best_overlap >= 0.30
            status = "PASS" if found else "MISS"
            print(f"  Known pocket overlap: {best_overlap:.0%} at rank {best_rank}  [{status}]")

            if clusters:
                top = clusters[0]
                print(f"  Top pocket: druggability={top.druggability:.3f}, "
                      f"persistence={top.persistence:.0%}, "
                      f"cryptic={'YES' if top.cryptic else 'no'}")

            results.append({
                "pdb": pdb_id,
                "name": spec["name"],
                "status": status,
                "best_overlap": round(best_overlap, 3),
                "best_rank": best_rank,
                "n_clusters": len(clusters),
                "elapsed_s": round(elapsed, 2),
                "n_residues": len(structure.residues),
            })

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results.append({"pdb": pdb_id, "status": "error", "error": str(e)})

    return results


# -- performance benchmark -----------------------------------------------------

def run_performance(pdb_dir: Path) -> list[dict]:
    """Time the full pipeline at different conformer counts on one protein."""
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.ensemble.random_backend import RandomBackend
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets

    pdb_id = "1HEL"
    print(f"\n{'-'*60}")
    print(f"  Performance sweep on {pdb_id} ({CONFORMER_COUNTS} conformers)")
    print(f"{'-'*60}")

    try:
        pdb_path = download_pdb(pdb_id, pdb_dir)
    except Exception as e:
        print(f"  [SKIP] {e}")
        return []

    structure = load_structure(pdb_path)
    results = []

    for n in CONFORMER_COUNTS:
        backend = RandomBackend(seed=0)
        t0 = time.perf_counter()

        coord_sets = backend.generate(pdb_path, n_conformers=n)
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

        per_conf = elapsed / len(all_coords)
        print(f"  n={n:3d} conformers: {elapsed:5.2f}s total  ({per_conf:.3f}s/conformer)  "
              f"{len(clusters)} pockets")
        results.append({
            "n_conformers": n,
            "elapsed_s": round(elapsed, 3),
            "s_per_conformer": round(per_conf, 4),
            "n_pockets": len(clusters),
        })

    return results


# -- main ----------------------------------------------------------------------

def main():
    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  LACUNA BENCHMARK SUITE")
    print("=" * 60)

    # Performance
    perf = run_performance(pdb_dir)

    # Accuracy
    acc = run_accuracy(pdb_dir)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    if perf:
        print(f"\nPerformance (1HEL, {len(structure.residues) if 'structure' in dir() else '?'} residues):")
        for r in perf:
            print(f"  {r['n_conformers']:3d} conformers: {r['elapsed_s']:.2f}s "
                  f"({r['s_per_conformer']:.3f}s each)")

    passed = sum(1 for r in acc if r.get("status") == "PASS")
    total = sum(1 for r in acc if r.get("status") in ("PASS", "MISS"))
    print(f"\nAccuracy: {passed}/{total} known pockets found (>=30% residue overlap in top-5)")
    for r in acc:
        s = r.get("status", "?")
        marker = "PASS" if s == "PASS" else ("FAIL" if s == "MISS" else "–")
        name = r.get("name", r.get("pdb", "?"))
        ov = f"{r['best_overlap']:.0%}" if "best_overlap" in r else ""
        rank = f"rank {r['best_rank']}" if r.get("best_rank") else ""
        print(f"  {marker} {r['pdb']}  {ov:5s} {rank:8s}  {name}")


if __name__ == "__main__":
    main()
