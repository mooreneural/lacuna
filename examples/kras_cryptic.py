"""Example: discovering the cryptic switch-II pocket in K-Ras.

K-Ras (KRAS) was considered "undruggable" for decades because its GTP-binding
active site was too shallow and polar to drug. In 2019, Hallin et al. discovered
that a cryptic hydrophobic pocket opens transiently in the switch-II region
(residues ~25-40) near the G12C mutation site.

This pocket is the basis of sotorasib (Lumakras) and adagrasib (Krazati).

Usage:
    python examples/kras_cryptic.py /path/to/kras_apo.pdb

You can download the K-Ras apo structure from PDB: 4OBE
    wget https://files.rcsb.org/download/4OBE.pdb
"""

import sys
from pathlib import Path

from lacuna import load_structure, detect_pockets, cluster_pockets
from lacuna.ensemble.nma_backend import NMABackend
from lacuna.io.writers import write_report, write_boltz_constraint, write_vina_box


def main(pdb_path: str):
    path = Path(pdb_path)
    output_dir = Path("kras_lacuna_output")
    output_dir.mkdir(exist_ok=True)

    print(f"Loading structure: {path}")
    structure = load_structure(path)
    print(f"  {len(structure.residues)} residues, chains: {list(structure.sequence.keys())}")

    # Generate conformational ensemble (NMA is the zero-dependency default backend)
    print("\nGenerating ensemble (30 conformers, NMA backend)...")
    backend = NMABackend(seed=42)
    coord_sets = backend.generate(path, n_conformers=30)

    # Detect pockets in each conformer
    from lacuna.io.structure import coords_array
    import numpy as np

    all_coord_sets = [coords_array(structure)] + coord_sets
    print(f"Detecting pockets across {len(all_coord_sets)} conformers...")

    pocket_lists = []
    for ci, coords in enumerate(all_coord_sets):
        pockets = detect_pockets(coords, structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)
        if (ci + 1) % 10 == 0:
            print(f"  {ci + 1}/{len(all_coord_sets)} done")

    # Cluster and rank
    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coord_sets))
    print(f"\nFound {len(clusters)} pocket clusters.\n")

    # Print top 5
    print(f"{'Rank':<6}{'Druggability':<14}{'Crypticity':<12}{'Persistence':<13}{'Volume(Å³)':<12}Key Residues")
    print("-" * 75)
    for c in clusters[:5]:
        res_str = ", ".join(c.lining_residues[:4])
        print(f"{c.rank:<6}{max(c.druggability, c.max_druggability):<14.3f}"
              f"{c.crypticity:<12.2f}{c.persistence:<13.0%}{c.volume_a3:<12.0f}{res_str}")

    # Check if switch-II region appears in top 3
    switch_ii = {"25", "26", "27", "28", "29", "30", "31", "32",
                 "33", "34", "35", "36", "37", "38", "39", "40"}
    print("\nChecking for switch-II pocket (residues 25–40):")
    for c in clusters[:3]:
        res_numbers = set()
        for label in c.lining_residues:
            try:
                res_part = label.split(":")[0]
                num = "".join(ch for ch in res_part if ch.isdigit())
                res_numbers.add(num)
            except Exception:
                pass
        overlap = res_numbers & switch_ii
        if overlap:
            print(f"  Rank {c.rank} pocket contains switch-II residues: {sorted(overlap)}")
            print(f"  -> This is likely the cryptic pocket! (druggability={c.druggability:.3f})")
            break
    else:
        print("  Switch-II pocket not in top 3 (try more conformers or --backend boltz)")

    # Write outputs
    write_report(clusters, structure, len(all_coord_sets), output_dir)
    for i, c in enumerate(clusters[:3]):
        write_boltz_constraint(c, structure, output_dir, i)
        write_vina_box(c, output_dir, i)

    print(f"\nResults written to {output_dir}/")
    print("  pocket_report.json  — full ranked report")
    print("  pocket_0_constraint.yaml  — Boltz docking input for top pocket")
    print("  pocket_0_vina.conf  — AutoDock Vina box for top pocket")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kras_cryptic.py <path/to/kras.pdb>")
        sys.exit(1)
    main(sys.argv[1])
