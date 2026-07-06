# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Output writers for Lacuna discovery results."""

from __future__ import annotations

import json
from pathlib import Path

from lacuna.models import PocketCluster, Structure


def write_report(
    clusters: list[PocketCluster],
    structure: Structure,
    n_conformers: int,
    output_dir: Path,
    rank_by: str = "druggability",
) -> Path:
    """Write pocket_report.json with ranked cluster metadata."""
    report = {
        "protein": structure.path,
        "n_conformers": n_conformers,
        "ranked_by": rank_by,
        "n_pockets_found": len(clusters),
        "n_cryptic_pockets": sum(1 for c in clusters if c.crypticity >= 0.3),
        "pockets": [c.to_dict() for c in clusters],
    }
    out = output_dir / "pocket_report.json"
    out.write_text(json.dumps(report, indent=2))
    return out


def write_pocket_pdb(
    cluster: PocketCluster,
    output_dir: Path,
    index: int,
) -> Path:
    """Write a PDB file with pseudoatoms at the pocket centroid and extent.

    The pocket is represented as HETATM records so it can be loaded in
    PyMOL/ChimeraX alongside the protein for visualization.
    """
    cx, cy, cz = cluster.centroid
    radius = (cluster.volume_a3 * 3 / (4 * 3.14159)) ** (1 / 3)

    lines = [
        "REMARK  Lacuna pocket pseudoatoms",
        f"REMARK  Rank {cluster.rank}, druggability={cluster.druggability:.3f}, "
        f"persistence={cluster.persistence:.3f}, cryptic={cluster.cryptic}, "
        f"crypticity={cluster.crypticity:.3f}",
        f"REMARK  Volume {cluster.apo_volume_a3:.0f}->{cluster.volume_max_a3:.0f} A^3 "
        f"(apo->open)",
        f"REMARK  Lining residues: {', '.join(cluster.lining_residues[:10])}",
    ]

    # Central pseudoatom
    lines.append(
        f"HETATM    1  C   PKT A   1    "
        f"{cx:8.3f}{cy:8.3f}{cz:8.3f}  1.00{cluster.druggability:6.2f}           C"
    )

    # 6 pseudoatoms at ±radius on each axis (mark the pocket extent)
    serial = 2
    for axis in range(3):
        for sign in (-1, 1):
            pos = [cx, cy, cz]
            pos[axis] += sign * radius
            lines.append(
                f"HETATM{serial:5d}  C   PKT A{serial:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00{cluster.druggability:6.2f}           C"
            )
            serial += 1

    lines.append("END")
    out = output_dir / f"pocket_{index}_site.pdb"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_boltz_constraint(
    cluster: PocketCluster,
    structure: Structure,
    output_dir: Path,
    index: int,
) -> Path:
    """Write a Boltz YAML constraint file that targets this pocket for docking.

    The user can drop a ligand SMILES into this file and run:
        boltz predict pocket_0_constraint.yaml
    to dock specifically into the discovered pocket.
    """
    # Determine residue numbers and chain from consensus lining residues
    # Format is "RES_NAME SEQ_NUM:CHAIN_ID"
    pocket_residues: dict[str, list[int]] = {}
    for label in cluster.lining_residues:
        # label format: "ALA123:A"
        try:
            res_part, chain = label.split(":")
            # Strip the three-letter residue name prefix (3 chars)
            seq_num = int("".join(c for c in res_part if c.isdigit()))
            pocket_residues.setdefault(chain, []).append(seq_num)
        except (ValueError, IndexError):
            continue

    # Build YAML manually (avoid pyyaml dependency)
    lines = [
        "# Lacuna-generated Boltz constraint file",
        f"# Pocket rank {cluster.rank}: druggability={cluster.druggability:.3f}, "
        f"persistence={cluster.persistence:.3f}, crypticity={cluster.crypticity:.3f}",
        "#",
        "# Replace <SMILES_HERE> with your ligand SMILES string",
        "# Replace <CHAIN_ID> with your protein chain identifier",
        "",
        "sequences:",
    ]

    # Add protein chain(s) from structure
    for chain_id, seq in structure.sequence.items():
        lines += [
            f"  - protein:",
            f"      id: {chain_id}",
            f"      sequence: {seq}",
        ]

    # Placeholder ligand entry
    lines += [
        "  - ligand:",
        "      id: L",
        "      smiles: <SMILES_HERE>",
    ]

    lines += ["", "constraints:"]

    # Add pocket constraint for each chain that has lining residues
    for chain_id, residues in pocket_residues.items():
        residue_list = ", ".join(str(r) for r in sorted(residues))
        lines += [
            "  - pocket:",
            f"      binder: L",
            f"      contacts:",
            f"        - ['{chain_id}', [{residue_list}]]",
        ]

    out = output_dir / f"pocket_{index}_constraint.yaml"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_structure_pdb(structure: "Structure", path: Path) -> None:
    """Write a Structure object to PDB format for use as a temporary input file."""
    lines: list[str] = []
    for atom in structure.atoms:
        x, y, z = atom.coords
        name = atom.name
        # PDB atom name convention: 1-char elements get a leading space in col 13
        if len(name) < 4:
            name_str = f" {name:<3}"
        else:
            name_str = f"{name:<4}"
        chain = (atom.chain_id[0] if atom.chain_id else "A")
        elem = atom.element[:2].rjust(2) if atom.element else " C"
        # Column layout (PDB spec): name cols 13-16, altLoc col 17 (blank),
        # resName cols 18-20. The space before res_name is the altLoc placeholder -
        # without it the residue name shifts one column left and strict parsers
        # (e.g. OpenMM) reject the file.
        lines.append(
            f"ATOM  {atom.serial + 1:5d} {name_str} {atom.res_name:>3s} {chain}"
            f"{atom.res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            f"          {elem}  "
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_vina_box(
    cluster: PocketCluster,
    output_dir: Path,
    index: int,
    box_padding: float = 8.0,
) -> Path:
    """Write an AutoDock Vina config file centered on the pocket.

    Works with Vina, Gnina, QuickVina, and any tool using AutoDock-style box definitions.
    box_padding: Å added in each direction beyond the estimated pocket radius.
    """
    cx, cy, cz = cluster.centroid
    radius = (cluster.volume_a3 * 3 / (4 * 3.14159)) ** (1 / 3)
    box_size = (radius + box_padding) * 2

    lines = [
        f"# Lacuna pocket {index} - AutoDock Vina box",
        f"# Rank {cluster.rank}, druggability={cluster.druggability:.3f}, "
        f"crypticity={cluster.crypticity:.3f}",
        f"center_x = {cx:.3f}",
        f"center_y = {cy:.3f}",
        f"center_z = {cz:.3f}",
        f"size_x = {box_size:.1f}",
        f"size_y = {box_size:.1f}",
        f"size_z = {box_size:.1f}",
        "exhaustiveness = 8",
        "num_modes = 9",
    ]
    out = output_dir / f"pocket_{index}_vina.conf"
    out.write_text("\n".join(lines) + "\n")
    return out
