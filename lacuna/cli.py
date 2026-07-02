# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Lacuna CLI — cryptic binding pocket discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from lacuna import __version__

console = Console()


def _resolve_backend(name: str):
    if name == "random":
        from lacuna.ensemble.random_backend import RandomBackend
        return RandomBackend()
    elif name == "nma":
        from lacuna.ensemble.nma_backend import NMABackend
        return NMABackend()
    elif name == "openmm":
        from lacuna.ensemble.openmm_backend import OpenMMBackend
        return OpenMMBackend()
    elif name == "boltz":
        from lacuna.ensemble.boltz_backend import BoltzBackend
        return BoltzBackend()
    else:
        raise click.BadParameter(f"Unknown backend '{name}'. Choose: random, nma, openmm, boltz")


def _auto_backend():
    """Pick the best available backend at runtime."""
    for name in ("boltz", "openmm", "nma", "random"):
        try:
            return _resolve_backend(name)
        except ImportError:
            continue
    raise RuntimeError("No ensemble backend available.")


@click.group()
@click.version_option(version=__version__, prog_name="lacuna")
def main():
    """Lacuna — cryptic binding pocket discovery via conformational ensemble analysis.

    \b
    Typical workflow:
      lacuna discover protein.pdb            # run with defaults
      lacuna discover protein.pdb --backend boltz --conformers 30
      lacuna discover protein.pdb --emit-boltz-constraints --emit-vina-boxes
    """


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--backend", "-b",
    type=click.Choice(["auto", "random", "nma", "openmm", "boltz"]),
    default="auto",
    show_default=True,
    help="Ensemble generation backend.",
)
@click.option("--conformers", "-n", default=20, show_default=True,
              help="Number of conformers to generate.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output directory (default: <input_stem>_lacuna/).")
@click.option("--min-druggability", default=0.0, show_default=True,
              help="Filter: minimum druggability score [0–1].")
@click.option("--min-persistence", default=0.0, show_default=True,
              help="Filter: minimum persistence (fraction of conformers).")
@click.option("--emit-boltz-constraints", is_flag=True, default=False,
              help="Write Boltz YAML constraint files for each pocket.")
@click.option("--emit-vina-boxes", is_flag=True, default=False,
              help="Write AutoDock Vina box config files for each pocket.")
@click.option("--emit-pocket-pdbs", is_flag=True, default=False,
              help="Write pocket pseudoatom PDB files for visualization.")
@click.option("--top", default=10, show_default=True,
              help="Maximum number of pockets to report.")
@click.option(
    "--rank-by", "rank_by",
    type=click.Choice(["crypticity", "druggability", "persistence", "balanced"]),
    default="crypticity", show_default=True,
    help=(
        "Pocket ranking strategy. 'crypticity' (default) surfaces transiently-open "
        "cryptic sites first; 'druggability' ranks by peak open-state druggability "
        "(better for always-open/orthosteric sites); 'balanced' adds a mild "
        "persistence bonus; 'persistence' is the legacy persistence x druggability rule."
    ),
)
@click.option("--min-crypticity", default=0.0, show_default=True,
              help="Filter: minimum crypticity score [0-1].")
@click.option("--quiet", is_flag=True, default=False, help="Suppress progress output.")
@click.option(
    "--homodimer", is_flag=True, default=False,
    help=(
        "Analyze the biological assembly rather than the asymmetric unit. "
        "Reads BIOMT records (PDB) or _pdbx_struct_oper_list (mmCIF) to create "
        "the full assembly; required to detect pockets at dimer interfaces. "
        "For best results, use the biological assembly download from RCSB."
    ),
)
def discover(
    input_path: Path,
    backend: str,
    conformers: int,
    output: Path | None,
    min_druggability: float,
    min_persistence: float,
    emit_boltz_constraints: bool,
    emit_vina_boxes: bool,
    emit_pocket_pdbs: bool,
    top: int,
    rank_by: str,
    min_crypticity: float,
    quiet: bool,
    homodimer: bool,
):
    """Discover cryptic binding pockets in a protein structure.

    INPUT_PATH: Path to a PDB or mmCIF file (from AlphaFold, Boltz, Chai, or PDB).
    """
    import tempfile

    from lacuna.io.structure import load_structure, coords_array, make_biological_assembly
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets
    from lacuna.io.writers import (
        write_report, write_pocket_pdb, write_boltz_constraint, write_vina_box,
        write_structure_pdb,
    )

    output_dir = output or Path(f"{input_path.stem}_lacuna")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        console.print(f"\n[bold cyan]Lacuna[/bold cyan] — cryptic pocket discovery")
        console.print(f"  Input:    [green]{input_path}[/green]")
        console.print(f"  Backend:  {backend}")
        console.print(f"  Output:   {output_dir}\n")

    # Load structure
    if not quiet:
        console.print("[dim]Loading structure...[/dim]")
    structure = load_structure(input_path)

    # Optionally expand to biological assembly for dimer-interface pocket detection
    effective_path = input_path
    with tempfile.TemporaryDirectory() as _tmpdir:
        if homodimer:
            assembly = make_biological_assembly(input_path, structure)
            if len(assembly.atoms) > len(structure.atoms):
                tmp_pdb = Path(_tmpdir) / f"{input_path.stem}_assembly.pdb"
                write_structure_pdb(assembly, tmp_pdb)
                structure = assembly
                effective_path = tmp_pdb
                if not quiet:
                    console.print(
                        f"  [dim]Homodimer: biological assembly built "
                        f"({len(structure.sequence)} chains, {len(structure.residues)} residues)[/dim]"
                    )
            elif not quiet:
                console.print(
                    "  [yellow]--homodimer: no BIOMT symmetry records found. "
                    "For dimer-interface pockets, download the biological assembly "
                    "PDB from RCSB (use the 'Download Files → Biological Assembly' option).[/yellow]"
                )

        if not quiet:
            n_res = len(structure.residues)
            n_chains = len(structure.sequence)
            console.print(f"  [dim]{n_res} residues, {n_chains} chain(s)[/dim]")

        # Resolve backend
        if backend == "auto":
            be = _auto_backend()
            backend = be.name
        else:
            be = _resolve_backend(backend)

        if not quiet:
            console.print(f"\n[dim]Generating {conformers} conformers with '{backend}' backend...[/dim]")

        coord_sets = be.generate(effective_path, conformers)

    if not quiet:
        console.print(f"  [dim]Generated {len(coord_sets)} conformers.[/dim]")
        console.print("\n[dim]Detecting pockets across ensemble...[/dim]")

    # Detect pockets in each conformer
    pocket_lists = []
    base_coords = coords_array(structure)

    # Always include pockets from the input structure itself (conformer 0)
    all_coord_sets = [base_coords] + list(coord_sets)

    for ci, coords in enumerate(all_coord_sets):
        pockets = detect_pockets(coords, structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)
        if not quiet and (ci + 1) % 5 == 0:
            console.print(f"  [dim]{ci + 1}/{len(all_coord_sets)} conformers processed[/dim]")

    total_pockets = sum(len(pl) for pl in pocket_lists)
    if not quiet:
        console.print(f"  [dim]Found {total_pockets} raw pockets across ensemble.[/dim]")
        console.print("\n[dim]Clustering and ranking pockets...[/dim]")

    # Cluster across ensemble
    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coord_sets), rank_by=rank_by)

    # Apply filters
    clusters = [
        c for c in clusters
        if c.druggability >= min_druggability
        and c.persistence >= min_persistence
        and c.crypticity >= min_crypticity
    ][:top]

    if not quiet:
        console.print(f"  [dim]{len(clusters)} pocket clusters after filtering.[/dim]\n")

    # Write outputs
    report_path = write_report(clusters, structure, len(all_coord_sets), output_dir, rank_by=rank_by)

    written: list[str] = [f"[green]{report_path.name}[/green]"]

    for i, cluster in enumerate(clusters):
        if emit_pocket_pdbs:
            p = write_pocket_pdb(cluster, output_dir, i)
            written.append(p.name)
        if emit_boltz_constraints:
            p = write_boltz_constraint(cluster, structure, output_dir, i)
            written.append(p.name)
        if emit_vina_boxes:
            p = write_vina_box(cluster, output_dir, i)
            written.append(p.name)

    # Print results table
    if not quiet:
        _print_table(clusters)
        console.print(f"\n[bold]Output written to:[/bold] {output_dir}/")
        for name in written[:6]:
            console.print(f"  {name}")
        if len(written) > 6:
            console.print(f"  [dim]... and {len(written) - 6} more[/dim]")
        console.print()

    return clusters


@main.command("dock-prep")
@click.argument("report_path", type=click.Path(exists=True, path_type=Path))
@click.argument("protein_path", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt",
              type=click.Choice(["boltz", "vina", "pdb", "all"]),
              default="all", show_default=True,
              help="Docking tool output format.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--top", default=5, show_default=True,
              help="Prepare files for top N pockets.")
def dock_prep(
    report_path: Path,
    protein_path: Path,
    fmt: str,
    output: Path | None,
    top: int,
):
    """Generate docking input files from a pocket report.

    REPORT_PATH: pocket_report.json from a previous 'lacuna discover' run.
    PROTEIN_PATH: The protein structure file (PDB or mmCIF).
    """
    from lacuna.io.structure import load_structure
    from lacuna.io.writers import write_boltz_constraint, write_vina_box, write_pocket_pdb
    from lacuna.models import PocketCluster

    report = json.loads(report_path.read_text())
    structure = load_structure(protein_path)

    output_dir = output or report_path.parent / "docking_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pockets_data = report["pockets"][:top]

    for i, pd in enumerate(pockets_data):
        cluster = PocketCluster(
            rank=pd["rank"],
            centroid=tuple(pd["centroid"]),
            volume_a3=pd["volume_A3"],
            druggability=pd["druggability"],
            persistence=pd["persistence"],
            cryptic=pd["cryptic"],
            lining_residues=pd["lining_residues"],
            appears_in_conformers=pd["appears_in_conformers"],
        )

        if fmt in ("boltz", "all"):
            write_boltz_constraint(cluster, structure, output_dir, i)
        if fmt in ("vina", "all"):
            write_vina_box(cluster, output_dir, i)
        if fmt in ("pdb", "all"):
            write_pocket_pdb(cluster, output_dir, i)

    console.print(f"Docking inputs written to [green]{output_dir}[/green]")


def _print_table(clusters: list) -> None:
    if not clusters:
        console.print("[yellow]No pockets found.[/yellow]")
        return

    table = Table(title="Discovered Pockets", show_lines=True)
    table.add_column("Rank", style="bold", justify="right")
    table.add_column("Druggability", justify="right")
    table.add_column("Crypticity", justify="right")
    table.add_column("Persistence", justify="right")
    table.add_column("Volume (Å³)", justify="right")
    table.add_column("Key Residues")

    for c in clusters:
        drug = max(c.druggability, c.max_druggability)
        drug_color = "green" if drug > 0.6 else ("yellow" if drug > 0.3 else "red")
        cryp_color = "bold yellow" if c.crypticity > 0.4 else ("yellow" if c.crypticity > 0.2 else "dim")
        # Show volume as apo -> open when the pocket breathes meaningfully.
        if c.volume_max_a3 - c.volume_min_a3 > 50:
            vol_str = f"{c.apo_volume_a3:.0f} -> {c.volume_max_a3:.0f}"
        else:
            vol_str = f"{c.volume_a3:.0f}"
        key_res = ", ".join(c.lining_residues[:5])
        if len(c.lining_residues) > 5:
            key_res += f" (+{len(c.lining_residues) - 5})"

        table.add_row(
            str(c.rank),
            f"[{drug_color}]{drug:.3f}[/{drug_color}]",
            f"[{cryp_color}]{c.crypticity:.2f}[/{cryp_color}]",
            f"{c.persistence:.0%}",
            vol_str,
            key_res,
        )

    console.print(table)


if __name__ == "__main__":
    main()
