# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""OpenMM implicit-solvent MD ensemble backend.

Runs short MD trajectories at physiological (or elevated) temperature using the
GBn2 implicit-solvent model. Each snapshot is an independent conformer. Elevated
temperature is a light "enhanced sampling" knob for opening transient cavities
that plain 310 K sampling leaves shut.

The backend feeds detection the same atom set (and order) as the rest of the
pipeline: it starts from ``load_structure`` (which drops HETATM/water and can
select a single chain), so the force field never sees ligands/ions, and it maps
the MD positions back onto the original heavy-atom order by (chain, resSeq,
atom-name). Atoms with no MD match (rare — e.g. an alternate-name terminal atom)
keep their input coordinate.

Requires: pip install "lacuna-pockets[openmm]"  (openmm + pdbfixer)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from lacuna.ensemble.base import EnsembleBackend
from lacuna.io.structure import load_structure, coords_array
from lacuna.io.writers import write_structure_pdb


class OpenMMBackend(EnsembleBackend):
    """Short implicit-solvent MD for physically realistic conformational sampling.

    Parameters
    ----------
    temperature_k : float
        Simulation temperature in kelvin. Raise above ~310 K for a simple
        enhanced-sampling boost that helps prise open transient cavities.
    simulation_time_ps : float
        Total MD time per conformer (production, after equilibration).
    timestep_fs : float
        Integration timestep in femtoseconds.
    equilibrate_ps : float
        Short equilibration run once, before collecting conformers.
    """

    def __init__(
        self,
        temperature_k: float = 310.0,
        simulation_time_ps: float = 50.0,
        timestep_fs: float = 2.0,
        equilibrate_ps: float = 10.0,
    ):
        self.temperature_k = temperature_k
        self.simulation_time_ps = simulation_time_ps
        self.timestep_fs = timestep_fs
        self.equilibrate_ps = equilibrate_ps

    @property
    def name(self) -> str:
        return "openmm"

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        chain: str | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        try:
            import openmm as mm
            import openmm.app as app
            import openmm.unit as unit
            from pdbfixer import PDBFixer
        except ImportError as e:
            raise ImportError(
                "OpenMM backend requires openmm and pdbfixer. "
                'Install with: pip install "lacuna-pockets[openmm]"'
            ) from e

        # Source of truth = the same Structure detection uses (HETATM/water already
        # dropped, chain filtered). Write it to a clean temp PDB for PDBFixer so the
        # force field never meets a ligand/ion it has no template for.
        structure = load_structure(structure_path, chain=chain)
        base_coords = coords_array(structure)

        # ignore_cleanup_errors: on Windows PDBFixer can keep a handle on the temp
        # PDB, so auto-deletion may raise even though the MD completed fine.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            clean_pdb = Path(td) / "clean.pdb"
            write_structure_pdb(structure, clean_pdb)

            fixer = PDBFixer(filename=str(clean_pdb))
            fixer.findMissingResidues()
            # Do not model in whole missing loops — only complete partial residues.
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)

            # The GBn2 implicit solvent is supplied by the implicit/gbn2.xml force
            # field itself in OpenMM 8.x — createSystem must NOT also receive the
            # legacy implicitSolvent= kwargs (they are rejected as unused).
            ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
            system = ff.createSystem(
                fixer.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
            )

            integrator = mm.LangevinMiddleIntegrator(
                self.temperature_k * unit.kelvin,
                1.0 / unit.picosecond,
                self.timestep_fs * unit.femtosecond,
            )
            platform = _select_platform(mm)
            simulation = app.Simulation(fixer.topology, system, integrator, platform)
            simulation.context.setPositions(fixer.positions)

            simulation.minimizeEnergy(maxIterations=500)
            simulation.context.setVelocitiesToTemperature(self.temperature_k * unit.kelvin, 0)
            if self.equilibrate_ps > 0:
                simulation.step(int(self.equilibrate_ps * 1000 / self.timestep_fs))

            # Map MD topology atoms -> index into the original structure atom order.
            remap = _build_atom_remap(fixer.topology, structure)

            snap_steps = max(1, int(self.simulation_time_ps * 1000 / self.timestep_fs / n_conformers))
            conformers: list[np.ndarray] = []
            for _ in range(n_conformers):
                simulation.step(snap_steps)
                state = simulation.context.getState(getPositions=True)
                md_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
                conformers.append(_reorder(md_pos, remap, base_coords))

        return conformers


def _select_platform(mm):
    """Prefer CUDA > OpenCL > CPU, whichever is available."""
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    return mm.Platform.getPlatformByName("CPU")


def _build_atom_remap(topology, structure) -> list[int]:
    """For each original structure atom, the MD-position index (or -1 if unmatched).

    Keyed on (chain_id, res_seq, atom_name) — stable across PDBFixer, which
    preserves residue ids and standard atom names.
    """
    md_index: dict[tuple[str, int, str], int] = {}
    for i, atom in enumerate(topology.atoms()):
        try:
            key = (atom.residue.chain.id, int(atom.residue.id), atom.name)
        except (ValueError, TypeError):
            continue
        md_index.setdefault(key, i)

    remap: list[int] = []
    for a in structure.atoms:
        remap.append(md_index.get((a.chain_id, a.res_seq, a.name), -1))
    return remap


def _reorder(md_pos: np.ndarray, remap: list[int], base_coords: np.ndarray) -> np.ndarray:
    """Assemble coords in the original atom order; fall back to input for unmatched."""
    out = base_coords.copy()
    for orig_i, md_i in enumerate(remap):
        if md_i >= 0:
            out[orig_i] = md_pos[md_i]
    return out.astype(np.float32)
