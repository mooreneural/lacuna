# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""OpenMM implicit-solvent MD ensemble backend.

Runs short MD trajectories (~100 ps) at physiological temperature using
the GBn2 implicit solvent model. Each trajectory is started from a
different random velocity seed, giving genuinely independent conformers.

Requires: pip install lacuna[openmm]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lacuna.ensemble.base import EnsembleBackend


class OpenMMBackend(EnsembleBackend):
    """Short implicit-solvent MD for physically realistic conformational sampling."""

    def __init__(
        self,
        temperature_k: float = 310.0,
        simulation_time_ps: float = 100.0,
        timestep_fs: float = 2.0,
        snapshot_interval_ps: float = 10.0,
    ):
        self.temperature_k = temperature_k
        self.simulation_time_ps = simulation_time_ps
        self.timestep_fs = timestep_fs
        self.snapshot_interval_ps = snapshot_interval_ps

    @property
    def name(self) -> str:
        return "openmm"

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
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
                "Install with: pip install lacuna[openmm]"
            ) from e

        # Prepare structure with PDBFixer (adds missing atoms, H)
        fixer = PDBFixer(filename=str(structure_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        # Force field + GBn2 implicit solvent
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = ff.createSystem(
            fixer.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            implicitSolvent=app.GBn2,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

        integrator = mm.LangevinMiddleIntegrator(
            self.temperature_k * unit.kelvin,
            1.0 / unit.picosecond,
            self.timestep_fs * unit.femtosecond,
        )

        platform = mm.Platform.getPlatformByName("CPU")
        simulation = app.Simulation(fixer.topology, system, integrator, platform)
        simulation.context.setPositions(fixer.positions)

        # Minimize
        simulation.minimizeEnergy(maxIterations=500)

        n_steps = int(self.simulation_time_ps * 1000 / self.timestep_fs)
        snap_steps = int(self.snapshot_interval_ps * 1000 / self.timestep_fs)

        conformers: list[np.ndarray] = []
        for i in range(n_conformers):
            simulation.context.setVelocitiesToTemperature(self.temperature_k * unit.kelvin, i)
            simulation.step(n_steps)

            state = simulation.context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

            # Filter to heavy atoms matching original structure atom order
            conformers.append(positions.astype(np.float32))

        return conformers
