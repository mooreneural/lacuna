# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Zero-dependency ensemble backend: random backbone dihedral perturbation.

Uses a simple coarse approach:
  1. Parse Cα coordinates from the structure.
  2. Apply small random rigid-body perturbations to each chain segment,
     scaled by a noise level parameter.
  3. Propagate atom displacements using a Gaussian decay from each Cα.

This is intentionally lightweight - no force field, no minimization.
It's good enough to open/close shallow pockets and test the pipeline.
For real cryptic pocket work, use BoltzBackend or OpenMMBackend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lacuna.ensemble.base import EnsembleBackend
from lacuna.io.structure import load_structure, coords_array


class RandomBackend(EnsembleBackend):
    """Backbone perturbation ensemble - no external dependencies required."""

    def __init__(self, noise_levels: list[float] | None = None, seed: int | None = None):
        # noise_levels: per-conformer RMS displacement in Å
        # If None, linearly spaced from 0.3 to 2.0 Å
        self.noise_levels = noise_levels
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "random"

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        chain: str | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        structure = load_structure(structure_path, chain=chain)
        base_coords = coords_array(structure)  # (N, 3)

        if self.noise_levels is not None:
            levels = np.array(self.noise_levels)
            # Tile levels to cover n_conformers
            levels = np.tile(levels, (n_conformers // len(levels)) + 1)[:n_conformers]
        else:
            levels = np.linspace(0.3, 2.0, n_conformers)

        conformers: list[np.ndarray] = []
        for sigma in levels:
            coords = self._perturb(base_coords, structure, sigma)
            conformers.append(coords)

        return conformers

    def _perturb(
        self,
        base_coords: np.ndarray,
        structure,
        sigma: float,
    ) -> np.ndarray:
        """Apply correlated Gaussian perturbation decaying with distance from Cα."""
        coords = base_coords.copy()
        n_atoms = len(coords)

        # Find Cα indices for anchoring perturbations
        ca_indices = [
            a.serial for a in structure.atoms if a.name == "CA"
        ]

        if not ca_indices:
            # Fallback: fully uncorrelated noise (e.g., for coarse structure)
            coords += self.rng.normal(0, sigma, size=coords.shape).astype(np.float32)
            return coords

        # Sample independent displacements at each Cα
        ca_coords = base_coords[ca_indices]  # (M, 3)
        ca_displacements = self.rng.normal(0, sigma, size=ca_coords.shape)  # (M, 3)

        # Propagate to each atom weighted by inverse-distance to nearest Cα
        # Distance matrix: (N_atoms, N_ca)
        diff = coords[:, None, :] - ca_coords[None, :, :]  # (N, M, 3)
        dists = np.linalg.norm(diff, axis=-1)  # (N, M)

        # Gaussian weighting: correlation length ~5 Å
        correlation_length = 5.0
        weights = np.exp(-(dists**2) / (2 * correlation_length**2))  # (N, M)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8

        # Interpolate displacements
        atom_displacements = weights @ ca_displacements  # (N, 3)
        coords = coords + atom_displacements.astype(np.float32)
        return coords
