"""Elastic network model ensemble backend using anisotropic normal mode analysis.

Generates physically meaningful conformational diversity by displacing along
the lowest-frequency normal modes of the protein's elastic network. These modes
correspond to collective motions — hinge bending, domain twisting, breathing —
that are the primary mechanisms by which cryptic pockets open and close.

Zero additional dependencies beyond the core lacuna install (numpy + scipy).

Reference: Atilgan et al. (2001) Anisotropy of fluctuation dynamics of proteins
with an elastic network model. Biophys. J. 80(1):505-515.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from lacuna.ensemble.base import EnsembleBackend
from lacuna.io.structure import load_structure, coords_array


class NMABackend(EnsembleBackend):
    """Anisotropic Network Model (ANM) conformational sampling.

    Samples conformers by displacing Cα atoms along the lowest-frequency
    normal modes of an elastic network, then propagates to all atoms using
    Gaussian-weighted interpolation. Physically grounded alternative to
    RandomBackend — generates hinge/breathing motions rather than random noise.

    Parameters
    ----------
    cutoff : float
        ENM contact cutoff in Å. Pairs within this distance are connected by springs.
    n_modes : int
        Number of low-frequency modes to sample (beyond the 6 rigid-body modes).
    max_rmsd : float
        Maximum Cα RMSD in Å for the most-displaced conformer.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cutoff: float = 8.0,
        n_modes: int = 10,
        max_rmsd: float = 2.0,
        seed: int | None = None,
    ):
        self.cutoff = cutoff
        self.n_modes = n_modes
        self.max_rmsd = max_rmsd
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "nma"

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        chain: str | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        structure = load_structure(structure_path, chain=chain)
        base_coords = coords_array(structure)

        ca_indices = [a.serial for a in structure.atoms if a.name == "CA"]
        if len(ca_indices) < 6:
            from lacuna.ensemble.random_backend import RandomBackend
            seed = int(self.rng.integers(2**31))
            return RandomBackend(seed=seed).generate(structure_path, n_conformers, chain=chain)

        ca_coords = base_coords[ca_indices]  # (M, 3)
        modes, freqs = self._compute_modes(ca_coords)

        conformers = []
        for i in range(n_conformers):
            amplitude = (i + 1) / n_conformers * self.max_rmsd
            ca_displaced = self._sample_ca(ca_coords, modes, freqs, amplitude, seed=i)
            all_coords = self._propagate(base_coords, ca_coords, ca_displaced)
            conformers.append(all_coords)

        return conformers

    def _compute_modes(
        self, ca_coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ANM Hessian and return low-frequency eigenmodes.

        Requests 4x the needed modes so degenerate zero modes (rigid body +
        z-displacement modes in flat structures) can be filtered before
        returning. Only modes with eigenvalue above a relative threshold are
        treated as non-trivial.
        """
        n = len(ca_coords)
        # Extra headroom handles degenerate test structures (planar, linear)
        # where more than 6 eigenvalues are near-zero.
        n_request = min(6 + self.n_modes * 4, 3 * n - 1)

        diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # (N, N, 3)
        r = np.linalg.norm(diff, axis=-1)  # (N, N)

        contact = (r > 1e-6) & (r <= self.cutoff)
        r_safe = np.where(r > 1e-6, r, 1.0)

        # ANM off-diagonal super-element: M[i,j] = -1/r² * (r_vec ⊗ r_vec)
        M = np.where(
            contact[:, :, None, None],
            -1.0 / r_safe[:, :, None, None] ** 2
            * (diff[:, :, :, None] * diff[:, :, None, :]),
            0.0,
        )  # (N, N, 3, 3)

        # Assemble 3N×3N Hessian
        # Off-diagonal: H[3i+a, 3j+b] = M[i,j,a,b] via axes (i,a,j,b) → (3N,3N)
        H = M.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        # Diagonal: H[3i:3i+3, 3i:3i+3] = -sum_j M[i,j]
        diag_blocks = -M.sum(axis=1)  # (N, 3, 3)
        for i in range(n):
            H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += diag_blocks[i]

        # Partial eigendecomposition — only the lowest n_request modes
        eigenvalues, eigenvectors = eigh(H, subset_by_index=[0, n_request - 1])

        # Filter: keep only genuinely non-trivial modes.
        # Threshold at 0.1% of the max eigenvalue removes rigid-body modes and
        # any extra zero modes from degenerate structures (e.g. flat z=0 fixtures).
        max_eval = float(np.abs(eigenvalues).max()) or 1.0
        valid = np.where(eigenvalues > max_eval * 1e-3)[0]

        if len(valid) == 0:
            # Completely degenerate: use highest-magnitude modes as fallback
            valid = np.argsort(np.abs(eigenvalues))[-self.n_modes:]

        n_use = min(self.n_modes, len(valid))
        idx = valid[:n_use]
        modes = eigenvectors[:, idx].T               # (n_use, 3N)
        freqs = np.maximum(eigenvalues[idx], 1e-6)   # ensure strictly positive

        return modes, freqs

    def _sample_ca(
        self,
        ca_coords: np.ndarray,
        modes: np.ndarray,
        freqs: np.ndarray,
        amplitude: float,
        seed: int,
    ) -> np.ndarray:
        """Displace Cα along a Boltzmann-weighted combination of normal modes."""
        n = len(ca_coords)
        rng = np.random.default_rng(seed)

        # Thermal amplitude: lower-frequency modes have larger fluctuations (∝ 1/√freq)
        weights = 1.0 / np.sqrt(freqs + 1e-10)
        coeffs = rng.normal(0.0, weights)  # (n_modes,)

        # Linear combination → (3N,) → reshape to (N, 3)
        displacement = (coeffs[:, None] * modes).sum(axis=0).reshape(n, 3)

        rmsd = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))
        if rmsd > 1e-8:
            displacement *= amplitude / rmsd

        return (ca_coords + displacement).astype(np.float32)

    def _propagate(
        self,
        base_coords: np.ndarray,
        ca_coords: np.ndarray,
        ca_displaced: np.ndarray,
    ) -> np.ndarray:
        """Propagate Cα displacements to all atoms via Gaussian-weighted interpolation."""
        ca_disp = ca_displaced - ca_coords  # (M, 3)

        diff = base_coords[:, None, :] - ca_coords[None, :, :]  # (N, M, 3)
        dists = np.linalg.norm(diff, axis=-1)  # (N, M)

        weights = np.exp(-(dists**2) / (2 * 5.0**2))  # 5 Å correlation length
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8

        atom_disp = weights @ ca_disp  # (N, 3)
        return (base_coords + atom_disp).astype(np.float32)
