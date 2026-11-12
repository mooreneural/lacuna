# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Elastic network model ensemble backend using anisotropic normal mode analysis.

Generates physically meaningful conformational diversity by displacing along
the lowest-frequency normal modes of the protein's elastic network. These modes
correspond to collective motions - hinge bending, domain twisting, breathing -
that are the primary mechanisms by which cryptic pockets open and close.

Zero additional dependencies beyond the core lacuna install (numpy + scipy).

Reference: Atilgan et al. (2001) Anisotropy of fluctuation dynamics of proteins
with an elastic network model. Biophys. J. 80(1):505-515.

Reproducibility
---------------
Every pocket Lacuna reports is carved out of these conformers, so one structure
has to give one ensemble on every machine. The obstacle is the eigensolver: its
output depends on how many threads BLAS runs, in two independent ways, and both
used to reach the conformers.

Precision. The Hessian was built from float32 coordinates, so the eigensolver
ran in single precision, where its thread-dependent rounding is large enough to
move atoms. Every intermediate is now float64: the pairwise distances, the
Hessian, the eigendecomposition, the mode combination and the propagation of
bead displacements to all atoms. A conformer is cast to float32 once, as the
last thing that happens to it.

Sign. An eigenvector is defined only up to sign, and the sign LAPACK returns
also varies with threading. A conformer is a seeded Gaussian combination of the
modes, so a flipped mode gives a different conformer rather than a mirrored
one. Every mode now leaves ``_compute_modes`` with the sign fixed by
``canonical_sign``.

The eigensolver is the only step that calls BLAS. The Hessian it is handed is
built from elementwise IEEE arithmetic in a fixed order, so it is the same
matrix bit for bit wherever it is built, and the mode combination, the
propagation weights and the propagation itself are elementwise NumPy
arithmetic and ``einsum``, which do not run on a thread pool. What is left is
float64 rounding from two places: the solver itself, and the last bit of
``exp`` in the propagation weights, which NumPy evaluates with whichever
vector kernel the CPU offers (so it can differ between two machines, though
not between two thread counts on one). Neither moves a coordinate by more
than about 1e-11 A, far below float32 resolution, so it shows only where it
tips a coordinate across a float32 rounding boundary. Two runs can then
disagree by one unit in the last place of an occasional coordinate, or by a
few units, still about 1e-11 A, in a coordinate that happens to lie within
about 1e-4 A of zero, where the float32 grid is that fine. Measured on 184
structures between 1, 2, 8 and the default number of threads, and between a
Windows and a Linux build: at most 40 of 26 million coordinates differed, and
none by more than 1.9e-6 A.

What this does not cover
------------------------
All of the above assumes each mode is individually well defined. It is not
when eigenvalues coincide: any rotation of the basis inside a degenerate set
is an equally valid answer, LAPACK returns whichever one its rounding lands
on, and no sign rule can pin a rotation. The seeded coefficients belong to
individual modes, so a rotated basis is a different ensemble. There are two
ways to get there, and neither is detected or handled here.

Exact symmetry. A structure with a three-fold or higher rotation axis (a C3
trimer, a D3 hexamer, anything tetrahedral, octahedral or icosahedral) has
modes that come in degenerate pairs or triplets. Two-fold symmetry is not
affected: its modes are non-degenerate, and the tie rule in ``canonical_sign``
exists for exactly that case.

An under-constrained network. A bead held by fewer than three springs, or by
springs that all lie in one plane, adds a zero mode to the six rigid-body
ones. A handful is harmless, because zero modes are filtered out before
sampling. When they fill the whole window of eigenvalues that
``_compute_modes`` requests (40 or more of them at the default ``n_modes``),
nothing in the window is a vibration and the fallback there samples an
arbitrary basis of the null space, rigid-body motion included. A predicted
model whose disordered regions are drawn as long extended ribbons does this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from lacuna.ensemble.base import EnsembleBackend
from lacuna.io.structure import load_structure, coords_array

#: Two components of a mode count as tied for largest when their magnitudes
#: agree to this relative tolerance. See ``canonical_sign``.
_SIGN_TIE_RTOL = 1e-6

#: Largest scratch array, in float64 elements, used while assembling the
#: Hessian and the propagation weights (2**17 elements is 1 MB, which stays in
#: cache). Both are built a slab of rows at a time so that peak memory is the
#: result itself, not a multiple of it. The slab size changes memory and speed
#: only: every element is produced by the same operations in the same order
#: whatever it is.
_BLOCK_ELEMENTS = 1 << 17


def _slab_rows(n_rows: int, row_elements: int) -> int:
    """How many of an array's ``n_rows`` rows to process at once.

    At most ``_BLOCK_ELEMENTS`` elements, which bounds the scratch memory of a
    large structure, and at most an eighth of the rows, which keeps the scratch
    small next to the result on a small one.
    """
    return max(1, min(_BLOCK_ELEMENTS // max(row_elements, 1), n_rows // 8))


def canonical_sign(modes: np.ndarray) -> np.ndarray:
    """Give every mode the sign that makes its largest component positive.

    An eigenvector is defined only up to sign, and which of the two LAPACK
    returns depends on its rounding, hence on the BLAS thread count. The
    sampler draws one seeded coefficient per mode, so the sign has to be fixed
    by a rule that reads nothing but the mode itself.

    The rule: find the largest-magnitude component and flip the mode if that
    component is negative. "Largest" is taken with a relative tolerance of
    1e-6, and among the components inside the tolerance the lowest index
    decides. The tolerance is what makes the rule safe on a symmetric
    structure. There the components of symmetry-related atoms are equal in
    magnitude in exact arithmetic, so which one is larger in floating point is
    decided by rounding, and when the two have opposite signs a plain argmax
    flips the mode from one run to the next. Taking the lowest index among
    near-equals removes that coin toss: to bring it back, a component would
    have to sit within rounding error of exactly (1 - 1e-6) times the maximum.

    Parameters
    ----------
    modes : np.ndarray
        (n_modes, 3N) array, one mode per row.

    Returns
    -------
    np.ndarray
        A new float64 array of the same shape. Each row is the input row or
        its exact negation.
    """
    modes = np.asarray(modes, dtype=np.float64)
    if modes.size == 0:
        return modes.copy()
    magnitude = np.abs(modes)
    peak = magnitude.max(axis=1, keepdims=True)
    # argmax over booleans returns the first True, which is the lowest index.
    pivot = (magnitude >= peak * (1.0 - _SIGN_TIE_RTOL)).argmax(axis=1)
    negative = modes[np.arange(len(modes)), pivot] < 0.0
    return np.where(negative[:, None], -modes, modes)


class NMABackend(EnsembleBackend):
    """Anisotropic Network Model (ANM) conformational sampling.

    Samples conformers by displacing Cα atoms along the lowest-frequency
    normal modes of an elastic network, then propagates to all atoms using
    Gaussian-weighted interpolation. Physically grounded alternative to
    RandomBackend - generates hinge/breathing motions rather than random noise.

    The ensemble is a function of the input coordinates and the parameters
    below and of nothing else, the BLAS thread count included. See
    "Reproducibility" in the module docstring for what that rests on, and
    "What this does not cover" for the two kinds of input it fails on.

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

    #: Propagation weights of the ``generate`` call in progress, as
    #: ``(base_coords, bead_coords, weights)``, with ``weights`` None until
    #: the first conformer asks for them. None outside a call. See
    #: ``_sample_ensemble``.
    _weights_memo: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None

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

        return self._sample_ensemble(base_coords, base_coords[ca_indices], n_conformers)

    def _sample_ensemble(
        self,
        base_coords: np.ndarray,
        bead_coords: np.ndarray,
        n_conformers: int,
    ) -> list[np.ndarray]:
        """Sample ``n_conformers`` conformers from the network on ``bead_coords``.

        The one loop behind every network backend: the protein path calls it
        with Cα beads and the nucleic-acid path with nucleotide beads, so the
        two cannot drift apart in how they sample or in how reproducible they
        are.

        The modes and the propagation weights depend only on the input
        structure, so each is computed once per call. The weights are an
        (N_atoms, N_beads) array that used to be rebuilt for every conformer.
        They reach ``_propagate`` through ``_weights_memo`` rather than through
        a new argument, so the three hooks (``_compute_modes``, ``_sample_ca``,
        ``_propagate``) keep their signatures and a subclass or a patch that
        replaces one of them still takes effect. The memo is matched on object
        identity and cleared when the call ends, so it cannot serve weights for
        a different structure and it does not keep a large array alive after
        the ensemble is returned.
        """
        base = np.asarray(base_coords, dtype=np.float64)
        beads = np.asarray(bead_coords, dtype=np.float64)

        modes, freqs = self._compute_modes(beads)

        self._weights_memo = (base, beads, None)
        try:
            conformers = []
            for i in range(n_conformers):
                amplitude = (i + 1) / n_conformers * self.max_rmsd
                displaced = self._sample_ca(beads, modes, freqs, amplitude, seed=i)
                conformers.append(self._propagate(base, beads, displaced))
        finally:
            self._weights_memo = None

        return conformers

    def _hessian(
        self, ca_coords: np.ndarray, gamma: np.ndarray | None = None
    ) -> np.ndarray:
        """Assemble the 3N×3N ANM Hessian in float64, whatever the input dtype.

        Built one slab of rows at a time, so the largest scratch array is
        ``_BLOCK_ELEMENTS`` float64 values rather than the (N, N, 3, 3) tensor
        of super-elements, which is as large as the Hessian itself and which
        the unblocked expression holds several copies of at once.

        Every entry comes from the same elementwise operations in the same
        order as the unblocked expression, so the result is bit-identical to
        it and does not depend on the slab size. Nothing here calls BLAS. The
        matrix is symmetric to the last bit whenever ``gamma`` is.
        """
        coords = np.asarray(ca_coords, dtype=np.float64)
        n = len(coords)
        if gamma is not None:
            gamma = np.asarray(gamma, dtype=np.float64)

        hessian = np.empty((3 * n, 3 * n), dtype=np.float64)
        step = _slab_rows(n, 9 * n)
        for start in range(0, n, step):
            stop = min(start + step, n)
            rows = stop - start

            diff = coords[start:stop, None, :] - coords[None, :, :]  # (rows, N, 3)
            r = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)

            contact = (r > 1e-6) & (r <= self.cutoff)
            r_safe = np.where(r > 1e-6, r, 1.0)

            # ANM off-diagonal super-element: M[i,j] = -γ_ij / r² * (r_vec ⊗ r_vec)
            strength = 1.0 if gamma is None else gamma[start:stop, :, None, None]
            block = np.where(
                contact[:, :, None, None],
                -strength / r_safe[:, :, None, None] ** 2
                * (diff[:, :, :, None] * diff[:, :, None, :]),
                0.0,
            )  # (rows, N, 3, 3)

            # Off-diagonal: H[3i+a, 3j+b] = M[i,j,a,b] via axes (i,a,j,b)
            slab = hessian[3 * start : 3 * stop].reshape(rows, 3, n, 3)
            slab[...] = block.transpose(0, 2, 1, 3)

            # Diagonal: H[3i:3i+3, 3i:3i+3] = -sum_j M[i,j]. A running sum
            # rather than .sum(): a prefix sum is sequential by definition, so
            # the rounding of the diagonal cannot depend on how a NumPy build
            # chooses to order a reduction.
            diag_blocks = -np.add.accumulate(block, axis=1)[:, -1]  # (rows, 3, 3)
            for k in range(rows):
                i = 3 * (start + k)
                hessian[i : i + 3, i : i + 3] += diag_blocks[k]

        return hessian

    def _compute_modes(
        self, ca_coords: np.ndarray, gamma: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ANM Hessian and return low-frequency eigenmodes.

        Requests 4x the needed modes so degenerate zero modes (rigid body +
        z-displacement modes in flat structures) can be filtered before
        returning. Only modes with eigenvalue above a relative threshold are
        treated as non-trivial.

        The Hessian and the eigendecomposition are float64 whatever the dtype
        of ``ca_coords``, and both returned arrays are float64. Each mode has
        the sign fixed by ``canonical_sign``, so the same network gives the
        same modes, not the same modes up to a sign per mode.

        Parameters
        ----------
        gamma : np.ndarray | None
            Optional (N, N) per-pair spring-constant multiplier. ``None`` (the
            default) means a uniform network (all springs = 1.0), which
            reproduces the standard ANM exactly. Values < 1 soften specific
            contacts - a hook for spring-perturbation experiments (e.g. releasing
            the local "cage" of contacts that hold a candidate cryptic pocket
            shut) that recompute the modes of a locally softened network.
        """
        n = len(ca_coords)
        # Extra headroom handles degenerate test structures (planar, linear)
        # where more than 6 eigenvalues are near-zero.
        n_request = min(6 + self.n_modes * 4, 3 * n - 1)

        if gamma is not None:
            gamma = np.asarray(gamma, dtype=np.float64)
        hessian = self._hessian(ca_coords, gamma)

        # Partial eigendecomposition - only the lowest n_request modes.
        #
        # eigh reads the lower triangle and LAPACK wants Fortran order, so a
        # C-ordered matrix is copied before it is factored, which doubles the
        # peak. The Hessian is symmetric to the last bit whenever gamma is, so
        # its transpose is the same matrix already in Fortran order, and LAPACK
        # can then work in place. An asymmetric gamma is not a physical spring
        # network, but it keeps the old meaning (lower triangle) at the old
        # cost (one copy) rather than silently reading the other triangle.
        symmetric = gamma is None or np.array_equal(gamma, gamma.T)
        eigenvalues, eigenvectors = eigh(
            hessian.T if symmetric else hessian,
            subset_by_index=[0, n_request - 1],
            overwrite_a=True,
        )
        del hessian

        # Filter: keep only genuinely non-trivial modes.
        # Threshold at 0.1% of the max eigenvalue removes rigid-body modes and
        # any extra zero modes from degenerate structures (e.g. flat z=0 fixtures).
        max_eval = float(np.abs(eigenvalues).max()) or 1.0
        valid = np.where(eigenvalues > max_eval * 1e-3)[0]

        if len(valid) == 0:
            # Completely degenerate: use highest-magnitude modes as fallback.
            # Every eigenvalue in the window is rounding noise around zero
            # here, so these are null-space vectors in whatever basis LAPACK
            # returned, and the ensemble built on them is not reproducible
            # ("What this does not cover" in the module docstring). The stable
            # sort at least keeps equal magnitudes in index order.
            valid = np.argsort(np.abs(eigenvalues), kind="stable")[-self.n_modes:]

        n_use = min(self.n_modes, len(valid))
        idx = valid[:n_use]
        modes = canonical_sign(eigenvectors[:, idx].T)  # (n_use, 3N)
        freqs = np.maximum(eigenvalues[idx], 1e-6)      # ensure strictly positive

        return modes, freqs

    def _sample_ca(
        self,
        ca_coords: np.ndarray,
        modes: np.ndarray,
        freqs: np.ndarray,
        amplitude: float,
        seed: int,
    ) -> np.ndarray:
        """Displace Cα along a Boltzmann-weighted combination of normal modes.

        Returns the displaced bead coordinates as float64. They are an
        intermediate: rounding them to float32 here, as this method used to,
        would put a second rounding step between the modes and the conformer.
        """
        n = len(ca_coords)
        rng = np.random.default_rng(seed)

        # Thermal amplitude: lower-frequency modes have larger fluctuations (∝ 1/√freq)
        weights = 1.0 / np.sqrt(np.asarray(freqs, dtype=np.float64) + 1e-10)
        coeffs = rng.normal(0.0, weights)  # (n_modes,)

        # Linear combination → (3N,) → reshape to (N, 3). Accumulated one mode
        # at a time, in mode order: no matrix product, so BLAS stays out, and
        # no reduction whose order of additions is left to NumPy.
        modes = np.asarray(modes, dtype=np.float64)
        displacement = np.zeros(3 * n, dtype=np.float64)
        for coeff, mode in zip(coeffs, modes):
            displacement += coeff * mode
        displacement = displacement.reshape(n, 3)

        rmsd = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))
        if rmsd > 1e-8:
            displacement *= amplitude / rmsd

        return np.asarray(ca_coords, dtype=np.float64) + displacement

    def _propagation_weights(
        self, base_coords: np.ndarray, ca_coords: np.ndarray
    ) -> np.ndarray:
        """Row-normalised Gaussian weights from every atom to every bead.

        Returns an (N_atoms, N_beads) float64 array: the weight of bead m in
        the displacement of atom n, a Gaussian of their distance with a 5 Å
        correlation length, normalised over beads. It depends on the input
        structure only, which is why ``_sample_ensemble`` computes it once.

        Squared distances are accumulated one axis at a time into the result,
        a slab of rows at a time, so the (N_atoms, N_beads, 3) difference
        tensor is never formed and peak memory is the result plus one slab.
        """
        base = np.asarray(base_coords, dtype=np.float64)
        beads = np.asarray(ca_coords, dtype=np.float64)
        n_atoms, n_beads = len(base), len(beads)
        bead_axes = [np.ascontiguousarray(beads[:, k]) for k in range(3)]

        weights = np.empty((n_atoms, n_beads), dtype=np.float64)
        step = _slab_rows(n_atoms, n_beads)
        scratch = np.empty((step, n_beads), dtype=np.float64)
        for start in range(0, n_atoms, step):
            atoms = base[start : start + step]
            w = weights[start : start + step]        # a view, filled in place
            s = scratch[: len(w)]

            np.subtract(atoms[:, 0, None], bead_axes[0], out=w)
            np.square(w, out=w)
            for k in (1, 2):
                np.subtract(atoms[:, k, None], bead_axes[k], out=s)
                np.square(s, out=s)
                w += s                               # squared distance

            np.divide(w, -(2 * 5.0**2), out=w)       # 5 Å correlation length
            np.exp(w, out=w)
            w /= w.sum(axis=1, keepdims=True) + 1e-8

        return weights

    def _propagate(
        self,
        base_coords: np.ndarray,
        ca_coords: np.ndarray,
        ca_displaced: np.ndarray,
    ) -> np.ndarray:
        """Propagate Cα displacements to all atoms via Gaussian-weighted interpolation.

        The arithmetic is float64 and the returned conformer is float32: this
        cast is the only place the sampler rounds to single precision.

        Inside ``generate`` the weights are computed by the first call and
        kept for the rest of the ensemble in the memo that ``_sample_ensemble``
        sets up. Called on its own, with any other arrays, it computes them
        itself and gives the same answer.
        """
        memo = self._weights_memo
        if memo is not None and memo[0] is base_coords and memo[1] is ca_coords:
            weights = memo[2]
            if weights is None:
                weights = self._propagation_weights(base_coords, ca_coords)
                self._weights_memo = (base_coords, ca_coords, weights)
        else:
            weights = self._propagation_weights(base_coords, ca_coords)

        base = np.asarray(base_coords, dtype=np.float64)
        ca_disp = (np.asarray(ca_displaced, dtype=np.float64)
                   - np.asarray(ca_coords, dtype=np.float64))  # (M, 3)
        disp_axes = [np.ascontiguousarray(ca_disp[:, k]) for k in range(3)]

        # atom_disp = weights @ ca_disp, written as einsum because matmul hands
        # the product to BLAS, whose rounding can depend on the thread count.
        # Slabs keep each block of weights in cache for its three passes.
        n_atoms, n_beads = weights.shape
        atom_disp = np.empty((n_atoms, 3), dtype=np.float64)
        step = max(1, _BLOCK_ELEMENTS // max(n_beads, 1))
        for start in range(0, n_atoms, step):
            w = weights[start : start + step]
            for k in range(3):
                atom_disp[start : start + step, k] = np.einsum("nm,m->n", w, disp_axes[k])

        return (base + atom_disp).astype(np.float32)
