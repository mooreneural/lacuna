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
has to give the same pockets on every machine, which takes conformers that
agree to within float32 rounding. The obstacle is the eigensolver: its
output depends on how many threads BLAS runs, in two independent ways, and both
used to reach the conformers. (A third way, which only symmetric assemblies
meet, has its own section below.)

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
none by more than 1.9e-6 A. The solver's share of that rounding grows as two
sampled eigenvalues approach each other (to nearly 1e-9 A on the symmetric
assemblies of the next section, where they come within 1e-3 of each other),
which makes these one-unit disagreements more frequent; the next section is
about where that stops being rounding.

Degenerate modes
----------------
A sign rule is enough only while each mode is individually well defined. It
is not when eigenvalues coincide: any rotation of the basis inside a
degenerate set is an equally valid answer, LAPACK returns whichever one its
rounding lands on, and the seeded coefficients belong to individual modes, so
a rotated basis is a different ensemble. A structure with a three-fold or
higher rotation axis (a C3 trimer, a C4 or D4 assembly, anything tetrahedral,
octahedral or icosahedral) has modes that come in such pairs and triplets,
and an assembly built by ``--homodimer`` from crystallographic operators can
carry that symmetry to the last bit. Left alone, such an assembly gave
conformers up to 5.8 A apart between thread counts, and a different top
pocket. Two-fold symmetry is another matter: its modes are non-degenerate,
and the tie rule in ``canonical_sign`` is all it needs.

What a degenerate set does define is the space it spans. ``_compute_modes``
groups neighbouring eigenvalues that agree to a relative 1e-4
(``_DEGENERACY_RTOL``) and replaces the solver's basis of each group by
``canonical_basis``, which is built from the projector onto that space and so
comes out the same whichever basis the solver returned. The members of a
group share one frequency to within the tolerance, so every orthonormal basis
of it is a set of normal modes as good as the solver's, and the ensemble is
drawn from the same distribution. A structure with no two eigenvalues that
close is not touched: it gets the solver's eigenvectors, sign fixed, exactly
as before.

The tolerance is wide on purpose. Symmetry that is exact in the molecule is
only approximate in a file: a copy placed by a translation is rounded to
float32, and one placed by a general rotation to the 0.001 A of the format.
That splits each degenerate set by a relative 1e-7 or so in the first case
and by 1e-5 to 1e-4 in the second (the 1OK8 trimer built from its own
operators comes out split by 2.7e-5), and the closer the split, the more of
the solver's rounding reaches the vectors. Measured on ten symmetric
assemblies with seeded noise of 0 to 0.01 A added to break the symmetry by a
known amount (80 cases, 1 and 2 threads against the default): left to the
solver, 13 cases moved by more than 1e-5 A and the worst by 5.7 A; with a
tolerance of 1e-6, 126,600 coordinates still differed, by up to four float32
units; with 1e-4, 3,811 of 80 million differed, none by more than 7.7e-6 A,
and no coordinate larger than 1 A by more than one unit, which is the residue
described under "Reproducibility". Ordinary structures stay well clear of the
tolerance: across 749 CryptoBench train-fold chains the closest two sampled
eigenvalues are 2.4e-3 apart.

When the group of the last sampled mode runs to the edge of the eigenvalue
window, the solve is repeated with twice the window, so that a group is never
cut by what happened to be requested. It can still be cut by ``n_modes``; the
first vectors of the canonical basis are then the ones kept. The second solve
costs as much as the first and is needed only when zero modes or a very large
degenerate set have used up the headroom in the window.

What this does not cover
------------------------
An under-constrained network. A bead held by fewer than three springs, or by
springs that all lie in one plane, adds a zero mode to the six rigid-body
ones. A handful is harmless, because zero modes are filtered out before
sampling. When they fill the whole window of eigenvalues that
``_compute_modes`` requests (40 or more of them at the default ``n_modes``),
nothing in the window is a vibration and what gets sampled is an arbitrary
basis of the null space, rigid-body motion included. A predicted model whose
disordered regions are drawn as long extended ribbons does this: of 157
structures in the benchmark cache, the two it happens to are AlphaFold models
of GPCRs with more than a hundred zero modes each. The sign and basis rules
above do not reach this case, because the window is an arbitrary slice of a
larger null space, so such an ensemble is not reproducible across thread
counts. It is not covered here: it is detected and reported with a
``RuntimeWarning``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from lacuna.ensemble.base import EnsembleBackend
from lacuna.io.structure import load_structure, coords_array

#: Two components of a mode count as tied for largest when their magnitudes
#: agree to this relative tolerance. See ``canonical_sign``.
_SIGN_TIE_RTOL = 1e-6

#: Neighbouring eigenvalues that agree to this relative tolerance are treated
#: as one degenerate set, whose basis is fixed by ``canonical_basis`` instead
#: of being taken from the solver. See "Degenerate modes" in the module
#: docstring for why it is this wide.
_DEGENERACY_RTOL = 1e-4

#: The eigenvalue window holds no vibration at all when its largest eigenvalue
#: is below this fraction of the largest diagonal entry of the Hessian. Real
#: soft modes sit many orders of magnitude above it and the rounding noise of
#: a zero mode several below.
_NULL_WINDOW_RTOL = 1e-9

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


def canonical_basis(vectors: np.ndarray) -> np.ndarray:
    """Replace an orthonormal basis by one that depends only on its span.

    When several modes share one eigenvalue the solver returns an arbitrary
    orthonormal basis of the space they span, a different one for every
    rounding pattern, hence for every BLAS thread count. The space itself is
    well defined, and so is the projector onto it, P = sum_k v_k v_k^T. This
    builds a basis from P alone:

    1. Take the coordinate axis that P shrinks least, that is the largest
       diagonal entry of P. The first basis vector is that axis projected into
       the space and normalised.
    2. Remove that vector from the space and repeat on what is left.

    This is a Cholesky factorisation of P with diagonal pivoting, and its
    columns are orthonormal because P is a projector. "Largest" is taken with
    the tolerance and the lowest-index rule of ``canonical_sign``, for the
    same reason: in a symmetric structure the diagonal entries of P that
    belong to symmetry-related atoms are equal in exact arithmetic, and
    rounding must not get to choose among them.

    Each vector comes out with a positive component on its own pivot axis, so
    the sign is fixed too. Nothing here calls BLAS: every sum is accumulated
    one vector at a time in index order.

    Parameters
    ----------
    vectors : np.ndarray
        (d, 3N) array, one vector per row, orthonormal.

    Returns
    -------
    np.ndarray
        A new (d, 3N) float64 array with orthonormal rows spanning the same
        space. Two inputs that span the same space give the same output up to
        float64 rounding.
    """
    work = np.array(vectors, dtype=np.float64)        # a copy, deflated in place
    basis = np.empty_like(work)
    if work.size == 0:
        return basis
    for k in range(len(work)):
        # Diagonal of the projector onto what is left of the space.
        weight = np.zeros(work.shape[1], dtype=np.float64)
        for row in work:
            weight += row * row
        # argmax over booleans returns the first True, which is the lowest index.
        pivot = int((weight >= weight.max() * (1.0 - _SIGN_TIE_RTOL)).argmax())
        coeffs = work[:, pivot] / np.sqrt(weight[pivot])

        vector = np.zeros(work.shape[1], dtype=np.float64)
        for coeff, row in zip(coeffs, work):
            vector += coeff * row
        basis[k] = vector
        for coeff, row in zip(coeffs, work):
            row -= coeff * vector
    return basis


def _degenerate_runs(
    eigenvalues: np.ndarray, first: int, last: int
) -> list[tuple[int, int]]:
    """Runs of tied eigenvalues that between them cover ``first`` to ``last``.

    ``eigenvalues`` is ascending. Two neighbours are tied when they differ by
    no more than ``_DEGENERACY_RTOL`` of the larger one, and a run is a maximal
    chain of tied neighbours. Returns inclusive ``(start, stop)`` index pairs
    in ascending order. The first run may start below ``first`` and the last
    may stop above ``last``: a run is reported whole, wherever the range cuts
    it. A mode tied with nothing is a run of one.
    """
    def tied(i: int) -> bool:
        gap = eigenvalues[i + 1] - eigenvalues[i]
        return bool(gap <= _DEGENERACY_RTOL * abs(eigenvalues[i + 1]))

    start = first
    while start > 0 and tied(start - 1):
        start -= 1
    runs = []
    while start <= last:
        stop = start
        while stop + 1 < len(eigenvalues) and tied(stop):
            stop += 1
        runs.append((start, stop))
        start = stop + 1
    return runs


class NMABackend(EnsembleBackend):
    """Anisotropic Network Model (ANM) conformational sampling.

    Samples conformers by displacing Cα atoms along the lowest-frequency
    normal modes of an elastic network, then propagates to all atoms using
    Gaussian-weighted interpolation. Physically grounded alternative to
    RandomBackend - generates hinge/breathing motions rather than random noise.

    The pocket report built from the ensemble is identical at every BLAS
    thread count and between a Windows and a Linux build. The conformer
    coordinates themselves agree to float32 rounding: on an ordinary
    structure about one coordinate in a million differs, by one unit in its
    last place (a few units for a coordinate that lies almost at zero). See
    "Reproducibility" in the module docstring for what that rests on and for
    the measured bounds, "Degenerate modes" for how symmetric assemblies are
    kept inside it, and "What this does not cover" for the one kind of input
    that is outside it, which raises a warning.

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
        the sign fixed by ``canonical_sign``, and each set of modes that share
        an eigenvalue has its basis fixed by ``canonical_basis``, so the same
        network gives the same modes, not the same modes up to a sign per
        mode or a rotation per degenerate set.

        Warns (``RuntimeWarning``) when the network is so under-constrained
        that the requested eigenvalues are all zero modes.

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
        n_window = min(6 + self.n_modes * 4, 3 * n - 1)

        if gamma is not None:
            gamma = np.asarray(gamma, dtype=np.float64)
        symmetric = gamma is None or np.array_equal(gamma, gamma.T)

        # Nearly always one pass. A second, with more eigenvalues, is needed
        # only when the degenerate set of the last sampled mode runs to the
        # edge of what was requested, so that it cannot be seen whole.
        n_request = n_window
        while True:
            hessian = self._hessian(ca_coords, gamma)
            # Scale of the spectrum, read before LAPACK overwrites the matrix.
            # A maximum has no order of operations, so no rounding of its own.
            stiffest = float(hessian.diagonal().max()) if n else 0.0

            # Partial eigendecomposition - only the lowest n_request modes.
            #
            # eigh reads the lower triangle and LAPACK wants Fortran order, so
            # a C-ordered matrix is copied before it is factored, which doubles
            # the peak. The Hessian is symmetric to the last bit whenever gamma
            # is, so its transpose is the same matrix already in Fortran order,
            # and LAPACK can then work in place. An asymmetric gamma is not a
            # physical spring network, but it keeps the old meaning (lower
            # triangle) at the old cost (one copy) rather than silently
            # reading the other triangle.
            eigenvalues, eigenvectors = eigh(
                hessian.T if symmetric else hessian,
                subset_by_index=[0, n_request - 1],
                overwrite_a=True,
            )
            del hessian

            # Filter: keep only genuinely non-trivial modes.
            # Threshold at 0.1% of the max eigenvalue removes rigid-body modes
            # and any extra zero modes from degenerate structures (e.g. flat
            # z=0 fixtures). The filter reads the first n_window eigenvalues
            # only, so a widened request never changes which modes are chosen.
            window = eigenvalues[:n_window]
            max_eval = float(np.abs(window).max()) or 1.0
            valid = np.where(window > max_eval * 1e-3)[0]

            if len(valid) == 0 or max_eval <= stiffest * _NULL_WINDOW_RTOL:
                # No vibration in the window: every eigenvalue in it is
                # rounding noise around zero, and what gets sampled is the
                # null space in whatever basis LAPACK returned, rigid-body
                # motion included ("What this does not cover" in the module
                # docstring). The rules below do not cover that, so say so.
                warnings.warn(
                    "elastic network is under-constrained: its zero modes fill "
                    f"the {n_window} lowest eigenvalues, so there is no "
                    "vibration to sample. The ensemble is built from rigid-body "
                    "and free-bead motion, means nothing physically, and is "
                    "not reproducible from one machine to the next.",
                    RuntimeWarning, stacklevel=2,
                )
                if len(valid) == 0:
                    # Use the highest-magnitude modes. The stable sort at
                    # least keeps equal magnitudes in index order.
                    valid = np.argsort(np.abs(window), kind="stable")[-self.n_modes:]
                runs = []
                break

            # valid is a contiguous, ascending stretch of the window from here
            # on, so the sampled modes are too.
            n_use = min(self.n_modes, len(valid))
            if n_use == 0:
                runs = []
                break
            runs = _degenerate_runs(eigenvalues, int(valid[0]), int(valid[n_use - 1]))
            seen_whole = runs[-1][1] < len(eigenvalues) - 1
            if seen_whole or n_request >= 3 * n:
                break
            n_request = min(2 * n_request, 3 * n)

        n_use = min(self.n_modes, len(valid))
        idx = valid[:n_use]
        modes = eigenvectors[:, idx].T.copy()           # (n_use, 3N)
        freqs = np.maximum(eigenvalues[idx], 1e-6)      # ensure strictly positive

        # A degenerate set is sampled in the basis its span defines, not the
        # one the solver happened to return. Where n_modes cuts a set, the
        # first vectors of that basis are the ones kept.
        for start, stop in runs:
            if stop == start:
                continue
            basis = canonical_basis(eigenvectors[:, start : stop + 1].T)
            low, high = max(start, int(idx[0])), min(stop, int(idx[-1]))
            modes[low - idx[0] : high - idx[0] + 1] = basis[: high - low + 1]

        return canonical_sign(modes), freqs

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
