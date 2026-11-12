# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""One structure has to give one ensemble.

Every pocket Lacuna reports is carved out of the conformers ``NMABackend``
returns, and those conformers used to depend on how many threads BLAS was
running. Measured on 184 CryptoBench train-fold targets, default threads
against one thread: 2 ensembles were bit-identical, 56 differed by more than
0.1 A and the worst by 24 A. Downstream, 17 of 46 targets produced different
candidate pockets.

There were two independent causes, and a third that only symmetric assemblies
have. The tests are split the same way.

  sign        An eigenvector is defined only up to sign, and the sign LAPACK
              returns varies with threading. A conformer is a seeded Gaussian
              combination of the modes, so a flipped mode is a different
              conformer. Pinning the sign alone left no ensemble more than
              0.07 A apart, and still only 2 bit-identical.
  precision   The Hessian was built from float32 coordinates, so the solver ran
              in single precision and its thread-dependent rounding was large
              enough to see. In float64 with the sign pinned, 181 of the 184
              are bit-identical and the other three agree to 2.4e-7 A.
  degeneracy  Modes that share an eigenvalue (any structure with a three-fold
              or higher axis has them) come back in an arbitrary basis of the
              space they span, which no sign rule can pin. Assemblies with an
              exact four-fold axis still differed by up to 5.8 A between
              thread counts after the two fixes above. Each degenerate set
              now gets a basis computed from its projector.

A unit test cannot change the thread count of the process it runs in, so most
of these replace ``eigh`` with a stand-in that returns what a differently
threaded LAPACK would: the same vectors with other signs, with other rounding,
or with another basis for each degenerate set. The last two tests do it for
real, in two subprocesses each.

One kind of input is outside what any of this can fix: a network so
under-constrained that zero modes fill the eigenvalue window. It is described
in the ``nma_backend`` module docstring, and the test for it is that the
sampler warns. The fixtures for the sign and precision tests are checked to be
neither that nor degenerate, see ``_well_posed``; the symmetric fixtures are
checked to be degenerate.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

import lacuna
from lacuna.ensemble import nma_backend
from lacuna.ensemble.nma_backend import NMABackend, canonical_basis, canonical_sign
from lacuna.io.structure import coords_array, load_structure
from lacuna.nucleic.anm import RNAAnmBackend

ROOT = Path(__file__).resolve().parent.parent
#: 167 residues, tracked in the repository. On the machine this was written on
#: the unfixed sampler moved it by 3.7 A between one BLAS thread and four.
KRAS = ROOT / "benchmarks" / "pdb_cache" / "KRAS_G12D.pdb"


# ------------------------------------------------------------------ fixtures

_ATOM = "ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"


def _write_pdb(path: Path, atoms) -> Path:
    """atoms: (name, residue name, chain, residue number, xyz, element)."""
    lines = [
        _ATOM % (i + 1, name if len(name) == 4 else " " + name, res, chain, num, *xyz, el)
        for i, (name, res, chain, num, xyz, el) in enumerate(atoms)
    ]
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _residue(rng, centre, parts, jitter):
    """One residue: every atom at centre + offset, jittered so nothing is regular."""
    out = []
    for name, element, offset in parts:
        xyz = np.asarray(centre) + np.asarray(offset, dtype=float) + rng.uniform(-jitter, jitter, 3)
        out.append((name, element, tuple(float(v) for v in np.round(xyz, 3))))
    return out


_AMINO = (("N", "N", (-1.2, 0.5, 0.3)), ("CA", "C", (0.0, 0.0, 0.0)),
          ("C", "C", (1.1, 0.8, -0.4)), ("O", "O", (1.6, 1.7, 0.5)))
_NUCLEOTIDE = (("P", "P", (-2.4, 0.6, 0.2)), ("O2'", "O", (-0.6, -1.9, 0.8)),
               ("C1'", "C", (0.0, 0.0, 0.0)), ("N1", "N", (1.1, 0.9, -0.5)),
               ("C2", "C", (2.2, 1.5, 0.6)))


@pytest.fixture
def protein_pdb(tmp_path: Path) -> Path:
    """27 residues on a jittered 3x3x3 lattice: compact, rigid, no symmetry."""
    rng = np.random.default_rng(11)
    atoms = []
    cells = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    for num, cell in enumerate(cells, start=1):
        centre = 3.8 * np.array(cell, dtype=float) + rng.uniform(-0.5, 0.5, 3)
        for name, element, xyz in _residue(rng, centre, _AMINO, 0.2):
            atoms.append((name, "ALA", "A", num, xyz, element))
    return _write_pdb(tmp_path / "protein.pdb", atoms)


@pytest.fixture
def dimer_pdb(tmp_path: Path) -> Path:
    """A homodimer with an exact two-fold axis along z.

    Chain B is chain A under (x, y, z) -> (-x, -y, z), applied to the rounded
    coordinates, so the symmetry is exact in the file and in float32. Every
    mode is then symmetric or antisymmetric, which makes the x and y (or z)
    components of partner atoms equal in magnitude and opposite in sign: the
    case the tie rule in ``canonical_sign`` is for.
    """
    rng = np.random.default_rng(5)
    half = []
    cells = [(i, j, k) for i in range(2) for j in range(3) for k in range(2)]
    for num, cell in enumerate(cells, start=1):
        centre = (np.array([2.0, -3.8, 0.0]) + 3.8 * np.array(cell, dtype=float)
                  + rng.uniform(-0.5, 0.5, 3))
        for name, element, xyz in _residue(rng, centre, _AMINO[:3], 0.2):
            half.append((name, num, xyz, element))
    atoms = [(name, "ALA", "A", num, xyz, el) for name, num, xyz, el in half]
    atoms += [(name, "ALA", "B", num, (-x, -y, z), el) for name, num, (x, y, z), el in half]
    return _write_pdb(tmp_path / "dimer.pdb", atoms)


@pytest.fixture
def rna_pdb(tmp_path: Path) -> Path:
    """Eight nucleotides, three beads each (P, C1', C2) and two atoms that are not."""
    rng = np.random.default_rng(3)
    atoms = []
    cells = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    for num, cell in enumerate(cells, start=1):
        centre = 6.0 * np.array(cell, dtype=float) + rng.uniform(-0.6, 0.6, 3)
        for name, element, xyz in _residue(rng, centre, _NUCLEOTIDE, 0.3):
            atoms.append((name, "ACGU"[num % 4], "A", num, xyz, element))
    return _write_pdb(tmp_path / "rna.pdb", atoms)


def _backend(kind: str):
    return RNAAnmBackend() if kind == "rna" else NMABackend()


def _beads(backend, path: Path):
    """(float32 coordinates of every atom, indices of the network beads)."""
    structure = load_structure(path)
    if isinstance(backend, RNAAnmBackend):
        index = backend.bead_indices(structure)
    else:
        index = [a.serial for a in structure.atoms if a.name == "CA"]
    return coords_array(structure), index


def _well_posed(backend, beads: np.ndarray) -> None:
    """Fail unless the fixture is one the sampler can be reproducible on.

    Exactly six zero modes (a rigid network), a full set of kept modes, and
    kept eigenvalues separated from each other and from the first one dropped.
    Without this a later edit to a fixture could make it degenerate, and the
    sign and precision tests would then pass or fail on the strength of the
    canonical basis instead of the thing they are named for.
    """
    values = scipy.linalg.eigvalsh(backend._hessian(beads))
    assert int((np.abs(values) < 1e-9 * values[-1]).sum()) == 6, "fixture is not rigid"
    window = values[: min(6 + 4 * backend.n_modes, len(values) - 1)]
    kept = window[window > np.abs(window).max() * 1e-3]
    assert len(kept) > backend.n_modes
    kept = kept[: backend.n_modes + 1]
    assert (np.diff(kept) / kept[1:]).min() > 1e-3, "fixture has near-degenerate modes"


class _ReplayEigh:
    """Stand-in for ``scipy.linalg.eigh`` that solves once and replays the answer.

    The first call runs the real solver and keeps its output. Every call
    returns that one output passed through ``transform``. Two ``generate``
    calls therefore see literally the same eigendecomposition apart from the
    transform, so "the conformers are identical" can be asserted bit for bit
    without assuming LAPACK itself is repeatable.
    """

    def __init__(self, transform=None):
        self.transform = transform or (lambda values, vectors: (values, vectors))
        self.solution = None
        self.inputs = []          # (dtype, shape) of every matrix handed over

    def __call__(self, a, *args, **kwargs):
        self.inputs.append((a.dtype, a.shape))
        if self.solution is None:
            self.solution = scipy.linalg.eigh(a, *args, **kwargs)
        values, vectors = self.solution
        return self.transform(values.copy(), vectors.copy())


def _generate(monkeypatch, backend, path, n, transform=None, solution=None):
    """Run ``generate`` with ``eigh`` replaced; returns (conformers, the stand-in)."""
    stand_in = _ReplayEigh(transform)
    stand_in.solution = solution
    monkeypatch.setattr(nma_backend, "eigh", stand_in)
    return backend.generate(path, n), stand_in


def _same(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def _worst(a, b) -> float:
    return max(float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
               for x, y in zip(a, b))


def _reference_ensemble(backend, base32: np.ndarray, index, n_conformers: int):
    """The model written out plainly in float64, never rounded: a list of float64 arrays.

    Dense arrays, a full eigendecomposition, a matrix product. It shares no
    code with the sampler apart from NumPy and LAPACK, and it states the sign
    rule the simple way (largest component positive), which is the same rule
    wherever no two components tie.
    """
    base = base32.astype(np.float64)
    beads = base[index]
    n = len(beads)
    diff = beads[:, None, :] - beads[None, :, :]
    r = np.sqrt((diff ** 2).sum(axis=-1))
    contact = (r > 1e-6) & (r <= backend.cutoff)
    r_safe = np.where(contact, r, 1.0)
    outer = diff[:, :, :, None] * diff[:, :, None, :]
    blocks = np.where(contact[:, :, None, None], -outer / r_safe[:, :, None, None] ** 2, 0.0)
    hessian = blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n).copy()
    for i in range(n):
        hessian[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= blocks[i].sum(axis=0)

    values, vectors = scipy.linalg.eigh(hessian)
    n_request = min(6 + 4 * backend.n_modes, 3 * n - 1)
    values, vectors = values[:n_request], vectors[:, :n_request]
    kept = np.where(values > np.abs(values).max() * 1e-3)[0][: backend.n_modes]
    modes = vectors[:, kept].T
    pivot = np.abs(modes).argmax(axis=1)
    modes = modes * np.sign(modes[np.arange(len(modes)), pivot])[:, None]
    freqs = values[kept]

    d = np.sqrt(((base[:, None, :] - beads[None, :, :]) ** 2).sum(axis=-1))
    weights = np.exp(-d ** 2 / (2 * 5.0 ** 2))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8

    out = []
    for i in range(n_conformers):
        amplitude = (i + 1) / n_conformers * backend.max_rmsd
        coeffs = np.random.default_rng(i).normal(0.0, 1.0 / np.sqrt(freqs + 1e-10))
        disp = (coeffs[:, None] * modes).sum(axis=0).reshape(n, 3)
        disp *= amplitude / np.sqrt((disp ** 2).sum(axis=1).mean())
        out.append(base + weights @ disp)
    return out


# ------------------------------------------------------------------ the sign rule

def test_the_largest_component_of_every_mode_is_made_positive():
    modes = np.array([[0.1, -0.9, 0.4],
                      [0.2, 0.3, -0.1],
                      [-0.5, 0.2, 0.1]])
    out = canonical_sign(modes)
    np.testing.assert_array_equal(out, [[-0.1, 0.9, -0.4],
                                        [0.2, 0.3, -0.1],
                                        [0.5, -0.2, -0.1]])
    assert out.dtype == np.float64 and out.shape == modes.shape
    assert out is not modes, "the input must not be rewritten in place"


def test_a_mode_and_its_negation_get_the_same_sign():
    """The whole point: v and -v are the same eigenvector, so they must come
    out as the same array, bit for bit, not merely close."""
    rng = np.random.default_rng(0)
    modes = rng.normal(size=(12, 90))
    flips = np.where(rng.random(12) < 0.5, -1.0, 1.0)[:, None]
    np.testing.assert_array_equal(canonical_sign(modes), canonical_sign(modes * flips))
    np.testing.assert_array_equal(canonical_sign(modes), canonical_sign(-modes))


def test_single_precision_modes_are_promoted_not_rounded():
    modes = np.array([[0.25, -0.75, 0.5]], dtype=np.float32)
    out = canonical_sign(modes)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [[-0.25, 0.75, -0.5]])


def test_near_ties_for_largest_go_to_the_lowest_index():
    """Two components within 1e-6 of each other count as tied, and the earlier
    one decides. Rounding can make either of them the strict maximum, so a
    plain argmax would give these two rows opposite signs."""
    lower_wins = np.array([[0.3, -0.7 * (1 + 1e-9), 0.7, 0.1]])
    upper_wins = np.array([[0.3, -0.7, 0.7 * (1 + 1e-9), 0.1]])
    assert np.abs(lower_wins).argmax() == 1 and np.abs(upper_wins).argmax() == 2

    a, b = canonical_sign(lower_wins), canonical_sign(upper_wins)
    # Index 1 decides in both: it is negative, so both rows are flipped.
    assert a[0, 1] > 0 and b[0, 1] > 0
    np.testing.assert_allclose(a, b, rtol=1e-8)
    np.testing.assert_array_equal(a, -lower_wins)
    np.testing.assert_array_equal(b, -upper_wins)


def test_a_clear_maximum_is_not_overruled_by_an_earlier_component():
    """The tolerance is relative 1e-6 and no wider: a component that is larger
    by one part in 1e5 is the maximum, whatever comes before it."""
    modes = np.array([[0.3, -0.7, 0.7 * (1 + 1e-5), 0.1]])
    np.testing.assert_array_equal(canonical_sign(modes), modes)
    np.testing.assert_array_equal(canonical_sign(-modes), modes)


def test_zero_and_empty_modes_pass_through():
    np.testing.assert_array_equal(canonical_sign(np.zeros((2, 6))), np.zeros((2, 6)))
    assert canonical_sign(np.zeros((0, 9))).shape == (0, 9)


# ------------------------------------------------------------------ sign, end to end

def _negate_all(values, vectors):
    return values, -vectors


def _negate_alternate(values, vectors):
    vectors[:, ::2] *= -1.0
    return values, vectors


def _negate_first(values, vectors):
    # Past the six rigid-body modes, so this one is a mode that gets sampled.
    vectors[:, 6] *= -1.0
    return values, vectors


@pytest.mark.parametrize("kind", ["protein", "rna"])
@pytest.mark.parametrize("flip", [_negate_all, _negate_alternate, _negate_first])
def test_conformers_do_not_depend_on_the_sign_eigh_returns(
        monkeypatch, request, kind, flip):
    """Hand the sampler the same eigenvectors with other signs, which is what a
    differently threaded LAPACK does, and every conformer must be bit-identical."""
    path = request.getfixturevalue(kind + "_pdb")
    backend = _backend(kind)
    base, index = _beads(backend, path)
    _well_posed(backend, base[index])

    plain, first = _generate(monkeypatch, backend, path, 6)
    flipped, _ = _generate(monkeypatch, backend, path, 6, flip, first.solution)
    assert _same(plain, flipped), "worst difference %.3g A" % _worst(plain, flipped)

    # Control: with the sign rule switched off the same flip does move atoms,
    # so the assertion above is the rule working and not a patch that missed.
    monkeypatch.setattr(nma_backend, "canonical_sign",
                        lambda modes: np.asarray(modes, dtype=np.float64))
    loose, _ = _generate(monkeypatch, backend, path, 6, None, first.solution)
    loose_flipped, _ = _generate(monkeypatch, backend, path, 6, flip, first.solution)
    assert _worst(loose, loose_flipped) > 1e-3


def test_modes_do_not_depend_on_the_sign_eigh_returns(monkeypatch, protein_pdb):
    backend = NMABackend()
    base, index = _beads(backend, protein_pdb)
    stand_in = _ReplayEigh()
    monkeypatch.setattr(nma_backend, "eigh", stand_in)
    modes, freqs = backend._compute_modes(base[index])

    stand_in.transform = _negate_alternate
    modes_flipped, freqs_flipped = backend._compute_modes(base[index])
    np.testing.assert_array_equal(modes, modes_flipped)
    np.testing.assert_array_equal(freqs, freqs_flipped)
    # And the sign is the documented one, not just a consistent one.
    pivot = np.abs(modes).argmax(axis=1)
    assert (modes[np.arange(len(modes)), pivot] > 0).all()


# ------------------------------------------------------------------ precision

@pytest.mark.parametrize("kind", ["protein", "rna"])
def test_the_eigensolver_is_handed_a_float64_hessian(monkeypatch, request, kind):
    """The coordinates arrive as float32 (``coords_array`` returns nothing
    else) and that used to decide the precision of the whole solve."""
    path = request.getfixturevalue(kind + "_pdb")
    backend = _backend(kind)
    base, index = _beads(backend, path)
    assert base.dtype == np.float32

    _, stand_in = _generate(monkeypatch, backend, path, 2)
    n = len(index)
    assert stand_in.inputs == [(np.dtype(np.float64), (3 * n, 3 * n))]
    assert backend._hessian(base[index]).dtype == np.float64


def test_no_step_before_the_final_cast_is_single_precision(protein_pdb):
    """Every hook hands on float64, and only the finished conformer is float32."""
    backend = NMABackend()
    base, index = _beads(backend, protein_pdb)
    beads = base[index]
    assert base.dtype == beads.dtype == np.float32

    modes, freqs = backend._compute_modes(beads)
    assert modes.dtype == np.float64 and freqs.dtype == np.float64
    displaced = backend._sample_ca(beads, modes, freqs, 1.0, seed=0)
    assert displaced.dtype == np.float64
    assert backend._propagation_weights(base, beads).dtype == np.float64
    assert backend._propagate(base, beads, displaced).dtype == np.float32


def test_single_precision_input_is_promoted_exactly(protein_pdb):
    """float32 coordinates and the same numbers held as float64 are one input:
    the Hessian is the same matrix bit for bit, so the modes are the same too
    (to float64 rounding, since that is two separate LAPACK calls)."""
    backend = NMABackend()
    base, index = _beads(backend, protein_pdb)
    beads32, beads64 = base[index], base[index].astype(np.float64)
    np.testing.assert_array_equal(backend._hessian(beads32), backend._hessian(beads64))

    modes32, freqs32 = backend._compute_modes(beads32)
    modes64, freqs64 = backend._compute_modes(beads64)
    np.testing.assert_allclose(modes32, modes64, rtol=0, atol=1e-9)
    np.testing.assert_allclose(freqs32, freqs64, rtol=1e-12)


def _textbook_hessian(coords, cutoff, gamma=None):
    """The (N, N, 3, 3) expression the Hessian was first written as, in float64."""
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    r = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
    contact = (r > 1e-6) & (r <= cutoff)
    r_safe = np.where(r > 1e-6, r, 1.0)
    strength = 1.0 if gamma is None else gamma[:, :, None, None]
    blocks = np.where(
        contact[:, :, None, None],
        -strength / r_safe[:, :, None, None] ** 2 * (diff[:, :, :, None] * diff[:, :, None, :]),
        0.0)
    hessian = blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n).copy()
    diagonal = -np.add.accumulate(blocks, axis=1)[:, -1]
    for i in range(n):
        hessian[3 * i:3 * i + 3, 3 * i:3 * i + 3] += diagonal[i]
    return hessian


@pytest.mark.parametrize("block_elements", [None, 64, 1000])
def test_the_blockwise_hessian_is_the_textbook_matrix_bit_for_bit(
        monkeypatch, protein_pdb, block_elements):
    """Assembling it a slab at a time is a memory measure and nothing else: the
    matrix is the same to the last bit, whatever the slab size, with or
    without per-pair spring constants."""
    if block_elements is not None:
        monkeypatch.setattr(nma_backend, "_BLOCK_ELEMENTS", block_elements)
    backend = NMABackend()
    base, index = _beads(backend, protein_pdb)
    beads = base[index]

    np.testing.assert_array_equal(
        backend._hessian(beads), _textbook_hessian(beads, backend.cutoff))

    rng = np.random.default_rng(1)
    gamma = rng.uniform(0.1, 1.0, size=(len(beads), len(beads)))
    gamma = (gamma + gamma.T) / 2
    np.testing.assert_array_equal(
        backend._hessian(beads, gamma), _textbook_hessian(beads, backend.cutoff, gamma))


def test_the_hessian_is_symmetric_to_the_last_bit(protein_pdb):
    """``_compute_modes`` hands LAPACK the transpose so that it can work in
    place, which is only the same matrix if this holds exactly."""
    backend = NMABackend()
    base, index = _beads(backend, protein_pdb)
    hessian = backend._hessian(base[index])
    np.testing.assert_array_equal(hessian, hessian.T)

    gamma = np.ones((len(index), len(index)))
    gamma[0, 1] = gamma[1, 0] = 0.05
    softened = backend._hessian(base[index], gamma)
    np.testing.assert_array_equal(softened, softened.T)
    assert not np.array_equal(softened, hessian)


def test_an_asymmetric_gamma_still_reads_the_lower_triangle(monkeypatch, protein_pdb):
    """Not a physical network, but it had a defined meaning before (LAPACK
    read the lower triangle) and the in-place shortcut must not change it."""
    backend = NMABackend()
    base, index = _beads(backend, protein_pdb)
    beads = base[index]
    gamma = np.ones((len(beads), len(beads)))
    gamma[np.triu_indices(len(beads), 1)] = 0.5      # upper triangle only

    _, freqs = backend._compute_modes(beads, gamma=gamma)
    lower = np.tril(backend._hessian(beads, gamma))
    lower = lower + np.tril(lower, -1).T
    window = scipy.linalg.eigvalsh(lower)[: min(6 + 4 * backend.n_modes, 3 * len(beads) - 1)]
    expected = window[window > np.abs(window).max() * 1e-3][: backend.n_modes]
    np.testing.assert_allclose(freqs, expected, rtol=1e-9)


@pytest.mark.parametrize("kind", ["protein", "rna"])
def test_a_conformer_is_the_float64_model_rounded_once(request, kind):
    """Compare against the model written out plainly in float64 and never
    rounded. Each coordinate must be the nearest float32 to it, give or take
    1e-9 A for the two routes rounding differently in float64. A float32
    intermediate anywhere would cost far more than that."""
    path = request.getfixturevalue(kind + "_pdb")
    backend = _backend(kind)
    base, index = _beads(backend, path)
    _well_posed(backend, base[index])

    conformers = backend.generate(path, 5)
    reference = _reference_ensemble(_backend(kind), base, index, 5)
    for got, want in zip(conformers, reference):
        error = np.abs(got.astype(np.float64) - want)
        half_ulp = 0.5 * np.spacing(np.abs(got)).astype(np.float64)
        assert (error <= half_ulp + 1e-9).all(), "worst %.3g A" % error.max()
        # And they did move: this is not two copies of the input agreeing.
        assert np.abs(want - base).max() > 1e-2


# ------------------------------------------------------------------ the tie rule

def _tied_pairs(modes: np.ndarray):
    """For each mode, its two largest components as (low index, high index)."""
    order = np.argsort(-np.abs(modes), axis=1)[:, :2]
    return np.sort(order, axis=1)


def test_the_symmetric_fixture_really_has_opposite_ties(dimer_pdb):
    """What makes the fixture a test of anything. Under a two-fold axis the two
    largest components of every mode are the same component of two partner
    atoms, equal in magnitude to rounding, and in some modes opposite in sign.
    Those are the modes a plain argmax flips on a coin toss."""
    backend = NMABackend()
    base, index = _beads(backend, dimer_pdb)
    _well_posed(backend, base[index])
    modes, _ = backend._compute_modes(base[index])
    half = 3 * len(index) // 2

    opposite = 0
    for mode, (low, high) in zip(modes, _tied_pairs(modes)):
        assert high - low == half, "largest components are not symmetry partners"
        assert abs(abs(mode[high]) / abs(mode[low]) - 1.0) < 1e-9
        opposite += mode[low] * mode[high] < 0
    assert opposite >= 3, "fixture no longer exercises the tie rule"


def test_the_tie_rule_holds_on_a_symmetric_structure(dimer_pdb):
    """Nudge each tied pair one way and then the other, by far more than
    rounding would and far less than the tolerance. The canonical sign must
    not notice. A plain argmax does, and the last lines check that it does, so
    this cannot pass by the nudge being too small to matter."""
    backend = NMABackend()
    base, index = _beads(backend, dimer_pdb)
    modes, _ = backend._compute_modes(base[index])
    rows = np.arange(len(modes))
    low, high = _tied_pairs(modes).T

    low_larger, high_larger = modes.copy(), modes.copy()
    low_larger[rows, low] *= 1 + 1e-9
    high_larger[rows, high] *= 1 + 1e-9

    def plain(m):
        pivot = np.abs(m).argmax(axis=1)
        return m * np.sign(m[rows, pivot])[:, None]

    for nudged in (low_larger, high_larger, -low_larger, -high_larger):
        np.testing.assert_allclose(canonical_sign(nudged), modes, rtol=1e-8, atol=0)
    flipped = np.sign(plain(low_larger)[rows, low]) != np.sign(plain(high_larger)[rows, low])
    assert flipped.sum() >= 3


def _scale_chain(first_half: bool, factor: float):
    """Make one chain of the dimer carry slightly larger components everywhere."""
    def transform(values, vectors):
        half = vectors.shape[0] // 2
        if first_half:
            vectors[:half] *= factor
        else:
            vectors[half:] *= factor
        return values, vectors
    return transform


def test_conformers_of_a_symmetric_structure_survive_solver_rounding(
        monkeypatch, dimer_pdb):
    """The same thing end to end: eigenvectors whose tied components lean one
    way, then the other, must give the same ensemble. The lean is 1e-9, so
    "the same" means to that, not bit for bit."""
    backend = NMABackend()
    lean_a, first = _generate(monkeypatch, backend, dimer_pdb, 6, _scale_chain(True, 1 + 1e-9))
    lean_b, _ = _generate(monkeypatch, backend, dimer_pdb, 6,
                          _scale_chain(False, 1 + 1e-9), first.solution)
    assert _worst(lean_a, lean_b) < 1e-5

    # Control: with the tolerance set to zero the rule is a plain argmax, and
    # the same two solutions give ensembles that are visibly different.
    monkeypatch.setattr(nma_backend, "_SIGN_TIE_RTOL", 0.0)
    plain_a, _ = _generate(monkeypatch, backend, dimer_pdb, 6,
                           _scale_chain(True, 1 + 1e-9), first.solution)
    plain_b, _ = _generate(monkeypatch, backend, dimer_pdb, 6,
                           _scale_chain(False, 1 + 1e-9), first.solution)
    assert _worst(plain_a, plain_b) > 1e-3


# ------------------------------------------------------------------ degenerate modes

def _fourfold(xyz):
    """(x, y, z) -> (-y, x, z): a quarter turn about z, exact in decimal and in binary."""
    x, y, z = xyz
    return (-y, x, z)


def _threefold(xyz):
    """(x, y, z) -> (y, z, x): a third of a turn about the (1, 1, 1) diagonal, exact."""
    x, y, z = xyz
    return (y, z, x)


def _assembly(tmp_path: Path, name: str, parts, cells, origin, spacing, operation,
              copies: int, residue, seed: int) -> Path:
    """One jittered chain and its images under an exact symmetry operation.

    The operation is applied to the rounded coordinates and only permutes or
    negates them, so the copies are symmetric to the last bit in the file and
    in float32, the way an assembly built from crystallographic operators
    along a cell axis is.
    """
    rng = np.random.default_rng(seed)
    chain = []
    for num, cell in enumerate(cells, start=1):
        centre = (np.asarray(origin) + spacing * np.array(cell, dtype=float)
                  + rng.uniform(-0.5, 0.5, 3))
        for atom_name, element, xyz in _residue(rng, centre, parts, 0.2):
            chain.append((atom_name, num, xyz, element))
    atoms = []
    for copy in range(copies):
        atoms += [(atom_name, residue(num), "ABCD"[copy], num, xyz, el)
                  for atom_name, num, xyz, el in chain]
        chain = [(atom_name, num, operation(xyz), el) for atom_name, num, xyz, el in chain]
    return _write_pdb(tmp_path / name, atoms)


@pytest.fixture
def tetramer_pdb(tmp_path: Path) -> Path:
    """Four chains about an exact four-fold axis along z (C4)."""
    cells = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    return _assembly(tmp_path, "tetramer.pdb", _AMINO[:3], cells, (2.5, 2.5, 0.0), 3.8,
                     _fourfold, 4, lambda num: "ALA", seed=5)


@pytest.fixture
def trimer_pdb(tmp_path: Path) -> Path:
    """Three chains about an exact three-fold axis along the body diagonal (C3)."""
    cells = [(i, j, k) for i in range(2) for j in range(2) for k in range(3)]
    return _assembly(tmp_path, "trimer.pdb", _AMINO[:3], cells, (5.0, -1.0, -4.0), 3.8,
                     _threefold, 3, lambda num: "ALA", seed=5)


@pytest.fixture
def rna_trimer_pdb(tmp_path: Path) -> Path:
    """Three RNA chains about an exact three-fold axis (C3), three beads a nucleotide."""
    cells = [(i, j, 0) for i in range(2) for j in range(2)]
    return _assembly(tmp_path, "rna_trimer.pdb", _NUCLEOTIDE, cells, (5.0, 0.0, -4.0), 6.0,
                     _threefold, 3, lambda num: "ACGU"[num % 4], seed=3)


_SYMMETRIC = {"tetramer": "tetramer_pdb", "trimer": "trimer_pdb", "rna_trimer": "rna_trimer_pdb"}


def _symmetric(request, kind):
    """(backend, path, float32 coordinates, bead indices) of a symmetric fixture."""
    backend = RNAAnmBackend() if kind.startswith("rna") else NMABackend()
    path = request.getfixturevalue(_SYMMETRIC[kind])
    base, index = _beads(backend, path)
    return backend, path, base, index


def _tied_runs(values, rtol=1e-9):
    """Inclusive (start, stop) runs of eigenvalues equal to ``rtol``, longer than one."""
    runs, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] - values[i - 1] > rtol * abs(values[i]):
            if i - 1 > start:
                runs.append((start, i - 1))
            start = i
    return runs


def _rotate_tied(seed: int, rtol: float = 1e-9):
    """What another thread count does to degenerate modes: the same space, another basis.

    Every run of eigenvalues equal to ``rtol`` gets its eigenvectors multiplied
    by a seeded random orthogonal matrix, reflections included.
    """
    def transform(values, vectors):
        rng = np.random.default_rng(seed)
        for start, stop in _tied_runs(values, rtol):
            q, _ = np.linalg.qr(rng.normal(size=(stop - start + 1,) * 2))
            vectors[:, start:stop + 1] = vectors[:, start:stop + 1] @ q
        return values, vectors
    return transform


def _assert_same_to_rounding(ensemble_a, ensemble_b) -> None:
    """Equal up to float64 rounding that tipped a float32 coordinate: one unit
    in the last place, or 1e-9 A for a coordinate that close to zero."""
    assert len(ensemble_a) == len(ensemble_b)
    for a, b in zip(ensemble_a, ensemble_b):
        gap = np.abs(a.astype(np.float64) - b.astype(np.float64))
        ulp = np.spacing(np.maximum(np.abs(a), np.abs(b))).astype(np.float64)
        assert (gap <= np.maximum(ulp, 1e-9)).all(), "worst difference %.3g A" % gap.max()


def _random_subspace(rng, d: int, size: int) -> np.ndarray:
    """(d, size) orthonormal rows."""
    q, _ = np.linalg.qr(rng.normal(size=(size, d)))
    return q.T.copy()


@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_the_canonical_basis_is_an_orthonormal_basis_of_the_same_space(d):
    vectors = _random_subspace(np.random.default_rng(d), d, 60)
    basis = canonical_basis(vectors)
    assert basis.dtype == np.float64 and basis.shape == vectors.shape
    np.testing.assert_allclose(basis @ basis.T, np.eye(d), atol=1e-13)
    np.testing.assert_allclose(basis.T @ basis, vectors.T @ vectors, atol=1e-13)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_the_canonical_basis_does_not_depend_on_the_basis_it_is_given(d):
    """The whole point: any basis of the space goes in, one basis comes out."""
    rng = np.random.default_rng(100 + d)
    vectors = _random_subspace(rng, d, 60)
    expected = canonical_basis(vectors)
    for _ in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        np.testing.assert_allclose(canonical_basis(q @ vectors), expected, atol=1e-13)
    # The input is not modified.
    np.testing.assert_array_equal(vectors, _random_subspace(np.random.default_rng(100 + d), d, 60))


def test_each_canonical_vector_is_a_projected_axis_and_positive_on_it():
    """Pivoted Cholesky of the projector, stated the slow way."""
    vectors = _random_subspace(np.random.default_rng(8), 3, 40)
    basis = canonical_basis(vectors)
    projector = vectors.T @ vectors
    for vector in basis:
        pivot = int(np.argmax(np.diag(projector)))
        expected = projector[pivot] / np.sqrt(projector[pivot, pivot])
        np.testing.assert_allclose(vector, expected, atol=1e-13)
        assert vector[pivot] > 0
        projector = projector - np.outer(expected, expected)


def test_empty_and_single_vector_inputs_to_the_canonical_basis():
    assert canonical_basis(np.zeros((0, 12))).shape == (0, 12)
    vector = _random_subspace(np.random.default_rng(2), 1, 12)
    basis = canonical_basis(-vector)
    np.testing.assert_allclose(np.abs(basis), np.abs(vector), atol=1e-15)
    assert basis[0, np.abs(vector[0]).argmax()] > 0


def test_tied_eigenvalues_are_grouped_into_runs(monkeypatch):
    monkeypatch.setattr(nma_backend, "_DEGENERACY_RTOL", 1e-4)
    values = np.array([0.0, 1.0, 1.0 + 5e-5, 1.0 + 1e-4, 2.0, 3.0, 3.0, 4.0])
    runs = nma_backend._degenerate_runs
    assert runs(values, 1, 5) == [(1, 3), (4, 4), (5, 6)]
    # A run is reported whole wherever the range cuts it, at either end.
    assert runs(values, 2, 5) == [(1, 3), (4, 4), (5, 6)]
    assert runs(values, 4, 4) == [(4, 4)]
    # Neighbours 2e-4 apart are not tied, and a chain of ties is one run.
    assert runs(np.array([1.0, 1.0002, 1.0004]), 0, 2) == [(0, 0), (1, 1), (2, 2)]
    assert runs(np.array([1.0, 1.00009, 1.00018]), 0, 0) == [(0, 2)]


@pytest.mark.parametrize("kind", sorted(_SYMMETRIC))
def test_the_symmetric_fixtures_really_have_degenerate_modes(request, kind):
    """What makes the fixtures a test of anything: a rigid network whose
    sampled modes include pairs with one eigenvalue, to rounding."""
    backend, _, base, index = _symmetric(request, kind)
    values = scipy.linalg.eigvalsh(backend._hessian(base[index]))
    assert int((np.abs(values) < 1e-9 * values[-1]).sum()) == 6, "fixture is not rigid"
    sampled = values[6:6 + backend.n_modes + 1]
    gaps = np.diff(sampled) / sampled[1:]
    assert (gaps < 1e-9).sum() >= 2, "fixture has no degenerate pairs"
    assert gaps[gaps >= 1e-9].min() > 1e-3, "fixture has a gap that is neither"


@pytest.mark.parametrize("kind", sorted(_SYMMETRIC))
def test_conformers_do_not_depend_on_the_basis_eigh_returns_for_degenerate_modes(
        monkeypatch, request, kind):
    """A three-fold or four-fold assembly, solved once, sampled from that
    solution and from the same solution with every degenerate pair turned to
    another basis. The float64 arithmetic differs by rounding only, so the
    conformers agree to a float32 unit in the last place."""
    backend, path, _, _ = _symmetric(request, kind)
    plain, first = _generate(monkeypatch, backend, path, 6)
    assert _tied_runs(first.solution[0][6:16]), "nothing to rotate"
    for seed in (1, 2, 3):
        turned, _ = _generate(monkeypatch, backend, path, 6, _rotate_tied(seed), first.solution)
        _assert_same_to_rounding(plain, turned)

    # Control: with degenerate sets left as the solver returned them, the same
    # rotation gives a visibly different ensemble.
    monkeypatch.setattr(nma_backend, "_DEGENERACY_RTOL", -1.0)
    plain, _ = _generate(monkeypatch, backend, path, 6, None, first.solution)
    turned, _ = _generate(monkeypatch, backend, path, 6, _rotate_tied(1), first.solution)
    assert _worst(plain, turned) > 1e-3


def test_degenerate_modes_are_still_normal_modes_of_the_network(request):
    """The canonical basis replaces eigenvectors by other eigenvectors: every
    returned mode satisfies H v = lambda v, the modes are orthonormal, and the
    frequencies are the solver's."""
    backend, _, base, index = _symmetric(request, "tetramer")
    beads = base[index].astype(np.float64)
    modes, freqs = backend._compute_modes(beads)
    hessian = backend._hessian(beads)
    np.testing.assert_allclose(modes @ modes.T, np.eye(len(modes)), atol=1e-12)
    np.testing.assert_allclose(modes @ hessian, freqs[:, None] * modes, atol=1e-11)
    np.testing.assert_allclose(freqs, scipy.linalg.eigvalsh(hessian)[6:6 + len(freqs)],
                               rtol=1e-10)
    np.testing.assert_array_equal(modes, canonical_sign(modes))


def test_the_tie_rule_holds_inside_a_degenerate_set(monkeypatch, request):
    """Under a four-fold axis the projector onto a degenerate pair has the
    same diagonal entry on four symmetry-related coordinates. Lean the
    solver's vectors towards one chain and then another, by far more than
    rounding and far less than the tolerance: the basis must not move. With
    the tolerance at zero the pivot follows the lean and the basis jumps."""
    backend, path, _, _ = _symmetric(request, "tetramer")

    def lean(chain):
        def transform(values, vectors):
            quarter = vectors.shape[0] // 4
            vectors[chain * quarter:(chain + 1) * quarter] *= 1 + 1e-9
            return values, vectors
        return transform

    first, stand_in = _generate(monkeypatch, backend, path, 6, lean(0))
    for chain in (1, 2, 3):
        other, _ = _generate(monkeypatch, backend, path, 6, lean(chain), stand_in.solution)
        assert _worst(first, other) < 1e-5

    monkeypatch.setattr(nma_backend, "_SIGN_TIE_RTOL", 0.0)
    ensembles = [_generate(monkeypatch, backend, path, 6, lean(chain), stand_in.solution)[0]
                 for chain in range(4)]
    assert max(_worst(ensembles[0], other) for other in ensembles[1:]) > 1e-3


def test_a_degenerate_set_cut_by_n_modes_is_cut_the_same_way_every_time(monkeypatch, request):
    """The trimer's tenth and eleventh modes are a degenerate pair, so the
    default ten modes take one vector out of a two-dimensional space. Which
    one must not depend on the solver's basis either."""
    backend, path, base, index = _symmetric(request, "trimer")
    values = scipy.linalg.eigvalsh(backend._hessian(base[index]))
    assert abs(values[16] - values[15]) < 1e-12 * values[16], "fixture: modes 10 and 11 not a pair"
    assert values[15] - values[14] > 1e-3 * values[15]

    stand_in = _ReplayEigh()
    monkeypatch.setattr(nma_backend, "eigh", stand_in)
    modes, _ = backend._compute_modes(base[index])
    for seed in (1, 2, 3):
        stand_in.transform = _rotate_tied(seed)
        turned, _ = backend._compute_modes(base[index])
        np.testing.assert_allclose(turned, modes, atol=1e-12)
    # The vector kept lies in the pair's space.
    pair = stand_in.solution[1][:, 15:17]
    np.testing.assert_allclose(pair @ (pair.T @ modes[-1]), modes[-1], atol=1e-12)


def test_broken_symmetry_is_still_one_degenerate_set(monkeypatch, request):
    """Symmetry that is exact in the molecule is approximate in a file. Noise
    of 1e-7 A splits the pairs by far less than the tolerance, the solver's
    vectors inside each pair are then decided by its rounding, and the
    ensemble must not follow them."""
    backend, _, base, index = _symmetric(request, "tetramer")
    noisy = base.astype(np.float64) + 1e-7 * np.random.default_rng(4).normal(size=base.shape)
    values = scipy.linalg.eigvalsh(backend._hessian(noisy[index]))[6:17]
    gaps = np.diff(values) / values[1:]
    assert ((gaps > 1e-12) & (gaps < 1e-6)).sum() >= 2, "noise did not split the pairs"

    stand_in = _ReplayEigh()
    monkeypatch.setattr(nma_backend, "eigh", stand_in)
    plain = backend._sample_ensemble(noisy, noisy[index], 6)
    stand_in.transform = _rotate_tied(7, rtol=1e-6)
    turned = backend._sample_ensemble(noisy, noisy[index], 6)
    _assert_same_to_rounding(plain, turned)

    # Control: left to the solver, the same two solutions are angstroms apart.
    monkeypatch.setattr(nma_backend, "_DEGENERACY_RTOL", -1.0)
    stand_in.transform = lambda values, vectors: (values, vectors)
    plain = backend._sample_ensemble(noisy, noisy[index], 6)
    stand_in.transform = _rotate_tied(7, rtol=1e-6)
    turned = backend._sample_ensemble(noisy, noisy[index], 6)
    assert _worst(plain, turned) > 1e-3


def test_modes_without_a_tie_are_left_to_the_solver(monkeypatch, protein_pdb, dimer_pdb):
    """No degenerate set, no canonical basis: an ordinary structure, and one
    with a two-fold axis, get the solver's own eigenvectors with the sign
    fixed, exactly as before degenerate sets were handled."""
    def refuse(vectors):
        raise AssertionError("canonical_basis called on a structure with no degenerate modes")

    for path in (protein_pdb, dimer_pdb):
        backend = NMABackend()
        base, index = _beads(backend, path)
        with_rule, stand_in = _generate(monkeypatch, backend, path, 4)
        monkeypatch.setattr(nma_backend, "canonical_basis", refuse)
        monkeypatch.setattr(nma_backend, "_DEGENERACY_RTOL", -1.0)
        without, _ = _generate(monkeypatch, backend, path, 4, None, stand_in.solution)
        assert _same(with_rule, without)
        assert len(stand_in.inputs) == 1, "one eigensolve per call"
        monkeypatch.undo()


def _centred_pentagon() -> np.ndarray:
    """Six beads in a plane: nine zero modes, then a degenerate pair."""
    angle = 2 * np.pi * np.arange(5) / 5
    ring = np.stack([5.0 * np.cos(angle), 5.0 * np.sin(angle), np.zeros(5)], axis=1)
    return np.concatenate([np.zeros((1, 3)), ring])


class _CountingEigh:
    """The real solver, with every requested window recorded and tied sets rotated."""

    def __init__(self, seed=None):
        self.windows = []
        self.transform = _rotate_tied(seed) if seed is not None else None

    def __call__(self, a, subset_by_index=None, **kwargs):
        self.windows.append(tuple(subset_by_index))
        values, vectors = scipy.linalg.eigh(a, subset_by_index=subset_by_index, **kwargs)
        if self.transform is not None:
            values, vectors = self.transform(values, vectors)
        return values, vectors


def test_a_degenerate_set_at_the_edge_of_the_window_is_fetched_whole(monkeypatch):
    """With one mode asked for, the window is ten eigenvalues: nine zero modes
    and the first of a degenerate pair. Its partner is outside. The solve is
    repeated with a wider window, and the mode returned is then the same
    whatever basis the solver gives the pair."""
    beads = _centred_pentagon()
    backend = NMABackend(n_modes=1)
    values = scipy.linalg.eigvalsh(backend._hessian(beads))
    assert int((np.abs(values) < 1e-9 * values[-1]).sum()) == 9
    assert abs(values[10] - values[9]) < 1e-12 * values[10] < values[11] - values[10]

    counting = _CountingEigh()
    monkeypatch.setattr(nma_backend, "eigh", counting)
    modes, freqs = backend._compute_modes(beads)
    assert counting.windows == [(0, 9), (0, 17)]
    assert modes.shape == (1, 18) and freqs.shape == (1,)
    for seed in (1, 2, 3):
        monkeypatch.setattr(nma_backend, "eigh", _CountingEigh(seed))
        turned, _ = backend._compute_modes(beads)
        np.testing.assert_allclose(turned, modes, atol=1e-12)

    # Control: with the pair left to the solver, its basis comes straight through.
    monkeypatch.setattr(nma_backend, "_DEGENERACY_RTOL", -1.0)
    results = []
    for seed in (1, 2, 3):
        monkeypatch.setattr(nma_backend, "eigh", _CountingEigh(seed))
        results.append(backend._compute_modes(beads)[0])
    assert max(np.abs(r - results[0]).max() for r in results[1:]) > 1e-3


def test_a_wider_window_does_not_change_which_modes_are_chosen(monkeypatch):
    """Zero modes are told from vibrations relative to the largest eigenvalue
    of the original window. If the second, wider solve fed that filter, asking
    for more eigenvalues would raise the bar and could drop a soft mode. Here
    the eigenvalues beyond the original window are made a million times
    larger: the mode chosen and its frequency must not notice."""
    beads = _centred_pentagon()
    backend = NMABackend(n_modes=1)
    modes, freqs = backend._compute_modes(beads)

    def stiff_beyond_the_window(a, subset_by_index=None, **kwargs):
        values, vectors = scipy.linalg.eigh(a, subset_by_index=subset_by_index, **kwargs)
        values[11:] *= 1e6            # past the window (10) and the pair's partner
        return values, vectors

    monkeypatch.setattr(nma_backend, "eigh", stiff_beyond_the_window)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stiff_modes, stiff_freqs = backend._compute_modes(beads)
    np.testing.assert_allclose(stiff_modes, modes, atol=1e-12)
    np.testing.assert_allclose(stiff_freqs, freqs, rtol=1e-12)


# ------------------------------------------------------------------ what cannot be fixed

def test_a_network_with_no_vibration_in_the_window_warns(tmp_path):
    """Thirty pairs of beads far apart from each other: 30 springs, 150 zero
    modes. The 46 lowest eigenvalues are all zero, there is nothing to sample,
    and the caller is told so instead of being handed rigid-body noise."""
    rng = np.random.default_rng(1)
    beads = []
    for k in range(30):
        centre = 40.0 * np.array([k % 5, (k // 5) % 3, k // 15], dtype=float)
        beads += [centre, centre + rng.normal(size=3) * 2.0 + np.array([3.0, 0.0, 0.0])]
    backend = NMABackend()
    with pytest.warns(RuntimeWarning, match="under-constrained"):
        modes, freqs = backend._compute_modes(np.array(beads))
    assert modes.shape == (backend.n_modes, 180) and (freqs > 0).all()

    # No springs at all is the same case.
    with pytest.warns(RuntimeWarning, match="under-constrained"):
        backend._compute_modes(100.0 * np.arange(8)[:, None] * np.ones(3))

    # So is a straight rod, whose only vibrations are along its length. Its
    # zero modes come out as rounding noise of either sign, and the positive
    # ones pass the relative filter for vibrations: it is the scale of the
    # Hessian that gives them away.
    rod = 3.8 * np.arange(60)[:, None] * np.array([0.36, 0.48, 0.8])
    window = scipy.linalg.eigh(backend._hessian(rod), subset_by_index=[0, 45], eigvals_only=True)
    assert (window > np.abs(window).max() * 1e-3).any() and np.abs(window).max() < 1e-12
    with pytest.warns(RuntimeWarning, match="under-constrained"):
        backend._compute_modes(rod)


@pytest.mark.parametrize("kind", ["protein", "dimer", "rna", "tetramer", "trimer", "rna_trimer"])
def test_an_ordinary_structure_does_not_warn(request, kind):
    if kind in _SYMMETRIC:
        backend, path, _, _ = _symmetric(request, kind)
    else:
        backend, path = _backend(kind), request.getfixturevalue(kind + "_pdb")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.generate(path, 2)


# ------------------------------------------------------------------ what did not change

@pytest.mark.parametrize("kind", ["protein", "dimer", "rna"])
def test_output_dtype_and_shapes_are_unchanged(request, kind):
    path = request.getfixturevalue(kind + "_pdb")
    backend = _backend(kind)
    base, index = _beads(backend, path)

    conformers = backend.generate(path, 4)
    assert isinstance(conformers, list) and len(conformers) == 4
    for conformer in conformers:
        assert isinstance(conformer, np.ndarray)
        assert conformer.dtype == np.float32
        assert conformer.shape == base.shape == (len(base), 3)
        assert np.isfinite(conformer).all()

    modes, freqs = backend._compute_modes(base[index])
    assert modes.shape == (backend.n_modes, 3 * len(index))
    assert freqs.shape == (backend.n_modes,)
    assert (freqs > 0).all() and (np.diff(freqs) >= 0).all()
    displaced = backend._sample_ca(base[index], modes, freqs, 1.5, seed=3)
    assert displaced.shape == (len(index), 3)
    # The amplitude is still the bead RMSD of the displacement.
    rmsd = np.sqrt(((displaced - base[index]) ** 2).sum(axis=1).mean())
    assert rmsd == pytest.approx(1.5, rel=1e-6)
    assert backend._propagate(base, base[index], displaced).shape == base.shape
    assert backend.generate(path, 0) == []


@pytest.mark.parametrize("kind", ["protein", "rna"])
def test_the_hooks_called_one_by_one_give_the_same_ensemble(monkeypatch, request, kind):
    """A subclass that runs its own loop over the three hooks, as the RNA
    backend used to and as analysis code does, gets the conformers ``generate``
    returns, bit for bit, whether it passes float32 or float64 arrays."""
    path = request.getfixturevalue(kind + "_pdb")
    backend = _backend(kind)
    base, index = _beads(backend, path)
    n = 5
    generated, stand_in = _generate(monkeypatch, backend, path, n)

    for dtype in (np.float32, np.float64):
        coords, beads = base.astype(dtype), base[index].astype(dtype)
        modes, freqs = backend._compute_modes(beads)
        by_hand = []
        for i in range(n):
            amplitude = (i + 1) / n * backend.max_rmsd
            displaced = backend._sample_ca(beads, modes, freqs, amplitude, seed=i)
            by_hand.append(backend._propagate(coords, beads, displaced))
        assert _same(generated, by_hand), dtype
    assert len(stand_in.inputs) == 3


def test_propagation_weights_are_built_once_per_generate(monkeypatch, protein_pdb):
    """They depend on the input structure only. They used to be rebuilt, at
    (N_atoms, N_beads, 3) a time, for every conformer."""
    backend = NMABackend()
    calls = []
    real = backend._propagation_weights

    def counted(base_coords, ca_coords):
        calls.append((len(base_coords), len(ca_coords)))
        return real(base_coords, ca_coords)

    monkeypatch.setattr(backend, "_propagation_weights", counted)
    backend.generate(protein_pdb, 7)
    assert calls == [(108, 27)]
    assert backend._weights_memo is None, "the weights must not outlive the call"

    backend.generate(protein_pdb, 0)
    assert len(calls) == 1, "no conformers, so no weights"


def test_the_weights_are_dropped_when_sampling_fails(monkeypatch, protein_pdb):
    backend = NMABackend()

    def broken(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(backend, "_sample_ca", broken)
    with pytest.raises(RuntimeError):
        backend.generate(protein_pdb, 3)
    assert backend._weights_memo is None


def test_a_replaced_propagate_hook_still_takes_effect(protein_pdb):
    """The weights travel in a memo rather than a new argument so that the
    hooks keep their signatures. A subclass written against the old ones must
    still be called, once per conformer, and must not pay for weights it
    never asks for."""
    seen = []

    class Frozen(NMABackend):
        def _propagate(self, base_coords, ca_coords, ca_displaced):
            seen.append(ca_displaced.shape)
            return np.asarray(base_coords, dtype=np.float32)

        def _propagation_weights(self, base_coords, ca_coords):
            raise AssertionError("weights computed for a hook that does not use them")

    base, index = _beads(NMABackend(), protein_pdb)
    conformers = Frozen().generate(protein_pdb, 4)
    assert seen == [(len(index), 3)] * 4
    assert all(np.array_equal(c, base) for c in conformers)


def test_the_slab_size_does_not_change_the_ensemble(monkeypatch, protein_pdb):
    backend = NMABackend()
    default, first = _generate(monkeypatch, backend, protein_pdb, 5)
    monkeypatch.setattr(nma_backend, "_BLOCK_ELEMENTS", 64)
    small, _ = _generate(monkeypatch, backend, protein_pdb, 5, None, first.solution)
    assert _same(default, small)
    base, index = _beads(backend, protein_pdb)
    slabbed = backend._propagation_weights(base, base[index])
    monkeypatch.undo()
    np.testing.assert_array_equal(slabbed, backend._propagation_weights(base, base[index]))


# ------------------------------------------------------------------ the thread count itself

_CHILD = """
import pathlib, sys
tree, pdb, out = sys.argv[1:4]
sys.path.insert(0, tree)
import numpy as np
import lacuna
assert pathlib.Path(tree) in pathlib.Path(lacuna.__file__).resolve().parents, lacuna.__file__
from lacuna.ensemble.nma_backend import NMABackend
np.save(out, np.stack(NMABackend().generate(pdb, 5)))
"""


def _ensemble_at(threads: int, pdb: Path, out: Path):
    """Start the sampler in a fresh interpreter whose BLAS is capped at ``threads``.

    The child imports lacuna from the tree this test module imported it from,
    not from whatever happens to be installed.
    """
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = str(threads)
    tree = str(Path(lacuna.__file__).resolve().parents[1])
    return subprocess.Popen([sys.executable, "-c", _CHILD, tree, str(pdb), str(out)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _assert_one_thread_and_four_agree(pdb: Path, tmp_path: Path) -> None:
    outs = [tmp_path / "t1.npy", tmp_path / "t4.npy"]
    children = [_ensemble_at(t, pdb, out) for t, out in zip((1, 4), outs)]
    for child in children:
        log = child.communicate(timeout=300)[0].decode(errors="replace")
        assert child.returncode == 0, log
    one, four = (np.load(out) for out in outs)

    assert one.dtype == four.dtype == np.float32 and one.shape == four.shape
    gap = np.abs(one.astype(np.float64) - four.astype(np.float64))
    ulp = np.spacing(np.maximum(np.abs(one), np.abs(four))).astype(np.float64)
    assert (gap <= np.maximum(ulp, 1e-9)).all(), "worst difference %.3g A" % gap.max()


@pytest.mark.skipif(not KRAS.exists(), reason="KRAS_G12D.pdb not available")
def test_one_blas_thread_and_four_give_the_same_ensemble(tmp_path):
    """The defect itself, run for real: the same structure at two thread counts.

    The unfixed sampler fails this by angstroms wherever BLAS threads a matrix
    this size. The fixed one may still differ in the last float32 bit of an
    occasional coordinate, which is the solver's float64 rounding (of the
    order of 1e-11 A) crossing a float32 rounding boundary, so one unit in the
    last place is allowed. So is 1e-9 A outright, for a coordinate so close to
    zero that the float32 grid is finer than the float64 rounding. Nothing
    more. On a BLAS that ignores these variables the two runs are the same run
    and this passes without testing anything.
    """
    _assert_one_thread_and_four_agree(KRAS, tmp_path)


def _kras_tetramer(path: Path) -> Path:
    """KRAS and three images of it about an exact four-fold axis along z.

    The chain is moved off the axis far enough to leave a pore and near enough
    for neighbours to touch, rounded to the three decimals of the file, and
    then turned by (x, y, z) -> (-y, x, z), which is exact. This is what
    ``--homodimer`` produces from a crystal whose four-fold lies along a cell
    axis, and its modes include degenerate pairs.
    """
    lines = [line for line in KRAS.read_text().splitlines() if line.startswith("ATOM")]
    xyz = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    for line in lines])
    xyz = np.round(xyz - xyz.mean(axis=0) + np.array([24.0, 0.0, 0.0]), 3)
    out = []
    for copy in range(4):
        out += ["%s%s%s%8.3f%8.3f%8.3f%s" % (line[:21], "ABCD"[copy], line[22:30], x, y, z,
                                             line[54:])
                for line, (x, y, z) in zip(lines, xyz)]
        xyz = np.stack([-xyz[:, 1], xyz[:, 0], xyz[:, 2]], axis=1)
    path.write_text("\n".join(out) + "\nEND\n")
    return path


@pytest.mark.skipif(not KRAS.exists(), reason="KRAS_G12D.pdb not available")
def test_one_blas_thread_and_four_give_the_same_ensemble_of_a_fourfold_assembly(tmp_path):
    """The same, on a structure with an exact four-fold axis.

    Here the sign rule and double precision are not enough: the solver returns
    a different basis of every degenerate pair at each thread count, and on
    the machine this was written on the ensemble moved by 2.7 A between one
    thread and four until degenerate sets were given a canonical basis.
    """
    pdb = _kras_tetramer(tmp_path / "kras_c4.pdb")
    backend = NMABackend()
    base, index = _beads(backend, pdb)
    values = scipy.linalg.eigvalsh(backend._hessian(base[index]),
                                   subset_by_index=[0, 6 + backend.n_modes])
    assert int((np.abs(values) < 1e-9 * values[-1]).sum()) == 6, "assembly is not one rigid network"
    assert _tied_runs(values[6:]), "assembly has no degenerate modes among the sampled ones"
    _assert_one_thread_and_four_agree(pdb, tmp_path)
