"""Tests for the ensemble sanity guard.

Every quantity Lacuna reports is relative to the input structure, so an ensemble
that stops resembling it makes the whole protein read as cryptic while the
report still looks well formed. These cover the four ways that has actually
happened in this project, plus the requirement that the guard stay silent on a
healthy ensemble, which is what makes it worth having.
"""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.ensemble.sanity import (
    MAX_DRIFT,
    MAX_OFFSET,
    check_ensemble,
    representative_indices,
)
from lacuna.io.structure import coords_array, load_structure

_ATOM = ("ATOM  {serial:>5}  {name:<3} {res} A{seq:>4}    "
         "{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C")


def _globule(n: int = 60, seed: int = 0) -> np.ndarray:
    """A compact, non-collinear blob with realistic CA spacing."""
    rng = np.random.default_rng(seed)
    xyz = [np.zeros(3)]
    for _ in range(n - 1):
        step = rng.normal(size=3)
        step /= np.linalg.norm(step)
        nxt = xyz[-1] + 3.8 * step
        # Keep it folded rather than letting the walk run away.
        if np.linalg.norm(nxt) > 12.0:
            nxt = xyz[-1] - 3.8 * step
        xyz.append(nxt)
    return np.array(xyz)


@pytest.fixture
def protein(tmp_path):
    """A CA-only structure and its coordinates."""
    xyz = _globule()
    lines = [_ATOM.format(serial=i + 1, name="CA", res="ALA", seq=i + 1,
                          x=p[0], y=p[1], z=p[2]) for i, p in enumerate(xyz)]
    p = tmp_path / "prot.pdb"
    p.write_text("\n".join(lines) + "\nEND\n")
    st = load_structure(p)
    return st, coords_array(st)


def _breathe(base, k, scale=0.25):
    """A small, shape-changing perturbation: a healthy ensemble."""
    rng = np.random.default_rng(100 + k)
    return np.asarray(base, float) + rng.normal(scale=scale, size=np.shape(base))


def _rigid(xyz, seed):
    rng = np.random.default_rng(seed)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = rng.uniform(0.5, 2.5)
    k = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    rot = np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)
    return np.asarray(xyz, float) @ rot.T + rng.uniform(-60.0, 60.0, size=3)


class TestHealthyEnsemble:
    def test_small_motions_are_not_reported(self, protein):
        """The guard is only useful if it stays quiet on a good ensemble."""
        st, base = protein
        h = check_ensemble(st, [_breathe(base, k) for k in range(10)])
        assert h.ok, h.messages
        assert h.drift < MAX_DRIFT
        # A random perturbation moves the centroid a little, so the offset is
        # small but not zero. What matters is that it stays far under the bar.
        assert h.offset < MAX_OFFSET / 10
        assert h.n_conformers == 10

    def test_summary_is_reportable(self, protein):
        st, base = protein
        h = check_ensemble(st, [_breathe(base, k) for k in range(4)])
        s = h.summary()
        assert "conformers" in s and "drift" in s

    def test_empty_and_single_conformer_do_not_crash(self, protein):
        st, base = protein
        assert check_ensemble(st, []).ok
        # One conformer has no pairwise spread, which must not read as collapse.
        assert check_ensemble(st, [_breathe(base, 1)]).ok


class TestDrift:
    def test_an_ensemble_far_from_the_input_is_reported(self, protein):
        """Unfolding ensemble: 26 A from a structure whose radius of gyration is 18."""
        st, base = protein
        rng = np.random.default_rng(7)
        wandered = [np.asarray(base, float) + rng.normal(scale=8.0, size=np.shape(base))
                    for _ in range(6)]
        h = check_ensemble(st, wandered)
        assert not h.ok
        assert h.drift > MAX_DRIFT
        assert any("radii from the input" in m for m in h.messages)


class TestOffset:
    def test_a_correctly_shaped_frame_in_the_wrong_place_is_reported(self, protein):
        """The external backend's old state: right shape, arbitrary frame.

        Drift is measured after superposition and so cannot see this, which is
        the point of measuring placement separately.
        """
        st, base = protein
        good = [_breathe(base, k) for k in range(8)]
        moved = [_rigid(c, i) for i, c in enumerate(good)]

        healthy = check_ensemble(st, good)
        h = check_ensemble(st, moved)

        assert not h.ok
        assert h.offset > MAX_OFFSET
        assert any("not in the input's frame" in m for m in h.messages)
        # Shape is unchanged, so drift must be unaffected by the displacement.
        assert h.drift == pytest.approx(healthy.drift, rel=1e-6)


class TestRadiusOfGyration:
    def test_an_expanding_chain_is_reported(self, protein):
        st, base = protein
        blown = [np.asarray(base, float) * 1.6 for _ in range(5)]
        h = check_ensemble(st, blown)
        assert not h.ok
        assert h.rg_max > 1.2
        assert any("expanding rather than breathing" in m for m in h.messages)

    def test_a_collapsing_chain_is_reported(self, protein):
        st, base = protein
        squashed = [np.asarray(base, float) * 0.5 for _ in range(5)]
        h = check_ensemble(st, squashed)
        assert not h.ok
        assert h.rg_min < 0.85
        assert any("collapsing rather than breathing" in m for m in h.messages)


class TestCollapse:
    def test_duplicate_conformers_are_reported(self, protein):
        """Mode collapse: every pocket is present in every conformer, so
        nothing can register as cryptic."""
        st, base = protein
        h = check_ensemble(st, [np.array(base, float) for _ in range(5)])
        assert not h.ok
        assert any("near-duplicates" in m for m in h.messages)


class TestRepresentativeAtoms:
    def test_protein_uses_ca(self, protein):
        st, _ = protein
        idx = representative_indices(st)
        assert len(idx) == 60
        assert all(st.atoms[i].name == "CA" for i in idx)

    def test_nucleic_falls_back_to_c1_prime(self, tmp_path):
        """Nucleic input has no CA, and the elastic network uses C1' beads."""
        xyz = _globule(20, seed=3)
        lines = []
        for i, p in enumerate(xyz):
            lines.append(_ATOM.format(serial=2 * i + 1, name="P", res="  A",
                                      seq=i + 1, x=p[0], y=p[1], z=p[2]))
            lines.append(_ATOM.format(serial=2 * i + 2, name="C1'", res="  A",
                                      seq=i + 1, x=p[0] + 1.0, y=p[1], z=p[2]))
        f = tmp_path / "rna.pdb"
        f.write_text("\n".join(lines) + "\nEND\n")
        st = load_structure(f)
        idx = representative_indices(st)
        assert len(idx) == 20
        assert all(st.atoms[i].name == "C1'" for i in idx)


class TestMixedPolymerRepresentatives:
    """A nucleic structure with a small bound protein must measure both.

    The first version preferred CA whenever at least ``_MIN_ATOMS`` of them
    existed, so a 71-nucleotide riboswitch beside a 6-residue peptide was judged
    entirely on the peptide. That failed in both directions on the same input:
    an unmoved ensemble warned, and one with the RNA exploded by 12 A per atom
    stayed silent and reported only that the sampler had collapsed.
    """

    @staticmethod
    def _mixed(tmp_path, n_nt: int = 40, n_res: int = 6):
        """Nucleic chain A, plus a small protein chain P standing 60 A away."""
        rna = _globule(n_nt, seed=11)
        pep = _globule(n_res, seed=12) + np.array([60.0, 0.0, 0.0])
        lines, serial = [], 1
        for i, p in enumerate(rna):
            for nm in ("P", "C1'"):
                lines.append(_ATOM.format(serial=serial, name=nm, res="  A",
                                          seq=i + 1, x=p[0], y=p[1], z=p[2]))
                serial += 1
        for i, p in enumerate(pep):
            # Chain is fixed to A by the _ATOM template; relabel the tail.
            for nm in ("N", "CA"):
                lines.append(_ATOM.format(serial=serial, name=nm, res="ALA",
                                          seq=500 + i, x=p[0], y=p[1], z=p[2])
                             .replace(" A 5", " P 5", 1))
                serial += 1
        f = tmp_path / "mixed.pdb"
        f.write_text("\n".join(lines) + "\nEND\n")
        return load_structure(f), n_nt, n_res

    def test_both_polymers_are_represented(self, tmp_path):
        st, n_nt, n_res = self._mixed(tmp_path)
        idx = representative_indices(st)
        names = [st.atoms[i].name for i in idx]
        assert len(idx) == n_nt + n_res
        assert names.count("C1'") == n_nt
        assert names.count("CA") == n_res

    def test_a_healthy_mixed_ensemble_stays_silent(self, tmp_path):
        """The false positive: ordinary motion in both polymers, nothing wrong.

        The peptide stands 60 A from the nucleic chain, so judging the ensemble
        on the peptide alone reads its small local motion as a rigid-body
        displacement of the whole structure.
        """
        st, _, _ = self._mixed(tmp_path)
        base = coords_array(st)
        rng = np.random.default_rng(7)
        sets = [base + rng.normal(scale=0.4, size=base.shape) for _ in range(5)]
        health = check_ensemble(st, sets)
        assert health.ok, health.messages

    def test_a_destroyed_nucleic_chain_is_caught(self, tmp_path):
        """The false negative: the RNA is wrecked while the peptide holds still.

        Measured on the peptide alone this looked like a collapsed sampler.
        """
        st, _, _ = self._mixed(tmp_path)
        base = coords_array(st)
        rna_atoms = [i for i, a in enumerate(st.atoms) if a.chain_id != "P"]
        rng = np.random.default_rng(0)
        sets = []
        for _ in range(5):
            c = base.copy()
            c[rna_atoms] += rng.normal(scale=12.0, size=(len(rna_atoms), 3))
            sets.append(c)
        health = check_ensemble(st, sets)
        assert not health.ok
        joined = " ".join(health.messages)
        assert "near-duplicates" not in joined, joined
