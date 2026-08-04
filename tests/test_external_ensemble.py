"""Tests for consuming an externally generated conformational ensemble.

The point of this backend is that Lacuna's own samplers are replaceable, so these
cover the cases real ensembles actually arrive in: a multi-model file, a directory
of files, and frames whose atoms do not line up with the reference.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lacuna.ensemble.external_backend import ExternalEnsembleBackend
from lacuna.io.structure import iter_model_coords, load_structure

_ATOM = ("ATOM  {serial:>5}  CA  {res} A{seq:>4}    "
         "{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C")


def _model(offset: float, n_res: int = 4, start: int = 1) -> str:
    lines = []
    for i in range(n_res):
        lines.append(_ATOM.format(serial=i + 1, res="ALA", seq=start + i,
                                  x=i * 3.8 + offset, y=0.0, z=0.0))
    return "\n".join(lines)


def _multi_model(offsets, n_res: int = 4, start: int = 1) -> str:
    out = []
    for i, off in enumerate(offsets, 1):
        out += [f"MODEL     {i:>4}", _model(off, n_res, start), "ENDMDL"]
    out.append("END")
    return "\n".join(out) + "\n"


@pytest.fixture
def reference(tmp_path):
    """A single-model reference structure and its path."""
    p = tmp_path / "ref.pdb"
    p.write_text(_model(0.0) + "\nEND\n")
    return p


class TestIterModelCoords:
    def test_yields_one_map_per_model(self, tmp_path):
        p = tmp_path / "ens.pdb"
        p.write_text(_multi_model([0.0, 0.5, 1.0]))
        frames = list(iter_model_coords(p))
        assert len(frames) == 3
        # Keyed on (chain, resseq, atom name), and the offset is visible.
        assert frames[0][("A", 1, "CA")][0] == pytest.approx(0.0)
        assert frames[2][("A", 1, "CA")][0] == pytest.approx(1.0)

    def test_load_structure_still_takes_only_the_first_model(self, tmp_path):
        """The single-conformation path must not change behaviour."""
        p = tmp_path / "ens.pdb"
        p.write_text(_multi_model([0.0, 9.0]))
        s = load_structure(p)
        assert len(s.residues) == 4
        assert s.atoms[0].coords[0] == pytest.approx(0.0)


class TestMultiModelFile:
    def test_reads_every_model(self, tmp_path, reference):
        ens = tmp_path / "ens.pdb"
        ens.write_text(_multi_model([0.0, 0.5, 1.0]))
        got = ExternalEnsembleBackend(ens).generate(reference, 0)
        assert len(got) == 3
        assert all(c.shape == (4, 3) for c in got)
        assert got[1][0][0] == pytest.approx(0.5)

    def test_n_conformers_caps_the_ensemble(self, tmp_path, reference):
        ens = tmp_path / "ens.pdb"
        ens.write_text(_multi_model([0.0, 0.5, 1.0, 1.5, 2.0]))
        assert len(ExternalEnsembleBackend(ens).generate(reference, 2)) == 2

    def test_coords_are_float32_in_reference_order(self, tmp_path, reference):
        ens = tmp_path / "ens.pdb"
        ens.write_text(_multi_model([0.25]))
        c = ExternalEnsembleBackend(ens).generate(reference, 0)[0]
        assert c.dtype == np.float32
        # Reference order is ascending residue number, so x increases by 3.8.
        assert c[3][0] - c[0][0] == pytest.approx(3.8 * 3, abs=1e-3)


class TestDirectory:
    def test_reads_each_file_as_a_conformer(self, tmp_path, reference):
        d = tmp_path / "frames"
        d.mkdir()
        for i, off in enumerate([0.0, 0.4, 0.8]):
            (d / f"frame_{i}.pdb").write_text(_model(off) + "\nEND\n")
        got = ExternalEnsembleBackend(d).generate(reference, 0)
        assert len(got) == 3

    def test_empty_directory_is_an_error(self, tmp_path, reference):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="no structure files"):
            ExternalEnsembleBackend(d).generate(reference, 0)

    def test_missing_source_is_an_error(self, tmp_path, reference):
        with pytest.raises(ValueError, match="not found"):
            ExternalEnsembleBackend(tmp_path / "nope.pdb").generate(reference, 0)


class TestAtomCorrespondence:
    def test_partial_frame_keeps_reference_coords_for_missing_atoms(
            self, tmp_path, reference):
        """A frame missing a residue should not shift every other atom."""
        ens = tmp_path / "partial.pdb"
        # 3 of the reference's 4 residues, all displaced by 1.0.
        ens.write_text(_multi_model([1.0], n_res=3))
        c = ExternalEnsembleBackend(ens).generate(reference, 0)[0]
        # Matched atoms moved; the unmatched one kept its reference position.
        assert c[0][0] == pytest.approx(1.0)
        assert c[2][0] == pytest.approx(2 * 3.8 + 1.0)
        assert c[3][0] == pytest.approx(3 * 3.8)

    def test_numbering_mismatch_warns_and_is_skipped(self, tmp_path, reference):
        """Wrong residue numbering is a mismatch, not a conformational change."""
        ens = tmp_path / "shifted.pdb"
        good = _multi_model([0.5])
        # Second file: same atoms renumbered from 500, so nothing matches.
        bad = _multi_model([0.5], start=500)
        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a_good.pdb").write_text(good)
        (d / "b_bad.pdb").write_text(bad)
        with pytest.warns(RuntimeWarning, match="numbering probably differs"):
            got = ExternalEnsembleBackend(d).generate(reference, 0)
        assert len(got) == 1  # only the matching frame survived

    def test_strict_raises_on_mismatch(self, tmp_path, reference):
        ens = tmp_path / "shifted.pdb"
        ens.write_text(_multi_model([0.5], start=500))
        with pytest.raises(ValueError, match="matched only"):
            ExternalEnsembleBackend(ens, strict=True).generate(reference, 0)

    def test_all_frames_unusable_is_an_error(self, tmp_path, reference):
        ens = tmp_path / "shifted.pdb"
        ens.write_text(_multi_model([0.5], start=500))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="no usable conformers"):
                ExternalEnsembleBackend(ens).generate(reference, 0)


class TestChainFiltering:
    def test_chain_argument_restricts_frames(self, tmp_path):
        two_chain = []
        for i in range(3):
            two_chain.append(_ATOM.format(serial=i + 1, res="ALA", seq=i + 1,
                                          x=i * 3.8, y=0.0, z=0.0))
        for i in range(3):
            two_chain.append(_ATOM.format(serial=i + 4, res="GLY", seq=i + 1,
                                          x=i * 3.8, y=20.0, z=0.0)
                             .replace(" A", " B", 1))
        text = "\n".join(two_chain) + "\nEND\n"
        ref = tmp_path / "dimer.pdb"
        ref.write_text(text)
        ens = tmp_path / "ens.pdb"
        ens.write_text(text)
        got = ExternalEnsembleBackend(ens, chain="A").generate(ref, 0, chain="A")
        assert got[0].shape[0] == 3  # chain A only
