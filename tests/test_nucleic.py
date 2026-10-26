# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Tests for nucleic-acid support.

The behaviour under test is mostly a bug fix. ``NMABackend`` builds its elastic
network from atoms named CA, RNA has none, so nucleic input fell through to
``RandomBackend`` and the run still printed a ranked table with crypticity and
persistence columns computed from Gaussian noise. Measured on 10 held-out
structures whose ligand site is shut in apo, that fallback recovered 0 of them
while stretching C1'-C1' spacing 25%.

So the first requirement is that ``auto`` never routes nucleic input to a
protein sampler, and the second is that the protein path is untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lacuna.cli import _auto_backend, _resolve_backend
from lacuna.io.structure import coords_array, load_structure
from lacuna.nucleic import (classify, heavy_atoms_only, is_nucleic, is_nucleotide,
                            RNAAnmBackend)
from lacuna.nucleic.chem import is_aromatic_base, is_purine, parent_base, ring_count

ROOT = Path(__file__).resolve().parent.parent
PROTEIN = ROOT / "benchmarks" / "pdb_cache" / "1JWP.pdb"
# RNA fixtures live in the staging repository rather than here, so these skip
# cleanly on a machine that only has the main checkout.
RNA_DIR = Path(r"C:/Users/clayt/Documents/GitHub/lacuna-rna/data/pdb")
XRAY_RNA = RNA_DIR / "6VUH.cif"      # 33 nt, single model
NMR_RNA = RNA_DIR / "1QC8.cif"       # 25 nt, 29 deposited models, hydrogens

needs = lambda p: pytest.mark.skipif(not p.exists(), reason=f"{p.name} not available")


# ------------------------------------------------------------------ chemistry

def test_standard_and_modified_nucleotides_are_recognised():
    for base in ("A", "C", "G", "U", "DA", "DC", "DG", "DT", "PSU", "5MC"):
        assert is_nucleotide(base), base


def test_amino_acids_are_not_nucleotides():
    for aa in ("ALA", "GLY", "TRP", "HIS", "PHE", "CYS"):
        assert not is_nucleotide(aa), aa


def test_all_four_bases_count_as_aromatic():
    """Lacuna's aromatic lookup is over the 20 amino acids, so every nucleotide
    answered False and pinned the term that should be highest at zero."""
    for base in ("A", "C", "G", "U"):
        assert is_aromatic_base(base)


def test_purines_have_two_rings_and_pyrimidines_one():
    assert ring_count("A") == ring_count("G") == 2
    assert ring_count("C") == ring_count("U") == 1
    assert is_purine("A") and is_purine("G")
    assert not (is_purine("C") or is_purine("U"))


def test_modified_bases_resolve_to_a_parent():
    assert parent_base("PSU") == "U"
    assert parent_base("5MC") == "C"
    assert parent_base("ALA") is None


def test_classify_separates_nucleic_protein_and_mixed():
    assert classify(["A", "U", "G", "C"])["kind"] == "nucleic"
    assert classify(["ALA", "GLY", "TRP"])["kind"] == "protein"
    assert classify(["A", "U", "ALA", "GLY"])["kind"] == "mixed"
    assert classify([])["kind"] == "empty"


# ------------------------------------------------------------------ routing

@needs(PROTEIN)
def test_auto_still_picks_a_protein_backend_for_protein():
    st = load_structure(PROTEIN)
    assert not is_nucleic(st)
    assert _auto_backend(st).name != "rna-anm"


@needs(XRAY_RNA)
def test_auto_routes_nucleic_input_to_the_rna_network():
    """The bug fix. Without this, auto reaches NMABackend, finds no CA atoms and
    silently returns RandomBackend."""
    st = heavy_atoms_only(load_structure(XRAY_RNA))
    assert is_nucleic(st)
    assert "rna-anm" in _auto_backend(st).name


def test_auto_without_a_structure_keeps_the_old_behaviour():
    """Called with no structure it must behave exactly as before, since other
    call sites and older code pass nothing."""
    assert "rna-anm" not in _auto_backend().name


def test_rna_backend_is_selectable_by_name():
    assert isinstance(_resolve_backend("rna-nma"), RNAAnmBackend)


def test_unknown_backend_still_raises():
    import click
    with pytest.raises(click.BadParameter):
        _resolve_backend("not-a-backend")


# ------------------------------------------------------------------ structure prep

@needs(NMR_RNA)
def test_hydrogens_are_dropped_from_nucleic_input():
    """6VUH reads 1062 atoms with hydrogens and 702 without, on the same 33
    residues, which changes the pocket count 2 to 1. Every nucleic benchmark was
    measured on heavy atoms."""
    raw = load_structure(NMR_RNA)
    if not any((a.element or "").upper() == "H" for a in raw.atoms):
        pytest.skip("no hydrogens in this file")
    clean = heavy_atoms_only(raw)
    assert len(clean.atoms) < len(raw.atoms)
    assert not any((a.element or "").upper() == "H" for a in clean.atoms)


@needs(PROTEIN)
def test_protein_structures_are_returned_untouched():
    """Protein benchmarks were measured on whatever the loader returns, so this
    must not quietly change them."""
    st = load_structure(PROTEIN)
    assert heavy_atoms_only(st) is st or len(heavy_atoms_only(st).atoms) == len(st.atoms)


@needs(NMR_RNA)
def test_atom_and_residue_indices_stay_consistent():
    """Residue.atom_indices points into the atom list by serial, so atoms and
    residues must be renumbered together or lining-residue lookups read the
    wrong atoms and nothing visibly fails."""
    st = heavy_atoms_only(load_structure(NMR_RNA))
    assert [a.serial for a in st.atoms] == list(range(len(st.atoms)))
    for res in st.residues:
        assert res.atom_indices
        assert all(0 <= i < len(st.atoms) for i in res.atom_indices)


# ------------------------------------------------------------------ the sampler

@needs(XRAY_RNA)
def test_the_network_is_built_on_nucleotide_beads():
    st = heavy_atoms_only(load_structure(XRAY_RNA))
    be = RNAAnmBackend()
    idx = be.bead_indices(st)
    assert idx, "no beads selected"
    names = {st.atoms[i].name for i in idx}
    assert names <= set(be.bead_names)


@needs(PROTEIN)
def test_the_network_refuses_protein_rather_than_falling_back():
    """Raising is the point. The shipped NMA backend returns a RandomBackend
    here and says nothing."""
    with pytest.raises(ValueError):
        RNAAnmBackend().generate(PROTEIN, 5)


@needs(XRAY_RNA)
def test_conformers_keep_the_backbone_intact_at_the_default_amplitude():
    """Judged as percent stretch against this structure's own resting spacing:
    6VUH rests at 6.32 A where 1Y26 rests at 5.79, so an absolute threshold
    calibrated on one is nearly tripped at rest by the other. Past about 5% a
    pocket has been torn open rather than opened."""
    import tempfile

    from lacuna.io.writers import write_structure_pdb

    st = heavy_atoms_only(load_structure(XRAY_RNA))
    X = coords_array(st)
    c1 = [a.serial for a in st.atoms if a.name == "C1'"]
    base = float(np.mean(np.linalg.norm(np.diff(X[c1], axis=0), axis=1)))
    # Sample from the PREPARED structure, as the CLI does. Generating from the
    # original file while scoring against a hydrogen-stripped Structure hands the
    # sampler a different atom count than the detector uses, and this assertion
    # is what caught that in the first port of the CLI.
    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "heavy.pdb"
        write_structure_pdb(st, prepared)
        sets = list(RNAAnmBackend(seed=0).generate(prepared, 10))
    assert sets and all(c.shape == X.shape for c in sets), "sampler/detector atom mismatch"
    worst = max(float(np.mean(np.linalg.norm(np.diff(c[c1], axis=0), axis=1)))
                for c in sets)
    assert worst / base - 1.0 < 0.05, "backbone torn, not opened"


@needs(XRAY_RNA)
def test_generation_is_deterministic():
    import tempfile

    from lacuna.io.writers import write_structure_pdb

    st = heavy_atoms_only(load_structure(XRAY_RNA))
    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "heavy.pdb"
        write_structure_pdb(st, prepared)
        a = list(RNAAnmBackend(seed=0).generate(prepared, 5))
        b = list(RNAAnmBackend(seed=0).generate(prepared, 5))
    assert all(np.allclose(x, y) for x, y in zip(a, b))


# ------------------------------------------------------- the fitted-geometry guard

def test_the_guard_covers_every_constant_that_changes_carving():
    """The learned ranker reads pocket size, burial and depth, so any constant that
    changes how cavities are carved changes what those features mean, and the
    documented failure is SILENT: ranking drops to the random null with no error.

    The alpha-sphere bounds decide which grid points become alpha points at all
    and were missing from the guard. That omission was not hypothetical: an RNA
    study swept ALPHA_DIST_MAX from 6.0 to 4.0 across 96 settings with the learned
    ranker scoring every one, and nothing warned."""
    import lacuna.pockets.detector as detector
    from lacuna.pockets.clusterer import _FITTED_GEOMETRY

    carving = {"ALPHA_DIST_MIN", "ALPHA_DIST_MAX", "CLUSTER_RADIUS_A",
               "MIN_VOLUME_A3", "MAX_VOLUME_A3", "GRID_SPACING", "LINING_CONTACT_A"}
    present = {c for c in carving if hasattr(detector, c)}
    missing = present - set(_FITTED_GEOMETRY)
    assert not missing, "these change carving but are unguarded: %s" % sorted(missing)


def test_the_guard_actually_fires_on_an_alpha_bound_change():
    import warnings as _w

    import lacuna.pockets.detector as detector
    import lacuna.pockets.clusterer as clusterer

    old_val = detector.ALPHA_DIST_MAX
    old_flag = clusterer._geometry_warned
    detector.ALPHA_DIST_MAX = old_val - 2.0
    clusterer._geometry_warned = False
    try:
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            clusterer._check_fitted_geometry()
        assert caught, "changing ALPHA_DIST_MAX produced no warning"
        assert "ALPHA_DIST_MAX" in str(caught[0].message)
    finally:
        detector.ALPHA_DIST_MAX = old_val
        clusterer._geometry_warned = old_flag


def test_the_guard_is_quiet_at_the_shipped_geometry():
    """It must not cry wolf on a default install."""
    import warnings as _w

    import lacuna.pockets.clusterer as clusterer

    old = clusterer._geometry_warned
    clusterer._geometry_warned = False
    try:
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            clusterer._check_fitted_geometry()
        assert not [c for c in caught if "detection geometry" in str(c.message)]
    finally:
        clusterer._geometry_warned = old
