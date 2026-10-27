# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Tests for the nucleic correction to the ranker's burial term.

`enc` and `bur_raw` are one measurement entered twice, at +2.003 and -15.465, so
together they penalise burial. That is right for protein, where the ligand site
is the shallower of the candidates a structure yields, and backwards for nucleic
acid, where it is the more buried one. On 660 open nucleic sites the uncorrected
ranker reached top-5 in 39% against a 47% random-ordering baseline, significantly
WORSE than shuffling the list; zeroing bur_raw gives 67%.

Two things make this worth testing carefully rather than trusting by inspection.

First, the suite had no coverage of the bur_raw weight's VALUE: zeroing it
unconditionally, for protein and nucleic alike, left the summary line unchanged
at 227 passed. A silent 17-point protein regression was therefore available.

Second, `Pocket.buriedness_raw` defaults to 0.0, so a cluster built from the
usual test helper cannot see this gate at all. Every cluster here sets it
explicitly.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from lacuna.models import Atom, Pocket, Residue, Structure
from lacuna.nucleic import is_nucleic, is_nucleic_only
from lacuna.pockets import clusterer
from lacuna.pockets.clusterer import (
    _NUCLEIC_RANKER_WEIGHTS, _NUCLEIC_ZEROED_FEATURES, _RANKER_FEATURES,
    _RANKER_WEIGHTS, cluster_pockets, learned_score, ranker_features,
)

BUR = 0.31          # below the 0.4 clip, where nucleic clusters actually sit
W_BUR = _RANKER_WEIGHTS[_RANKER_FEATURES.index("bur_raw")]


def _pocket(centroid, volume=200.0, conformer_idx=0, buriedness_raw=BUR):
    return Pocket(
        centroid=centroid,
        volume_a3=volume,
        enclosure=min(buriedness_raw / 0.4, 1.0),
        hydrophobic_fraction=0.5,
        aromatic_count=2,
        lining_residues=["ALA1:A", "GLY2:A"],
        conformer_idx=conformer_idx,
        buriedness_raw=buriedness_raw,
    )


def _cluster(**kw):
    return cluster_pockets([[_pocket((0.0, 0.0, 0.0), **kw)]], n_conformers=1)[0]


# --------------------------------------------------------------- the weights

def test_gated_weights_stay_aligned_with_the_feature_names():
    assert len(_NUCLEIC_RANKER_WEIGHTS) == len(_RANKER_FEATURES)
    assert len(_NUCLEIC_RANKER_WEIGHTS) == len(_RANKER_WEIGHTS)


def test_exactly_the_named_features_are_zeroed_and_nothing_else_moves():
    """Guards against gating the wrong index, which a hand-copied weight list
    would make easy and which no other test would notice."""
    differ = [name for name, a, b in zip(_RANKER_FEATURES, _RANKER_WEIGHTS,
                                         _NUCLEIC_RANKER_WEIGHTS) if a != b]
    assert differ == list(_NUCLEIC_ZEROED_FEATURES) == ["bur_raw"]
    for name in _NUCLEIC_ZEROED_FEATURES:
        assert _NUCLEIC_RANKER_WEIGHTS[_RANKER_FEATURES.index(name)] == 0.0


def test_the_burial_term_was_actually_negative_before_the_gate():
    """If a refit ever makes bur_raw positive on protein, the premise of this
    correction is gone and it must be re-measured, not silently kept."""
    assert W_BUR < 0.0
    assert _RANKER_WEIGHTS[_RANKER_FEATURES.index("enc")] > 0.0


# ------------------------------------------------------------- the gate itself

def test_gate_removes_exactly_the_burial_penalty():
    c = _cluster()
    feats = ranker_features(c)
    assert feats["bur_raw"] == pytest.approx(BUR), "fixture must exercise bur_raw"
    delta = learned_score(c, nucleic=True) - learned_score(c, nucleic=False)
    assert delta == pytest.approx(-W_BUR * feats["bur_raw"])
    assert delta > 0.0, "removing a penalty must raise the score"


def test_protein_scoring_is_unchanged_to_the_bit():
    c = _cluster()
    feats = ranker_features(c)
    expected = clusterer._RANKER_INTERCEPT + sum(
        w * feats[name] for name, w in zip(_RANKER_FEATURES, _RANKER_WEIGHTS))
    assert learned_score(c, nucleic=False).hex() == expected.hex()


def test_the_default_is_protein():
    """The public one-argument signature must keep its old meaning."""
    c = _cluster()
    assert learned_score(c).hex() == learned_score(c, nucleic=False).hex()


def test_gate_is_a_noop_when_there_is_no_burial_to_penalise():
    c = _cluster(buriedness_raw=0.0)
    assert learned_score(c, nucleic=True).hex() == learned_score(c, nucleic=False).hex()


def test_scoring_order_cannot_leak_the_nucleic_weights():
    """A module-level weight swap would pass every test above and still corrupt
    the next protein run in the same process."""
    c = _cluster()
    first_nucleic = learned_score(c, nucleic=True)
    protein = learned_score(c, nucleic=False)
    second_nucleic = learned_score(c, nucleic=True)
    assert first_nucleic.hex() == second_nucleic.hex()
    assert learned_score(c, nucleic=False).hex() == protein.hex()
    assert clusterer._RANKER_WEIGHTS[_RANKER_FEATURES.index("bur_raw")] == W_BUR


def test_gate_emits_no_warning():
    clusterer._geometry_warned = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            learned_score(_cluster(), nucleic=True)
    finally:
        clusterer._geometry_warned = False


# ------------------------------------------------------- threading through the API

def test_cluster_pockets_threads_the_flag_into_the_ranking():
    """The gate is useless if the flag never reaches the sort. Two sites that the
    burial penalty orders one way must swap when it is removed: the deeper one
    wins once burial stops counting against it."""
    deep = _pocket((0.0, 0.0, 0.0), volume=200.0, buriedness_raw=0.38)
    shallow = _pocket((40.0, 0.0, 0.0), volume=200.0, buriedness_raw=0.02)
    pockets = [[deep, shallow]]

    as_protein = cluster_pockets(pockets, 1, rank_by="learned", nucleic=False)
    as_nucleic = cluster_pockets(pockets, 1, rank_by="learned", nucleic=True)

    assert as_protein[0].centroid == shallow.centroid, "penalty should bury the deep site"
    assert as_nucleic[0].centroid == deep.centroid, "correction should surface it"


def test_cluster_pockets_default_matches_explicit_protein():
    pockets = [[_pocket((0.0, 0.0, 0.0), volume=200.0, buriedness_raw=0.38),
                _pocket((40.0, 0.0, 0.0), volume=200.0, buriedness_raw=0.02)]]
    assert ([c.centroid for c in cluster_pockets(pockets, 1, rank_by="learned")]
            == [c.centroid for c in cluster_pockets(pockets, 1, rank_by="learned",
                                                    nucleic=False)])


@pytest.mark.parametrize("strategy",
                         ["crypticity", "druggability", "persistence", "balanced",
                          "learned-fused"])
def test_gate_touches_only_the_learned_strategy(strategy):
    """The measurement covers _RANKER_WEIGHTS. learned-fused is a separate fit
    that gives both burial terms a negative coefficient, so it is not the same
    failure and must not move until it has its own benchmark."""
    pockets = [[_pocket((0.0, 0.0, 0.0), volume=200.0, buriedness_raw=0.38),
                _pocket((40.0, 0.0, 0.0), volume=180.0, buriedness_raw=0.02)]]
    try:
        a = [c.centroid for c in cluster_pockets(pockets, 1, rank_by=strategy,
                                                 nucleic=False)]
        b = [c.centroid for c in cluster_pockets(pockets, 1, rank_by=strategy,
                                                 nucleic=True)]
    except FileNotFoundError:
        pytest.skip(f"{strategy} needs a model file absent from this install")
    assert a == b


# ------------------------------------------------- which predicate gates the weight

def _structure(residue_names):
    """Minimal structure with one atom per named residue."""
    atoms, residues = [], []
    for i, name in enumerate(residue_names):
        atoms.append(Atom(serial=i, name="C1'", res_name=name, chain_id="A",
                          res_seq=i + 1, coords=(float(i), 0.0, 0.0), element="C"))
        residues.append(Residue(chain_id="A", seq_num=i + 1, name=name,
                                atom_indices=[i]))
    return Structure(path=Path("synthetic.pdb"), atoms=atoms,
                     residues=residues)


def test_pure_nucleic_gates_and_pure_protein_does_not():
    assert is_nucleic_only(_structure(["A", "U", "G", "C"])) is True
    assert is_nucleic_only(_structure(["ALA", "GLY", "SER"])) is False


def test_mixed_input_keeps_the_protein_weighting():
    """The regression this file exists for. `classify` calls a structure "mixed"
    with no minimum nucleic fraction, so a protein target carrying one incidental
    nucleotide is "mixed", and 4 of the 210 CryptoBench test-fold structures are
    exactly that (1g59, 6du4, 5hto, 4gv9, at 14% nucleotide and below). Gating
    the weight on is_nucleic would apply the nucleic correction to those protein
    pockets, where removing bur_raw costs 17 points.

    It would also be invisible: the protein benchmark harness loads those
    structures one chain at a time, so all 210 classify as pure protein there."""
    mostly_protein = _structure(["ALA"] * 300 + ["G"])
    assert is_nucleic(mostly_protein) is True, "backend still needs nucleotide beads"
    assert is_nucleic_only(mostly_protein) is False, "but the ranker must not switch"


def test_the_two_predicates_are_not_interchangeable():
    """Stated as a test so that collapsing them back into one gets caught."""
    mixed = _structure(["ALA", "GLY", "A", "U"])
    assert is_nucleic(mixed) != is_nucleic_only(mixed)
