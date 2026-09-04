"""Multi-chain selection, and the silence that hid it.

An interface site names its chains as a hyphenated pair ("G-H"). Comparing that
string to a chain id matched nothing, so load_structure returned an empty
Structure and the failure surfaced much later, somewhere that could not say
which chain was missing. Those entries were absent from every benchmark rather
than reported as errors.
"""
from __future__ import annotations

import pytest

from lacuna.io.structure import chain_selector


def test_none_means_every_chain():
    assert chain_selector(None) is None


def test_single_chain():
    assert chain_selector("A") == frozenset({"A"})


@pytest.mark.parametrize("sel", ["G-H", "H-G", "G,H", " G - H "])
def test_pair_parses_to_both_chains(sel):
    assert chain_selector(sel) == frozenset({"G", "H"})


def test_three_chains():
    assert chain_selector("A-B-C") == frozenset({"A", "B", "C"})


def test_repeated_chain_collapses():
    assert chain_selector("A-A") == frozenset({"A"})


@pytest.mark.parametrize("sel", ["", "-", " ", ",,"])
def test_selector_naming_no_chain_raises(sel):
    with pytest.raises(ValueError, match="names no chain"):
        chain_selector(sel)


def test_multi_chain_structure_loads_both(tmp_path):
    pdb = tmp_path / "two.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA G   1      0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  GLY H   1      3.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  SER I   1      6.000   0.000   0.000  1.00  0.00           C\n"
        "END\n")
    from lacuna.io.structure import load_structure

    both = load_structure(pdb, chain="G-H")
    assert sorted({a.chain_id for a in both.atoms}) == ["G", "H"]
    one = load_structure(pdb, chain="G")
    assert sorted({a.chain_id for a in one.atoms}) == ["G"]
    assert len(load_structure(pdb).atoms) == 3


def test_absent_chain_raises_naming_what_is_present(tmp_path):
    pdb = tmp_path / "one.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n"
        "END\n")
    from lacuna.io.structure import load_structure

    with pytest.raises(ValueError, match=r"matched nothing.*\['A'\]"):
        load_structure(pdb, chain="Z")
