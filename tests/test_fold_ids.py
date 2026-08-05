"""The benchmark fold lookup keys on the PDB id, not on "everything but one char".

Chain IDs are not all one character. When ``split_of`` stripped a single
trailing character, structures with author chain IDs like ``AAA`` looked up a
five-character key, matched nothing, and were reported "unmapped". Unmapped
structures are dropped from both sides of every split, so they vanished from
fitting and from cross-validation without any warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from train_ranker import pdb_id_of  # noqa: E402


@pytest.mark.parametrize("structure_id,expected", [
    ("3hi6A", "3hi6"),      # ordinary single-character chain
    ("7p75AAA", "7p75"),    # three-character author chain, the case that broke
    ("7oapEEE", "7oap"),
    ("1JPM", "1jpm"),       # lowercased for lookup
    ("4x19", "4x19"),       # bare id, no chain
])
def test_pdb_id_of(structure_id, expected):
    assert pdb_id_of(structure_id) == expected


def test_multichar_chain_is_not_truncated_into_the_key():
    """The specific regression: a 3-char chain must not leak into the fold key."""
    assert pdb_id_of("7p75AAA") != "7p75aa"
    assert len(pdb_id_of("7p75AAA")) == 4
