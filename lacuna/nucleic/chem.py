# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Nucleic-acid residue chemistry.

Lacuna's own ``is_hydrophobic`` / ``is_aromatic`` are lookups against sets of the
twenty amino acids, so every nucleotide falls through as False. That is wrong in
a way that matters: the aromatic term feeds druggability, which feeds the learned
ranker, so every RNA pocket is scored as if it were lined by nothing a ligand
could stack against. Stacking is the dominant interaction RNA ligands actually
make, so the one term that should be highest is pinned at zero.

What is settled here:
  - All four standard bases are planar aromatic heterocycles. Purines (A, G) are
    bicyclic, pyrimidines (C, U) monocyclic, so they do not present equal
    stacking area and are not counted equally.
  - RNA has no hydrophobic residue in the protein sense. The protein
    hydrophobic-burial term does not transfer and is not reused; see
    ``stacking_fraction`` for what replaces it.

What is not settled here: how these terms should be weighted into a druggability
score. That needs the RNA ligandability literature and a fitted model, and until
then ``rna_lacuna.score`` reports the components without collapsing them.
"""
from __future__ import annotations

# Standard ribonucleotides, as they appear in PDB/mmCIF residue names.
RNA_BASES = frozenset({"A", "C", "G", "U"})
# Deoxy equivalents, so a DNA or hybrid construct is not silently treated as protein.
DNA_BASES = frozenset({"DA", "DC", "DG", "DT", "DU"})

# Common modified nucleotides mapped to the parent base. Deliberately short: these
# are the ones that show up in riboswitch and rRNA entries. An unknown modified
# residue is reported by `classify` rather than guessed at.
MODIFIED_TO_PARENT = {
    "PSU": "U",    # pseudouridine
    "5MC": "C", "OMC": "C", "4SU": "U", "5MU": "U", "2MU": "U",
    "1MA": "A", "2MA": "A", "6MA": "A", "OMG": "G", "2MG": "G",
    "7MG": "G", "1MG": "G", "M2G": "G", "YG": "G", "H2U": "U",
    "UR3": "U", "G7M": "G", "5BU": "U", "CBV": "C", "I": "A",
}

PURINES = frozenset({"A", "G"})
PYRIMIDINES = frozenset({"C", "U", "T"})

# Fused-ring count, used as a proxy for stacking surface area. A purine presents
# roughly twice the face of a pyrimidine, which is why they are not counted alike.
RING_COUNT = {"A": 2, "G": 2, "C": 1, "U": 1, "T": 1}

# Base atoms, for separating a stacking face from backbone contact. Sugar and
# phosphate atoms carry a prime or are phosphorus/OP, so the base is what is left.
_SUGAR_PHOSPHATE_SUFFIXED = ("'", "*")
_PHOSPHATE_ATOMS = frozenset({"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"})


def parent_base(res_name: str) -> str | None:
    """Standard base letter for a residue name, or None if it is not a nucleotide."""
    n = res_name.strip().upper()
    if n in RNA_BASES:
        return n
    if n in DNA_BASES:
        return "U" if n == "DU" else n[1]
    return MODIFIED_TO_PARENT.get(n)


def is_nucleotide(res_name: str) -> bool:
    return parent_base(res_name) is not None


def is_aromatic_base(res_name: str) -> bool:
    """True for every standard and recognised modified nucleotide.

    Each of A, G, C and U is an aromatic ring system. This is the correction to
    the protein lookup, which returns False for all of them.
    """
    return parent_base(res_name) is not None


def ring_count(res_name: str) -> int:
    """Aromatic rings on the base: 2 for a purine, 1 for a pyrimidine, 0 otherwise."""
    b = parent_base(res_name)
    return RING_COUNT.get(b, 0) if b else 0


def is_purine(res_name: str) -> bool:
    b = parent_base(res_name)
    return b in PURINES if b else False


def is_base_atom(atom_name: str) -> bool:
    """True for an atom of the nucleobase, as opposed to the sugar or phosphate.

    Sugar atoms carry a prime (C1', O2'); phosphate atoms are P or OPn. Anything
    else on a nucleotide belongs to the base.
    """
    n = atom_name.strip().upper()
    if n in _PHOSPHATE_ATOMS:
        return False
    return not n.endswith(_SUGAR_PHOSPHATE_SUFFIXED)


def classify(res_names) -> dict:
    """Summarise a residue-name collection as nucleic, protein, or mixed.

    Used to decide whether an RNA path applies at all, and to surface unknown
    modified residues instead of letting them fall through as non-aromatic.
    """
    names = [str(r).strip().upper() for r in res_names]
    nuc = [n for n in names if is_nucleotide(n)]
    # The twenty standard amino acids, enough to tell the two polymer types apart.
    aa = frozenset({
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"})
    prot = [n for n in names if n in aa]
    unknown = sorted({n for n in names if n not in aa and not is_nucleotide(n)})
    total = max(len(names), 1)
    kind = "empty"
    if nuc and prot:
        kind = "mixed"
    elif nuc:
        kind = "nucleic"
    elif prot:
        kind = "protein"
    return {
        "kind": kind,
        "n_nucleotide": len(nuc),
        "n_protein": len(prot),
        "nucleotide_fraction": len(nuc) / total,
        "unknown_residues": unknown,
    }


def stacking_fraction(lining_res_names) -> float:
    """Share of a pocket's lining that can present an aromatic face, purine-weighted.

    This is the RNA analogue of the protein hydrophobic fraction, and it is a
    different quantity rather than a renamed one: it counts stacking surface, not
    burial of apolar side chains. Normalised so an all-purine lining scores 1.0.
    """
    names = list(lining_res_names)
    if not names:
        return 0.0
    return sum(ring_count(n) for n in names) / (2.0 * len(names))
