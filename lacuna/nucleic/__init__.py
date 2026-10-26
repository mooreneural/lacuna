# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Nucleic-acid support.

Detection, clustering and ranking are shared with the protein path unchanged,
because they already work on RNA: on 120 held-out apo/holo records the true
ligand site is ranked top-5 in 80% of cases. What RNA needs is narrower than it
first appears.

  chem   nucleotide chemistry, which the 20-amino-acid lookups elsewhere miss
  anm    an elastic network built on nucleotide beads rather than CA atoms

The reason this exists at all is a silent failure. ``NMABackend`` builds its
network from atoms named CA; RNA has none, so it fell through to
``RandomBackend`` and the run completed, printing a ranked table with crypticity
and persistence columns computed from Gaussian noise. Measured on 10 held-out
structures whose site is shut, that fallback recovered 0 of them while stretching
C1'-C1' spacing 25%. It was worse than refusing the input while looking like it
had worked.

Measured performance, all on held-out structures:

  120 apo/holo records, 37 structures   top-5 80%, top-1 50%
  31 apo/holo pairs over 12 independent
  BGSU equivalence classes, 62-459 nt   site found 87%, top-10 65%, top-5 39%

Large RNA is therefore supported and worth running, but ranks the site deeper:
a 400 nt RNA yields 40 or more clusters, so read further down the list than you
would for a 30-mer. Above roughly 120 nt a bigger ensemble stops helping, since
the sites there are already open in the input rather than cryptic; the
single-structure result is what the 87% figure measures.

Evidence, including the stages that were tried and reversed, is in the lacuna-rna
staging repository.
"""
from dataclasses import replace

from lacuna.nucleic.anm import RNAAnmBackend
from lacuna.nucleic.chem import classify, is_nucleotide

__all__ = ["RNAAnmBackend", "classify", "is_nucleotide", "is_nucleic",
           "heavy_atoms_only"]


def is_nucleic(structure) -> bool:
    """True when the structure is RNA, DNA, or a nucleic/protein mixture."""
    return classify([r.name for r in structure.residues])["kind"] in ("nucleic", "mixed")


def heavy_atoms_only(structure):
    """Drop hydrogens, renumbering atoms and residues together.

    mmCIF entries from RCSB often carry hydrogens and the loader keeps them:
    6VUH reads 1062 atoms with them against 702 without, on the same 33
    residues, and that changes the van der Waals surface and so the pocket count
    (2 against 1 on that structure). Every nucleic benchmark was measured on
    heavy atoms, so this keeps the tool behaving the way it was validated.

    ``Residue.atom_indices`` points into the atom list by serial, so both have to
    be renumbered together or lining-residue lookups silently read the wrong
    atoms and nothing visibly fails.

    Applied to nucleic input only. Protein behaviour is deliberately untouched,
    since the protein benchmarks were measured with whatever the loader returns.
    """
    keep = [a for a in structure.atoms if (a.element or "").upper() != "H"]
    if len(keep) == len(structure.atoms):
        return structure
    remap = {a.serial: i for i, a in enumerate(keep)}
    atoms = [replace(a, serial=i) for i, a in enumerate(keep)]
    residues = [replace(r, atom_indices=[remap[s] for s in r.atom_indices if s in remap])
                for r in structure.residues]
    return replace(structure, atoms=atoms,
                   residues=[r for r in residues if r.atom_indices])
