# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Nucleic-acid support.

Detection and clustering are shared with the protein path unchanged, because they
already work on RNA. Ranking is shared too, with one weight corrected: the
ranker's burial term is fitted on protein, where the ligand site is the
SHALLOWER of the candidates a structure yields, and it inverts on nucleic acid,
where the site is the more buried one. See ``_NUCLEIC_RANKER_WEIGHTS`` in
``lacuna.pockets.clusterer``. What RNA needs is narrower than it first appears.

  chem   nucleotide chemistry, which the 20-amino-acid lookups elsewhere miss
  anm    an elastic network built on nucleotide beads rather than CA atoms

The reason this exists at all is a silent failure. ``NMABackend`` builds its
network from atoms named CA; RNA has none, so it fell through to
``RandomBackend`` and the run completed, printing a ranked table with crypticity
and persistence columns computed from Gaussian noise. Measured on 10 held-out
structures whose site is shut, that fallback recovered 0 of them while stretching
C1'-C1' spacing 25%. It was worse than refusing the input while looking like it
had worked.

DETECTION, measured on held-out structures: across 31 structures from 12
independent BGSU equivalence classes spanning 62 to 459 nt, a single pass finds
the ligand site for 87% of structures. Detection does not depend on the ranker,
so the correction below leaves it untouched.

RANKING. Every ranking figure quoted here before 1.3.1 was measured with the
uncorrected burial term, against no baseline, so they are withdrawn rather than
restated. Ranking is now reported against the random-ordering baseline it has to
beat to mean anything, which is ``1 - C(N-k, m)/C(N, m)`` averaged over
structures, per structure and held out:

  population                              n    baseline   1.3.0   1.3.1
  nucleic sites open in the input       660         47%     39%     67%
  nucleic sites shut in the input        61         50%     64%     61%

The first row is the correction: at 39% the shipped ranker was significantly
WORSE than shuffling the list (p = 1.3e-07), because it was penalising exactly
the burial that identifies a nucleic site. The second row is unchanged within
noise (paired sign test p = 0.75) and was already above baseline.

Large RNA is supported and worth running. It still ranks the site deeper than a
short one does, because a 400 nt RNA yields 40 or more clusters while a 30-mer
yields a handful, so read further down the list; the correction shortens that
distance rather than removing it. Above roughly 120 nt a bigger ensemble stops
helping, since the sites there are already open in the input rather than cryptic,
and the 87% detection figure is a single-structure result.

Evidence, including the stages that were tried and reversed, is in the lacuna-rna
staging repository.
"""
from dataclasses import replace

from lacuna.nucleic.anm import RNAAnmBackend
from lacuna.nucleic.chem import classify, is_nucleotide

__all__ = ["RNAAnmBackend", "classify", "is_nucleotide", "is_nucleic",
           "is_nucleic_only", "heavy_atoms_only"]


def is_nucleic(structure) -> bool:
    """True when the structure is RNA, DNA, or a nucleic/protein mixture.

    This is the ensemble-backend question: it decides whether the elastic network
    needs nucleotide beads, and a mixture does, because ``NMABackend`` would
    leave every nucleotide rigid. For the RANKER gate use ``is_nucleic_only``.
    """
    return classify([r.name for r in structure.residues])["kind"] in ("nucleic", "mixed")


def is_nucleic_only(structure) -> bool:
    """True when the structure is RNA or DNA with no protein in it.

    Deliberately stricter than ``is_nucleic``, and the difference is load-bearing
    rather than pedantic. ``classify`` reports "mixed" whenever both polymers are
    present with no minimum fraction, so 300 amino acids beside a single guanine
    is "mixed", and 4 of the 210 structures in the CryptoBench test fold are
    protein targets carrying a small nucleic co-crystal: 1g59 at 14% nucleotide,
    6du4 at 9%, 5hto at 5%, 4gv9 at 4%. Their annotated sites are protein
    pockets, and the ranker's burial term is right for those.

    So the two questions differ. Which beads the elastic network needs is about
    whether nucleotides are present at all, and a mixture needs them. Which way
    the burial term should point is about what walls the pocket, and in a mostly
    protein structure that is protein. Gating the weight on ``is_nucleic`` would
    apply the nucleic correction to 1.9% of the protein benchmark, where it costs
    17 points, and the regression would be invisible to the benchmark harness,
    which loads those structures one chain at a time and so sees all 210 as pure
    protein.

    Mixed protein/nucleic input therefore keeps the protein weighting. That is
    the conservative choice rather than a measured one: the correction is
    validated on nucleic-only structures, and a separate survey of 478 ligand
    copies in protein/RNA complexes found their sites sharply bimodal, about 44%
    walled almost entirely by protein and 36% almost entirely by RNA, so a single
    structure-level answer cannot be right for both. Per-pocket gating is the
    open question, not a shipped behaviour.
    """
    return classify([r.name for r in structure.residues])["kind"] == "nucleic"


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
