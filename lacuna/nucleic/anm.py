# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Elastic network sampling for nucleic acids.

Lacuna's NMA backend builds its network from atoms named CA. RNA has none, so
``len(ca_indices) < 6`` is true and it falls back to RandomBackend without
warning: the run completes and reports a ranked, "cryptic"-flagged pocket list
built on Gaussian noise. Everything downstream of that is uninterpretable, and
nothing on screen says so. This module refuses that input instead.

Only the bead selection is overridden. The Hessian, the mode sampling and the
bead-to-all-atom propagation are Lacuna's own methods called unchanged, so a
protein-versus-RNA comparison measures the beads rather than two different
implementations.

Parameters follow Pinamonti et al., NAR 2015 (doi:10.1093/nar/gkv708), the only
systematic bead-and-cutoff scan for RNA. Three of its results contradict the
obvious port of the protein model and are worth stating, because each one would
otherwise be got wrong by analogy:

  1. Swapping CA for P is the worst available choice. P-only scores lowest of the
     eight networks tested. Protein structure is organised by backbone
     self-interaction, so a backbone bead carries it; RNA structure is organised
     by base-base interaction, so a backbone bead misses it.
  2. RNA does not want a longer cutoff. The widely quoted 20 to 24 A figures are
     P-only values. The recommended three-bead network optimises at 9 A, shorter
     than any protein ANM cutoff, and the all-atom optimum is 7 A.
  3. There is no cutoff plateau. Protein ANM is robust across roughly 15 to 24 A;
     RNA accuracy degrades monotonically past the optimum, because springs appear
     between nucleotides that are not in contact. The cutoff is a tighter
     parameter here than in the protein case, so it is not a knob to raise
     casually.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lacuna.ensemble.nma_backend import NMABackend
from lacuna.io.structure import load_structure, coords_array

from lacuna.nucleic.chem import classify, is_nucleotide

# Pinamonti Table II: optimal cutoff per network, scanned 3 to 30 A and averaged
# over four RNAs. "SBP" is sugar + base + phosphate, their recommended model.
# The base bead is the C2 ATOM, not a base centroid.
BEAD_SETS: dict[str, tuple[str, ...]] = {
    "SBP": ("C1'", "C2", "P"),     # recommended: 9 A
    "SB": ("C1'", "C2"),           # 11 A
    "S": ("C1'",),                 # 15 A
    "B": ("C2",),                  # 17 A
    "P": ("P",),                   # 20 A, worst-performing
}
OPTIMAL_CUTOFF_A: dict[str, float] = {"SBP": 9.0, "SB": 11.0, "S": 15.0, "B": 17.0, "P": 20.0}

DEFAULT_BEAD_SET = "SBP"

# LOWERED from 4.0 to 3.0 by exp29.
#
# 4.0 was chosen in exp05 on a single target (preQ1), where the backbone stretched
# only 2.3%. Swept across the 30 structures that have a ligand-bound relative,
# 4.0 reaches 6.0% worst-case stretch, past the 5% limit where a site is torn
# rather than opened, and the tool's own guard warns at that point.
#
#   amplitude   blind recovery   worst C1'-C1' stretch   backbone
#     2.0 A        10.8%                2.3%             intact
#     3.0 A        36.9%                4.0%             intact
#     4.0 A        32.3%                6.0%             TORN
#     5.0 A        40.0%                8.4%             TORN
#     6.0 A        52.3%               11.6%             TORN
#
# 3.0 dominates 4.0 outright: higher recovery AND an intact backbone, so this is
# not a trade. Recovery keeps climbing above 4.0 but only by distorting the
# structure, which is the artifact the backbone criterion exists to catch; the
# random-perturbation fallback "recovered" sites the same way at 25% stretch.
#
# The recovery figures are measured on structures with ground truth, so they are
# fitted. The backbone measurement is not: 6.0% stretch at 4.0 A is a physical
# property of the sampler and is reason enough on its own.
DEFAULT_MAX_RMSD_A = 3.0


class RNAAnmBackend(NMABackend):
    """Anisotropic network sampling with nucleic-acid beads.

    Parameters
    ----------
    bead_set : str
        Key of ``BEAD_SETS``. Default "SBP", the three-bead model.
    cutoff : float | None
        Spring cutoff in A. None takes the published optimum for the bead set.
    n_modes, max_rmsd, seed
        As ``NMABackend``.
    """

    def __init__(
        self,
        bead_set: str = DEFAULT_BEAD_SET,
        cutoff: float | None = None,
        n_modes: int = 10,
        max_rmsd: float = DEFAULT_MAX_RMSD_A,
        seed: int | None = None,
    ):
        if bead_set not in BEAD_SETS:
            raise ValueError(f"bead_set must be one of {sorted(BEAD_SETS)}, got {bead_set!r}")
        super().__init__(
            cutoff=OPTIMAL_CUTOFF_A[bead_set] if cutoff is None else cutoff,
            n_modes=n_modes, max_rmsd=max_rmsd, seed=seed)
        self.bead_set = bead_set
        self.bead_names = BEAD_SETS[bead_set]

    @property
    def name(self) -> str:
        return f"rna-anm[{self.bead_set}@{self.cutoff:g}A]"

    def bead_indices(self, structure) -> list[int]:
        """Network node indices, restricted to nucleotide residues.

        The residue restriction matters for a protein/RNA complex: a protein
        chain also contains an atom named C2 in some residues, and a bead set
        mixing the two polymers would couple them through springs that do not
        exist.
        """
        names = set(self.bead_names)
        return [a.serial for a in structure.atoms
                if a.name in names and is_nucleotide(a.res_name)]

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        chain: str | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        structure = load_structure(structure_path, chain=chain)
        base_coords = coords_array(structure)

        idx = self.bead_indices(structure)
        if len(idx) < 6:
            # Refuse rather than fall back. A silent fallback to random
            # perturbation is the failure mode this module exists to remove.
            info = classify([r.name for r in structure.residues])
            raise ValueError(
                f"RNA network needs at least 6 beads from {self.bead_names}, found {len(idx)}. "
                f"Residue content: {info['kind']} "
                f"({info['n_nucleotide']} nucleotide, {info['n_protein']} protein). "
                f"Unknown residues: {info['unknown_residues'][:8]}")

        bead_coords = base_coords[idx]
        modes, freqs = self._compute_modes(bead_coords)

        conformers = []
        for i in range(n_conformers):
            amplitude = (i + 1) / n_conformers * self.max_rmsd
            displaced = self._sample_ca(bead_coords, modes, freqs, amplitude, seed=i)
            conformers.append(self._propagate(base_coords, bead_coords, displaced))
        return conformers


def network_report(structure_path: Path, bead_set: str = DEFAULT_BEAD_SET,
                   cutoff: float | None = None) -> dict:
    """Describe the network a given bead set and cutoff would build.

    Contact statistics are worth seeing before trusting an ensemble: a cutoff too
    small leaves a chain-like network whose low modes are end-flapping, and one
    too large couples nucleotides that never touch, which is the specific failure
    Pinamonti reports for RNA.
    """
    structure = load_structure(structure_path)
    coords = coords_array(structure)
    be = RNAAnmBackend(bead_set=bead_set, cutoff=cutoff)
    idx = be.bead_indices(structure)
    if len(idx) < 2:
        return {"bead_set": bead_set, "n_beads": len(idx), "error": "too few beads"}
    pts = coords[idx]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    contacts = (d < be.cutoff).sum(1) - 1
    n_res = sum(1 for r in structure.residues if is_nucleotide(r.name))
    return {
        "bead_set": bead_set,
        "beads": list(be.bead_names),
        "cutoff_A": be.cutoff,
        "n_beads": len(idx),
        "beads_per_nucleotide": round(len(idx) / max(n_res, 1), 2),
        "contacts_per_bead": {
            "mean": round(float(contacts.mean()), 1),
            "min": int(contacts.min()),
            "max": int(contacts.max()),
        },
        "isolated_beads": int((contacts == 0).sum()),
    }
