# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Use a conformational ensemble somebody else generated.

Lacuna's own samplers are a convenience, not the point. The detection, clustering
and ranking stages do not care where the conformers came from, so the most useful
thing the tool can do is consume a better ensemble than it can produce. An
elastic-network model cannot generate large hinge or interface motions; a long
molecular dynamics trajectory or a generative model such as AlphaFlow or BioEmu
can, and this backend is how that output gets in.

Accepts either shape ensembles usually arrive in:

  * one multi-model file (NMR deposition, trajectory exported to PDB, generative
    model output written as models)
  * a directory of single-structure files, one per conformer

Atom correspondence
-------------------
External frames rarely match the reference atom-for-atom. A predicted model may
be missing a disordered loop the crystal resolved, or number residues from a
construct rather than the deposition. Frames are therefore matched on
``(chain_id, res_seq, atom_name)`` rather than by position, and any reference atom
a frame does not contain keeps its reference coordinate. That degrades gracefully
instead of silently pairing the wrong atoms, which is the failure that would
quietly corrupt every pocket downstream.

A frame matching very little of the reference is reported rather than used
silently, because that almost always means a numbering mismatch rather than a
genuinely different conformation.

Global frame
------------
Every accepted frame is rigid-body superposed onto the reference on its matched
atoms. Lacuna decides whether two detections are the same site by comparing
pocket centroids in absolute Cartesian space, so a frame that arrives in its own
global frame puts each of its pockets in a cluster of its own, leaves the input
structure's pockets matching nothing, and reports the entire protein as cryptic.
Multi-model NMR depositions are usually pre-aligned and are unaffected, but a
molecular dynamics trajectory exported without fitting, and the output of any
generative model that folds in its own frame, are not.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from lacuna.ensemble.base import MIN_MATCH_FRACTION, EnsembleBackend, align_frame
from lacuna.io.structure import coords_array, iter_model_coords, load_structure

#: Extensions treated as structures when a directory is supplied.
_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif")

#: Below this fraction of reference atoms matched, a frame is reported as
#: suspicious. Shared with the other backends; see ``base.MIN_MATCH_FRACTION``.
_MIN_MATCH_FRACTION = MIN_MATCH_FRACTION

#: Below this fraction of paired residues sharing a residue name, a pairing is
#: judged wrong. Atom-name counting alone cannot detect a numbering offset,
#: because every residue carries N/CA/C/O and so matches across a shift.
_MIN_IDENTITY_FRACTION = 0.9


def _residues_in_order(
    items,
) -> tuple[list[tuple[str, int]], dict, dict]:
    """Group ``(chain, res_seq, atom_name) -> value`` pairs into residues.

    Returns the residue keys in the order first seen, each residue's
    ``atom_name -> value`` map, and each residue's residue name where one is
    available. File order is observation order, which is what lets a frame be
    paired positionally when its numbering cannot be trusted.
    """
    order: list[tuple[str, int]] = []
    atoms: dict[tuple[str, int], dict] = {}
    for key, value in items:
        rk = (key[0], key[1])
        if rk not in atoms:
            order.append(rk)
            atoms[rk] = {}
        atoms[rk][key[2]] = value
    return order, atoms, {}


def _identity(ref_names: list[str], frame_names: list[str]) -> float:
    """Fraction of positionally paired residues that are the same residue type."""
    if not ref_names:
        return 0.0
    same = sum(1 for a, b in zip(ref_names, frame_names) if a == b)
    return same / min(len(ref_names), max(len(frame_names), 1))


class ExternalEnsembleBackend(EnsembleBackend):
    """Read conformers from a file or directory instead of generating them.

    Parameters
    ----------
    source : Path
        A multi-model structure file, or a directory of structure files.
    chain : str | None
        Restrict to one chain, matching the chain used for the reference.
    strict : bool
        Raise if a frame matches less than half the reference atoms, instead of
        warning and using it.
    """

    def __init__(self, source: str | Path, chain: str | None = None,
                 strict: bool = False):
        self.source = Path(source)
        self.chain = chain
        self.strict = strict
        self._skipped: list[str] = []

    @property
    def name(self) -> str:
        return "external"

    def _frame_paths(self) -> list[Path]:
        if self.source.is_dir():
            files = sorted(p for p in self.source.iterdir()
                           if p.suffix.lower() in _SUFFIXES)
            if not files:
                raise ValueError(
                    f"no structure files ({', '.join(_SUFFIXES)}) in {self.source}")
            return files
        if not self.source.exists():
            raise ValueError(f"ensemble source not found: {self.source}")
        return [self.source]

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        chain: str | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Return external conformers in the reference structure's atom order.

        ``n_conformers`` caps how many frames are read; the whole ensemble is used
        when it is 0 or negative. Frames are taken in file order, which for a
        multi-model file is deposition order.
        """
        chain = chain if chain is not None else self.chain
        reference = load_structure(structure_path, chain=chain)
        base = coords_array(reference)
        cap = n_conformers if n_conformers and n_conformers > 0 else None

        # The reference as ordered residues, each owning its atom slots.
        ref_order: list[tuple[str, int]] = []
        ref_slots: dict[tuple[str, int], dict[str, int]] = {}
        ref_name: dict[tuple[str, int], str] = {}
        for i, a in enumerate(reference.atoms):
            rk = (a.chain_id, a.res_seq)
            if rk not in ref_slots:
                ref_order.append(rk)
                ref_slots[rk] = {}
                ref_name[rk] = a.res_name
            ref_slots[rk][a.name] = i

        conformers: list[np.ndarray] = []
        self._skipped = []
        for path in self._frame_paths():
            # Read every chain rather than pre-filtering to the reference's chain
            # letter. Structure predictors emit a single chain and almost always
            # label it "A", so filtering on a reference chain of "B" would discard
            # a frame that is in fact the right protein. Deciding per frame needs
            # to see which chains it actually contains.
            for model_i, (frame, frame_names) in enumerate(
                    iter_model_coords(path, with_names=True)):
                if cap is not None and len(conformers) >= cap:
                    return conformers
                label = f"{path.name}[model {model_i}]"
                f_order, f_atoms, _ = _residues_in_order(frame.items())
                frame_chains = {c for c, _ in f_order}

                # Candidate pairings, tried in order of how much they assume.
                cands: list[tuple[str, list]] = [
                    ("numbering", [(rk, rk) for rk in ref_order if rk in f_atoms]),
                ]
                if len(frame_chains) == 1 and frame_chains != {chain}:
                    # One chain under a different letter: the mapping is
                    # unambiguous, so try the numbers with the letter ignored.
                    # Only for single-chain frames, since with two or more chains
                    # dropping the letter would merge residues sharing a number.
                    only = next(iter(frame_chains))
                    cands.append(("relabelled chain",
                                  [(rk, (only, rk[1])) for rk in ref_order
                                   if (only, rk[1]) in f_atoms]))
                if len(f_order) == len(ref_order):
                    # Same residue count: position is a candidate key in its own
                    # right, and the only one that survives renumbering. aSAM,
                    # Boltz and AlphaFold all renumber their output from 1, and
                    # a numbering offset still matches most atom names, because
                    # every residue carries N/CA/C/O. Residue identity is what
                    # separates a real pairing from a shifted one.
                    cands.append(("order", list(zip(ref_order, f_order))))

                chosen = chosen_how = None
                fallback = None
                for how, pairing in cands:
                    if not pairing:
                        continue
                    ident = _identity([ref_name[a] for a, _ in pairing],
                                      [frame_names.get(b, "") for _, b in pairing])
                    if ident >= _MIN_IDENTITY_FRACTION:
                        chosen, chosen_how = pairing, how
                        break
                    if fallback is None or ident > fallback[0]:
                        fallback = (ident, how, pairing)

                if chosen is None:
                    ident = fallback[0] if fallback else 0.0
                    msg = (f"{label}: no residue pairing agrees with the reference "
                           f"sequence (best {ident:.0%} of paired residues share a "
                           "residue name); numbering or sequence differs")
                    if self.strict:
                        raise ValueError(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    self._skipped.append(label)
                    continue
                if chosen_how == "order":
                    # Worth saying out loud: the frame's own residue numbers were
                    # unusable and position was trusted instead. A relabelled
                    # single chain is the documented case and stays quiet.
                    warnings.warn(
                        f"{label}: residue numbers did not agree with the "
                        "reference sequence, so residues were paired by order; "
                        "the frame appears to have been renumbered",
                        RuntimeWarning, stacklevel=2)

                out = base.copy()
                matched_idx: list[int] = []
                for rk, fk in chosen:
                    got = f_atoms.get(fk) or {}
                    for aname, slot in ref_slots[rk].items():
                        xyz = got.get(aname)
                        if xyz is not None:
                            out[slot] = xyz
                            matched_idx.append(slot)
                frac = len(matched_idx) / max(len(base), 1)
                if frac < _MIN_MATCH_FRACTION:
                    msg = (f"{label} matched only {frac:.0%} of reference atoms; "
                           "residue numbering probably differs")
                    if self.strict:
                        raise ValueError(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    self._skipped.append(label)
                    continue
                # Into the reference frame. See the module docstring: without
                # this an unaligned trajectory reports the whole protein cryptic.
                out = align_frame(out, base, matched_idx)
                conformers.append(out.astype(np.float32))

        if not conformers:
            raise ValueError(
                f"no usable conformers read from {self.source}"
                + (f" (skipped: {', '.join(self._skipped)})" if self._skipped else "")
            )
        return conformers
