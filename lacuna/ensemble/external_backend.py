# SPDX-License-Identifier: AGPL-3.0-or-later
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
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from lacuna.ensemble.base import EnsembleBackend
from lacuna.io.structure import coords_array, iter_model_coords, load_structure

#: Extensions treated as structures when a directory is supplied.
_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif")

#: Below this fraction of reference atoms matched, a frame is reported as
#: suspicious. Chosen loosely: real ensembles of the same protein match almost
#: everything, so anything this low indicates a mismatch rather than motion.
_MIN_MATCH_FRACTION = 0.5


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
        keys = [(a.chain_id, a.res_seq, a.name) for a in reference.atoms]
        cap = n_conformers if n_conformers and n_conformers > 0 else None

        conformers: list[np.ndarray] = []
        self._skipped = []
        for path in self._frame_paths():
            # Read every chain rather than pre-filtering to the reference's chain
            # letter. Structure predictors emit a single chain and almost always
            # label it "A", so filtering on a reference chain of "B" would discard
            # a frame that is in fact the right protein. Deciding per frame needs
            # to see which chains it actually contains.
            for model_i, frame in enumerate(iter_model_coords(path)):
                if cap is not None and len(conformers) >= cap:
                    return conformers
                frame_chains = {c for c, _, _ in frame}
                if len(frame_chains) == 1 and frame_chains != {chain}:
                    # One chain, and not the reference's letter: the mapping is
                    # unambiguous, so match on residue and atom name alone. Doing
                    # this only for single-chain frames keeps it safe, since with
                    # two or more chains dropping the chain would silently merge
                    # residues that share a number across them.
                    by_res = {(r, n): xyz for (_, r, n), xyz in frame.items()}
                    lookup = [(a.res_seq, a.name) for a in reference.atoms]
                else:
                    by_res = frame
                    lookup = keys
                out = base.copy()
                matched = 0
                for i, key in enumerate(lookup):
                    got = by_res.get(key)
                    if got is not None:
                        out[i] = got
                        matched += 1
                frac = matched / max(len(keys), 1)
                label = f"{path.name}[model {model_i}]"
                if frac < _MIN_MATCH_FRACTION:
                    msg = (f"{label} matched only {frac:.0%} of reference atoms; "
                           "residue numbering probably differs")
                    if self.strict:
                        raise ValueError(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    self._skipped.append(label)
                    continue
                conformers.append(out.astype(np.float32))

        if not conformers:
            raise ValueError(
                f"no usable conformers read from {self.source}"
                + (f" (skipped: {', '.join(self._skipped)})" if self._skipped else "")
            )
        return conformers
