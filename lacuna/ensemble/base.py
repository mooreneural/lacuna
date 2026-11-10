# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Abstract base class for ensemble generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

#: Below this fraction of reference atoms matched, a frame is suspicious: real
#: ensembles of the same protein match almost everything, so anything this low
#: means a numbering mismatch rather than a genuinely different conformation.
MIN_MATCH_FRACTION = 0.5


def superpose(
    mobile: np.ndarray, target: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """Kabsch-superpose ``mobile`` onto ``target`` using only the atoms in ``idx``.

    Returns the whole ``mobile`` array rotated and translated into the target
    frame. Superposition is rigid, so it changes a conformer's global placement
    and never its internal geometry: pocket volumes, depths and lining residues
    are all invariant under it.

    Every backend that reads coordinates from somewhere else needs this. Lacuna
    decides whether two detections are the same site by absolute-Cartesian
    centroid proximity, so a conformer that arrives in a different global frame
    puts each of its pockets in a cluster of its own. The input structure's
    pockets then match nothing, ``apo_volume`` reads 0 everywhere, ``opening``
    is 1.0 by construction and the whole protein is reported as cryptic.
    Trajectories and generative-model output routinely arrive unaligned, so the
    guard belongs here rather than in each caller.

    Parameters
    ----------
    mobile : np.ndarray
        (N, 3) coordinates to move.
    target : np.ndarray
        (N, 3) reference coordinates, same atom order as ``mobile``.
    idx : np.ndarray
        Indices of the atoms to fit on. Fewer than three defines no unique
        rotation, so ``mobile`` is returned unchanged.
    """
    if idx.size < 3:
        return mobile
    m, t = mobile[idx], target[idx]
    mc, tc = m.mean(axis=0), t.mean(axis=0)
    cov = (m - mc).T @ (t - tc)
    u, _s, vt = np.linalg.svd(cov)
    # Reflection guard: without it an improper rotation can mirror the structure,
    # which preserves RMSD while inverting every chiral centre.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (mobile - mc) @ rot.T + tc


def align_frame(
    filled: np.ndarray, base: np.ndarray, matched_idx: list[int] | np.ndarray
) -> np.ndarray:
    """Bring a partly-filled external frame into the reference frame.

    ``filled`` is a copy of ``base`` whose matched slots carry the external
    frame's coordinates. Those need rotating into the reference frame; the
    unmatched slots already hold reference coordinates and must not be rotated,
    so they are restored after the transform. Doing it in that order matters:
    applying the rotation to the whole array and leaving it there would drag the
    unmatched atoms away from the positions they were meant to keep.
    """
    idx = np.asarray(matched_idx, dtype=int)
    if idx.size < 3:
        return filled
    out = superpose(filled, base, idx)
    if idx.size < len(base):
        unmatched = np.ones(len(base), dtype=bool)
        unmatched[idx] = False
        out[unmatched] = base[unmatched]
    return out


class EnsembleBackend(ABC):
    """Generates a conformational ensemble from a starting structure.

    Each conformer is returned as a (N_atoms, 3) float32 coordinate array
    in the same atom order as the input structure.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        **kwargs,
    ) -> list[np.ndarray]:
        """Return list of coordinate arrays, one per conformer."""
        ...
