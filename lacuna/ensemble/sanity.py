# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Is this ensemble worth analysing?

Every quantity Lacuna reports is relative to the input structure. ``opening`` is
``(max_volume - apo_volume) / max_volume``, so a conformer set that no longer
resembles the input makes every pocket look newly opened, and ``crypticity``
collapses to druggability. The failure is silent: the report is well formed, the
numbers are in range, and the whole protein is cryptic.

That has now happened four separate ways in this project.

* An externally supplied ensemble that was never superposed onto the reference.
  Six real sites became 28 clusters, 93% of them flagged cryptic, with median
  persistence 0.048 because every pocket landed in a cluster of its own.
* The Boltz backend pairing atoms across a residue-numbering offset, yielding
  19 A conformers from a fold that was correct to 0.92 A.
* Structures repaired with terminal residues built as extended tails, which took
  one 90-residue chain from a radius of gyration of 15.4 A to 24.9 A.
* An aSAM ensemble sampled at 600 K, where the chain is unfolding (radius of
  gyration 1.66x the input, 26 A from it) and Lacuna reported 151 cryptic
  pockets per target without complaint.

Three measurements separate those from a healthy ensemble, and none of them
needs to know where the conformers came from.

``drift``
    Mean RMSD of each conformer to the input over representative atoms, after
    optimal superposition, divided by the input's radius of gyration. Dividing
    makes it dimensionless and comparable across protein sizes.
``spread``
    Mean pairwise RMSD within the ensemble. Near zero means the sampler
    collapsed and there is no conformational variety to analyse, which is the
    documented failure of co-folding models asked for many samples.
``rg_ratio``
    Each conformer's radius of gyration over the input's. Catches unfolding and
    collapse, which drift alone can miss when the fold inflates evenly.
``offset``
    How much of a conformer's distance from the input is rigid-body placement
    rather than shape, as a fraction of the input's radius of gyration. Drift is
    measured after superposition and so is blind to a frame that is the right
    shape in the wrong place, which is exactly the state that puts every pocket
    in a cluster of its own, because cluster identity is centroid proximity in
    absolute Cartesian space. The backends now superpose, so this should always
    read zero; it is here because it did not always, and a caller building
    coordinate sets by hand bypasses the backends entirely.

Thresholds were calibrated on 12 CryptoBench targets across five ensembles of
known quality: Lacuna's NMA default and aSAM at 250 and 320 K, where the fold is
intact and recovery matches the default, against aSAM at 450 and 600 K, where
recovery falls to 5 and 1 of 12. Drift separated them cleanly, worst healthy
0.243 against best unhealthy 0.294, so the bar sits between at 0.27.

This warns rather than refuses. A genuine large hinge or interdomain motion can
exceed these bars legitimately, and that is a case worth looking at rather than
one worth blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lacuna.ensemble.base import superpose

#: Dimensionless drift above which an ensemble is reported. Calibrated: the
#: worst healthy arm reached 0.243 and the best unhealthy one 0.294.
MAX_DRIFT = 0.27

#: Radius of gyration, as a fraction of the input's, outside which a conformer
#: is reported. The worst healthy conformer measured 1.089 and the unhealthiest
#: 2.909, so 1.20 leaves healthy ensembles alone.
RG_RANGE = (0.85, 1.20)

#: Mean pairwise RMSD below which the ensemble has no usable variety. A
#: structurally identical set makes every pocket persistent and nothing cryptic.
MIN_SPREAD_A = 0.10

#: Rigid-body offset, as a fraction of the input's radius of gyration, above
#: which conformers are reported as not being in the input's frame. Generous:
#: the backends superpose, so anything beyond numerical noise here means the
#: coordinates did not come through a backend.
MAX_OFFSET = 0.05

#: Fewer representative atoms than this and the measurements are meaningless.
_MIN_ATOMS = 4


@dataclass
class EnsembleHealth:
    """What the guard measured, and what it wants to say about it."""

    n_conformers: int
    n_atoms: int
    drift: float
    spread: float
    rg_min: float
    rg_max: float
    offset: float = 0.0
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.messages

    def summary(self) -> str:
        return (
            f"{self.n_conformers} conformers over {self.n_atoms} representative "
            f"atoms: drift {self.drift:.3f} (bar {MAX_DRIFT}), spread "
            f"{self.spread:.2f} A, radius of gyration "
            f"{self.rg_min:.3f} to {self.rg_max:.3f}x the input"
        )


def representative_indices(structure) -> list[int]:
    """Atom indices to measure on: CA for protein, C1' for nucleic acid.

    One atom per residue keeps the measurement independent of how many side
    chain atoms a residue happens to carry, so a structure with missing side
    chains is not judged differently from a complete one.
    """
    idx = [i for i, a in enumerate(structure.atoms) if a.name == "CA"]
    if len(idx) >= _MIN_ATOMS:
        return idx
    idx = [i for i, a in enumerate(structure.atoms) if a.name == "C1'"]
    if len(idx) >= _MIN_ATOMS:
        return idx
    return list(range(len(structure.atoms)))


def _rg(xyz: np.ndarray) -> float:
    xyz = np.asarray(xyz, dtype=float)
    return float(np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=1).mean()))


def _rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD after optimal superposition, so only shape is compared."""
    idx = np.arange(len(target))
    moved = superpose(np.asarray(mobile, dtype=float),
                      np.asarray(target, dtype=float), idx)
    return float(np.sqrt(((moved - target) ** 2).sum(axis=1).mean()))


def check_ensemble(structure, conformers, reference=None) -> EnsembleHealth:
    """Measure an ensemble against the structure it is supposed to describe.

    Args:
        structure: The input structure the ensemble was generated from.
        conformers: Generated coordinate sets, each (N_atoms, 3) in the
            structure's atom order. The input itself should not be included.
        reference: Input coordinates, if already to hand. Defaults to the
            structure's own.

    Returns:
        An :class:`EnsembleHealth`. Its ``messages`` are empty when nothing is
        worth saying, which is the common case.
    """
    from lacuna.io.structure import coords_array

    base = coords_array(structure) if reference is None else np.asarray(reference)
    idx = representative_indices(structure)
    ref = np.asarray(base, dtype=float)[idx]
    sets = [np.asarray(c, dtype=float)[idx] for c in conformers]

    if not sets or len(ref) < _MIN_ATOMS:
        return EnsembleHealth(len(sets), len(ref), 0.0, 0.0, 1.0, 1.0)

    rg0 = _rg(ref)
    shape = [_rmsd(c, ref) for c in sets]
    # Unsuperposed, so rigid-body placement counts. The difference against the
    # superposed value is placement rather than shape.
    placed = [float(np.sqrt(((c - ref) ** 2).sum(axis=1).mean())) for c in sets]
    drift = float(np.mean(shape) / rg0) if rg0 > 0 else 0.0
    offset = (float(np.mean([p - s for p, s in zip(placed, shape)]) / rg0)
              if rg0 > 0 else 0.0)
    pair = [_rmsd(sets[i], sets[j])
            for i in range(len(sets)) for j in range(i + 1, len(sets))]
    spread = float(np.mean(pair)) if pair else 0.0
    ratios = [_rg(c) / rg0 for c in sets] if rg0 > 0 else [1.0]

    health = EnsembleHealth(
        n_conformers=len(sets), n_atoms=len(ref), drift=drift, spread=spread,
        rg_min=float(min(ratios)), rg_max=float(max(ratios)), offset=offset,
    )

    if offset > MAX_OFFSET:
        health.messages.append(
            f"conformers sit {offset:.2f} input radii away as rigid-body "
            "placement rather than shape, so they are not in the input's "
            "frame. Pockets are matched across conformers by centroid "
            "proximity in absolute coordinates, so each one will land in a "
            "cluster of its own and the whole structure will read as cryptic")

    if drift > MAX_DRIFT:
        health.messages.append(
            f"the ensemble sits {drift:.2f} input radii from the input structure "
            f"on average (bar {MAX_DRIFT}). Every pocket will look newly opened "
            "and crypticity will reduce to druggability. Either the conformers "
            "do not correspond to this structure, or the sampler was driven too "
            "hard; a genuine large domain motion can also read this way")
    lo, hi = RG_RANGE
    if health.rg_max > hi:
        health.messages.append(
            f"a conformer is {health.rg_max:.2f}x the input's radius of gyration "
            f"(bar {hi}), so the chain is expanding rather than breathing")
    if health.rg_min < lo:
        health.messages.append(
            f"a conformer is {health.rg_min:.2f}x the input's radius of gyration "
            f"(bar {lo}), so the chain is collapsing rather than breathing")
    if len(sets) > 1 and spread < MIN_SPREAD_A:
        health.messages.append(
            f"the conformers differ by {spread:.3f} A on average, so the sampler "
            "returned near-duplicates. Nothing will register as cryptic because "
            "every pocket is present in every conformer")
    return health
