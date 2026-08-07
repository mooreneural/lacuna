# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Propose pocket locations from sequence where geometry proposes none.

A cryptic site in its apo form often has no concavity for an alpha-sphere detector
to find, which is the definition of the problem rather than a shortcoming of the
detector. On a single structure that caps coverage at 46.5% of CryptoBench targets
against P2Rank's 61.9% on identical input, and the shortfall is entirely detection:
at one conformer the ranker already converts 97.6% of the coverage it is given, so
there is nothing left to win by ranking better.

The sequence head does not need a pocket to be open. Handing its highest-scoring
residues to ``characterize_pockets`` (the entry point the P2Rank fusion path uses)
produces proposals described by the same geometry code as detected ones, so the two
are on one scale in the clusterer.

Measured on CryptoBench, one conformer, held-out test fold: +8.5% CI[+4.0, +13.6]
top-5 recovery with the sequence ranker, from 18 structures gained against 3 lost,
for 1.2 extra candidates per structure.

When this helps, and when it does not
-------------------------------------
Added coverage converts to recovery only while the candidate set is small. Holding
this intervention fixed and varying ensemble size, so that candidate count is the
only thing changing:

    candidates    oracle gain    top-5 gain    converts
    6.6 -> 7.8      +14.3%         +12.3%         86%
    14.1 -> 15.3     +6.4%          +5.5%         85%
    17.5 -> 18.7     +5.2%          +2.7%         51%
    21.1 -> 22.2     +3.9%          +2.0%         52%

Conversion holds near 85% up to roughly fifteen candidates and halves above it. So
seeding substitutes for conformers rather than adding to them: at five conformers
it closes the whole gap to the twenty-conformer ensemble (-0.1% CI[-2.9, +2.6]) at
a third of the wall clock, while at twenty it adds nothing separable from zero.
That is why this is opt-in and why turning it on is worth pairing with a smaller
ensemble rather than layering it on top of a large one.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from lacuna.models import Structure

#: Residues considered as seed material, from the top of the sequence ranking.
#: Twenty matched a selection sized to each true site (31.1% against 31.3% of
#: geometry-missed structures rescued), so nothing is gained by using site size,
#: which a predictor would not know anyway.
TOP_N = 20

#: Seed residues within this distance of one another describe the same proposed
#: site. Roughly the span of a small pocket's lining.
LINK_A = 8.0

#: Fewer residues than this is a scattered high-scoring position rather than a
#: localized site.
MIN_GROUP = 3

#: Hard cap on proposals per conformer. Candidate count governs whether added
#: coverage converts at all, so the budget is fixed in advance rather than left to
#: however many groups happen to form on a given structure.
MAX_SEEDS = 3


def seed_centers(
    structure: Structure,
    residue_probs: dict[int, float],
    coords: np.ndarray | None = None,
    top_n: int = TOP_N,
    max_seeds: int = MAX_SEEDS,
) -> list[tuple[float, float, float]]:
    """Locations to characterize, from the highest-scoring residues.

    Parameters
    ----------
    structure : Structure
        Supplies the residue list and its atom indices.
    residue_probs : dict[int, float]
        Residue sequence number to the sequence head's probability, as returned by
        ``lacuna.pockets.plm.residue_probabilities``.
    coords : np.ndarray, optional
        Coordinates for the conformer being seeded. Defaults to the structure's own
        atom coordinates. Passing the conformer's coordinates matters: a seed is a
        location in *this* frame, and using the reference frame throughout would
        place every proposal at the apo position.
    top_n, max_seeds : int
        Overrides for the module constants above.

    Returns
    -------
    list of (x, y, z)
        At most ``max_seeds`` centers, ordered by summed residue probability.
        Empty when the sequence head supplied nothing or its top residues are too
        scattered to form a group.
    """
    if not residue_probs:
        return []
    if coords is None:
        coords = np.asarray([a.coords for a in structure.atoms], dtype=float)
    coords = np.asarray(coords, dtype=float)

    ranked = sorted(residue_probs.items(), key=lambda kv: -kv[1])[:top_n]
    wanted = {num for num, _ in ranked}

    centroids: list[np.ndarray] = []
    nums: list[int] = []
    for res in structure.residues:
        if res.seq_num not in wanted or not res.atom_indices:
            continue
        idx = [i for i in res.atom_indices if i < len(coords)]
        if idx:
            centroids.append(coords[idx].mean(axis=0))
            nums.append(res.seq_num)
    if len(centroids) < MIN_GROUP:
        return []

    pts = np.asarray(centroids)
    groups = _link(pts, LINK_A)

    scored = []
    for members in groups:
        if len(members) < MIN_GROUP:
            continue
        weight = sum(residue_probs[nums[i]] for i in members)
        scored.append((weight, pts[members].mean(axis=0)))
    scored.sort(key=lambda t: -t[0])
    return [tuple(float(x) for x in c) for _, c in scored[:max_seeds]]


def _link(pts: np.ndarray, radius: float) -> list[list[int]]:
    """Single-linkage grouping of points within ``radius``.

    Union-find over the pair graph. Single linkage rather than k-means because the
    number of sites is not known in advance and chaining along a groove is the
    desired behaviour, not a failure mode.
    """
    parent = list(range(len(pts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(pts).query_pairs(radius):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def seeded_pockets(
    coords: np.ndarray,
    structure: Structure,
    residue_probs: dict[int, float],
    **kwargs,
) -> list:
    """Pockets characterized at the sequence head's proposed locations.

    Returns an empty list when nothing is proposed, and drops proposals where the
    frame has no open void at that location, which ``characterize_pockets`` reports
    as ``None``.
    """
    from lacuna.pockets.detector import characterize_pockets

    centers = seed_centers(structure, residue_probs, coords, **kwargs)
    if not centers:
        return []
    return [p for p in characterize_pockets(coords, structure, centers)
            if p is not None]
