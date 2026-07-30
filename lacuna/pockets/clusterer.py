# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Cluster pockets across the conformational ensemble.

Each conformer produces a list of Pocket objects. This module:
  1. Pools all pockets across all conformers.
  2. Clusters by centroid proximity using DBSCAN (eps=5 Å, min_samples=1).
  3. Computes per-cluster statistics: persistence, druggability, volume
     dynamics, a continuous crypticity score, and consensus residues.
  4. Ranks clusters by a configurable strategy (see ``rank_by``).
  5. Flags clusters as "cryptic" if persistence < 0.9 (not open in all conformers).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from lacuna.models import Pocket, PocketCluster
from lacuna.pockets.scorer import score_pocket

_DBSCAN_EPS = 5.0    # Å - pockets within 5 Å centroid distance are the same pocket
_CRYPTIC_THRESHOLD = 0.9  # persistence below this → cryptic

# Ranking strategies. The default "learned" is a fitted model (see below) and
# recovers roughly twice as many known sites as the analytic rules on CryptoBench
# (32.2% vs 17.8% on the held-out test fold, n=180). "crypticity" is the previous
# default and ranks purely by how much a site opens relative to the input;
# "druggability" ranks by peak open-state druggability (preferable for always-open
# / orthosteric sites); the legacy "persistence" strategy multiplies druggability
# by persistence, demoting the very transient pockets the tool targets; "balanced"
# keeps druggability primary with a mild persistence bonus.
RANK_STRATEGIES = ("learned", "crypticity", "druggability", "persistence", "balanced")
_DEFAULT_RANK_BY = "learned"

# ── learned ranker ──────────────────────────────────────────────────────────────
# A linear model over cluster features, fitted to predict whether a cluster is the
# true binding site under the size-robust criterion (Jaccard >= 0.25).
#
# Why this exists: on CryptoBench the analytic strategies above rank at the
# random-selection null. Measured on 150 shuffled structures, an oracle over all
# clusters recovers 75.3% of sites, picking 5 clusters at random recovers 13.0%,
# and ranking by crypticity recovers 12.7%. The detector already produces a
# well-localized pocket for roughly three quarters of structures; the analytic
# scores simply cannot find it among a median of 64 candidates (median 1 of which
# is correct).
#
# Fitted with logistic regression on the 322 available CryptoBench train-fold
# structures (20839 clusters, 464 positives) and measured on the dataset's own
# held-out test fold (180 structures). Splitting on CryptoBench's folds rather
# than at random keeps homologous proteins out of the evaluation. Test-fold top-5
# recovery 32.2% (95% CI 25.6-38.9) against 17.8% for crypticity and a 14.4%
# random null; an identical fit on shuffled labels scores 12.8%, confirming the
# gain is signal rather than an artifact of the evaluation.
#
# Standardization is folded into the weights so scoring is a dot product on raw
# features: no scikit-learn or other runtime dependency, and the coefficients stay
# readable. Retrain with benchmarks/train_ranker.py.
_RANKER_FEATURES = (
    "vol", "vol_max", "vol_min", "apo_vol", "drug", "max_drug", "cryp",
    "pers", "n_lin", "vol_per_lin", "enc", "hyd", "aro", "n_mem",
)
_RANKER_WEIGHTS = (
    0.0054610928119365534,    # vol
    0.001000443151555848,     # vol_max
    -0.0007162512347846649,   # vol_min
    0.0001635217366979712,    # apo_vol
    3.2045748644036958,       # drug
    1.0083869415868223,       # max_drug
    -0.1144220848072665,      # cryp
    -5.434517954848764,       # pers
    -0.06637447464600439,     # n_lin
    -0.07240682820499411,     # vol_per_lin
    -2.402750210437795,       # enc
    -1.593739396310968,       # hyd
    0.027505584308715823,     # aro
    0.18838365771604712,      # n_mem
)
_RANKER_INTERCEPT = 1.0588632130688596


def ranker_features(c: PocketCluster) -> dict[str, float]:
    """Feature values the learned ranker consumes, keyed by ``_RANKER_FEATURES``.

    Enclosure, hydrophobicity and aromatic count are averaged over the cluster's
    member pockets so the ranker sees the pocket's typical character across the
    ensemble rather than any single conformer.
    """
    mem = c.member_pockets
    n_mem = max(len(mem), 1)
    n_lin = len(c.lining_residues)
    return {
        "vol": c.volume_a3,
        "vol_max": c.volume_max_a3,
        "vol_min": c.volume_min_a3,
        "apo_vol": c.apo_volume_a3,
        "drug": c.druggability,
        "max_drug": c.max_druggability,
        "cryp": c.crypticity,
        "pers": c.persistence,
        "n_lin": float(n_lin),
        "vol_per_lin": c.volume_a3 / max(n_lin, 1),
        "enc": sum(getattr(p, "enclosure", 0.0) for p in mem) / n_mem,
        "hyd": sum(getattr(p, "hydrophobic_fraction", 0.0) for p in mem) / n_mem,
        "aro": sum(getattr(p, "aromatic_count", 0) for p in mem) / n_mem,
        "n_mem": float(len(mem)),
    }


def learned_score(c: PocketCluster) -> float:
    """Learned ranking score (higher is better).

    The linear pre-activation of the fitted logistic model. Monotonic in the
    predicted probability, so it orders identically without needing the sigmoid.
    """
    f = ranker_features(c)
    return _RANKER_INTERCEPT + sum(
        w * f[name] for name, w in zip(_RANKER_FEATURES, _RANKER_WEIGHTS)
    )


def compute_crypticity(apo_volume: float, max_volume: float, max_druggability: float) -> float:
    """Continuous crypticity score in [0, 1].

    A site is cryptic to the degree that it (a) opens up relative to the input/apo
    state and (b) is druggable once open - the conformational-selection signature
    of a cryptic pocket (Cimermancic 2016; Vajda 2018; Meller 2023).

        opening    = (max_volume - apo_volume) / max_volume   # 1.0 if absent in apo
        crypticity = opening × max_druggability

    A pocket that is already fully formed in the apo structure has opening ≈ 0 and
    so crypticity ≈ 0 (it is a constitutive site, not a cryptic one), regardless of
    how druggable it is. A pocket absent in the apo structure that opens into a
    druggable cavity scores near 1.
    """
    if max_volume <= 0.0:
        return 0.0
    opening = (max_volume - apo_volume) / max_volume
    opening = min(max(opening, 0.0), 1.0)
    return round(opening * max_druggability, 4)


def _rank_key(c: PocketCluster, rank_by: str) -> float:
    if rank_by == "persistence":
        return c.persistence * c.druggability
    if rank_by == "balanced":
        return c.max_druggability * (0.5 + 0.5 * c.persistence)
    if rank_by == "druggability":
        return c.max_druggability
    if rank_by == "crypticity":
        return c.crypticity
    if rank_by == "learned":
        return learned_score(c)
    raise ValueError(
        f"Unknown rank_by={rank_by!r}; choose from {RANK_STRATEGIES}"
    )


def cluster_pockets(
    pocket_lists: list[list[Pocket]],
    n_conformers: int,
    rank_by: str = _DEFAULT_RANK_BY,
) -> list[PocketCluster]:
    """Aggregate pockets across ensemble conformers into ranked clusters.

    Args:
        pocket_lists: One list of Pocket objects per conformer.
        n_conformers: Total number of conformers (denominator for persistence).
        rank_by: Ranking strategy - one of ``RANK_STRATEGIES``. ``"crypticity"``
            (default) surfaces transiently-open cryptic sites first;
            ``"druggability"`` ranks by peak open-state druggability (better for
            always-open/orthosteric sites); ``"persistence"`` is the legacy
            persistence × druggability ranking; ``"balanced"`` keeps druggability
            primary with a mild persistence bonus; ``"learned"`` uses the fitted
            linear ranker, which substantially outperforms the analytic strategies
            on CryptoBench (see ``_RANKER_WEIGHTS``).

    Returns:
        Ranked list of PocketCluster objects (rank 1 = best under ``rank_by``).
    """
    if rank_by not in RANK_STRATEGIES:
        raise ValueError(
            f"Unknown rank_by={rank_by!r}; choose from {RANK_STRATEGIES}"
        )
    all_pockets: list[Pocket] = []
    for ci, pockets in enumerate(pocket_lists):
        for p in pockets:
            p.conformer_idx = ci
            all_pockets.append(p)

    if not all_pockets:
        return []

    centroids = np.array([p.centroid for p in all_pockets])  # (N, 3)

    # DBSCAN without sklearn: greedy centroid merging with union-find
    labels = _greedy_cluster(centroids, eps=_DBSCAN_EPS)
    n_clusters = labels.max() + 1 if len(labels) > 0 else 0

    clusters: list[PocketCluster] = []
    for cid in range(n_clusters):
        members = [all_pockets[i] for i in range(len(all_pockets)) if labels[i] == cid]
        if not members:
            continue

        # Consensus centroid: mean over members
        centroid = tuple(np.mean([p.centroid for p in members], axis=0).tolist())

        # Volume statistics across the ensemble
        member_volumes = [p.volume_a3 for p in members]
        volume = float(np.mean(member_volumes))
        volume_min = float(min(member_volumes))
        volume_max = float(max(member_volumes))

        # Druggability: score a "representative" pocket (closest to mean)
        mean_c = np.array(centroid)
        dists = [np.linalg.norm(np.array(p.centroid) - mean_c) for p in members]
        rep = members[int(np.argmin(dists))]
        drug_score = score_pocket(rep).composite

        # Peak druggability - the pocket scored in its most-open conformer. This
        # is the relevant figure for a transiently-open cryptic site, which may be
        # half-collapsed in the representative (mean-centroid) member.
        max_drug_score = max(score_pocket(p).composite for p in members)

        # Volume in the input/apo structure (conformer 0). 0.0 if the pocket is
        # absent there - the strongest signal of crypticity.
        apo_members = [p.volume_a3 for p in members if p.conformer_idx == 0]
        apo_volume = float(max(apo_members)) if apo_members else 0.0

        crypticity = compute_crypticity(apo_volume, volume_max, max_drug_score)

        # Persistence: unique conformers that have this pocket
        conformer_set = sorted({p.conformer_idx for p in members})
        persistence = len(conformer_set) / max(n_conformers, 1)

        # Consensus lining residues: appear in ≥ 50% of conformers with this pocket
        from collections import Counter
        res_counts: Counter[str] = Counter()
        for p in members:
            for r in p.lining_residues:
                res_counts[r] += 1
        threshold = len(conformer_set) * 0.5
        consensus_residues = sorted(
            r for r, cnt in res_counts.items() if cnt >= threshold
        )

        clusters.append(PocketCluster(
            rank=0,  # set after sorting
            centroid=centroid,
            volume_a3=round(volume, 1),
            volume_min_a3=round(volume_min, 1),
            volume_max_a3=round(volume_max, 1),
            druggability=round(drug_score, 3),
            max_druggability=round(max_drug_score, 3),
            apo_volume_a3=round(apo_volume, 1),
            crypticity=crypticity,
            persistence=round(persistence, 3),
            cryptic=persistence < _CRYPTIC_THRESHOLD,
            lining_residues=consensus_residues,
            appears_in_conformers=conformer_set,
            member_pockets=members,
        ))

    # Rank by the chosen strategy (ties broken by peak druggability for stability)
    clusters.sort(key=lambda c: (_rank_key(c, rank_by), c.max_druggability), reverse=True)
    for i, c in enumerate(clusters):
        c.rank = i + 1

    return clusters


def _greedy_cluster(centroids: np.ndarray, eps: float) -> np.ndarray:
    """Simple greedy clustering: assign each point to the nearest existing cluster
    centroid within eps, or start a new cluster.

    This is O(N²) but fine for the typical N < 500 pockets per protein.
    """
    n = len(centroids)
    labels = np.full(n, -1, dtype=int)
    cluster_centers: list[np.ndarray] = []

    for i in range(n):
        if not cluster_centers:
            labels[i] = 0
            cluster_centers.append(centroids[i].copy())
            continue

        centers_arr = np.array(cluster_centers)
        dists = np.linalg.norm(centers_arr - centroids[i], axis=1)
        nearest = int(np.argmin(dists))

        if dists[nearest] <= eps:
            labels[i] = nearest
            # Update centroid (running mean)
            n_in_cluster = (labels[:i] == nearest).sum() + 1
            cluster_centers[nearest] = (
                cluster_centers[nearest] * (n_in_cluster - 1) / n_in_cluster
                + centroids[i] / n_in_cluster
            )
        else:
            new_id = len(cluster_centers)
            labels[i] = new_id
            cluster_centers.append(centroids[i].copy())

    return labels
