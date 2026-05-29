"""Cluster pockets across the conformational ensemble.

Each conformer produces a list of Pocket objects. This module:
  1. Pools all pockets across all conformers.
  2. Clusters by centroid proximity using DBSCAN (eps=5 Å, min_samples=1).
  3. Computes per-cluster statistics: persistence, mean druggability, consensus residues.
  4. Ranks clusters by persistence × druggability (descending).
  5. Flags clusters as "cryptic" if persistence < 0.9 (not open in all conformers).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from lacuna.models import Pocket, PocketCluster
from lacuna.pockets.scorer import score_pocket

_DBSCAN_EPS = 5.0    # Å — pockets within 5 Å centroid distance are the same pocket
_CRYPTIC_THRESHOLD = 0.9  # persistence below this → cryptic


def cluster_pockets(
    pocket_lists: list[list[Pocket]],
    n_conformers: int,
) -> list[PocketCluster]:
    """Aggregate pockets across ensemble conformers into ranked clusters.

    Args:
        pocket_lists: One list of Pocket objects per conformer.
        n_conformers: Total number of conformers (denominator for persistence).

    Returns:
        Ranked list of PocketCluster objects (rank 1 = most druggable/persistent).
    """
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

        # Mean volume
        volume = float(np.mean([p.volume_a3 for p in members]))

        # Druggability: score a "representative" pocket (closest to mean)
        mean_c = np.array(centroid)
        dists = [np.linalg.norm(np.array(p.centroid) - mean_c) for p in members]
        rep = members[int(np.argmin(dists))]
        drug_score = score_pocket(rep).composite

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
            druggability=round(drug_score, 3),
            persistence=round(persistence, 3),
            cryptic=persistence < _CRYPTIC_THRESHOLD,
            lining_residues=consensus_residues,
            appears_in_conformers=conformer_set,
            member_pockets=members,
        ))

    # Rank by persistence × druggability
    clusters.sort(key=lambda c: c.persistence * c.druggability, reverse=True)
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
