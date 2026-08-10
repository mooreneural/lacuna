# SPDX-License-Identifier: MIT
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

import warnings

import numpy as np

from lacuna.models import Pocket, PocketCluster
from lacuna.pockets.scorer import score_pocket

_DBSCAN_EPS = 5.0    # Å - pockets within 5 Å centroid distance are the same pocket
_CRYPTIC_THRESHOLD = 0.9  # persistence below this → cryptic

# Ranking strategies. The default "learned" is a fitted model (see below) and
# recovers roughly three times as many known sites as the analytic rules on
# CryptoBench (57.0% vs 17.8% on the held-out test fold, n=180). "crypticity" is the previous
# default and ranks purely by how much a site opens relative to the input;
# "druggability" ranks by peak open-state druggability (preferable for always-open
# / orthosteric sites); the legacy "persistence" strategy multiplies druggability
# by persistence, demoting the very transient pockets the tool targets; "balanced"
# keeps druggability primary with a mild persistence bonus.
RANK_STRATEGIES = ("learned", "learned-plm", "crypticity", "druggability",
                   "persistence", "balanced")
DEFAULT_RANK_BY = "learned"
_DEFAULT_RANK_BY = DEFAULT_RANK_BY  # backwards-compatible alias

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
# Trained on within-structure pairs (linear RankNet) over 741 CryptoBench
# train-fold structures, and measured on the dataset's own held-out test fold
# (180 structures). Splitting on CryptoBench's folds rather than at random keeps
# homologous proteins out of the evaluation.
#
# Two choices were made by cross-validation over the train folds, never on the
# test fold: the geometry features below are worth ~+6.9 points (CI +4.0 to +9.7),
# and training on pairs rather than on individual clusters adds a further +2.4
# (CI +0.7 to +4.2). Pairs are the better fit to the task because the decision is
# "which of this protein's ~64 candidates is the site", and differencing two
# clusters from the same structure cancels every protein-level nuisance term.
# Gradient boosting was not separable from this linear model (+0.7, CI -1.6 to
# +3.0), so the simpler, dependency-free scorer ships.
#
# Test-fold top-5 recovery 57.0% (95% CI 49.7-64.2), a paired gain of +24.0
# (CI +16.2 to +32.4) over the ranking the same pipeline produces without it. An
# identical fit on shuffled labels scores 38.0%, well below, confirming the gain
# is signal rather than an artifact of the evaluation.
#
# The weights are specific to the detector geometry they were fitted on. Halving
# CLUSTER_RADIUS_A changed pocket sizes enough that the previous weights fell to
# the random null on the new pockets, so any change to detection constants means
# refitting via benchmarks/train_ranker.py --fit.
#
# Standardization is folded into the weights so scoring is a dot product on raw
# features: no scikit-learn or other runtime dependency, and the coefficients stay
# readable. Retrain with benchmarks/train_ranker.py.
# Every feature here is invariant to the size of the ensemble. The obvious
# alternatives are not: a raw member count scales with the conformer count
# outright, and a max or min over members drifts into the tails as more conformers
# are drawn. A model fitted at one ensemble size then misreads another, which cost
# 17.5 points of recovery at 80 conformers against the 20 it was fitted on, with
# no error and no warning. Quantiles and per-conformer rates carry the same
# information and hold steady, so --conformers is safe to change.
_RANKER_FEATURES = (
    "vol", "vol_p90", "vol_p10", "apo_vol", "drug", "max_drug", "cryp",
    "pers", "n_lin", "vol_per_lin", "enc", "hyd", "aro", "mem_per_conf",
    "bur_raw", "depth", "depth_p90", "mouth", "elong", "flat", "dcen",
    "centroid_std", "vol_cv",
)
_RANKER_WEIGHTS = (
    0.007219713575962337,     # vol
    -0.0012338755899826907,   # vol_p90
    0.0026710529864424257,    # vol_p10
    -0.000557149448829811,    # apo_vol
    1.0540817689452078,       # drug
    2.5769849046013653,       # max_drug
    0.1591183395382916,       # cryp
    0.2927419226901292,       # pers
    -0.055120263275148663,    # n_lin
    -0.21437485823287503,     # vol_per_lin
    2.0033981765375746,       # enc
    -1.4132889715947423,      # hyd
    -0.015890154046957816,    # aro
    -2.5138826614863174,      # mem_per_conf
    -15.465042150906225,      # bur_raw
    0.035062052366515786,     # depth
    0.06769307580639192,      # depth_p90
    -0.1646410510807594,      # mouth
    1.3051173515736778,       # elong
    -4.741322319769843,       # flat
    -2.470522607918055,       # dcen
    -0.14961954545189457,     # centroid_std
    2.5880019412575415,       # vol_cv
)
# Ranking is invariant to a constant offset and the pairwise fit carries no
# intercept, so this is fixed at zero.
_RANKER_INTERCEPT = 0.0

# ── sequence-augmented ranker ("learned-plm") ───────────────────────────────────
# The same linear form, refitted with four extra features summarizing what a
# protein language model thinks of each pocket's lining residues (lacuna/pockets/
# plm.py). Test-fold top-5 recovery 66.5% (95% CI 59.2-73.2) against 55.9% for the
# ranking the same pipeline produces without it, a paired gain of +10.6
# (CI +6.1 to +15.1). Fitted on the same conformer-invariant geometry block as
# the strategy above, so both are robust to --conformers.
#
# Kept as a separate strategy rather than folded into "learned" so results never
# depend on whether an optional dependency happens to be installed: the same
# command gives the same ranking on every machine.
#
# Worth knowing before reading too much into the geometric terms: in this linear
# form the sequence features carry most of the ordering (plm_mean's weight dwarfs
# every geometric one, and a PLM-only fit ranks almost identically). Geometry is
# what proposes the candidates in the first place, which no sequence model can
# do; it just contributes little to ordering them once sequence signal is present.
#
# Named explicitly rather than derived from _RANKER_FEATURES, even though the two
# lists currently agree on their geometric block. Deriving it once meant that
# renaming a geometry feature silently repointed all 23 weights at different
# quantities: identical length, so the alignment test still passed while every
# weight applied to the wrong feature. Each list names what its own weights were
# fitted on, and tests/test_ranker_integrity.py pins both.
_PLM_RANKER_FEATURES = (
    "vol", "vol_p90", "vol_p10", "apo_vol", "drug", "max_drug", "cryp",
    "pers", "n_lin", "vol_per_lin", "enc", "hyd", "aro", "mem_per_conf",
    "bur_raw", "depth", "depth_p90", "mouth", "elong", "flat", "dcen",
    "centroid_std", "vol_cv",
    "plm_mean", "plm_max", "plm_top3", "plm_frac",
)
_PLM_RANKER_WEIGHTS = (
    0.0008587964251764705,   # vol
    -0.0008950110827481275,  # vol_p90
    0.002358314137086671,    # vol_p10
    0.0012741499866228824,   # apo_vol
    2.1803621582813997,      # drug
    -1.3452119229909634,     # max_drug
    0.9413291819163585,      # cryp
    0.5486897209489089,      # pers
    0.009155839144294388,    # n_lin
    -0.05304395351292021,    # vol_per_lin
    0.8778124616615285,      # enc
    0.41202655263874927,     # hyd
    -0.03115027735553585,    # aro
    -3.294394577013049,      # mem_per_conf
    -3.945607394062707,      # bur_raw
    -0.16095623004206333,    # depth
    0.11926279855681517,     # depth_p90
    -1.253317377193816,      # mouth
    0.1364262706597425,      # elong
    -1.1476449403961864,     # flat
    -0.4515404362280174,     # dcen
    0.007271336115707842,    # centroid_std
    1.1711487359654862,      # vol_cv
    10.929097322221535,      # plm_mean
    -3.847526128710042,      # plm_max
    2.502700611361364,       # plm_top3
    -1.0999725348834386,     # plm_frac
)


def ranker_features(c: PocketCluster) -> dict[str, float]:
    """Feature values describing one cluster, for ranking.

    Returns a superset of the features the shipped model uses: ``_RANKER_FEATURES``
    names the subset actually scored, while the extra keys are candidates that
    benchmarks/train_ranker.py can evaluate. Adding a key here is therefore safe;
    it changes what can be fitted, not what is currently scored.

    Per-conformer properties are averaged over the cluster's member pockets so the
    ranker sees the site's typical character across the ensemble rather than any
    single conformer.
    """
    mem = c.member_pockets
    n_mem = max(len(mem), 1)
    n_lin = len(c.lining_residues)

    def avg(attr: str, default: float = 0.0) -> float:
        return sum(getattr(p, attr, default) for p in mem) / n_mem

    # Spatial stability across the ensemble: a real site keeps the same location
    # while a spurious blob wanders between conformers. Single-structure detectors
    # have no equivalent, so this is information unique to the ensemble approach.
    if len(mem) > 1:
        pts = np.asarray([p.centroid for p in mem], dtype=float)
        centroid_std = float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()))
    else:
        centroid_std = 0.0

    vols = [p.volume_a3 for p in mem] or [0.0]
    vol_mean = sum(vols) / len(vols)
    vol_cv = (
        float(np.std(vols) / vol_mean) if vol_mean > 1e-6 and len(vols) > 1 else 0.0
    )

    return {
        # --- shipped subset -------------------------------------------------
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
        "enc": avg("enclosure"),
        "hyd": avg("hydrophobic_fraction"),
        "aro": avg("aromatic_count"),
        "n_mem": float(len(mem)),
        # --- candidate geometry (see detector._pocket_from_cavity) -----------
        # `enc` above is clipped at a buriedness of 0.4 and saturates for deeply
        # buried pockets; the raw value keeps that resolution.
        "bur_raw": avg("buriedness_raw"),
        "depth": avg("depth_a"),
        "depth_max": max((getattr(p, "depth_a", 0.0) for p in mem), default=0.0),
        "mouth": avg("mouth_frac"),
        "elong": avg("elongation", 1.0),
        "flat": avg("flatness", 1.0),
        "dcen": avg("dist_center_frac"),
        # --- candidate ensemble dynamics -------------------------------------
        "centroid_std": centroid_std,
        "vol_cv": vol_cv,
        # --- conformer-count-invariant candidates ----------------------------
        # Four of the features above drift with the size of the ensemble, which
        # silently degrades ranking when a user changes --conformers: `n_mem`
        # counts members so it scales with N outright, and `vol_max`, `vol_min`
        # and `depth_max` are extreme-value statistics, so drawing more conformers
        # pushes them further into the tails whatever the pocket is doing. A model
        # fitted at one ensemble size therefore misreads another (measured: -17.5%
        # at N=80 against N=20). Quantiles and per-conformer rates below are
        # stable as N grows and carry the same information.
        "vol_p90": _quantile(vols, 0.90),
        "vol_p10": _quantile(vols, 0.10),
        "depth_p90": _quantile([getattr(p, "depth_a", 0.0) for p in mem], 0.90),
        # Members per occupied conformer: ~1.0 normally, higher when the site
        # fragments into several pockets within a single conformer. `pers` already
        # carries the "how many conformers" part, so this is the residual signal
        # in `n_mem` with the ensemble size divided out.
        "mem_per_conf": len(mem) / max(len(c.appears_in_conformers), 1),
    }


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile, robust to the tiny lists clusters produce."""
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


#: Detection geometry the shipped weights were fitted against. The ranker reads
#: pocket size, burial and depth, so changing how pockets are carved changes what
#: those numbers mean: after CLUSTER_RADIUS_A went 4.0 -> 2.0 the previous weights
#: scored at the random null on the new pockets. That failure is silent, which is
#: the dangerous part, so a mismatch is reported once rather than left to be
#: discovered as an unexplained accuracy drop.
_FITTED_GEOMETRY = {
    "CLUSTER_RADIUS_A": 2.0,
    "MIN_VOLUME_A3": 80.0,
    "MAX_VOLUME_A3": 1500.0,
    "GRID_SPACING": 1.0,
    "LINING_CONTACT_A": 5.0,
}

_geometry_warned = False


def _check_fitted_geometry() -> None:
    """Warn once if detection constants no longer match the fitted model."""
    global _geometry_warned
    if _geometry_warned:
        return
    from lacuna.pockets import detector

    drifted = {
        name: (expected, getattr(detector, name))
        for name, expected in _FITTED_GEOMETRY.items()
        if getattr(detector, name, expected) != expected
    }
    if drifted:
        detail = ", ".join(f"{k}={got} (fitted at {exp})"
                           for k, (exp, got) in sorted(drifted.items()))
        warnings.warn(
            f"learned ranker was fitted on different detection geometry: {detail}. "
            "Ranking quality is not guaranteed; refit with "
            "benchmarks/train_ranker.py --fit or use an analytic --rank-by.",
            RuntimeWarning, stacklevel=3,
        )
    _geometry_warned = True


def learned_score(c: PocketCluster) -> float:
    """Learned ranking score (higher is better).

    The linear pre-activation of the fitted logistic model. Monotonic in the
    predicted probability, so it orders identically without needing the sigmoid.
    """
    _check_fitted_geometry()
    f = ranker_features(c)
    return _RANKER_INTERCEPT + sum(
        w * f[name] for name, w in zip(_RANKER_FEATURES, _RANKER_WEIGHTS)
    )


def learned_plm_score(c: PocketCluster, plm_features: dict[str, float]) -> float:
    """Sequence-augmented ranking score (higher is better).

    ``plm_features`` comes from ``lacuna.pockets.plm.pocket_features`` for this
    cluster's lining residues. Missing keys score 0, which is what an absent
    sequence signal should contribute rather than an error.
    """
    f = {**ranker_features(c), **plm_features}
    return sum(w * f.get(name, 0.0)
               for name, w in zip(_PLM_RANKER_FEATURES, _PLM_RANKER_WEIGHTS))


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
    if rank_by == "learned-plm":
        # Populated by cluster_pockets when residue probabilities are supplied.
        return learned_plm_score(c, getattr(c, "_plm_features", {}) or {})
    raise ValueError(
        f"Unknown rank_by={rank_by!r}; choose from {RANK_STRATEGIES}"
    )


def cluster_pockets(
    pocket_lists: list[list[Pocket]],
    n_conformers: int,
    rank_by: str = _DEFAULT_RANK_BY,
    plm_residue_probs: dict[int, float] | None = None,
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
    # Checked before any work: a caller who forgot the probabilities has a bug,
    # and should hear about it immediately rather than after clustering, or not
    # at all when the structure happens to yield no pockets.
    if rank_by == "learned-plm" and not plm_residue_probs:
        raise ValueError(
            'rank_by="learned-plm" needs plm_residue_probs; compute them once per '
            "structure with lacuna.pockets.plm.residue_probabilities"
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

    if rank_by == "learned-plm":
        from lacuna.pockets.plm import pocket_features as _plm_pocket_features

        for c in clusters:
            nums = set()
            for label in c.lining_residues:
                digits = "".join(ch for ch in label.split(":")[0] if ch.isdigit())
                if digits:
                    nums.add(int(digits))
            c._plm_features = _plm_pocket_features(nums, plm_residue_probs)

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
