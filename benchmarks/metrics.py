"""Size-robust evaluation metrics for the Lacuna benchmarks.

Every metric is computed from the same canonical representations so that no
experiment can improve the headline by exploiting a metric mismatch:

  * residue sets are sets of integer sequence numbers,
  * centroids and Cα coordinates are (x, y, z) tuples in Å.

The headline metric is deliberately size-robust. ``residue_recall`` is kept only
for backward comparison with older Lacuna runs; it is size-gameable (a large
pocket engulfs a small known site and scores high recall without being localized
on it), so it must never be the primary reported number.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

# Thresholds - the one prespecified headline set. Ablations may sweep around these
# but the reported headline stays fixed.
JACCARD_THRESHOLD = 0.25       # size-robust IoU
CENTROID_THRESHOLD = 4.0       # Å, field-standard localization
OVERLAP_THRESHOLD = 0.30       # legacy recall (backward comparison only)
CORE_RADIUS = 8.0              # Å, hotspot-core radius
STRICT_CENTROID = 6.0          # Å, looser centroid used with the AND-criterion


def found_resnums(cluster_residues: Iterable[str]) -> set[int]:
    """Parse residue sequence numbers from Lacuna lining-residue labels.

    Labels are ``NAME+seq:chain`` (e.g. ``ALA123:A``); we take the part before
    ``:`` and keep its digits. (Canonical (chain, resseq) IDs on the Pocket model
    would remove this parsing step - tracked as a follow-up.)
    """
    out: set[int] = set()
    for label in cluster_residues:
        try:
            out.add(int("".join(c for c in label.split(":")[0] if c.isdigit())))
        except (ValueError, IndexError):
            pass
    return out


def residue_recall(found: set[int], known: set[int]) -> float:
    """|found ∩ known| / |known| - SIZE-GAMEABLE, backward comparison only."""
    return len(found & known) / len(known) if known else 0.0


def jaccard(found: set[int], known: set[int]) -> float:
    """|found ∩ known| / |found ∪ known| - size-robust; anchors honest reporting."""
    union = found | known
    return len(found & known) / len(union) if union else 0.0


def centroid_distance(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> float:
    """Å distance between two centroids; ``inf`` if either is missing."""
    if a is None or b is None:
        return math.inf
    return math.dist(a, b)


def hotspot_core_hit(
    pocket_centroid: tuple[float, float, float] | None,
    known_ca: Sequence[tuple[float, float, float]],
    radius: float = CORE_RADIUS,
) -> float:
    """Fraction of known-site Cα within ``radius`` of the pocket hotspot centroid.

    Depends only on the single hotspot point, so it cannot be inflated by a larger
    pocket, and it tracks whether the detector centre falls on the ligandable core
    rather than the pocket mouth.
    """
    if pocket_centroid is None or not known_ca:
        return 0.0
    r2 = radius * radius
    cx, cy, cz = pocket_centroid
    hits = sum(
        1 for (x, y, z) in known_ca
        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2
    )
    return hits / len(known_ca)


def headline_hit(jac: float, centroid_dist: float,
                 jaccard_thr: float = JACCARD_THRESHOLD,
                 centroid_thr: float = CENTROID_THRESHOLD) -> bool:
    """Size-robust headline: Jaccard ≥ thr OR centroid ≤ thr."""
    return (jac >= jaccard_thr) or (centroid_dist <= centroid_thr)


def strict_localized_hit(jac: float, centroid_dist: float,
                         jaccard_thr: float = JACCARD_THRESHOLD,
                         centroid_thr: float = STRICT_CENTROID) -> bool:
    """Stricter AND-criterion, for ablations that claim localization gains."""
    return (jac >= jaccard_thr) and (centroid_dist <= centroid_thr)


def summarize_topk(per_target_hits_by_k: dict[int, list[bool]]) -> dict[int, float]:
    """Given {k: [hit per target]} return {k: hit-rate}. Should be monotone in k."""
    return {
        k: (sum(hits) / len(hits) if hits else 0.0)
        for k, hits in sorted(per_target_hits_by_k.items())
    }


def paired_bootstrap_ci(
    hits: Sequence[bool],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) of a per-target hit rate via bootstrap resampling.

    Resamples targets (not residues) so the interval reflects target-level
    uncertainty - the unit an honest cryptic-pocket claim is made over.
    """
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(hits) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(hits[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return mean, lo, hi
