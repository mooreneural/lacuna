"""Phase 14: is the irreducible remainder a detection limit or a metric artifact?

Between six detectors, 7.8% of test-fold targets have no qualifying candidate
anywhere. Read plainly that is the field's detection frontier. But those targets
have a median annotated site of 5 residues against 15 for the rest, which
suggests something duller: Jaccard cannot reward a hit on a very small
annotation, because

    J = |A n B| / |A u B| <= min(|A|, |B|) / max(|A|, |B|)

so a 2-residue site scored against a 15-residue pocket tops out at 0.13,
whatever the pocket is. The site is unhittable by construction, not undetected.

Three tests separate those readings.

1. The ceiling. For each target, the best Jaccard any of its actual candidates
   could reach if it contained the site perfectly. If that ceiling is below the
   threshold the target was never scoreable.

2. The centroid criterion. Distance is size-independent, so a site that is
   located but not scoreable will be found by centroid and missed by Jaccard.
   This is only possible now that centroids are collected per candidate.

3. The decomposition at a size floor. If the ranking conclusions hold on sites
   the metric can actually score, the argument survives; if they evaporate, the
   argument was resting on unscoreable targets.

    python analysis/moores_pocket_law/phase14_small_sites.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio_v2 as D
from dataio_v2 import JACCARD_THRESHOLD as T
from dataio_v2 import LABEL, METHODS, boot_ci

HERE = Path(__file__).resolve().parent
CENTROID_A = 4.0          # the repository's size-robust radius
FLOORS = (0, 4, 6, 8, 10)


def ceiling(rec):
    """Best Jaccard reachable given this target's actual candidate sizes."""
    s = rec["n_known"]
    best = 0.0
    for labels in rec["residues_by_rank"]:
        c = len(labels)
        if c:
            best = max(best, min(s, c) / max(s, c))
    return best


def centroid_hit(rec, k=None, radius=CENTROID_A):
    """Does any candidate centroid land within `radius` of the site centroid?"""
    site = rec.get("site_centroid")
    if not site:
        return False
    site = np.asarray(site, float)
    cs = D.centroids(rec)
    if k is not None:
        cs = cs[:k]
    if not len(cs):
        return False
    d = np.linalg.norm(cs - site, axis=1)
    return bool(np.nanmin(d) <= radius) if np.isfinite(d).any() else False


def main() -> None:
    out = {}
    for fold in ("test", "train"):
        data = D.paired(fold)
        ids = sorted(data)
        sizes = np.array([data[i]["fpocket"]["n_known"] for i in ids])
        inv = np.array([not any(D.covered(data[i][m]) for m in METHODS) for i in ids])

        print("=" * 80)
        print("%s fold, n = %d" % (fold, len(ids)))
        print("=" * 80)

        # 1. ceiling ---------------------------------------------------------
        ceil = np.array([max(ceiling(data[i][m]) for m in METHODS) for i in ids])
        unscoreable = ceil < T
        print("\n1. Was the target ever scoreable?")
        print("   targets whose best possible Jaccard is below %.2f : %d (%.1f%%)"
              % (T, unscoreable.sum(), 100 * unscoreable.mean()))
        print("   of the %d invisible targets, %d (%.0f%%) were unscoreable"
              % (inv.sum(), (inv & unscoreable).sum(),
                 100 * (inv & unscoreable).sum() / max(inv.sum(), 1)))
        print("   of the %d covered targets,   %d were unscoreable"
              % ((~inv).sum(), (~inv & unscoreable).sum()))

        # 2. centroid --------------------------------------------------------
        cen_any = np.array([any(centroid_hit(data[i][m]) for m in METHODS) for i in ids])
        cen5 = np.array([any(centroid_hit(data[i][m], 5) for m in METHODS) for i in ids])
        print("\n2. Located but not scoreable? (centroid within %.0f A)" % CENTROID_A)
        print("   union by Jaccard          : %.1f%%" % (100 * (~inv).mean()))
        print("   union by centroid, any rank: %.1f%%" % (100 * cen_any.mean()))
        print("   union by centroid, top 5  : %.1f%%" % (100 * cen5.mean()))
        rescued = inv & cen_any
        print("   invisible by Jaccard but located by centroid: %d of %d (%.0f%%)"
              % (rescued.sum(), inv.sum(), 100 * rescued.sum() / max(inv.sum(), 1)))
        if rescued.any():
            print("   their median site size: %.0f residues"
                  % np.median(sizes[rescued]))

        # 3. decomposition at a floor ---------------------------------------
        print("\n3. Do the ranking conclusions hold on scoreable sites?")
        print("   %-8s %5s %9s %9s %9s %11s" %
              ("floor", "n", "cov span", "conv span", "union", "best top-5"))
        rows = {}
        for f in FLOORS:
            keep = [i for i in ids if data[i]["fpocket"]["n_known"] >= f]
            if len(keep) < 40:
                continue
            covs, convs, t5s = [], [], []
            for m in METHODS:
                recs = [data[i][m] for i in keep]
                c = np.mean([D.covered(r) for r in recs])
                t = np.mean([D.hit(r, 5) for r in recs])
                covs.append(c); convs.append(t / c if c else np.nan); t5s.append(t)
            u = np.mean([any(D.covered(data[i][m]) for m in METHODS) for i in keep])
            rows[str(f)] = {"n": len(keep), "union": float(u),
                            "coverage_span": float(max(covs) - min(covs)),
                            "conversion_span": float(max(convs) - min(convs)),
                            "best_top5": float(max(t5s))}
            print("   >=%-6d %5d %8.1f%% %8.1f%% %8.1f%% %10.1f%%"
                  % (f, len(keep), 100 * (max(covs) - min(covs)),
                     100 * (max(convs) - min(convs)), 100 * u, 100 * max(t5s)))

        out[fold] = {
            "n": len(ids),
            "invisible": int(inv.sum()),
            "unscoreable": int(unscoreable.sum()),
            "invisible_and_unscoreable": int((inv & unscoreable).sum()),
            "union_jaccard": float((~inv).mean()),
            "union_centroid_any": float(cen_any.mean()),
            "union_centroid_top5": float(cen5.mean()),
            "rescued_by_centroid": int(rescued.sum()),
            "by_floor": rows,
            "site_size_invisible_median": float(np.median(sizes[inv])) if inv.any() else None,
            "site_size_covered_median": float(np.median(sizes[~inv])),
        }
        print()

    (HERE / "phase14_small_sites.json").write_text(json.dumps(out, indent=1))
    print("wrote %s" % (HERE / "phase14_small_sites.json"))


if __name__ == "__main__":
    main()
