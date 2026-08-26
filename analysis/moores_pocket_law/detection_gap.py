"""Why does Lacuna miss the sites it misses?

Ranking is exhausted: eleven contrasts, nothing beat the shipped design, and the
conversion ceiling turned out to belong to the cryptic regime rather than to the
ranker. Coverage was never touched, and that is where the room is. Lacuna covers
71.5% of the CryptoBench test fold while the union of five generators covers
92.2%, and union coverage reaches 98.6% once sites of at least eight residues
are required. Small sites are the named failure mode.

Before changing a detector, the failure has to be diagnosed, because two very
different problems produce the same missed target and they need opposite fixes:

  localisation   Lacuna proposes nothing at the site at all. The cavity is below
                 whatever threshold the detector applies, so the fix is
                 sensitivity: more candidates, at a cost in conversion.

  boundary       Lacuna proposes a candidate in the right place whose lining
                 residues do not overlap the annotation enough to qualify. The
                 fix is how a candidate's residues are assigned, which costs
                 nothing in candidate count and so nothing in conversion.

The distinction is measurable. Every row carries centroid_by_rank and
site_centroid, so distance from the annotated site to Lacuna's nearest proposal
separates the two directly. A miss with a proposal 3 A away is a boundary
problem. A miss with nothing within 15 A is a localisation problem.

Reported alongside: which other detectors cover the same targets, since a site
that only fpocket finds says something different from one nothing finds.

    python analysis/moores_pocket_law/detection_gap.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataio
from dataio import JACCARD_THRESHOLD as T

CAND = HERE / "data" / "candidates" / "candidates_train.jsonl"
METHODS = ("fpocket", "p2rank", "ifsitepred", "mdpocket", "lacuna")
NEAR = 8.0          # A; within this, a proposal is at the site


def load():
    by = {}
    with open(CAND, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by.setdefault(r["id"], {})[r["tool"]] = r
    return by


def nearest(rec):
    """Distance from the annotated site centroid to the closest proposal."""
    sc = rec.get("site_centroid")
    cents = [c for c in (rec.get("centroid_by_rank") or []) if c]
    if not sc or not cents:
        return None, None
    d = np.linalg.norm(np.asarray(cents, float) - np.asarray(sc, float), axis=1)
    k = int(np.argmin(d))
    return float(d[k]), k


def main() -> None:
    data = load()
    full = {s: v for s, v in data.items() if all(m in v for m in METHODS)}
    print("targets with all five generators: %d\n" % len(full))

    covered = {m: {s for s, v in full.items()
                   if any(float(x) >= T for x in v[m]["jac_by_rank"])}
               for m in METHODS}
    union = set().union(*covered.values())
    print("%-12s %8s" % ("detector", "coverage"))
    for m in METHODS:
        print("  %-10s %7.1f%%" % (m, 100 * len(covered[m]) / len(full)))
    print("  %-10s %7.1f%%" % ("union", 100 * len(union) / len(full)))

    # Targets Lacuna misses but something else finds: the recoverable gap.
    gap = sorted((union - covered["lacuna"]))
    lost = sorted(set(full) - union)
    print("\nLacuna misses %d targets; %d of those another detector covers,"
          % (len(full) - len(covered["lacuna"]), len(gap)))
    print("and %d are missed by everything.\n" % len(lost))

    rows = []
    for s in gap:
        rec = full[s]["lacuna"]
        d, k = nearest(rec)
        jac = [float(x) for x in rec["jac_by_rank"]]
        rows.append({"id": s, "dist": d, "best_jac": max(jac, default=0.0),
                     "n_site": len(rec.get("site_residues") or []),
                     "n_prop": len(jac),
                     "who": [m for m in METHODS if m != "lacuna" and s in covered[m]]})

    have = [r for r in rows if r["dist"] is not None]
    near = [r for r in have if r["dist"] <= NEAR]
    print("of the %d recoverable misses, Lacuna's nearest proposal is:" % len(have))
    for lo, hi, lab in ((0, 4, "within 4 A   (boundary)"),
                        (4, 8, "4 to 8 A     (boundary-ish)"),
                        (8, 15, "8 to 15 A"),
                        (15, 1e9, "beyond 15 A  (localisation)")):
        n = sum(1 for r in have if lo <= r["dist"] < hi)
        print("    %-28s %4d  %5.1f%%" % (lab, n, 100 * n / len(have)))
    print("\n  median distance to nearest proposal : %.1f A"
          % float(np.median([r["dist"] for r in have])))
    print("  median best Jaccard achieved        : %.3f"
          % float(np.median([r["best_jac"] for r in have])))
    print("  median annotated site size          : %d residues"
          % int(np.median([r["n_site"] for r in have])))
    print("  median candidates proposed          : %d"
          % int(np.median([r["n_prop"] for r in have])))

    print("\n  on the %d where a proposal sits within %.0f A:" % (len(near), NEAR))
    if near:
        print("    median best Jaccard   : %.3f"
              % float(np.median([r["best_jac"] for r in near])))
        print("    median site size      : %d residues"
              % int(np.median([r["n_site"] for r in near])))
        b = np.array([r["best_jac"] for r in near])
        print("    best Jaccard in [0.15, 0.25) : %d of %d  (%.1f%%)"
              % (int(((b >= 0.15) & (b < T)).sum()), len(b),
                 100 * ((b >= 0.15) & (b < T)).mean()))

    print("\n  which detectors cover the gap:")
    for m, n in Counter(w for r in rows for w in r["who"]).most_common():
        print("    %-12s %4d" % (m, n))

    # Site size is the stated failure mode; check it holds for Lacuna alone.
    cs = [len(full[s]["lacuna"].get("site_residues") or []) for s in covered["lacuna"]]
    ms = [len(full[s]["lacuna"].get("site_residues") or [])
          for s in set(full) - covered["lacuna"]]
    print("\n  annotated site size, median residues")
    print("    Lacuna covers    : %d  (n=%d)" % (int(np.median(cs)), len(cs)))
    print("    Lacuna misses    : %d  (n=%d)" % (int(np.median(ms)), len(ms)))

    out = {"n_targets": len(full), "n_gap": len(gap), "n_lost": len(lost),
           "near_frac": (len(near) / len(have)) if have else None,
           "median_dist": float(np.median([r["dist"] for r in have])) if have else None}
    (HERE / "detection_gap.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "detection_gap.json"))


if __name__ == "__main__":
    main()
