"""Every number the v2 manuscript quotes, emitted from the data in one place.

The v1 manuscript was written against analysis.json for the same reason: a
figure and a sentence that disagree is the failure mode this guards against.
Nothing here is recalled; each value is computed from the six-detector
collection or read from a phase result written by the analysis scripts.

    python analysis/moores_pocket_law/v2_numbers.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataio_v2 as D
from dataio_v2 import METHODS, boot_ci

OUT = HERE / "v2_numbers.json"


def per_detector(data):
    ids = sorted(data)
    out = {}
    for m in METHODS:
        recs = [data[i][m] for i in ids]
        cov = boot_ci([D.covered(r) for r in recs])
        t5 = boot_ci([D.hit(r, 5) for r in recs])
        out[m] = {
            "label": D.LABEL[m],
            "n_candidates": float(np.mean([r["n_prop"] for r in recs])),
            "coverage": cov[0], "coverage_ci": [cov[1], cov[2]],
            "top5": t5[0], "top5_ci": [t5[1], t5[2]],
            "conversion": t5[0] / cov[0] if cov[0] else None,
            "topk": {str(k): float(np.mean([D.hit(r, k) for r in recs]))
                     for k in D.KS},
        }
    return out


def union_curve(data):
    """Greedy: add whichever detector adds most coverage next."""
    ids = sorted(data)
    chosen, remaining, curve = [], list(METHODS), []
    while remaining:
        best, best_v = None, -1.0
        for m in remaining:
            v = float(np.mean([any(D.covered(data[i][t]) for t in chosen + [m])
                               for i in ids]))
            if v > best_v:
                best, best_v = m, v
        chosen.append(best)
        remaining.remove(best)
        curve.append({"added": best, "label": D.LABEL[best], "union": best_v})
    return curve


def size_floor(data):
    ids = sorted(data)
    rows = {}
    for f in (0, 4, 6, 8, 10, 12):
        keep = [i for i in ids if data[i]["fpocket"]["n_known"] >= f]
        if len(keep) < 40:
            continue
        u = float(np.mean([any(D.covered(data[i][m]) for m in METHODS) for i in keep]))
        best = max(float(np.mean([D.hit(data[i][m], 5) for i in keep])) for m in METHODS)
        rows[str(f)] = {"n": len(keep), "union": u, "best_top5": best}
    return rows


def invisible(data):
    ids = sorted(data)
    inv = [i for i in ids if not any(D.covered(data[i][m]) for m in METHODS)]
    vis = [i for i in ids if i not in set(inv)]
    si = [data[i]["fpocket"]["n_known"] for i in inv]
    sv = [data[i]["fpocket"]["n_known"] for i in vis]
    return {"n_invisible": len(inv), "fraction": len(inv) / len(ids),
            "median_site_invisible": float(np.median(si)) if si else None,
            "median_site_covered": float(np.median(sv)),
            "examples": inv[:6]}


def best_pairs(data, k=4):
    ids = sorted(data)
    scored = []
    for pair in combinations(METHODS, 2):
        v = float(np.mean([any(D.covered(data[i][m]) for m in pair) for i in ids]))
        scored.append({"pair": [D.LABEL[m] for m in pair], "union": v})
    return sorted(scored, key=lambda d: -d["union"])[:k]


def main() -> None:
    out = {"criterion": "Jaccard >= %.2f, no centroid clause" % D.JACCARD_THRESHOLD,
           "methods": list(METHODS)}
    for fold in ("test", "train"):
        data = D.paired(fold)
        ids = sorted(data)
        u = boot_ci([any(D.covered(data[i][m]) for m in METHODS) for i in ids])
        out[fold] = {
            "n": len(data),
            "detectors": per_detector(data),
            "union": u[0], "union_ci": [u[1], u[2]],
            "irreducible": 1 - u[0],
            "union_curve": union_curve(data),
            "size_floor": size_floor(data),
            "invisible": invisible(data),
            "best_pairs": best_pairs(data),
        }

    # Results already computed by earlier phases, carried through so the
    # manuscript has one place to read from.
    for name, key in (("phase6_sweep.json", "sweep"),
                      ("phase10b_within_target_decoy.json", "decoy_within"),
                      ("phase10_decoy.json", "decoy_foreign"),
                      ("phase11_budget.json", "budget"),
                      ("phase12_platform.json", "platform"),
                      ("phase14_small_sites.json", "small_sites"),
                      ("phase4_results.json", "h1_h6")):
        p = HERE / name
        if p.exists():
            out[key] = json.loads(p.read_text())

    OUT.write_text(json.dumps(out, indent=1, default=float))

    t = out["test"]
    print("test fold, n = %d" % t["n"])
    print("  %-13s %6s %9s %9s %8s" % ("detector", "cands", "coverage", "top-5", "conv"))
    for m in METHODS:
        d = t["detectors"][m]
        print("  %-13s %6.1f %8.1f%% %8.1f%% %7.1f%%"
              % (d["label"], d["n_candidates"], 100 * d["coverage"],
                 100 * d["top5"], 100 * d["conversion"]))
    print("  union %.1f%%  irreducible %.1f%%" % (100 * t["union"], 100 * t["irreducible"]))
    print("  invisible sites: median %.0f residues vs %.0f covered"
          % (t["invisible"]["median_site_invisible"],
             t["invisible"]["median_site_covered"]))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
