"""Phase 13: the six-detector table, and what the union is worth.

This is v2's headline. It differs from v1's in three ways worth stating plainly
when it is written up.

Six detectors rather than four. MDpocket is added because it is the closest
relative of an ensemble method and, given the identical ensemble, isolates
detection and clustering from sampling. Lacuna's two rankers are separated
because they share candidates and differ only in ordering, which makes them the
cleanest possible demonstration that coverage and conversion are independent.

One platform. Every row here was produced by the same linear algebra, which was
not true of v1.

And the union is computed over six, so the ceiling and the irreducible remainder
both move.

    python analysis/moores_pocket_law/phase13_v2_table.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio_v2 as D
from dataio_v2 import KS, LABEL, METHODS, boot_ci

HERE = Path(__file__).resolve().parent
V1_FOUR = ("fpocket", "p2rank", "ifsitepred", "lacuna_plm")


def table(data, methods, title):
    ids = sorted(data)
    print("\n" + "=" * 84)
    print("%s   n = %d" % (title, len(ids)))
    print("=" * 84)
    print("%-13s %6s %9s %22s %9s" %
          ("detector", "cands", "coverage", "top-5 (95% CI)", "conversion"))
    print("-" * 84)
    out = {}
    for m in methods:
        recs = [data[i][m] for i in ids]
        n_c = float(np.mean([r["n_prop"] for r in recs]))
        cov, clo, chi = boot_ci([D.covered(r) for r in recs])
        t5, tlo, thi = boot_ci([D.hit(r, 5) for r in recs])
        conv = t5 / cov if cov else float("nan")
        print("%-13s %6.1f %8.1f%% %10.1f%% [%4.1f, %4.1f] %8.1f%%"
              % (LABEL[m], n_c, 100 * cov, 100 * t5, 100 * tlo, 100 * thi, 100 * conv))
        out[m] = {"n_candidates": n_c, "coverage": cov, "coverage_ci": [clo, chi],
                  "top5": t5, "top5_ci": [tlo, thi], "conversion": conv,
                  "topk": {str(k): boot_ci([D.hit(r, k) for r in recs])[0] for k in KS}}
    covs = [out[m]["coverage"] for m in methods]
    convs = [out[m]["conversion"] for m in methods]
    print("-" * 84)
    print("  coverage spans %.1f points; conversion spans %.1f points"
          % (100 * (max(covs) - min(covs)), 100 * (max(convs) - min(convs))))
    return out


def union_analysis(data, methods, label):
    ids = sorted(data)
    print("\n" + "=" * 84)
    print("union coverage, %s" % label)
    print("=" * 84)
    u = [any(D.covered(data[i][m]) for m in methods) for i in ids]
    um, ulo, uhi = boot_ci(u)
    best_t5 = max(np.mean([D.hit(data[i][m], 5) for i in ids]) for m in methods)
    best_cov = max(np.mean([D.covered(data[i][m]) for i in ids]) for m in methods)
    print("  union of all %d          : %.1f%% [%.1f, %.1f]"
          % (len(methods), 100 * um, 100 * ulo, 100 * uhi))
    print("  best single coverage      : %.1f%%" % (100 * best_cov))
    print("  best single top-5         : %.1f%%" % (100 * best_t5))
    print("  invisible to everything   : %.1f%%" % (100 * (1 - um)))
    print("  headroom, best top-5 to union coverage: %.1f points"
          % (100 * (um - best_t5)))

    print("\n  best pair, by union coverage:")
    pairs = sorted(((np.mean([any(D.covered(data[i][m]) for m in pair) for i in ids]), pair)
                    for pair in combinations(methods, 2)), reverse=True)[:4]
    for v, pair in pairs:
        print("    %-28s %.1f%%" % (" + ".join(LABEL[m] for m in pair), 100 * v))
    return {"union": um, "union_ci": [ulo, uhi], "best_single_top5": best_t5,
            "best_single_coverage": best_cov, "irreducible": 1 - um,
            "headroom": um - best_t5}


def main() -> None:
    out = {}
    for fold in ("test", "train"):
        data = D.paired(fold)
        if not data:
            print("no paired data for %s" % fold)
            continue
        out[fold] = {
            "n": len(data),
            "six": table(data, METHODS, "%s fold, all six" % fold),
            "union_six": union_analysis(data, METHODS, "six detectors"),
            "union_v1_four": union_analysis(data, V1_FOUR,
                                            "the four in v1, on this data"),
        }
    (HERE / "phase13_v2_table.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote %s" % (HERE / "phase13_v2_table.json"))


if __name__ == "__main__":
    main()
