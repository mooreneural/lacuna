"""Phase 2: reproduce the published test-fold numbers from candidate-level data.

Confirmatory. If any value here disagrees with paper/data/analysis.json beyond
rounding, the rest of the analysis is built on a different dataset than the
paper and must stop.

    python analysis/moores_pocket_law/phase2_reproduce.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import JACCARD_THRESHOLD, METHODS, boot_ci

TARGET = {
    "fpocket":    {"coverage": 0.742, "top5": 0.438},
    "p2rank":     {"coverage": 0.663, "top5": 0.635},
    "ifsitepred": {"coverage": 0.708, "top5": 0.618},
    "lacuna":     {"coverage": 0.736, "top5": 0.663},
}
UNION_TARGET = 0.921
TOL = 0.002          # published values are quoted to one decimal place


def hit(jacs, k):
    return any(j >= JACCARD_THRESHOLD for j in jacs[:k])


def main() -> int:
    cands = dataio.load_candidates()
    folds = dataio.fold_map()
    paired = {s: d for s, d in cands.items()
              if all(m in d for m in METHODS)}
    test = {s: d for s, d in paired.items()
            if dataio.split_of(s, folds) == "test"}
    ids = sorted(test)
    n = len(ids)

    print("cohort: n = %d paired test-fold structures "
          "(published: 178)\n" % n)
    print("%-12s %-22s %-22s %8s %10s" %
          ("method", "coverage (95% CI)", "top-5 (95% CI)", "conv", "cands"))
    print("-" * 78)

    ok = True
    summary = {"n": n, "methods": {}}
    for m in METHODS:
        cov = [hit(test[s][m], 10 ** 6) for s in ids]
        t5 = [hit(test[s][m], 5) for s in ids]
        ncand = [len(test[s][m]) for s in ids]
        cm, clo, chi = boot_ci(cov)
        tm, tlo, thi = boot_ci(t5)
        conv = tm / cm if cm else float("nan")
        print("%-12s %5.1f%% [%4.1f, %4.1f]      %5.1f%% [%4.1f, %4.1f]     "
              "%5.1f%% %9.1f" %
              (m, 100 * cm, 100 * clo, 100 * chi,
               100 * tm, 100 * tlo, 100 * thi,
               100 * conv, sum(ncand) / n))
        summary["methods"][m] = {
            "coverage": cm, "coverage_ci": [clo, chi],
            "top5": tm, "top5_ci": [tlo, thi],
            "conversion_at_5": conv,
            "mean_candidates": sum(ncand) / n,
        }
        for field, got in (("coverage", cm), ("top5", tm)):
            want = TARGET[m][field]
            if abs(got - want) > TOL:
                print("    MISMATCH %s %s: got %.4f, published %.3f"
                      % (m, field, got, want))
                ok = False

    union = [any(hit(test[s][m], 10 ** 6) for m in METHODS) for s in ids]
    um, ulo, uhi = boot_ci(union)
    print("\nunion coverage: %.1f%% [%.1f, %.1f]   (published %.1f%%)"
          % (100 * um, 100 * ulo, 100 * uhi, 100 * UNION_TARGET))
    summary["union_coverage"] = um
    summary["union_coverage_ci"] = [ulo, uhi]
    if abs(um - UNION_TARGET) > TOL:
        print("    MISMATCH union coverage")
        ok = False

    # Cross-check directly against the paper's own derived artifact.
    ref = json.loads((dataio.REPO / "paper/data/analysis.json").read_text())
    rt = ref["splits"]["test"]
    print("\ncross-check against paper/data/analysis.json:")
    print("  n: %d vs %d" % (n, rt["n"]))
    for m in METHODS:
        got5 = summary["methods"][m]["top5"]
        want5 = rt["tools"][m]["topk"]["5"][0]
        gotc = summary["methods"][m]["coverage"]
        wantc = rt["tools"][m]["oracle"][0]
        flag = "" if (abs(got5 - want5) < 1e-9 and abs(gotc - wantc) < 1e-9) \
            else "   <-- DIFFERS"
        print("  %-11s top5 %.6f vs %.6f | coverage %.6f vs %.6f%s"
              % (m, got5, want5, gotc, wantc, flag))
        if flag:
            ok = False

    print("\nPHASE 2: %s" % ("REPRODUCED" if ok else "FAILED"))
    out = Path(__file__).resolve().parent / "phase2_reproduction.json"
    out.write_text(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
