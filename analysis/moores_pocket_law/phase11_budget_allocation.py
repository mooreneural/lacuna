"""Phase 11: how to spend a fixed candidate budget across four detectors.

The union of the four detectors covers 92.1% of targets while the best single
tool surfaces 66.3% in its top five. Turning any of that gap into recovery means
spending a budget across tools rather than within one, and the question is what
allocation to use.

This needs no cross-detector candidate matching, which is fortunate because the
stored dumps do not support it. Asking whether *any* selected candidate qualifies
requires only each candidate's overlap with the true site, which is stored. If
two tools both propose the correct site then both slots hit, so the hit rate is
exact; only the efficiency interpretation is affected.

Three strategies at each total budget B:

  single       the best individual tool's top-B
  round-robin  rank 1 from every tool, then rank 2 from every tool, and so on
  fitted       the fixed allocation chosen on the train fold

The fitted allocation is searched over the train fold and reported on the test
fold, so the reported number is not the maximum of a search on its own data.

    python analysis/moores_pocket_law/phase11_budget_allocation.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import JACCARD_THRESHOLD as T
from dataio import METHODS, boot_ci

HERE = Path(__file__).resolve().parent
SEED = 0
BUDGETS = list(range(1, 21))
HEADLINE_B = 5


def hit_from_allocation(per_method, alloc):
    """Does any candidate qualify, taking alloc[m] candidates from tool m?"""
    for m, k in zip(METHODS, alloc):
        if k and any(j >= T for j in per_method[m][:k]):
            return True
    return False


def round_robin_alloc(B):
    """Interleave the four lists: 1 each, then 2 each, and so on."""
    alloc = [0] * len(METHODS)
    for i in range(B):
        alloc[i % len(METHODS)] += 1
    return tuple(alloc)


def allocations(B):
    """Every way to split B candidates across four tools."""
    for a in product(range(B + 1), repeat=len(METHODS)):
        if sum(a) == B:
            yield a


def evaluate(targets, cands, alloc):
    return [hit_from_allocation(cands[t], alloc) for t in targets]


def main() -> None:
    cands = dataio.load_candidates()
    folds = dataio.fold_map()
    paired = {s: d for s, d in cands.items() if all(m in d for m in METHODS)}
    train = sorted(s for s in paired if dataio.split_of(s, folds) == "train")
    test = sorted(s for s in paired if dataio.split_of(s, folds) == "test")
    print("train %d, test %d\n" % (len(train), len(test)))

    union = np.mean([any(any(j >= T for j in paired[s][m]) for m in METHODS)
                     for s in test])
    print("union coverage on test (the ceiling): %.1f%%\n" % (100 * union))

    out = {"seed": SEED, "n_train": len(train), "n_test": len(test),
           "union_coverage_test": float(union), "budgets": {}}

    print("=" * 78)
    print("recovery at each total budget B, test fold")
    print("=" * 78)
    print("%-4s %22s %22s %22s" %
          ("B", "best single tool", "round-robin", "fitted on train"))
    for B in BUDGETS:
        # best single tool, chosen on train, reported on test
        best_single_m = max(
            METHODS,
            key=lambda m: np.mean([any(j >= T for j in paired[s][m][:B])
                                   for s in train]))
        single_alloc = tuple(B if m == best_single_m else 0 for m in METHODS)
        rr_alloc = round_robin_alloc(B)
        fitted_alloc = max(allocations(B),
                           key=lambda a: np.mean(evaluate(train, paired, a)))

        row = {}
        for name, alloc in (("single", single_alloc), ("round_robin", rr_alloc),
                            ("fitted", fitted_alloc)):
            m_, lo, hi = boot_ci(evaluate(test, paired, alloc), seed=SEED)
            row[name] = {"alloc": list(alloc), "mean": m_, "ci": [lo, hi]}
        out["budgets"][str(B)] = row
        row["single"]["tool"] = best_single_m

        def fmt(d):
            return "%5.1f%% [%4.1f,%5.1f]" % (100 * d["mean"], 100 * d["ci"][0],
                                              100 * d["ci"][1])
        print("%-4d %22s %22s %22s   %s" %
              (B, fmt(row["single"]), fmt(row["round_robin"]),
               fmt(row["fitted"]), fitted_alloc))

    print("\n" + "=" * 78)
    print("at the conventional budget of five")
    print("=" * 78)
    r = out["budgets"][str(HEADLINE_B)]
    names = dict(zip(METHODS, r["fitted"]["alloc"]))
    print("  best single tool  : %s top-5, %.1f%%" %
          (r["single"]["tool"], 100 * r["single"]["mean"]))
    print("  round-robin       : %.1f%%" % (100 * r["round_robin"]["mean"]))
    print("  fitted allocation : %.1f%%   %s"
          % (100 * r["fitted"]["mean"],
             ", ".join("%s=%d" % (k, v) for k, v in names.items() if v)))
    print("  union ceiling     : %.1f%%" % (100 * union))

    # Paired test of the fitted allocation against the best single tool.
    a = np.array(evaluate(test, paired, tuple(r["fitted"]["alloc"])), float)
    b = np.array(evaluate(test, paired,
                          tuple(HEADLINE_B if m == r["single"]["tool"] else 0
                                for m in METHODS)), float)
    d = a - b
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(20000, len(d)))
    bm = np.sort(d[idx].mean(axis=1))
    lo, hi = float(bm[500]), float(bm[19499])
    out["headline_paired_delta"] = {"delta": float(d.mean()), "ci": [lo, hi]}
    print("\n  fitted minus best single, paired on test: %+.1f points [%+.1f, %+.1f]%s"
          % (100 * d.mean(), 100 * lo, 100 * hi,
             "  <-- excludes 0" if (lo > 0 or hi < 0) else ""))

    (HERE / "phase11_budget.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase11_budget.json"))


if __name__ == "__main__":
    main()
