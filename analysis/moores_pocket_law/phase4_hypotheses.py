"""Phase 4: H1-H6, the observational tests.

Every result here is observational and shares one weakness, stated once so it is
not restated at every turn. Candidate burden N is not assigned; it is a property
of a method-target pair. Coverage rises with N almost mechanically, since more
candidates give more chances to hold a qualifying one. Conditioning on coverage
and then regressing conversion on N therefore conditions on a collider: among
covered targets, low-N cases were covered easily, while high-N cases may be
covered only because the extra candidates caught a marginal site that a ranker
was always going to bury. That alone produces a declining conversion curve with
no competition effect whatsoever.

Nothing in this file can separate those. phase6_sweep.py can, because it varies
N experimentally on fixed targets.

    python analysis/moores_pocket_law/phase4_hypotheses.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import KS, METHODS, boot_ci

HERE = Path(__file__).resolve().parent
SEED = 0
BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 10 ** 6)]


def load_rows():
    import csv
    rows = []
    with open(HERE / "results.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("protein_length", "site_size", "total_candidate_count",
                      "qualifying_candidate_exists", "best_qualifying_rank",
                      "number_of_qualifying_candidates", "hit_at_1", "hit_at_3",
                      "hit_at_5", "hit_at_10", "hit_at_20"):
                r[k] = int(r[k]) if r[k] not in ("", "None") else None
            r["best_jaccard"] = float(r["best_jaccard"])
            r["candidate_density"] = (float(r["candidate_density"])
                                      if r["candidate_density"] not in ("", "None")
                                      else None)
            rows.append(r)
    return rows


def bin_label(lo, hi):
    return "%d-%d" % (lo, hi) if hi < 10 ** 6 else "%d+" % lo


def h1_conversion_vs_burden(rows, out):
    """P(hit@k | a qualifying candidate exists, N)."""
    print("=" * 74)
    print("H1  conversion among covered targets, versus candidate burden")
    print("=" * 74)
    res = {}
    covered = [r for r in rows if r["qualifying_candidate_exists"]]

    for scope in ("pooled",) + METHODS:
        sub = covered if scope == "pooled" else [r for r in covered
                                                if r["method"] == scope]
        if len(sub) < 40:
            continue
        n = np.array([r["total_candidate_count"] for r in sub], dtype=float)
        entry = {"n_covered": len(sub), "spearman": {}, "bins": {}}
        print("\n%s  (n covered = %d)" % (scope, len(sub)))
        for k in KS:
            y = np.array([r["hit_at_%d" % k] for r in sub], dtype=float)
            rho, p = stats.spearmanr(n, y)
            entry["spearman"]["hit_at_%d" % k] = {"rho": float(rho),
                                                  "p": float(p)}
            print("   k=%-3d spearman(N, hit) rho = %+.3f   p = %.2g"
                  % (k, rho, p))

        for lo, hi in BINS:
            sel = [r for r in sub if lo <= r["total_candidate_count"] <= hi]
            if len(sel) < 15:
                continue
            row = {"n": len(sel),
                   "mean_N": float(np.mean([r["total_candidate_count"]
                                            for r in sel]))}
            for k in KS:
                m, lo_ci, hi_ci = boot_ci([r["hit_at_%d" % k] for r in sel],
                                          seed=SEED)
                row["hit_at_%d" % k] = [m, lo_ci, hi_ci]
            entry["bins"][bin_label(lo, hi)] = row
        if scope == "pooled" or scope == "lacuna":
            print("   %-8s %5s %24s" % ("N bin", "n", "conversion@5 (95% CI)"))
            for lab, row in entry["bins"].items():
                m, l, h = row["hit_at_5"]
                print("   %-8s %5d      %5.1f%% [%4.1f, %5.1f]"
                      % (lab, row["n"], 100 * m, 100 * l, 100 * h))
        res[scope] = entry
    out["H1"] = res
    return res


def h2_rank_vs_burden(rows, out):
    """Does a bigger candidate set bury the true site deeper?"""
    print("\n" + "=" * 74)
    print("H2  depth of the first qualifying candidate, versus burden")
    print("=" * 74)
    res = {}
    for scope in ("pooled",) + METHODS:
        sub = [r for r in rows if r["qualifying_candidate_exists"]
               and (scope == "pooled" or r["method"] == scope)]
        if len(sub) < 40:
            continue
        n = np.array([r["total_candidate_count"] for r in sub], dtype=float)
        rank = np.array([r["best_qualifying_rank"] for r in sub], dtype=float)
        frac = rank / n
        r_abs, p_abs = stats.spearmanr(n, rank)
        r_rel, p_rel = stats.spearmanr(n, frac)
        res[scope] = {
            "n": len(sub),
            "spearman_N_vs_absolute_rank": {"rho": float(r_abs), "p": float(p_abs)},
            "spearman_N_vs_relative_rank": {"rho": float(r_rel), "p": float(p_rel)},
            "median_absolute_rank": float(np.median(rank)),
            "median_relative_rank": float(np.median(frac)),
        }
        print("  %-11s rho(N, rank)=%+.3f  rho(N, rank/N)=%+.3f   "
              "median rank=%.0f  median rank/N=%.3f"
              % (scope, r_abs, r_rel, np.median(rank), np.median(frac)))
    out["H2"] = res
    return res


def h3_budget_curves(rows, out):
    """Recovery and conversion-of-coverage as functions of the budget k."""
    print("\n" + "=" * 74)
    print("H3  budget curves")
    print("=" * 74)
    cands = dataio.load_candidates()
    folds = dataio.fold_map()
    res = {}
    ks = list(range(1, 51))
    for scope in METHODS:
        ids = sorted({r["target_id"] for r in rows if r["method"] == scope})
        jl = {s: cands[s][scope] for s in ids}
        cov = np.mean([any(j >= dataio.JACCARD_THRESHOLD for j in jl[s])
                       for s in ids])
        rec, conv = [], []
        for k in ks:
            r = np.mean([any(j >= dataio.JACCARD_THRESHOLD for j in jl[s][:k])
                         for s in ids])
            rec.append(float(r))
            conv.append(float(r / cov) if cov else float("nan"))
        res[scope] = {"k": ks, "recovery": rec, "conversion_of_coverage": conv,
                      "coverage": float(cov), "n": len(ids)}
        print("  %-11s coverage %.3f | recovery@1 %.3f @5 %.3f @20 %.3f | "
              "conv@5 %.3f" % (scope, cov, rec[0], rec[4], rec[19], conv[4]))
    out["H3"] = res
    return res


def h4_marginal_yield(rows, out):
    """Probability the FIRST qualifying candidate lands at each rank."""
    print("\n" + "=" * 74)
    print("H4  marginal yield of one more rank")
    print("=" * 74)
    res = {}
    for scope in METHODS:
        sub = [r for r in rows if r["method"] == scope]
        n_t = len(sub)
        hist = np.zeros(51)
        for r in sub:
            rk = r["best_qualifying_rank"]
            if rk is not None and rk <= 50:
                hist[rk] += 1
        marginal = (hist / n_t).tolist()
        res[scope] = {"n": n_t, "marginal_gain_by_rank": marginal[1:]}
        head = "  ".join("%d:%.3f" % (i, marginal[i]) for i in range(1, 9))
        print("  %-11s %s" % (scope, head))
    out["H4"] = res
    return res


def h5_failure_decomposition(rows, out):
    """Split the miss at k into never-proposed versus proposed-but-outranked."""
    print("\n" + "=" * 74)
    print("H5  what the misses are made of, at k=5")
    print("=" * 74)
    res = {}
    print("  %-11s %9s %9s %9s %9s" %
          ("method", "hit@5", "rank-fail", "detect-fail", "conv"))
    for scope in METHODS:
        sub = [r for r in rows if r["method"] == scope]
        n = len(sub)
        hit = sum(r["hit_at_5"] for r in sub) / n
        det = sum(1 for r in sub if not r["qualifying_candidate_exists"]) / n
        rank = sum(1 for r in sub if r["qualifying_candidate_exists"]
                   and not r["hit_at_5"]) / n
        res[scope] = {"n": n, "hit_at_5": hit, "ranking_failure": rank,
                      "detection_failure": det,
                      "conversion": hit / (hit + rank) if (hit + rank) else None}
        print("  %-11s %8.1f%% %8.1f%% %8.1f%% %8.1f%%"
              % (scope, 100 * hit, 100 * rank, 100 * det,
                 100 * res[scope]["conversion"]))
    out["H5"] = res
    return res


def h6_breakpoint(rows, out):
    """Search for a burden threshold. Post-hoc by construction."""
    print("\n" + "=" * 74)
    print("H6  is there a defensible breakpoint in N?")
    print("=" * 74)
    covered = [r for r in rows if r["qualifying_candidate_exists"]]
    n = np.array([r["total_candidate_count"] for r in covered], dtype=float)
    y = np.array([r["hit_at_5"] for r in covered], dtype=float)

    grid = list(range(5, 31))
    gaps = {}
    for t in grid:
        lo, hi = y[n <= t], y[n > t]
        if len(lo) < 30 or len(hi) < 30:
            continue
        gaps[t] = float(lo.mean() - hi.mean())
    best = max(gaps, key=gaps.get)
    print("  descriptive best split: N <= %d  (gap %.1f points)"
          % (best, 100 * gaps[best]))

    rng = np.random.default_rng(SEED)
    picks = []
    idx = np.arange(len(n))
    for _ in range(2000):
        b = rng.choice(idx, size=len(idx), replace=True)
        nb, yb = n[b], y[b]
        g = {}
        for t in grid:
            lo, hi = yb[nb <= t], yb[nb > t]
            if len(lo) < 30 or len(hi) < 30:
                continue
            g[t] = lo.mean() - hi.mean()
        if g:
            picks.append(max(g, key=g.get))
    picks = np.array(picks)
    lo_q, hi_q = np.percentile(picks, [2.5, 97.5])
    mode = int(stats.mode(picks, keepdims=False).mode)
    print("  bootstrap over the same grid: selected threshold 95%% interval "
          "[%.0f, %.0f], modal %d" % (lo_q, hi_q, mode))
    print("  fraction of resamples selecting the grid edge (5 or 30): %.1f%%"
          % (100 * np.mean((picks == grid[0]) | (picks == grid[-1]))))

    # Is a step actually better than a smooth trend?
    from sklearn.linear_model import LogisticRegression
    Xc = np.column_stack([np.log(n)])
    Xs = np.column_stack([(n <= best).astype(float)])
    lls = {}
    for name, X in (("log-linear in N", Xc), ("step at best N", Xs)):
        m = LogisticRegression(max_iter=1000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        k_par = X.shape[1] + 1
        lls[name] = {"loglik": ll, "aic": float(2 * k_par - 2 * ll)}
        print("  %-18s loglik %.2f  AIC %.2f" % (name, ll, lls[name]["aic"]))

    out["H6"] = {"grid": grid, "gaps": gaps, "descriptive_best": best,
                 "bootstrap_ci": [float(lo_q), float(hi_q)], "modal": mode,
                 "edge_fraction": float(np.mean((picks == grid[0]) |
                                                (picks == grid[-1]))),
                 "model_comparison": lls}
    return out["H6"]


def main() -> None:
    rows = load_rows()
    out = {"seed": SEED, "n_rows": len(rows),
           "n_targets": len({r["target_id"] for r in rows}),
           "criterion": "jaccard >= %.2f, no centroid clause"
                        % dataio.JACCARD_THRESHOLD}
    h1_conversion_vs_burden(rows, out)
    h2_rank_vs_burden(rows, out)
    h3_budget_curves(rows, out)
    h4_marginal_yield(rows, out)
    h5_failure_decomposition(rows, out)
    h6_breakpoint(rows, out)
    (HERE / "phase4_results.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase4_results.json"))


if __name__ == "__main__":
    main()
