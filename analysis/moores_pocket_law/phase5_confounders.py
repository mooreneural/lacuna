"""Phase 5: confounders, and a direct test of the efficiency claim.

Two questions.

Does the burden effect survive protein length and method identity? Candidate
count correlates with both, and method identity is nearly collinear with burden
(P2Rank's median is 6, IF-SitePred's is 23), so a pooled coefficient could be
method identity wearing a numeric disguise.

And does eta = dR/dC actually decline with burden? That is the quantitative form
of the hypothesis, and it is separable from "conversion falls as N rises". The
sweep gives it directly: each conformer step buys some coverage and some
recovery, and their ratio is the efficiency at that burden.

    python analysis/moores_pocket_law/phase5_confounders.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import METHODS

HERE = Path(__file__).resolve().parent
SEED = 0


def load_covered():
    rows = []
    with open(HERE / "results.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["qualifying_candidate_exists"] != "1":
                continue
            if r["protein_length"] in ("", "None"):
                continue
            rows.append({
                "target_id": r["target_id"],
                "method": r["method"],
                "N": float(r["total_candidate_count"]),
                "L": float(r["protein_length"]),
                "site": float(r["site_size"]),
                "y": float(r["hit_at_5"]),
            })
    return rows


def design(rows, standardize=False, drop_method=False):
    N = np.array([r["N"] for r in rows])
    L = np.array([r["L"] for r in rows])
    S = np.array([r["site"] for r in rows])
    if standardize:
        N = (N - N.mean()) / N.std()
        L = (L - L.mean()) / L.std()
        S = (S - S.mean()) / S.std()
    cols = [N, L, S]
    names = ["candidate_count", "protein_length", "site_size"]
    if not drop_method:
        for m in METHODS[1:]:          # fpocket is the reference level
            cols.append(np.array([1.0 if r["method"] == m else 0.0
                                  for r in rows]))
            names.append("method=%s" % m)
    return np.column_stack(cols), names


def fit(rows, standardize=False, drop_method=False, n_boot=1000, seed=SEED):
    X, names = design(rows, standardize, drop_method)
    y = np.array([r["y"] for r in rows])
    model = LogisticRegression(max_iter=2000, C=1e6).fit(X, y)
    coef = model.coef_[0]

    # Cluster the bootstrap on target, since each target appears once per method.
    targets = sorted({r["target_id"] for r in rows})
    by_target = {}
    for i, r in enumerate(rows):
        by_target.setdefault(r["target_id"], []).append(i)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(targets), size=len(targets), replace=True)
        idx = [i for p in pick for i in by_target[targets[p]]]
        Xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            draws.append(LogisticRegression(max_iter=2000, C=1e6)
                         .fit(Xb, yb).coef_[0])
        except Exception:
            continue
    draws = np.array(draws)
    lo = np.percentile(draws, 2.5, axis=0)
    hi = np.percentile(draws, 97.5, axis=0)
    return names, coef, lo, hi


def report_fit(title, rows, **kw):
    print("\n%s   (n = %d rows, %d targets)"
          % (title, len(rows), len({r["target_id"] for r in rows})))
    names, coef, lo, hi = fit(rows, **kw)
    print("   %-22s %9s %20s" % ("term", "beta", "95% CI (clustered)"))
    out = {}
    for nm, c, l, h in zip(names, coef, lo, hi):
        star = "  *" if (l > 0 or h < 0) else ""
        print("   %-22s %+9.4f   [%+7.4f, %+7.4f]%s" % (nm, c, l, h, star))
        out[nm] = {"beta": float(c), "ci": [float(l), float(h)],
                   "excludes_zero": bool(l > 0 or h < 0)}
    return out


def efficiency_from_sweep(out):
    """eta = dRecovery / dCoverage at each conformer step, with paired CIs."""
    print("\n" + "=" * 74)
    print("eta = dR/dC across the burden sweep  (the quantitative claim)")
    print("=" * 74)
    sweep = json.loads((HERE / "phase6_sweep.json").read_text())
    folds = dataio.fold_map()
    ranked = {int(k): dataio.lacuna_ranked(dataio.DATA / v)
              for k, v in dataio.SWEEP.items()}
    ranked = {k: {s: j for s, j in v.items()
                  if dataio.split_of(s, folds) == "train"}
              for k, v in ranked.items()}
    common = sorted(set.intersection(*(set(v) for v in ranked.values())))
    levels = sorted(ranked)
    T = dataio.JACCARD_THRESHOLD

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(common))
    boot_idx = rng.integers(0, len(common), size=(4000, len(common)))

    print("  %-14s %8s %9s %9s %22s" %
          ("step", "dN", "dC (pts)", "dR (pts)", "eta = dR/dC (95% CI)"))
    res = {}
    for lo_l, hi_l in zip(levels, levels[1:]):
        cov_lo = np.array([any(j >= T for j in ranked[lo_l][s]) for s in common], float)
        cov_hi = np.array([any(j >= T for j in ranked[hi_l][s]) for s in common], float)
        r_lo = np.array([any(j >= T for j in ranked[lo_l][s][:5]) for s in common], float)
        r_hi = np.array([any(j >= T for j in ranked[hi_l][s][:5]) for s in common], float)
        dC = cov_hi.mean() - cov_lo.mean()
        dR = r_hi.mean() - r_lo.mean()
        eta = dR / dC if dC else float("nan")
        etas = []
        for b in boot_idx:
            c = cov_hi[b].mean() - cov_lo[b].mean()
            r = r_hi[b].mean() - r_lo[b].mean()
            if abs(c) > 1e-9:
                etas.append(r / c)
        elo, ehi = np.percentile(etas, [2.5, 97.5])
        dN = (np.mean([len(ranked[hi_l][s]) for s in common])
              - np.mean([len(ranked[lo_l][s]) for s in common]))
        res["%d_to_%d" % (lo_l, hi_l)] = {
            "delta_N": float(dN), "delta_C": float(dC), "delta_R": float(dR),
            "eta": float(eta), "eta_ci": [float(elo), float(ehi)]}
        print("  conf_%-2d -> %-3d %8.1f %8.1f %9.1f      %5.2f [%5.2f, %5.2f]"
              % (lo_l, hi_l, dN, 100 * dC, 100 * dR, eta, elo, ehi))

    etas = [v["eta"] for v in res.values()]
    burdens = [v["delta_N"] for v in res.values()]
    print("\n  eta sequence: %s" % ", ".join("%.2f" % e for e in etas))
    print("  monotonically declining? %s"
          % ("yes" if all(a > b for a, b in zip(etas, etas[1:])) else "NO"))
    out["efficiency"] = res
    return res


def main() -> None:
    rows = load_covered()
    out = {"seed": SEED}
    print("=" * 74)
    print("Does the burden effect survive protein length and method identity?")
    print("=" * 74)
    out["logit_raw"] = report_fit("raw units, with method fixed effects", rows)
    out["logit_std"] = report_fit("standardized, with method fixed effects",
                                  rows, standardize=True)
    out["logit_no_method"] = report_fit("standardized, method effects REMOVED",
                                        rows, standardize=True,
                                        drop_method=True)
    print("\n  per-method partial correlation of N with hit@5 "
          "(Spearman, within method):")
    per = {}
    for m in METHODS:
        sub = [r for r in rows if r["method"] == m]
        rho, p = stats.spearmanr([r["N"] for r in sub], [r["y"] for r in sub])
        per[m] = {"rho": float(rho), "p": float(p), "n": len(sub)}
        print("     %-11s rho = %+.3f  (n = %d)" % (m, rho, len(sub)))
    out["within_method_spearman"] = per

    efficiency_from_sweep(out)
    (HERE / "phase5_confounders.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase5_confounders.json"))


if __name__ == "__main__":
    main()
