"""What is the ~84% conversion wall actually made of?

Eight independent attacks now land within a couple of points of the same place:
sequence embeddings pooled or head-scored, fold-conditioned embeddings pooled or
head-scored, atom-level chemistry, five times the pocket labels, gradient
boosting, LambdaRank, and four different aggregations. When that many levers
move nothing, the constraint is usually not in the model.

So this stops trying to rank better and looks at what the failures are. Three
questions, all answerable from data already on disk.

  A  Anatomy. On a target that is covered but not converted at top-5, how close
     did the ranker get? If the highest-overlap candidate in the top five sits
     just under Jaccard 0.25, the model found the site and the criterion
     disqualified it. That is a labelling outcome, not a ranking failure.

  B  Threshold sensitivity. Conversion is recomputed across thresholds from 0.10
     to 0.40, refitting the ranker at each one so the labels it trains on match
     the labels it is scored against. A sharp rise as the threshold falls means
     the criterion is binding. A flat curve means the failures are real.

  C  Ambiguity. On failures, how many rival candidates score close to the
     qualifying one, and how large is the score gap it loses by? If the true
     site is separated from its rivals by nothing, no ranker can order them,
     because the label is not a function of anything the ranker can see.

Ranking throughout is ESM-IF1 pooled, cross-fitted leave-one-fold-out: it ties
the best honest configuration found (83.8% at top-5) and needs no trained head.

Protocol: train folds only. The designated test fold is never read.

    python analysis/moores_pocket_law/criterion_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
import plm_headroom as PH
import atomistic as A
import esmif_route2 as E

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def rows_with_jac():
    """Candidates carrying the pooled ESM-IF1 vector and their raw Jaccard."""
    folds = dataio.fold_map()
    recs = []
    with open(A.CAND, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tool") != "lacuna":
                continue
            f = folds.get(dataio.pdb_id_of(r["id"]))
            if f is None or not f.startswith("train"):
                continue
            if not (r.get("residues_by_rank") and r.get("jac_by_rank")):
                continue
            r["fold"] = f
            recs.append(r)

    out = []
    for r in recs:
        fi = E.CACHE / ("%s.npz" % r["id"])
        if not fi.exists():
            continue
        z = np.load(fi)
        emb = z["emb"].astype(np.float32)
        pos = {int(n): i for i, n in enumerate(z["nums"])}
        for rank, (res, jac) in enumerate(zip(r["residues_by_rank"],
                                              r["jac_by_rank"])):
            idx = [pos[n] for n in PH._resnums(res, r["chain"]) if n in pos]
            if not idx:
                continue
            out.append({"id": r["id"], "fold": r["fold"], "rank": rank,
                        "x": emb[idx].mean(0), "jac": float(jac)})
    return out


def crossfit_scores(rows, thr):
    """Scores from a ranker trained against this threshold's own labels."""
    X = np.vstack([r["x"] for r in rows])
    y = np.array([r["jac"] >= thr for r in rows], dtype=int)
    fold = np.array([r["fold"] for r in rows])
    s = np.zeros(len(rows))
    for h in sorted(set(fold)):
        tr, te = fold != h, fold == h
        if y[tr].sum() < 5:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def grouped(rows, scores):
    by = {}
    for r, s in zip(rows, scores):
        by.setdefault(r["id"], []).append((s, r["jac"]))
    return by


def main() -> None:
    rows = rows_with_jac()
    print("candidates: %d over %d structures\n"
          % (len(rows), len({r["id"] for r in rows})))

    # ── B: threshold sensitivity ────────────────────────────────────────────
    print("B. threshold sensitivity (ranker refitted at each threshold)\n")
    print("%8s %10s %12s %12s" % ("Jaccard", "covered", "conv@5", "recovery@5"))
    print("-" * 46)
    sweep = {}
    for thr in THRESHOLDS:
        by = grouped(rows, crossfit_scores(rows, thr))
        cov = [t for t, v in by.items() if any(j >= thr for _s, j in v)]
        hit = []
        for t in cov:
            v = sorted(by[t], key=lambda z: -z[0])[:5]
            hit.append(int(any(j >= thr for _s, j in v)))
        conv = float(np.mean(hit))
        cvr = len(cov) / len(by)
        sweep[thr] = {"covered": cvr, "conv_5": conv, "recovery_5": cvr * conv}
        print("%8.2f %9.1f%% %11.1f%% %11.1f%%"
              % (thr, 100 * cvr, 100 * conv, 100 * cvr * conv))

    # ── A and C: anatomy of failures at the published threshold ─────────────
    T = 0.25
    by = grouped(rows, crossfit_scores(rows, T))
    cov = [t for t, v in by.items() if any(j >= T for _s, j in v)]
    fails, wins = [], []
    for t in cov:
        v = sorted(by[t], key=lambda z: -z[0])
        top5 = v[:5]
        if any(j >= T for _s, j in top5):
            wins.append(t)
            continue
        best_in_top5 = max(j for _s, j in top5)
        qual = max((z for z in v if z[1] >= T), key=lambda z: z[1])
        qrank = v.index(qual) + 1
        fails.append({"id": t, "best_jac_top5": best_in_top5,
                      "qual_jac": qual[1], "qual_rank": qrank,
                      "gap": v[0][0] - qual[0],
                      "n_cand": len(v)})

    print("\nA. anatomy of the %d covered targets that fail at top-5\n" % len(fails))
    b = np.array([f["best_jac_top5"] for f in fails])
    for lo, hi, label in ((0.20, 0.25, "0.20 to 0.25  (near miss)"),
                          (0.15, 0.20, "0.15 to 0.20"),
                          (0.10, 0.15, "0.10 to 0.15"),
                          (0.00, 0.10, "below 0.10    (nowhere near)")):
        n = int(((b >= lo) & (b < hi)).sum())
        print("  best overlap in top-5 %-28s %4d  %5.1f%%"
              % (label, n, 100 * n / len(fails)))
    print("\n  median best overlap in top-5 : %.3f" % float(np.median(b)))
    print("  median rank of the qualifying candidate: %d"
          % int(np.median([f["qual_rank"] for f in fails])))
    print("  median candidates per failing target   : %d"
          % int(np.median([f["n_cand"] for f in fails])))

    print("\nC. how much score separates the winner from the answer\n")
    g = np.array([f["gap"] for f in fails])
    for lim in (0.01, 0.05, 0.10, 0.25):
        n = int((g <= lim).sum())
        print("  score gap <= %.2f : %4d of %d failures  (%.1f%%)"
              % (lim, n, len(fails), 100 * n / len(fails)))
    print("\n  median gap: %.4f" % float(np.median(g)))

    qj = np.array([f["qual_jac"] for f in fails])
    wq = []
    for t in wins:
        v = sorted(by[t], key=lambda z: -z[0])[:5]
        wq.append(max(j for _s, j in v if j >= T))
    print("\n  qualifying candidate's own overlap")
    print("    on failures: median %.3f" % float(np.median(qj)))
    print("    on wins    : median %.3f" % float(np.median(wq)))

    out = {"sweep": {str(k): v for k, v in sweep.items()},
           "n_covered": len(cov), "n_fail": len(fails),
           "near_miss_frac": float(((b >= 0.20) & (b < 0.25)).mean()),
           "median_best_jac_top5": float(np.median(b)),
           "median_gap": float(np.median(g))}
    (HERE / "criterion_analysis.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "criterion_analysis.json"))


if __name__ == "__main__":
    main()
