"""Where does the residue-head gain actually come from?

E1 beat pocket-level training by 4.5 points at top-5 and 9.5 at top-1, both
intervals clear of zero. But E1 changed two things at once. It refitted the
residue head per fold, where the shipped head is fitted on all four and so
carries leakage, and it replaced the four learned aggregates with a fixed rule
picked by hand and never tuned.

Either could be doing the work, and they imply different fixes. If the refit is
what matters, production is reporting an inflated number and the remedy is
honest fitting. If the aggregation is what matters, the four aggregates are
throwing away residue-level structure and the remedy is a better collapse.

So: a 2x2, everything else held constant.

                    fixed rule            learned aggregates
    shipped head    leakage, no learning  the production path
    refit head      E1 as run             refit with learning

Reading the grid:

  * refit minus shipped, at fixed aggregation, is the leakage.
  * fixed minus learned, at fixed head, is what the four aggregates cost.

Both are paired on target and bootstrapped, because point estimates decide
nothing here.

Protocol: train folds only, leave-one-fold-out over the four homology-separated
train folds. The designated test fold is never read.

    python analysis/moores_pocket_law/residue_head_isolation.py
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

import ranker_next as R
import plm_headroom as PH
from dataio import JACCARD_THRESHOLD as T

KS = (1, 5, 10)
SEED = 0


def shipped_probs(data):
    """The production head: one linear layer, fitted on all four folds."""
    w = np.load(REPO / "lacuna" / "pockets" / "plm_head.npz")
    W, B = w["w"].astype(np.float32), float(w["b"])
    out = {}
    for sid in data:
        f = PH.CACHE / ("%s.npz" % sid)
        if not f.exists():
            continue
        z = np.load(f)
        emb = z["emb"].astype(np.float32)
        out[sid] = (1.0 / (1.0 + np.exp(-(emb @ W + B))), z["nums"].astype(int))
    return out


def candidate_view(data, probs):
    """Per candidate: the four aggregates, the fixed-rule score, label, fold."""
    rows = []
    for sid, recs in data.items():
        if sid not in probs:
            continue
        p, nums = probs[sid]
        pos = {int(n): i for i, n in enumerate(nums)}
        lac = next((r for r in recs if r["tool"] == "lacuna"), None)
        if lac is None:
            continue
        for res, jac in zip(lac["residues_by_rank"], lac["jac_by_rank"]):
            idx = [pos[n] for n in PH._resnums(res, lac["chain"]) if n in pos]
            if not idx:
                continue
            v = np.sort(p[idx])[::-1]
            rows.append({
                "id": sid, "fold": lac["fold"],
                "agg": np.array([v.mean(), v[0], v[:3].mean(),
                                 float((v >= 0.5).mean())], dtype=np.float32),
                "fixed": float(0.6 * v[:3].mean() + 0.4 * v.mean()),
                "y": int(float(jac) >= T)})
    return rows


def score_fixed(rows):
    by = {}
    for r in rows:
        by.setdefault(r["id"], [[], []])
        by[r["id"]][0].append(r["fixed"])
        by[r["id"]][1].append(r["y"])
    return by


def score_learned(rows):
    """Cross-fitted logistic over the four aggregates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([r["agg"] for r in rows])
    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    s = np.zeros(len(rows))
    for h in sorted(set(fold)):
        tr, te = fold != h, fold == h
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    by = {}
    for r, sc in zip(rows, s):
        by.setdefault(r["id"], [[], []])
        by[r["id"]][0].append(sc)
        by[r["id"]][1].append(r["y"])
    return by


def hit(by, t, k):
    sc, ys = by[t]
    o = np.argsort(-np.asarray(sc), kind="stable")[:k]
    return int(any(ys[i] for i in o))


def main() -> None:
    data = R.load_all(None)
    print("computing shipped-head probabilities")
    ship = shipped_probs(data)
    print("refitting the residue head per fold")
    refit = R.residue_head(data, "linear")

    cells = {}
    for hname, probs in (("shipped", ship), ("refit", refit)):
        rows = candidate_view(data, probs)
        cells[(hname, "fixed")] = score_fixed(rows)
        cells[(hname, "learned")] = score_learned(rows)

    common = sorted(set.intersection(*(set(v) for v in cells.values())))
    common = [t for t in common if any(cells[("shipped", "fixed")][t][1])]
    print("\npaired on %d covered targets\n" % len(common))

    print("%-28s %8s %8s %8s" % ("cell", "conv@1", "conv@5", "conv@10"))
    print("-" * 56)
    res = {}
    for key in (("shipped", "learned"), ("shipped", "fixed"),
                ("refit", "learned"), ("refit", "fixed")):
        by = cells[key]
        res[key] = {k: np.array([hit(by, t, k) for t in common]) for k in KS}
        label = "%s head, %s agg" % key
        print("%-28s %7.1f%% %7.1f%% %7.1f%%"
              % (label, *(100 * res[key][k].mean() for k in KS)))

    rng = np.random.default_rng(SEED)
    n = len(common)
    bi = rng.integers(0, n, size=(20000, n))

    def ci(a, b, label):
        d = res[a] , res[b]
        for k in KS:
            dd = res[a][k] - res[b][k]
            bm = np.sort(dd[bi].mean(axis=1))
            lo, hi = bm[500], bm[19499]
            star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
            print("  top-%-3d %+6.1f  [%+6.1f, %+6.1f]%s"
                  % (k, 100 * dd.mean(), 100 * lo, 100 * hi, star))

    print("\nleakage: refit minus shipped, aggregation held fixed")
    ci(("refit", "fixed"), ("shipped", "fixed"), "")
    print("\ncost of the four aggregates: fixed minus learned, head held fixed")
    ci(("refit", "fixed"), ("refit", "learned"), "")
    print("\nproduction path vs E1 as run")
    ci(("refit", "fixed"), ("shipped", "learned"), "")

    out = {"n_covered": n,
           "cells": {"%s_%s" % k: {"conv_%d" % kk: float(v[kk].mean())
                                   for kk in KS} for k, v in res.items()}}
    (HERE / "residue_head_isolation.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "residue_head_isolation.json"))


if __name__ == "__main__":
    main()
