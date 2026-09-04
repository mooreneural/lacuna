"""Fit the fused-pool ranker on all training folds and export it.

`refit_fused_ranker.py` established the effect by cross-fitting: a ranker that
has seen candidates from both detectors takes top-five recovery from 55.0% to
67.6% on the training folds, +12.6 [+9.3, +16.0]. Cross-fitting proves the
effect; it does not produce a model. This fits one model on all four training
folds, which is the thing a test-fold measurement has to evaluate.

Deliberately the same shape as the shipped ranker: a standardised linear model
over the 23 geometry and ensemble features in `_RANKER_FEATURES`. Provenance
was worth about a point at top-one and nothing at top-five, so it is left out
rather than complicating the artifact for a difference that did not resolve.

Fitted on `end_to_end_fusion.jsonl`, which holds every fused candidate's
features and Jaccard from the 747-target training run. The test fold is not
read here and must not be read until the model is frozen.

    python analysis/moores_pocket_law/fit_fused_ranker.py
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

from dataio import JACCARD_THRESHOLD as T
from lacuna.pockets.clusterer import _RANKER_FEATURES

SRC = HERE / "end_to_end_fusion.jsonl"
OUT = HERE / "fused_ranker.npz"
FEATS = list(_RANKER_FEATURES)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X, y, n_t = [], [], 0
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            feat = r.get("fused_feat")
            if not feat:
                continue
            n_t += 1
            X.append([[float(c.get(f, 0.0)) for f in FEATS] for c in feat])
            y.append([1 if float(j) >= T else 0 for j in r["fused_jac"]])
    if n_t < 300:
        raise SystemExit("only %d targets in %s; refusing to fit." % (n_t, SRC))

    X = np.vstack([np.asarray(a) for a in X])
    y = np.concatenate([np.asarray(a) for a in y])
    print("targets %d   candidates %d   qualifying %d (%.1f%%)"
          % (n_t, len(y), y.sum(), 100 * y.mean()))

    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    p = m.predict_proba(sc.transform(X))[:, 1]
    # In-sample: says the fit converged, not that it generalises. The held-out
    # estimate is the cross-fitted one in refit_fused_ranker.
    print("in-sample AUC %.3f  AP %.3f" % (roc_auc_score(y, p),
                                           average_precision_score(y, p)))

    np.savez(OUT, mean=sc.mean_, scale=sc.scale_,
             coef=m.coef_[0], intercept=m.intercept_[0],
             features=np.array(FEATS))
    print("\nwrote %s" % OUT)
    print("\nlargest weights (standardised)")
    for nm, w in sorted(zip(FEATS, m.coef_[0]), key=lambda kv: -abs(kv[1]))[:8]:
        print("  %-16s %+.3f" % (nm, w))


if __name__ == "__main__":
    main()
