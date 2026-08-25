"""Is Lacuna's ranking limited by its model, or by its features?

The benchmarking study established that ranking, not detection, is what caps
top-k recovery: the same candidate set converts at 75.0% under the default
ranker and 89.8% under the PLM-assisted one. That leaves 7.3 points reachable
without touching detection at all. The open question is what to spend on
closing it.

Two answers imply very different work:

  * If a high-capacity model over the *existing* features beats the shipped
    linear one, the limit is model capacity. That is a CPU-afternoon fix and
    needs no new featurisation.
  * If it does not, the existing 27 features are exhausted, and the only way
    up is new signal: per-residue language-model embeddings rather than four
    aggregates of them, which is a GPU job.

This decides between them for free, because every feature is already stored
per candidate in the released JSONL. Nothing is re-run.

Protocol: train folds only. The designated test fold is never read here, and
cross-fitting is across CryptoBench's four homology-separated train folds, so
no model ever scores a target that shares homology with its training data.

    python analysis/moores_pocket_law/ranker_headroom.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import JACCARD_THRESHOLD as T

warnings.filterwarnings("ignore")

FEATS = list(dataio._PLM_RANKER_FEATURES)
GEOM = [f for f in FEATS if not f.startswith("plm_")]
W_SHIPPED = np.asarray(dataio._PLM_RANKER_WEIGHTS)
KS = (1, 5, 10)
SEED = 0


def load():
    """Per-target feature matrices and labels, train folds only."""
    folds = dataio.fold_map()
    out = []
    for rec in dataio._jsonl(dataio.SWEEP[20]):
        # folds.json is keyed by PDB id, not by the pdb+chain structure id.
        fold = folds.get(dataio.pdb_id_of(rec["id"]))
        if fold is None or not fold.startswith("train"):
            continue
        cl = rec.get("clusters") or []
        if not cl:
            continue
        X = np.array([[float(c.get(f, 0.0)) for f in FEATS] for c in cl])
        j = np.array([float(c.get("jac", 0.0)) for c in cl])
        out.append({"id": rec["id"], "fold": fold, "X": X, "jac": j,
                    "covered": bool((j >= T).any())})
    return out


def hits(targets, scores_by_id):
    """hit@k over all targets, and conversion among covered ones."""
    res = {}
    cov = [t for t in targets if t["covered"]]
    for k in KS:
        h = []
        for t in targets:
            s = scores_by_id[t["id"]]
            order = np.argsort(-s, kind="stable")
            h.append(int((t["jac"][order[:k]] >= T).any()))
        res["hit_%d" % k] = float(np.mean(h))
        hc = []
        for t in cov:
            s = scores_by_id[t["id"]]
            order = np.argsort(-s, kind="stable")
            hc.append(int((t["jac"][order[:k]] >= T).any()))
        res["conv_%d" % k] = float(np.mean(hc))
    return res


def crossfit(targets, make_model, feats, needs_group=False):
    """Leave-one-fold-out predictions, so no target is scored by a model
    that trained on its homology group."""
    idx = [FEATS.index(f) for f in feats]
    scores = {}
    for held in sorted({t["fold"] for t in targets}):
        tr = [t for t in targets if t["fold"] != held]
        te = [t for t in targets if t["fold"] == held]
        Xtr = np.vstack([t["X"][:, idx] for t in tr])
        ytr = np.concatenate([(t["jac"] >= T).astype(int) for t in tr])
        groups = [len(t["jac"]) for t in tr]
        m = make_model()
        # Only LambdaRank takes group sizes. Probing by exception does not work
        # here: a sklearn Pipeline rejects the kwarg with ValueError, not
        # TypeError, so the fallback never fires and the run dies.
        if needs_group:
            m.fit(Xtr, ytr, group=groups)
        else:
            m.fit(Xtr, ytr)
        for t in te:
            Xt = t["X"][:, idx]
            if hasattr(m, "predict_proba"):
                scores[t["id"]] = m.predict_proba(Xt)[:, 1]
            else:
                scores[t["id"]] = m.predict(Xt)
    return scores


def main() -> None:
    targets = load()
    # A verdict computed from an empty cohort is worse than no verdict: the
    # comparison below is False against NaN, which prints a confident and
    # entirely unfounded conclusion. Refuse to run instead.
    if len(targets) < 50:
        raise SystemExit("only %d train-fold targets loaded; expected ~740. "
                         "Check the fold key format before trusting anything "
                         "this script prints." % len(targets))
    cov = np.mean([t["covered"] for t in targets])
    print("train-fold targets: %d   coverage: %.1f%%  (fixed; same candidates "
          "throughout)\n" % (len(targets), 100 * cov))

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    rows = []

    # The shipped ranker: fixed weights, nothing fitted here.
    s = {t["id"]: t["X"] @ W_SHIPPED for t in targets}
    rows.append(("shipped linear (production)", hits(targets, s), 27))

    rows.append(("logistic, refit, cross-fitted",
                 hits(targets, crossfit(targets, lambda: make_pipeline(
                     StandardScaler(),
                     LogisticRegression(max_iter=2000, C=1.0)), FEATS)), 27))

    gbm = lambda: lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=30, subsample=0.9, colsample_bytree=0.8,
        random_state=SEED, verbose=-1)
    rows.append(("LightGBM, cross-fitted", hits(targets, crossfit(targets, gbm, FEATS)), 27))

    rows.append(("LightGBM, geometry only (no PLM)",
                 hits(targets, crossfit(targets, gbm, GEOM)), len(GEOM)))

    ranker = lambda: lgb.LGBMRanker(
        objective="lambdarank", n_estimators=400, learning_rate=0.05,
        num_leaves=31, min_child_samples=30, random_state=SEED, verbose=-1)
    rows.append(("LightGBM LambdaRank, cross-fitted",
                 hits(targets, crossfit(targets, ranker, FEATS,
                                        needs_group=True)), 27))

    print("%-34s %5s %8s %8s %8s" % ("ranker", "feat", "top-1", "top-5", "top-10"))
    print("-" * 68)
    for name, r, nf in rows:
        print("%-34s %5d %7.1f%% %7.1f%% %7.1f%%"
              % (name, nf, 100 * r["hit_1"], 100 * r["hit_5"], 100 * r["hit_10"]))
    print("%-34s %5s %7s  %7.1f%% %7.1f%%" % ("perfect ranking (oracle)", "-", "-",
                                              100 * cov, 100 * cov))

    print("\nconversion among covered targets (the quantity ranking controls)")
    print("-" * 68)
    for name, r, _nf in rows:
        print("%-34s %7.1f%% %7.1f%% %7.1f%%"
              % (name, 100 * r["conv_1"], 100 * r["conv_5"], 100 * r["conv_10"]))
    print("%-34s %7s %7.1f%% %7.1f%%" % ("perfect ranking (oracle)", "-", 100.0, 100.0))

    base = rows[0][1]["conv_5"]
    best = max(r["conv_5"] for _n, r, _f in rows[1:])
    gap = 1.0 - base
    print("\nheadroom at top-5: %.1f points of conversion remain unconverted." % (100 * gap))
    print("best model over existing features recovers %.1f of them (%.0f%%)."
          % (100 * (best - base), 100 * (best - base) / (100 * gap) * 100))
    print("\n%s" % ("=" * 68))
    if best - base >= 0.02:
        print("VERDICT: model capacity is a real limit. Worth expanding the ranker")
        print("on the features you already have, before spending GPU time.")
    else:
        print("VERDICT: the existing 27 features are exhausted. A bigger model on")
        print("them buys nothing, so the remaining headroom needs new signal:")
        print("per-residue PLM embeddings rather than four aggregates. That is")
        print("the GPU job, and this says it is the right one.")


if __name__ == "__main__":
    main()
