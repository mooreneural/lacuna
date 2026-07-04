# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Train a learned pocket re-ranker and evaluate it honestly on held-out data.

Trains a gradient-boosted classifier on pocket-cluster features from the
CryptoBench TRAIN folds (produced by extract_features.py) to predict whether a
cluster is a true cryptic site, then evaluates on the held-out TEST fold:

  * cluster-level ROC-AUC / average precision,
  * protein-level top-5 recall of the physics ranking (crypticity) vs the learned
    re-ranking, over all test proteins,
  * the "reachable ceiling": fraction of test proteins that have >=1 positive
    cluster detected at all (re-ranking cannot help beyond this).

No fitting on the test fold; CryptoBench folds are sequence-clustered to prevent
similarity leakage.

    python benchmarks/train_reranker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CB = Path(__file__).parent / "cb_data"
TRAIN_CSV = CB / "features_t0+t1+t2+t3.csv"
TEST_CSV = CB / "features_test.csv"
MODEL_OUT = Path(__file__).parent.parent / "lacuna" / "pockets" / "reranker.joblib"

FEATURES = [
    "crypticity", "persistence", "volume_mean", "volume_min", "volume_max",
    "apo_volume", "volume_range", "volume_delta_apo", "druggability",
    "max_druggability", "n_members", "n_lining", "enclosure_mean", "enclosure_max",
    "hydrophobic_mean", "aromatic_mean", "aromatic_max", "phys_rank", "phys_rank_frac",
]


def topk_recall(df, score_col, k=5):
    """Fraction of proteins with a label-1 cluster among the top-k by score_col
    (higher = better). Returns (recall_all, n_proteins, reachable_fraction)."""
    hit = reach = 0
    groups = df.groupby("protein")
    for _, g in groups:
        has_pos = (g["label"] == 1).any()
        reach += int(has_pos)
        top = g.nlargest(k, score_col)
        hit += int((top["label"] == 1).any())
    n = groups.ngroups
    return hit / n, n, reach / n


def main():
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        sys.exit(f"missing features CSVs: {TRAIN_CSV.name}, {TEST_CSV.name}")
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    print(f"train: {len(tr)} clusters / {tr.protein.nunique()} proteins "
          f"({tr.label.mean():.1%} positive)")
    print(f"test : {len(te)} clusters / {te.protein.nunique()} proteins "
          f"({te.label.mean():.1%} positive)")

    Xtr, ytr = tr[FEATURES], tr["label"]
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=0,
        early_stopping=True, validation_fraction=0.15,
    )
    clf.fit(Xtr, ytr)

    te = te.copy()
    te["learned"] = clf.predict_proba(te[FEATURES])[:, 1]
    # physics score = inverse rank (higher = better) so top-k picks rank 1..k
    te["phys_score"] = -te["phys_rank"]

    print("\n=== cluster-level (held-out test) ===")
    print(f"  ROC-AUC          : {roc_auc_score(te.label, te.learned):.3f}")
    print(f"  avg precision    : {average_precision_score(te.label, te.learned):.3f}")
    print(f"  (crypticity AUC  : {roc_auc_score(te.label, te.crypticity):.3f})")

    print("\n=== protein-level top-5 recall (held-out test) ===")
    print(f"  {'':6} {'physics':>8} {'volume':>8} {'nlining':>8} {'LEARNED':>8}")
    for k in (1, 3, 5):
        p_all, n, reach = topk_recall(te, "phys_score", k)
        v_all, _, _ = topk_recall(te, "volume_mean", k)
        nl_all, _, _ = topk_recall(te, "n_lining", k)
        l_all, _, _ = topk_recall(te, "learned", k)
        print(f"  top-{k:<3} {p_all:>7.1%} {v_all:>8.1%} {nl_all:>8.1%} {l_all:>8.1%}")
    _, _, reach = topk_recall(te, "learned", 5)
    print(f"  reachable ceiling (proteins with >=1 positive cluster): {reach:.1%}")
    # decisive anti-gaming verdict
    l5 = topk_recall(te, "learned", 5)[0]
    v5 = topk_recall(te, "volume_mean", 5)[0]
    p5 = topk_recall(te, "phys_score", 5)[0]
    print("\n=== verdict ===")
    if l5 - v5 < 0.05:
        print(f"  learned ({l5:.1%}) ~= volume-ranking ({v5:.1%}): STILL GAMING size, not a real gain.")
    elif l5 - p5 < 0.05:
        print(f"  learned ({l5:.1%}) ~= physics ({p5:.1%}): no real improvement over crypticity.")
    else:
        print(f"  learned ({l5:.1%}) beats BOTH physics ({p5:.1%}) and volume ({v5:.1%}): real signal.")

    print("\n=== feature importance (permutation on test) ===")
    from sklearn.inspection import permutation_importance
    r = permutation_importance(clf, te[FEATURES], te["label"], n_repeats=5,
                               random_state=0, scoring="average_precision")
    for idx in r.importances_mean.argsort()[::-1][:8]:
        print(f"  {FEATURES[idx]:18} {r.importances_mean[idx]:+.4f}")

    import joblib
    joblib.dump({"model": clf, "features": FEATURES}, MODEL_OUT)
    print(f"\nsaved model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
