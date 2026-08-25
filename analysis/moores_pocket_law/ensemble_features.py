"""Do the ensemble-derived features carry what the static ones miss?

The criterion analysis left one suspect standing. Every representation tested so
far is static and single-conformer: embeddings pooled over lining residues, atom
chemistry of one structure, geometry of one cavity. All of them plateau near 84%
top-5 conversion, and the failures are confident errors rather than ties, which
means there is signal to be found and none of those inputs contain it.

What was never in play is the thing Lacuna actually adds over a single-structure
detector: how a cavity behaves across the conformational ensemble. Whether it
persists, how its volume swings, whether its centroid wanders, how much it opens
relative to the unbound structure.

I claimed this needed a re-collection. That was wrong, and it conflated two
questions. Combining ensemble features with *embeddings* does need one, because
pooling embeddings needs residue lists that the feature file lacks. But asking
whether ensemble dynamics discriminate at all needs no residues: the released
feature file carries all 27 features and the per-candidate Jaccard together.

So the 27 split three ways and are compared under one cross-fitting protocol.

  static    13  one conformer: size, shape, burial, chemistry, position
  ensemble  10  across conformers: persistence, variability, crypticity, drift
  plm        4  the sequence head's aggregates

Protocol: train folds only, leave-one-fold-out over the four homology-separated
train folds, paired bootstrap intervals. The designated test fold is never read.

    python analysis/moores_pocket_law/ensemble_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataio
from dataio import JACCARD_THRESHOLD as T
import ranker_headroom as RH

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KS = (1, 5, 10)
SEED = 0

#: One conformer only. Nothing here needs an ensemble to compute.
STATIC = ["vol", "drug", "n_lin", "vol_per_lin", "enc", "hyd", "aro",
          "bur_raw", "depth", "mouth", "elong", "flat", "dcen"]
#: Only definable across conformers: percentiles and extrema over the ensemble,
#: persistence, membership, drift, and opening relative to the apo structure.
ENSEMBLE = ["vol_p90", "vol_p10", "apo_vol", "max_drug", "cryp", "pers",
            "mem_per_conf", "depth_p90", "centroid_std", "vol_cv"]
PLM = ["plm_mean", "plm_max", "plm_top3", "plm_frac"]


def scores(targets, feats, C=1.0):
    idx = [RH.FEATS.index(f) for f in feats]
    out = {}
    for held in sorted({t["fold"] for t in targets}):
        tr = [t for t in targets if t["fold"] != held]
        te = [t for t in targets if t["fold"] == held]
        X = np.vstack([t["X"][:, idx] for t in tr])
        y = np.concatenate([(t["jac"] >= T).astype(int) for t in tr])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C))
        m.fit(X, y)
        for t in te:
            out[t["id"]] = m.predict_proba(t["X"][:, idx])[:, 1]
    return out


def main() -> None:
    targets = RH.load()
    covered = [t for t in targets if t["covered"]]
    print("train-fold targets: %d   covered: %d\n" % (len(targets), len(covered)))

    sets = [
        ("static only", STATIC),
        ("ensemble only", ENSEMBLE),
        ("static + ensemble", STATIC + ENSEMBLE),
        ("static + plm", STATIC + PLM),
        ("static + ensemble + plm", STATIC + ENSEMBLE + PLM),
    ]

    res = {}
    for name, feats in sets:
        s = scores(targets, feats)
        res[name] = {}
        for k in KS:
            hit = []
            for t in covered:
                order = np.argsort(-s[t["id"]], kind="stable")[:k]
                hit.append(int((t["jac"][order] >= T).any()))
            res[name][k] = np.array(hit)

    print("%-28s %5s %8s %8s %8s" % ("features", "n", "conv@1", "conv@5", "conv@10"))
    print("-" * 62)
    for name, feats in sets:
        print("%-28s %5d %7.1f%% %7.1f%% %7.1f%%"
              % (name, len(feats), *(100 * res[name][k].mean() for k in KS)))

    rng = np.random.default_rng(SEED)
    n = len(covered)
    bi = rng.integers(0, n, size=(20000, n))

    def ci(a, b, label):
        print("  %s" % label)
        for k in KS:
            d = res[a][k] - res[b][k]
            bm = np.sort(d[bi].mean(axis=1))
            lo, hi = bm[500], bm[19499]
            star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
            print("    top-%-3d %+6.1f  [%+6.1f, %+6.1f]%s"
                  % (k, 100 * d.mean(), 100 * lo, 100 * hi, star))

    print("\npaired bootstrap, 20,000 resamples\n")
    ci("static + ensemble", "static only", "what ensemble dynamics add to static")
    ci("static + plm", "static only", "what the sequence head adds to static")
    ci("static + ensemble + plm", "static + ensemble", "what sequence adds on top of dynamics")
    ci("static + ensemble + plm", "static + plm", "what dynamics add on top of sequence")

    out = {"n_covered": n,
           "cells": {nm: {"conv_%d" % k: float(v[k].mean()) for k in KS}
                     for nm, v in res.items()}}
    (HERE / "ensemble_features.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "ensemble_features.json"))


if __name__ == "__main__":
    main()
