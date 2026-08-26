"""Does ESM-IF1 beat the four plm_* aggregates once geometry is in the model?

This is the question the whole investigation kept deferring, because answering
it needs ranker features and lining residues in the same row and no file had
both. The v3 collection has both, so it is answerable now.

Everything measured before was sequence-only, on a candidate set that carried no
geometry. In that setting ESM-IF1 pooled and the ESM-2 residue head tied at
83.8%. But the ensemble-feature split showed the sequence head contributing
+8.3 points *on top of* static geometry, so the interesting question was never
which encoder wins alone. It is which one wins next to the 27 geometric and
ensemble terms the shipped ranker actually uses.

Four models, one protocol:

    geometry                27 geometric and ensemble features
    geometry + plm4         plus the four aggregates production uses
    geometry + esmif        plus 512-dim pooled ESM-IF1
    geometry + plm4 + esmif both

The plm_* aggregates are recomputed here from a residue head refitted per fold
rather than taken from the shipped head, which is fitted on all four train folds
and worth about 1.9 points of inflation. Production's own number is not the
right comparator; an honest version of production is.

Regularisation is chosen by inner cross-validation inside each training split
rather than fixed by hand. An arbitrary C made pooled embeddings look 5.8 points
worse than they are earlier in this investigation, and the fix cost nothing.

Protocol: train folds only, leave-one-fold-out over the four homology-separated
train folds, paired bootstrap intervals over targets. The designated test fold
is never read.

    python analysis/moores_pocket_law/v3_sequence_vs_geometry.py
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
from dataio import JACCARD_THRESHOLD as T
import plm_headroom as PH
import esmif_route2 as E

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CAND = HERE / "data" / "candidates_v3" / "candidates_train.jsonl"
KS = (1, 5, 10)
SEED = 0
CS = [0.01, 0.05, 0.2, 1.0]

GEOM = ["vol", "vol_p90", "vol_p10", "vol_max", "vol_min", "apo_vol", "drug",
        "max_drug", "cryp", "pers", "n_lin", "n_mem", "vol_per_lin", "enc",
        "hyd", "aro", "mem_per_conf", "bur_raw", "depth", "depth_max",
        "depth_p90", "mouth", "elong", "flat", "dcen", "centroid_std", "vol_cv"]


def load():
    folds = dataio.fold_map()
    out = []
    with open(CAND, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tool") != "lacuna":
                continue
            f = folds.get(dataio.pdb_id_of(r["id"]))
            if f is None or not f.startswith("train"):
                continue
            if not (r.get("features_by_rank") and r.get("residues_by_rank")):
                continue
            r["fold"] = f
            out.append(r)
    return out


def esm2_head_probs(recs):
    """Per-fold refit residue head on ESM-2, so no fold scores its own head."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    store = {}
    for r in recs:
        f = PH.CACHE / ("%s.npz" % r["id"])
        if not f.exists():
            continue
        z = np.load(f)
        nums = z["nums"].astype(int)
        site = set(int(x) for x in r.get("site_residues") or [])
        store[r["id"]] = (z["emb"].astype(np.float32), nums,
                          np.isin(nums, list(site)).astype(np.float32), r["fold"])
    probs = {}
    for held in sorted({v[3] for v in store.values()}):
        tr = [k for k, v in store.items() if v[3] != held]
        te = [k for k, v in store.items() if v[3] == held]
        X = torch.tensor(np.vstack([store[k][0] for k in tr]), device=dev)
        y = torch.tensor(np.concatenate([store[k][2] for k in tr]), device=dev)
        pw = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=dev)
        torch.manual_seed(SEED)
        net = nn.Linear(1280, 1).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        n = len(X)
        for _ep in range(25):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 8192):
                b = perm[i:i + 8192]
                opt.zero_grad()
                lossf(net(X[b]).squeeze(-1), y[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            for k in te:
                e = torch.tensor(store[k][0], device=dev)
                probs[k] = (torch.sigmoid(net(e).squeeze(-1)).cpu().numpy(),
                            store[k][1])
    return probs


def build(recs, probs):
    rows = []
    for r in recs:
        fi = E.CACHE / ("%s.npz" % r["id"])
        if not fi.exists() or r["id"] not in probs:
            continue
        zi = np.load(fi)
        ei = zi["emb"].astype(np.float32)
        pi = {int(n): i for i, n in enumerate(zi["nums"])}
        p, pnums = probs[r["id"]]
        pp = {int(n): i for i, n in enumerate(pnums)}

        for feats, res, jac in zip(r["features_by_rank"], r["residues_by_rank"],
                                   r["jac_by_rank"]):
            nums = PH._resnums(res, r["chain"])
            ii = [pi[n] for n in nums if n in pi]
            ip = [pp[n] for n in nums if n in pp]
            if not ii or not ip:
                continue
            v = np.sort(p[ip])[::-1]
            rows.append({
                "id": r["id"], "fold": r["fold"],
                "geom": np.array([float(feats.get(k, 0.0)) for k in GEOM], np.float32),
                "plm4": np.array([v.mean(), v[0], v[:3].mean(),
                                  float((v >= 0.5).mean())], np.float32),
                "esmif": ei[ii].mean(0),
                "y": int(float(jac) >= T)})
    return rows


def per_target(rows, key):
    X = np.vstack([key(r) for r in rows])
    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    tid = np.array([r["id"] for r in rows])
    s = np.zeros(len(rows))
    for h in sorted(set(fold)):
        tr, te = fold != h, fold == h
        m = make_pipeline(StandardScaler(),
                          LogisticRegressionCV(Cs=CS, cv=3, max_iter=4000,
                                               scoring="roc_auc", n_jobs=-1))
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    by = {}
    for t, sc, yy in zip(tid, s, y):
        by.setdefault(t, [[], []])
        by[t][0].append(sc)
        by[t][1].append(yy)
    return by, X.shape[1]


def main() -> None:
    recs = load()
    print("v3 structures: %d" % len(recs))
    print("refitting the ESM-2 residue head per fold ...")
    probs = esm2_head_probs(recs)
    rows = build(recs, probs)
    print("candidates: %d  (%d qualifying)\n" % (len(rows), sum(r["y"] for r in rows)))

    variants = [
        ("geometry", lambda r: r["geom"]),
        ("geometry + plm4", lambda r: np.concatenate([r["geom"], r["plm4"]])),
        ("geometry + esmif", lambda r: np.concatenate([r["geom"], r["esmif"]])),
        ("geometry + plm4 + esmif",
         lambda r: np.concatenate([r["geom"], r["plm4"], r["esmif"]])),
    ]
    cells, dims = {}, {}
    for name, key in variants:
        cells[name], dims[name] = per_target(rows, key)
        print("  fitted %s" % name)

    common = sorted(set.intersection(*(set(v) for v in cells.values())))
    common = [t for t in common if any(cells["geometry"][t][1])]

    def hit(by, t, k):
        sc, ys = by[t]
        o = np.argsort(-np.asarray(sc), kind="stable")[:k]
        return int(any(ys[i] for i in o))

    res = {n: {k: np.array([hit(by, t, k) for t in common]) for k in KS}
           for n, by in cells.items()}

    print("\ncross-fitted, %d covered targets\n" % len(common))
    print("%-28s %6s %8s %8s %8s" % ("features", "dims", "conv@1", "conv@5", "conv@10"))
    print("-" * 64)
    for name, _k in variants:
        print("%-28s %6d %7.1f%% %7.1f%% %7.1f%%"
              % (name, dims[name], *(100 * res[name][k].mean() for k in KS)))

    rng = np.random.default_rng(SEED)
    n = len(common)
    bi = rng.integers(0, n, size=(20000, n))

    def ci(a, b, label):
        print("\n  %s" % label)
        for k in KS:
            d = res[a][k] - res[b][k]
            bm = np.sort(d[bi].mean(axis=1))
            lo, hi = bm[500], bm[19499]
            star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
            print("    top-%-3d %+6.1f  [%+6.1f, %+6.1f]%s"
                  % (k, 100 * d.mean(), 100 * lo, 100 * hi, star))

    print("\npaired bootstrap, 20,000 resamples")
    ci("geometry + esmif", "geometry + plm4", "ESM-IF1 against the plm4 aggregates")
    ci("geometry + plm4", "geometry", "what the aggregates add to geometry")
    ci("geometry + esmif", "geometry", "what ESM-IF1 adds to geometry")
    ci("geometry + plm4 + esmif", "geometry + esmif", "does keeping plm4 help")

    out = {"n_covered": n, "n_candidates": len(rows),
           "cells": {nm: {"dims": dims[nm],
                          **{"conv_%d" % k: float(res[nm][k].mean()) for k in KS}}
                     for nm in res}}
    (HERE / "v3_sequence_vs_geometry.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "v3_sequence_vs_geometry.json"))


if __name__ == "__main__":
    main()
