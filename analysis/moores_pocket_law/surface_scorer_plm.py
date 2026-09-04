"""Does sequence signal help a surface scorer find sites geometry discards?

`surface_scorer.py` showed a point classifier over eleven geometric and
physicochemical features reaching the annotated site on 22.4% [17.1, 28.3] of
the train-fold targets where the alpha detector proposes nothing qualifying at
all. Per-point AUC 0.760.

Every feature in that model is geometry or local atom composition. Meanwhile
`ranker_headroom.py` found that stripping the four `plm_*` features costs the
pocket ranker 8.7 points, so almost all of its discriminative power is
sequence-derived. If that holds at the point level too, projecting per-residue
sequence probabilities onto surface points should move the scorer, and it is
signal P2Rank structurally cannot use.

The arms are paired on an identical set of structures, so the only thing that
differs is whether the three sequence features are present:

    geometry        the eleven features from surface_scorer.py
    geometry + PLM  those, plus mean, max and inverse-distance-weighted mean of
                    P(residue lines a cryptic site) over the twelve nearest
                    residues to each point

One correctness point that decides whether any of this means anything. The
shipped head in `plm_head.npz` was fitted on all four training folds, so using
it here would leak held-out information into the sequence features while the
point classifier is cross-fitted honestly. `plm_headroom_fair.py` caught the
same problem and fixed it the same way: the residue head is refitted inside
each fold, from that fold's training structures only.

Protocol: train folds only. The designated test fold is never read here.

    python analysis/moores_pocket_law/surface_scorer_plm.py --limit 80
    python analysis/moores_pocket_law/surface_scorer_plm.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
import surface_scorer as ss
from dataio import JACCARD_THRESHOLD as T

EMB = HERE / "data" / "plm_emb"
NBR_CACHE = HERE / "data" / "surface_nbr"

K_RES = 12          # nearest residues summarised per surface point
MAX_RES_DIST = 12.0  # A; beyond this a residue tells you nothing about the point
NEG_PER_POS = 8
PLM_FEATS = ("plm_pt_mean", "plm_pt_max", "plm_pt_wmean")


def emb_id(pdb: str, chain: str) -> str:
    return "%s%s" % (pdb.lower(), chain)


def neighbours(pdb: str, entry: dict, xyz: np.ndarray):
    """For each surface point, the K nearest residues as embedding row indices.

    Cached because it is fixed geometry: only the probabilities laid over it
    change from fold to fold, and those are cheap once the mapping exists.
    """
    f = NBR_CACHE / ("%s.npz" % pdb)
    if f.exists():
        z = np.load(f)
        return z["idx"], z["dist"]

    from lacuna.io.structure import load_structure
    from scipy.spatial import cKDTree

    ez = np.load(EMB / ("%s.npz" % emb_id(pdb, entry["apo_chain"])))
    row_of = {int(n): i for i, n in enumerate(ez["nums"])}

    st = load_structure(ss.CIF / ("%s.cif" % pdb.upper()), chain=entry["apo_chain"])
    by_res: dict[int, list] = {}
    for a in st.atoms:
        by_res.setdefault(a.res_seq, []).append(a.coords)
    nums = [n for n in by_res if n in row_of]
    if len(nums) < 5:
        raise ValueError("only %d residues map to embedding rows" % len(nums))
    cent = np.array([np.mean(by_res[n], axis=0) for n in nums], float)
    rows = np.array([row_of[n] for n in nums], np.int32)

    k = min(K_RES, len(nums))
    dist, near = cKDTree(cent).query(xyz, k=k)
    if k == 1:
        dist, near = dist[:, None], near[:, None]
    idx = rows[near].astype(np.int32)
    # Mark residues too far to be informative; the aggregator drops them.
    idx[dist > MAX_RES_DIST] = -1
    if k < K_RES:                                  # pad short chains
        pad = K_RES - k
        idx = np.hstack([idx, -np.ones((len(xyz), pad), np.int32)])
        dist = np.hstack([dist, np.full((len(xyz), pad), 1e3)])

    NBR_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, idx=idx, dist=dist.astype(np.float32))
    return idx, dist.astype(np.float32)


def residue_labels(entry: dict, nums: np.ndarray) -> np.ndarray:
    want = {num for ch, num in dataio.cb_residues(entry["apo_pocket_selection"])
            if ch == entry["apo_chain"]}
    return np.array([1 if int(n) in want else 0 for n in nums], np.int8)


def plm_point_features(prob: np.ndarray, idx: np.ndarray, dist: np.ndarray):
    """mean, max and inverse-distance-weighted mean of P over nearby residues."""
    ok = idx >= 0
    p = np.where(ok, prob[np.clip(idx, 0, len(prob) - 1)], 0.0)
    n = ok.sum(axis=1).clip(min=1)
    mean = p.sum(axis=1) / n
    mx = np.where(ok, p, -1.0).max(axis=1)
    mx = np.where(mx < 0, 0.0, mx)
    w = np.where(ok, 1.0 / np.clip(dist, 1.0, None), 0.0)
    wmean = (p * w).sum(axis=1) / np.clip(w.sum(axis=1), 1e-6, None)
    return np.column_stack([mean, mx, wmean]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    folds = dataio.fold_map()
    ds = json.loads(ss.DATASET.read_text())
    zero = ss.zero_coverage_ids()

    targets = []
    for pdb, entries in ds.items():
        f = folds.get(pdb.lower())
        if f is None or not f.startswith("train"):
            continue
        e = entries[0] if isinstance(entries, list) else entries
        if not (e.get("apo_chain") and e.get("apo_pocket_selection")):
            continue
        # Both caches must exist for the pairing to be honest: an arm that
        # silently scored a different set of structures would not be comparable.
        if not (ss.CACHE / ("%s.npz" % pdb)).exists():
            continue
        if not (EMB / ("%s.npz" % emb_id(pdb, e["apo_chain"]))).exists():
            continue
        targets.append((pdb, e, f))
    targets.sort()
    if a.limit:
        z = [t for t in targets if t[0].lower() in zero][:a.limit // 2]
        o = [t for t in targets if t[0].lower() not in zero][:a.limit - len(z)]
        targets = sorted(z + o)

    print("targets with surface features and embeddings: %d   zero-coverage: %d\n"
          % (len(targets), sum(1 for p, _e, _f in targets if p.lower() in zero)))

    rng = np.random.default_rng(0)
    data, fail, t0 = {}, 0, time.time()
    for i, (pdb, e, f) in enumerate(targets, 1):
        try:
            d = ss.cached(pdb, e, rng)
            if d is None:
                continue
            idx, dist = neighbours(pdb, e, d["xyz"].astype(float))
            ez = np.load(EMB / ("%s.npz" % emb_id(pdb, e["apo_chain"])))
            d.update({"fold": f, "idx": idx, "dist": dist,
                      "emb": ez["emb"].astype(np.float32),
                      "res_y": residue_labels(e, ez["nums"])})
            data[pdb] = d
        except Exception as exc:                        # noqa: BLE001
            fail += 1
            if fail <= 5:
                print("  skip %s: %s" % (pdb, exc), flush=True)
        if i % 50 == 0 or i == len(targets):
            r = (time.time() - t0) / i
            print("  %4d/%d  ok=%d fail=%d  %.2f s/target  eta %.0f min"
                  % (i, len(targets), len(data), fail, r,
                     r * (len(targets) - i) / 60), flush=True)

    if len(data) < 40:
        raise SystemExit("only %d structures usable; too few to conclude." % len(data))

    fold_names = sorted({d["fold"] for d in data.values()})
    scores = {"geo": {}, "plm": {}}

    for held in fold_names:
        tr = [p for p, d in data.items() if d["fold"] != held]
        te = [p for p, d in data.items() if d["fold"] == held]

        # 1. Refit the residue head on this fold's training structures only.
        Xe, ye = [], []
        for p in tr:
            d = data[p]
            pos = np.flatnonzero(d["res_y"] == 1)
            neg = np.flatnonzero(d["res_y"] == 0)
            if not len(pos):
                continue
            take = rng.permutation(neg)[:NEG_PER_POS * len(pos)]
            keep = np.concatenate([pos, take])
            Xe.append(d["emb"][keep]); ye.append(d["res_y"][keep])
        head = LogisticRegression(max_iter=3000, C=0.01)
        head.fit(np.vstack(Xe), np.concatenate(ye))

        # 2. Sequence features for every structure this fold touches, from that
        #    refit head, so no structure is described by a head that saw it.
        for p in tr + te:
            d = data[p]
            prob = head.predict_proba(d["emb"])[:, 1]
            d["plm"] = plm_point_features(prob, d["idx"], d["dist"])

        # 3. The two point classifiers, identical but for those three columns.
        for arm in ("geo", "plm"):
            Xs, ys = [], []
            for p in tr:
                d = data[p]
                X = d["X"] if arm == "geo" else np.hstack([d["X"], d["plm"]])
                pos = np.flatnonzero(d["y"] == 1)
                neg = np.flatnonzero(d["y"] == 0)
                take = rng.permutation(neg)[:NEG_PER_POS * max(len(pos), 1)]
                keep = np.concatenate([pos, take])
                Xs.append(X[keep]); ys.append(d["y"][keep])
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                   num_leaves=63, min_child_samples=50,
                                   subsample=0.9, colsample_bytree=0.8,
                                   random_state=0, verbose=-1)
            m.fit(np.vstack(Xs), np.concatenate(ys))
            for p in te:
                d = data[p]
                X = d["X"] if arm == "geo" else np.hstack([d["X"], d["plm"]])
                scores[arm][p] = m.predict_proba(X)[:, 1]
        print("  fold %s done" % held, flush=True)

    ally = np.concatenate([data[p]["y"] for p in sorted(scores["geo"])])
    print("\nper-point classifier, cross-fitted over %d folds (base rate %.3f)"
          % (len(fold_names), ally.mean()))
    print("%-18s %8s %8s" % ("arm", "AUC", "AP"))
    print("-" * 38)
    for arm, lab in (("geo", "geometry"), ("plm", "geometry + PLM")):
        s = np.concatenate([scores[arm][p] for p in sorted(scores[arm])])
        print("%-18s %8.3f %8.3f"
              % (lab, roc_auc_score(ally, s), average_precision_score(ally, s)))

    ent = {p: e for p, e, _f in targets}
    res = {}
    for arm, lab in (("geo", "geometry"), ("plm", "geometry + PLM")):
        hits, checked = ss.recover_zero_coverage(data, scores[arm], zero, ent)
        res[arm] = (np.array(hits, float), checked)
        print("\n%-16s zero-coverage recovered: %d of %d  (%.1f%%)"
              % (lab, int(sum(hits)), len(hits), 100 * np.mean(hits)))

    # Paired over the same targets, so the difference is the sequence features
    # and nothing else.
    assert res["geo"][1] == res["plm"][1], "arms scored different targets"
    d = res["plm"][0] - res["geo"][0]
    n = len(d)
    rng2 = np.random.default_rng(0)
    bm = np.sort(d[rng2.integers(0, n, size=(20000, n))].mean(axis=1))
    lo, hi = 100 * bm[500], 100 * bm[19499]
    print("\npaired difference (PLM minus geometry), n=%d" % n)
    print("  %+.1f points  [%+.1f, %+.1f]%s"
          % (100 * d.mean(), lo, hi, "  resolved" if (lo > 0 or hi < 0) else ""))

    print("\n%s" % ("=" * 60))
    if lo > 0:
        print("Sequence signal helps at the point level too. The scorer should")
        print("carry it, and it is signal a geometry-only detector cannot have.")
    elif hi < 0:
        print("Sequence features make it worse here. Worth knowing: the residue")
        print("head predicts site lining, which is not the same target as")
        print("whether a surface point is ligandable.")
    else:
        print("Not resolved at this n. The point-level AUC delta above is the")
        print("more sensitive read; use it to decide whether to keep going.")

    (HERE / "surface_scorer_plm.json").write_text(json.dumps(
        {"n_structures": len(data), "n_zero_scored": n,
         "recovered": {k: float(v[0].mean()) for k, v in res.items()},
         "delta": {"mean": float(100 * d.mean()), "lo": lo, "hi": hi}}, indent=1))
    print("\nwrote %s" % (HERE / "surface_scorer_plm.json"))


if __name__ == "__main__":
    main()
