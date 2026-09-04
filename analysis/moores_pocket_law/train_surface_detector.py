"""Fit the learned surface detector and export it as plain NumPy arrays.

`surface_scorer.py` and `surface_scorer_plm.py` established that this works:
held-out per-point AUC 0.848 with sequence features, recovering a qualifying
proposal at the annotated site for 34.1% of the targets where the alpha
detector proposes nothing at all. This turns that into the artifact the package
ships.

Two models are fitted, because sequence signal is optional at runtime:

    geom   the eleven geometric and physicochemical point features
    plm    those, plus three fields derived from per-residue cryptic-site
           probabilities

Both are trained on all four CryptoBench training folds. The designated test
fold is never read, so it stays clean for evaluating the finished detector.
Using the shipped `plm_head.npz` for the sequence fields is correct here and
was not correct in the cross-fitted analysis: that head was fitted on the same
four training folds, so it leaks nothing the detector has not already seen, but
it would have leaked into a held-out evaluation.

LightGBM is used to fit and is not a dependency of the package. The ensemble is
flattened into arrays and evaluated by NumPy at inference, so Lacuna keeps
installing with numpy, scipy and biopython alone.

    python analysis/moores_pocket_law/train_surface_detector.py --limit 60
    python analysis/moores_pocket_law/train_surface_detector.py
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
from lacuna.pockets import surface_detector as sd

CIF = REPO / "benchmarks" / "cb_data" / "cif"
DATASET = REPO / "benchmarks" / "cb_data" / "dataset.json"
EMB = HERE / "data" / "plm_emb"
OUT = REPO / "lacuna" / "pockets" / "surface_head.npz"

POS_RADIUS = 5.0        # A from an annotated site atom; matches the analysis
MAX_POINTS = 4000
NEG_PER_POS = 8


def shipped_residue_probs(pdb: str, chain: str) -> dict[int, float] | None:
    """P(residue lines a cryptic site) from the shipped head, via cached embeddings.

    Reads the cache rather than running ESM-2, so training needs no GPU.
    """
    f = EMB / ("%s%s.npz" % (pdb.lower(), chain))
    if not f.exists():
        return None
    z = np.load(f)
    w = np.load(REPO / "lacuna" / "pockets" / "plm_head.npz")
    logit = z["emb"].astype(np.float32) @ w["w"].astype(np.float32) + float(w["b"])
    prob = 1.0 / (1.0 + np.exp(-logit))
    return {int(n): float(p) for n, p in zip(z["nums"], prob)}


def build(pdb: str, entry: dict, rng):
    """Point features and labels for one structure, geometry and PLM variants."""
    from scipy.spatial import cKDTree

    from lacuna.io.structure import coords_array, load_structure
    from lacuna.pockets.detector import GRID_SPACING, _build_grid_context

    chain = entry["apo_chain"]
    st = load_structure(CIF / ("%s.cif" % pdb.upper()), chain=chain)
    coords = coords_array(st)
    if len(coords) < 20:
        return None
    ctx = _build_grid_context(coords, st, GRID_SPACING)

    vox = sd.shell_voxels(ctx)
    if len(vox) < 50:
        return None
    if len(vox) > MAX_POINTS:
        vox = vox[rng.permutation(len(vox))[:MAX_POINTS]]
    xyz = ctx.lo + vox * GRID_SPACING

    want = dataio.cb_residues(entry["apo_pocket_selection"])
    site = np.array([a.coords for a in st.atoms
                     if (a.chain_id, a.res_seq) in want], float)
    if len(site) < 3:
        return None
    d, _ = cKDTree(site).query(xyz, k=1)
    y = (d <= POS_RADIUS).astype(np.int8)
    if y.sum() < 5:
        return None

    probs = shipped_residue_probs(pdb, chain)
    return {
        "geom": sd.surface_point_features(st, ctx, vox),
        "plm": None if probs is None
        else sd.surface_point_features(st, ctx, vox, probs),
        "y": y,
    }


def flatten(booster) -> dict:
    """LightGBM trees to flat arrays a vectorised NumPy walk can evaluate."""
    dump = booster.dump_model()
    feature, thr, left, right, is_leaf, value, roots = [], [], [], [], [], [], []

    def walk(node) -> int:
        i = len(feature)
        feature.append(0); thr.append(0.0); left.append(0); right.append(0)
        is_leaf.append(False); value.append(0.0)
        if "leaf_value" in node:
            is_leaf[i] = True
            value[i] = float(node["leaf_value"])
            return i
        # Only the '<=' decision type is emitted for numeric splits, which is
        # all these features produce. Anything else would silently mis-evaluate,
        # so it raises instead.
        if node.get("decision_type", "<=") != "<=":
            raise ValueError("unsupported split %r" % node.get("decision_type"))
        feature[i] = int(node["split_feature"])
        thr[i] = float(node["threshold"])
        left[i] = walk(node["left_child"])
        right[i] = walk(node["right_child"])
        return i

    for tree in dump["tree_info"]:
        roots.append(walk(tree["tree_structure"]))

    return {"feature": np.array(feature, np.int32),
            "threshold": np.array(thr, np.float64),
            "left": np.array(left, np.int32),
            "right": np.array(right, np.int32),
            "is_leaf": np.array(is_leaf, bool),
            "value": np.array(value, np.float64),
            "roots": np.array(roots, np.int32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    folds = dataio.fold_map()
    ds = json.loads(DATASET.read_text())
    targets = []
    for pdb, entries in ds.items():
        f = folds.get(pdb.lower())
        if f is None or not f.startswith("train"):
            continue
        e = entries[0] if isinstance(entries, list) else entries
        if not (e.get("apo_chain") and e.get("apo_pocket_selection")):
            continue
        if not (CIF / ("%s.cif" % pdb.upper())).exists():
            continue
        targets.append((pdb, e))
    targets.sort()
    if a.limit:
        targets = targets[:a.limit]
    print("training structures (train folds only): %d\n" % len(targets))

    rng = np.random.default_rng(0)
    Xg, Xp, yg, yp, fail, t0 = [], [], [], [], 0, time.time()
    for i, (pdb, e) in enumerate(targets, 1):
        try:
            d = build(pdb, e, rng)
            if d is None:
                continue
            pos = np.flatnonzero(d["y"] == 1)
            neg = np.flatnonzero(d["y"] == 0)
            keep = np.concatenate([pos, rng.permutation(neg)[:NEG_PER_POS * len(pos)]])
            Xg.append(d["geom"][keep]); yg.append(d["y"][keep])
            if d["plm"] is not None:
                Xp.append(d["plm"][keep]); yp.append(d["y"][keep])
        except Exception as exc:                        # noqa: BLE001
            fail += 1
            if fail <= 5:
                print("  skip %s: %s" % (pdb, exc), flush=True)
        if i % 50 == 0 or i == len(targets):
            r = (time.time() - t0) / i
            print("  %4d/%d  fail=%d  %.2f s/target  eta %.0f min"
                  % (i, len(targets), fail, r, r * (len(targets) - i) / 60), flush=True)

    if len(Xg) < 40:
        raise SystemExit("only %d structures featurised; refusing to fit." % len(Xg))

    out = {}
    for tag, X, y, names in (("geom", Xg, yg, sd.GEOM_FEATURES),
                             ("plm", Xp, yp, sd.GEOM_FEATURES + sd.PLM_FEATURES)):
        if not X:
            print("\n%s: no data, skipping" % tag)
            continue
        X = np.vstack(X); y = np.concatenate(y)
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                               min_child_samples=50, subsample=0.9,
                               colsample_bytree=0.8, random_state=0, verbose=-1)
        m.fit(X, y)
        flat = flatten(m.booster_)
        for k, v in flat.items():
            out["%s_%s" % (tag, k)] = v

        # In-sample only: this says the export is faithful, not that the model
        # generalises. The held-out numbers come from the analysis scripts.
        raw = m.predict_proba(X)[:, 1]
        mine = 1.0 / (1.0 + np.exp(-sd._predict_raw(X, flat)))
        print("\n%s: %d points, %d features, %d trees, %d nodes"
              % (tag, len(y), X.shape[1], len(flat["roots"]), len(flat["feature"])))
        print("  in-sample AUC %.3f  AP %.3f  (base rate %.3f)"
              % (roc_auc_score(y, raw), average_precision_score(y, raw), y.mean()))
        print("  numpy export matches LightGBM to %.2e"
              % float(np.abs(mine - raw).max()))
        for n, g in sorted(zip(names, m.feature_importances_),
                           key=lambda kv: -kv[1])[:6]:
            print("    %-22s %d" % (n, g))

    np.savez_compressed(OUT, **out)
    print("\nwrote %s (%.1f KB)" % (OUT, OUT.stat().st_size / 1024))


if __name__ == "__main__":
    main()
