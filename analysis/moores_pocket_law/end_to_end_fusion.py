"""Does the surface detector help the whole pipeline, not just the point model?

`surface_scorer_plm.py` measured whether a proposal lands on the annotated site.
That is a coverage question and it answered it: 34.1% of the targets the alpha
detector misses entirely get a qualifying cluster. It says nothing about what
happens next. Those proposals still enter the pool, get clustered across the
ensemble, and get ranked by a linear model fitted on alpha-detector candidates.
Three things could go wrong between there and top-five recovery, and only a run
through the real pipeline can tell:

  * the recovered sites might rank below five and change nothing
  * the extra candidates might crowd out sites the alpha detector already found
  * the fitted ranker might score surface candidates badly, having never seen one

So this runs the shipped path end to end, three ways, on the same targets:

    alpha    the shipped detector alone. The baseline that must be beaten.
    surface  the learned detector alone.
    fused    both detectors' pockets pooled into one clustering, which is what
             shipping this would actually mean.

Everything downstream is identical across arms: same NMA ensemble, same
conformers, same clusterer, same `learned` ranker. Only the candidate source
changes.

Two leaks this had to avoid, both of which would have made the result look
better than it is. The shipped `surface_head.npz` was fitted on all four
training folds, so scoring train-fold targets with it is in-sample; per-fold
heads are fitted here instead and a structure is only ever scored by the head
that did not see its fold. The same applies to `plm_head.npz`, which supplies
the sequence fields, so the residue head is refitted per fold too.
`plm_headroom_fair.py` found the same leak in the ranking analysis.

Protocol: train folds only. The designated test fold is never read here.

    python analysis/moores_pocket_law/end_to_end_fusion.py --limit 60 --conformers 5
    python analysis/moores_pocket_law/end_to_end_fusion.py --conformers 10
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
import train_surface_detector as tsd
from dataio import JACCARD_THRESHOLD as T
from lacuna.pockets import surface_detector as sd

CIF = REPO / "benchmarks" / "cb_data" / "cif"
DATASET = REPO / "benchmarks" / "cb_data" / "dataset.json"
EMB = HERE / "data" / "plm_emb"
FOLD_HEADS = HERE / "data" / "surface_fold_heads"
CACHE = HERE / "end_to_end_fusion.jsonl"
CACHE_NOPLM = HERE / "end_to_end_fusion_noplm.jsonl"

ARMS = ("alpha", "surface", "fused")
KS = (1, 5, 10)
SEED = 42
NEG_PER_POS = 8


# ─────────────────────── per-fold heads, fitted honestly ─────────────────────

def fold_residue_heads(targets, folds) -> dict:
    """A residue head per fold, fitted on the other folds' structures only."""
    from sklearn.linear_model import LogisticRegression

    by_fold: dict[str, list] = {}
    for pdb, e, f in targets:
        p = EMB / ("%s%s.npz" % (pdb.lower(), e["apo_chain"]))
        if p.exists():
            by_fold.setdefault(f, []).append((pdb, e, p))

    rng = np.random.default_rng(0)
    heads = {}
    for held in sorted(by_fold):
        X, y = [], []
        for f, rows in by_fold.items():
            if f == held:
                continue
            for pdb, e, p in rows:
                z = np.load(p)
                want = {num for ch, num in
                        dataio.cb_residues(e["apo_pocket_selection"])
                        if ch == e["apo_chain"]}
                lab = np.array([1 if int(n) in want else 0 for n in z["nums"]], np.int8)
                pos = np.flatnonzero(lab == 1)
                if not len(pos):
                    continue
                neg = rng.permutation(np.flatnonzero(lab == 0))[:NEG_PER_POS * len(pos)]
                keep = np.concatenate([pos, neg])
                X.append(z["emb"][keep].astype(np.float32)); y.append(lab[keep])
        m = LogisticRegression(max_iter=3000, C=0.01)
        m.fit(np.vstack(X), np.concatenate(y))
        heads[held] = (m.coef_[0].astype(np.float32), float(m.intercept_[0]))
        print("  residue head for %s fitted" % held, flush=True)
    return heads


def residue_probs(pdb, entry, head) -> dict | None:
    if head is None:
        return None
    p = EMB / ("%s%s.npz" % (pdb.lower(), entry["apo_chain"]))
    if not p.exists():
        return None
    z = np.load(p)
    w, b = head
    prob = 1.0 / (1.0 + np.exp(-(z["emb"].astype(np.float32) @ w + b)))
    return {int(n): float(x) for n, x in zip(z["nums"], prob)}


def fold_surface_heads(targets, res_heads) -> dict:
    """A surface head per fold, fitted on the other folds' structures only."""
    import lightgbm as lgb

    FOLD_HEADS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    per_fold_data: dict[str, list] = {}
    for i, (pdb, e, f) in enumerate(targets, 1):
        try:
            probs = residue_probs(pdb, e, res_heads[f])
            # A missing embedding is a reason to skip only when the run is
            # supposed to have one. In geometry-only mode every structure has
            # probs None by design, and skipping them all leaves no data and a
            # KeyError four steps later.
            if probs is None and res_heads[f] is not None:
                continue
            d = _build_with_probs(pdb, e, rng, probs)
            if d is not None:
                per_fold_data.setdefault(f, []).append(d)
        except Exception as exc:                        # noqa: BLE001
            if i < 5:
                print("  featurise skip %s: %s" % (pdb, exc), flush=True)
        if i % 100 == 0:
            print("  featurised %d/%d" % (i, len(targets)), flush=True)

    heads = {}
    for held in sorted(per_fold_data):
        X = np.vstack([d[0] for f, rows in per_fold_data.items() if f != held
                       for d in rows])
        y = np.concatenate([d[1] for f, rows in per_fold_data.items() if f != held
                            for d in rows])
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                               min_child_samples=50, subsample=0.9,
                               colsample_bytree=0.8, random_state=0, verbose=-1)
        m.fit(X, y)
        heads[held] = tsd.flatten(m.booster_)
        print("  surface head for %s fitted on %d points" % (held, len(y)), flush=True)
    return heads


def _build_with_probs(pdb, entry, rng, probs):
    """Point features and labels for one structure, using the given residue probs."""
    from scipy.spatial import cKDTree

    from lacuna.io.structure import coords_array, load_structure
    from lacuna.pockets.detector import GRID_SPACING, _build_grid_context

    st = load_structure(CIF / ("%s.cif" % pdb.upper()), chain=entry["apo_chain"])
    coords = coords_array(st)
    ctx = _build_grid_context(coords, st, GRID_SPACING)
    vox = sd.shell_voxels(ctx)
    if len(vox) < 50:
        return None
    if len(vox) > tsd.MAX_POINTS:
        vox = vox[rng.permutation(len(vox))[:tsd.MAX_POINTS]]
    xyz = ctx.lo + vox * GRID_SPACING

    want = dataio.cb_residues(entry["apo_pocket_selection"])
    site = np.array([a.coords for a in st.atoms
                     if (a.chain_id, a.res_seq) in want], float)
    if len(site) < 3:
        return None
    d, _ = cKDTree(site).query(xyz, k=1)
    y = (d <= tsd.POS_RADIUS).astype(np.int8)
    if y.sum() < 5:
        return None
    pos = np.flatnonzero(y == 1)
    neg = rng.permutation(np.flatnonzero(y == 0))[:NEG_PER_POS * len(pos)]
    keep = np.concatenate([pos, neg])
    return (sd.surface_point_features(st, ctx, vox, probs)[keep], y[keep])


# ──────────────────────────── the pipeline, three ways ───────────────────────

def run_target(pdb, entry, fold, n_conf, s_head, r_head) -> dict:
    from lacuna.ensemble.nma_backend import NMABackend
    from lacuna.io.structure import coords_array, load_structure
    from lacuna.pockets.clusterer import cluster_pockets, ranker_features
    from lacuna.pockets.detector import detect_pockets

    path = CIF / ("%s.cif" % pdb.upper())
    st = load_structure(path, chain=entry["apo_chain"])
    base = coords_array(st)
    confs = [base]
    if n_conf > 0:
        confs += NMABackend(seed=SEED, max_rmsd=2.0, n_modes=10).generate(
            path, n_conformers=n_conf, chain=entry["apo_chain"])

    probs = residue_probs(pdb, entry, r_head)
    per = {"alpha": [], "surface": [], "fused": []}
    for c in confs:
        a = detect_pockets(c, st)
        s = sd.detect_pockets_surface(c, st, plm_residue_probs=probs, head=s_head)
        per["alpha"].append(a)
        per["surface"].append(s)
        per["fused"].append(a + s)

    out = {"pdb": pdb, "fold": fold,
           "truth": sorted(entry["apo_pocket_selection"])}
    truth = as_set(entry["apo_pocket_selection"], lacuna_style=False)
    for arm in ARMS:
        cl = cluster_pockets(per[arm], n_conformers=len(confs), rank_by="learned")
        out[arm] = [sorted(str(r) for r in c.lining_residues) for c in cl]
        # The ranker cannot be refitted on pooled candidates without their
        # features, and regenerating them means rerunning the whole pipeline,
        # so they are written alongside the residues rather than recomputed.
        feats, jacs, srcs = [], [], []
        for c in cl:
            got = as_set(c.lining_residues, lacuna_style=True)
            feats.append(ranker_features(c))
            jacs.append(len(got & truth) / len(got | truth) if got else 0.0)
            srcs.append(sorted({p.source for p in (c.member_pockets or [])} or [""]))
        out[arm + "_feat"] = feats
        out[arm + "_jac"] = jacs
        out[arm + "_src"] = srcs
    return out


def as_set(entries, lacuna_style: bool) -> set:
    """Both label formats to {(chain, resnum)}. See dataio for the parsers."""
    return (dataio.lacuna_residues(entries) if lacuna_style
            else dataio.cb_residues(entries))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conformers", type=int, default=10)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-plm", action="store_true",
                    help="Train and run the surface head on geometry alone. "
                         "The measured gain used ESM-2, which needs the plm "
                         "extra and a GPU to be quick; this says what is left "
                         "without either.")
    a = ap.parse_args()

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
        if not (EMB / ("%s%s.npz" % (pdb.lower(), e["apo_chain"]))).exists():
            continue
        targets.append((pdb, e, f))
    targets.sort()
    if a.limit:
        targets = targets[:a.limit]
    print("targets: %d   conformers: %d\n" % (len(targets), a.conformers))

    print("fitting per-fold heads (nothing scores a structure it trained on)")
    if a.no_plm:
        # None everywhere: surface_point_features omits the plm_* columns,
        # and the per-fold heads are fitted on the same eleven features they
        # will be scored with. Mixing those would be train/serve skew.
        r_heads = {f: None for f in {t[2] for t in targets}}
    else:
        r_heads = fold_residue_heads(targets, folds)
    s_heads = fold_surface_heads(targets, r_heads)

    cache = CACHE_NOPLM if a.no_plm else CACHE
    done = {}
    if cache.exists() and not a.refresh:
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["pdb"]] = r
    todo = [t for t in targets if t[0] not in done]
    print("\ncached: %d   to run: %d" % (len(targets) - len(todo), len(todo)))

    fail, t0 = 0, time.time()
    with cache.open("a", encoding="utf-8") as fh:
        for i, (pdb, e, f) in enumerate(todo, 1):
            try:
                r = run_target(pdb, e, f, a.conformers, s_heads[f], r_heads[f])
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done[pdb] = r
            except Exception as exc:                    # noqa: BLE001
                fail += 1
                if fail <= 5:
                    print("  skip %s: %s" % (pdb, exc), flush=True)
            if i % 20 == 0 or i == len(todo):
                rt = (time.time() - t0) / i
                print("  %4d/%d  ok=%d fail=%d  %.1f s/target  eta %.0f min"
                      % (i, len(todo), i - fail, fail, rt,
                         rt * (len(todo) - i) / 60), flush=True)

    rows = [done[p] for p, _e, _f in targets if p in done]
    if len(rows) < 30:
        raise SystemExit("only %d targets scored; too few to read anything into."
                         % len(rows))

    per = {arm: {"cov": [], **{k: [] for k in KS}} for arm in ARMS}
    for r in rows:
        truth = as_set(r["truth"], lacuna_style=False)
        for arm in ARMS:
            j = [len(as_set(s, True) & truth) / len(as_set(s, True) | truth)
                 if as_set(s, True) else 0.0 for s in r.get(arm, [])]
            per[arm]["cov"].append(int(any(x >= T for x in j)))
            for k in KS:
                per[arm][k].append(int(any(x >= T for x in j[:k])))

    n = len(rows)
    print("\n%d targets scored, %d skipped\n" % (n, fail))
    print("%-10s %9s %8s %8s %8s" % ("arm", "coverage", "top-1", "top-5", "top-10"))
    print("-" * 48)
    for arm in ARMS:
        d = per[arm]
        print("%-10s %8.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (arm, 100 * np.mean(d["cov"]), 100 * np.mean(d[1]),
                 100 * np.mean(d[5]), 100 * np.mean(d[10])))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(20000, n))
    print("\npaired against alpha, 20,000-resample CI")
    print("-" * 48)
    res = {}
    for arm in ("surface", "fused"):
        for label, key in (("coverage", "cov"), ("top-5", 5)):
            d = np.array(per[arm][key], float) - np.array(per["alpha"][key], float)
            bm = np.sort(d[idx].mean(axis=1))
            lo, hi = 100 * bm[500], 100 * bm[19499]
            mark = "  resolved" if (lo > 0 or hi < 0) else ""
            print("  %-8s %-9s %+6.2f  [%+6.2f, %+6.2f]%s"
                  % (arm, label, 100 * d.mean(), lo, hi, mark))
            res["%s_%s" % (arm, label)] = (100 * d.mean(), lo, hi)

    print("\n%s" % ("=" * 48))
    cov_lo = res["fused_coverage"][1]
    top_m, top_lo, top_hi = res["fused_top-5"]
    if cov_lo > 0 and top_lo > 0:
        print("Fusion raises coverage and that reaches top five. Ship it.")
    elif cov_lo > 0 and top_hi < 0:
        print("Fusion finds more sites and recovers fewer. The extra candidates")
        print("crowd out sites the alpha detector already had, so the ranker,")
        print("not the detector, is what blocks this from shipping.")
    elif cov_lo > 0:
        print("Coverage rises but top-five does not move. The sites are in the")
        print("pool and the ranker is not surfacing them: a ranker refit that")
        print("has seen surface candidates is the next step, not more detection.")
    else:
        print("Fusion does not raise coverage in the full pipeline, despite doing")
        print("so on the zero-coverage subset. Worth checking whether the "
              "clusterer is merging surface proposals into alpha ones.")

    (HERE / ("end_to_end_fusion_noplm.json" if a.no_plm else "end_to_end_fusion.json")).write_text(json.dumps(
        {"n": n, "conformers": a.conformers,
         "arms": {arm: {"coverage": float(np.mean(per[arm]["cov"])),
                        **{"hit_%d" % k: float(np.mean(per[arm][k])) for k in KS}}
                  for arm in ARMS},
         "deltas": {k: {"mean": v[0], "lo": v[1], "hi": v[2]}
                    for k, v in res.items()}}, indent=1))
    print("\nwrote %s" % (HERE / ("end_to_end_fusion_noplm.json" if a.no_plm else "end_to_end_fusion.json")))


if __name__ == "__main__":
    main()
