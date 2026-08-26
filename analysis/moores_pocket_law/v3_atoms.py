"""Can 23 CPU-computable descriptors replace the language model entirely?

Sequence-only, the atomistic descriptors tied ESM-2 pooled: -1.3 at top-5
[-4.9, +2.2]. Alongside geometry, ESM-IF1 tied the four plm_* aggregates:
+0.4 [-1.8, +2.6]. So the open question is whether atoms also tie plm_*
alongside geometry, which is the only version of the question with a shipping
consequence: if they do, Lacuna drops torch, transformers, and a 2.5 GB model
download without losing ranking quality.

Same protocol as v3_sequence_vs_geometry: train folds only, leave-one-fold-out,
per-fold refit ESM-2 head for the aggregates, inner CV for regularisation,
paired bootstrap over targets.

    python analysis/moores_pocket_law/v3_atoms.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))

import atomistic as A
import v3_sequence_vs_geometry as V
import plm_headroom as PH
from dataio import JACCARD_THRESHOLD as T

CACHE = HERE / "data" / "atom_feat_v3"


def build_atoms(recs):
    """Atom descriptors for the v3 candidates. Ranks differ from the earlier
    collection, so the old cache cannot be reused: it is keyed by a rank that
    now means a different pocket."""
    from lacuna.io.structure import load_structure
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recs if not (CACHE / ("%s.npz" % r["id"])).exists()]
    print("atom cache: %d present, %d to compute" % (len(recs) - len(todo), len(todo)))
    ok = fail = 0
    for i, r in enumerate(todo, 1):
        try:
            st = load_structure(A.CIF / ("%s.cif" % r["pdb"].upper()), chain=r["chain"])
            xyz = np.array([a.coords for a in st.atoms], float)
            tree = cKDTree(xyz)
            burial = np.array([len(tree.query_ball_point(p, 8.0)) - 1 for p in xyz], float)
            by_label = {res.label: res for res in st.residues}
            rows, keep = [], []
            for rank, (res_list, cent) in enumerate(zip(r["residues_by_rank"],
                                                        r["centroid_by_rank"])):
                idx = []
                for lab in res_list:
                    res = by_label.get(str(lab))
                    if res is not None:
                        idx.extend(res.atom_indices)
                if len(idx) < 3:
                    continue
                idx = np.asarray(idx, int)
                keys = [(st.atoms[j].res_name, st.atoms[j].name) for j in idx]
                v = A._feats(xyz[idx], [st.atoms[j].element for j in idx],
                             keys, burial[idx], cent)
                if v is not None:
                    rows.append(v); keep.append(rank)
            if not rows:
                raise ValueError("no candidate resolved")
            np.savez_compressed(CACHE / ("%s.npz" % r["id"]),
                                X=np.asarray(rows, np.float32),
                                rank=np.asarray(keep, np.int32))
            ok += 1
        except Exception as exc:                       # noqa: BLE001
            fail += 1
            if fail <= 3:
                print("  skip %s: %s" % (r["id"], exc))
        if i % 150 == 0 or i == len(todo):
            print("  %4d/%d ok=%d fail=%d" % (i, len(todo), ok, fail))


def main():
    recs = V.load()
    build_atoms(recs)
    print("\nrefitting the ESM-2 residue head per fold ...")
    probs = V.esm2_head_probs(recs)

    rows = []
    for r in recs:
        fa = CACHE / ("%s.npz" % r["id"])
        if not fa.exists() or r["id"] not in probs:
            continue
        za = np.load(fa)
        atom = {int(k): v for k, v in zip(za["rank"], za["X"])}
        p, pnums = probs[r["id"]]
        pp = {int(n): i for i, n in enumerate(pnums)}
        for rank, (feats, res, jac) in enumerate(zip(r["features_by_rank"],
                                                      r["residues_by_rank"],
                                                      r["jac_by_rank"])):
            if rank not in atom:
                continue
            ip = [pp[n] for n in PH._resnums(res, r["chain"]) if n in pp]
            if not ip:
                continue
            v = np.sort(p[ip])[::-1]
            rows.append({"id": r["id"], "fold": r["fold"],
                         "geom": np.array([float(feats.get(k, 0.0)) for k in V.GEOM], np.float32),
                         "plm4": np.array([v.mean(), v[0], v[:3].mean(),
                                           float((v >= 0.5).mean())], np.float32),
                         "atom": atom[rank],
                         "y": int(float(jac) >= T)})
    print("candidates: %d (%d qualifying)\n" % (len(rows), sum(r["y"] for r in rows)))

    variants = [("geometry", lambda r: r["geom"]),
                ("geometry + plm4", lambda r: np.concatenate([r["geom"], r["plm4"]])),
                ("geometry + atoms", lambda r: np.concatenate([r["geom"], r["atom"]])),
                ("geometry + atoms + plm4",
                 lambda r: np.concatenate([r["geom"], r["atom"], r["plm4"]]))]
    cells, dims = {}, {}
    for name, key in variants:
        cells[name], dims[name] = V.per_target(rows, key)
        print("  fitted %s" % name)

    common = sorted(set.intersection(*(set(v) for v in cells.values())))
    common = [t for t in common if any(cells["geometry"][t][1])]
    def hit(by, t, k):
        sc, ys = by[t]; o = np.argsort(-np.asarray(sc), kind="stable")[:k]
        return int(any(ys[i] for i in o))
    res = {n: {k: np.array([hit(by, t, k) for t in common]) for k in V.KS}
           for n, by in cells.items()}

    print("\ncross-fitted, %d covered targets\n" % len(common))
    print("%-28s %6s %8s %8s %8s" % ("features", "dims", "conv@1", "conv@5", "conv@10"))
    print("-" * 64)
    for name, _k in variants:
        print("%-28s %6d %7.1f%% %7.1f%% %7.1f%%"
              % (name, dims[name], *(100 * res[name][k].mean() for k in V.KS)))

    rng = np.random.default_rng(V.SEED); n = len(common)
    bi = rng.integers(0, n, size=(20000, n))
    def ci(a, b, label):
        print("\n  %s" % label)
        for k in V.KS:
            d = res[a][k] - res[b][k]
            bm = np.sort(d[bi].mean(axis=1)); lo, hi = bm[500], bm[19499]
            star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
            print("    top-%-3d %+6.1f  [%+6.1f, %+6.1f]%s" % (k, 100*d.mean(), 100*lo, 100*hi, star))
    print("\npaired bootstrap, 20,000 resamples")
    ci("geometry + atoms", "geometry + plm4", "atoms instead of the language model")
    ci("geometry + atoms", "geometry", "what atoms add to geometry")
    ci("geometry + atoms + plm4", "geometry + plm4", "do atoms add on top of plm4")

    (HERE / "v3_atoms.json").write_text(json.dumps(
        {"n_covered": n, "cells": {nm: {"dims": dims[nm],
         **{"conv_%d" % k: float(res[nm][k].mean()) for k in V.KS}} for nm in res}}, indent=1))
    print("\nwrote %s" % (HERE / "v3_atoms.json"))


if __name__ == "__main__":
    main()
