"""Route 2: does ESM-IF1's structure-conditioned representation add to ranking?

Seven contrasts have now failed to move top-5 conversion: model capacity,
representation width, atom-level chemistry, label volume, within-target
normalisation, aggregation, and residue-level supervision. Every one of those
either reshuffled the same information or added more of a kind already present.

ESM-IF1 is the one remaining candidate that is different in kind. ESM-2 sees
sequence, the atomistic descriptors see local chemistry, and both are computed
from information a sequence model can largely infer. ESM-IF1 is an inverse
folding model: its encoder is conditioned on backbone geometry, so its residue
representations encode what the fold puts where, which is exactly the property
a cryptic site depends on.

The empirical argument is the union analysis. Adding IF-SitePred to fpocket
moved union coverage from 73.7% to 89.9%, the largest single jump in the greedy
sequence, so ESM-IF1 demonstrably carries detection signal nothing else has.
Whether that transfers from detection to ranking is what this measures, and the
prior after seven negatives is not favourable.

Extraction runs natively here. IF-SitePred needed WSL for its PyMOL clustering
step, not for the encoder, and the backbone array is built straight from
Lacuna's own Structure, which sidesteps the biotite version conflict that made
that pipeline painful.

Protocol: train folds only, leave-one-fold-out over the four homology-separated
train folds. The designated test fold is never read. Comparisons are paired on
target with 20,000 bootstrap resamples, and gated on the interval.

    python analysis/moores_pocket_law/esmif_route2.py --limit 20   # smoke
    python analysis/moores_pocket_law/esmif_route2.py              # full
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
from dataio import JACCARD_THRESHOLD as T
import plm_headroom as PH
import atomistic as A

warnings.filterwarnings("ignore")

CACHE = HERE / "data" / "esmif_emb"        # ~230 MB; kept out of git
KS = (1, 5, 10)
SEED = 0
BACKBONE = ("N", "CA", "C")


def backbone(st):
    """L x 3 x 3 array of N/CA/C, and the residue numbers it lines up with.

    Built from Lacuna's Structure rather than through biotite: ESM-IF1's own
    loader pulls in a biotite version that conflicts with the one IF-SitePred
    needs, and the array is three atom lookups per residue.
    """
    coords, nums = [], []
    for res in st.residues:
        d = {st.atoms[i].name: st.atoms[i].coords for i in res.atom_indices}
        if all(a in d for a in BACKBONE):
            coords.append([d[a] for a in BACKBONE])
            nums.append(res.seq_num)
    return np.asarray(coords, dtype=np.float32), nums


def _encode(model, conv, xyz, dev):
    """ESM-IF1 encoder output, per residue, on the model's device.

    This is `esm.inverse_folding.util.get_encoder_output` with one change: its
    batch converter accepts a `device` argument that the helper never passes, so
    on GPU it builds the batch on CPU and the forward pass dies on a device
    mismatch. Four lines, reproduced rather than monkey-patched.
    """
    coords, confidence, _, _, padding_mask = conv([(xyz, None, None)], device=dev)
    out = model.encoder.forward(coords, padding_mask, confidence,
                                return_all_hiddens=False)
    return out["encoder_out"][0][1:-1, 0]          # drop bos/eos


def build_cache(recs) -> None:
    """One ESM-IF1 encoder pass per structure, cached. Resumable."""
    todo = [r for r in recs if not (CACHE / ("%s.npz" % r["id"])).exists()]
    print("esm-if1 cache: %d present, %d to compute"
          % (len(recs) - len(todo), len(todo)))
    if not todo:
        return

    import torch
    import esm
    import esm.inverse_folding as IF
    from lacuna.io.structure import load_structure

    CACHE.mkdir(parents=True, exist_ok=True)
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    conv = IF.util.CoordBatchConverter(alphabet)
    print("  device: %s" % dev)

    ok = fail = 0
    for i, r in enumerate(todo, 1):
        try:
            st = load_structure(A.CIF / ("%s.cif" % r["pdb"].upper()), chain=r["chain"])
            xyz, nums = backbone(st)
            if len(nums) < 10:
                raise ValueError("only %d complete backbone residues" % len(nums))
            with torch.no_grad():
                rep = _encode(model, conv, xyz, dev)
            rep = rep.detach().cpu().numpy()
            if len(rep) != len(nums):
                raise ValueError("encoder/residue mismatch %d vs %d"
                                 % (len(rep), len(nums)))
            np.savez_compressed(CACHE / ("%s.npz" % r["id"]),
                                emb=rep.astype(np.float16),
                                nums=np.asarray(nums, dtype=np.int32))
            ok += 1
        except Exception as exc:                        # noqa: BLE001
            fail += 1
            if fail <= 5:
                print("  skip %s: %s" % (r["id"], exc))
        if i % 50 == 0 or i == len(todo):
            print("  %4d/%d  ok=%d fail=%d" % (i, len(todo), ok, fail))


def assemble(recs):
    """Rows carrying ESM-2, ESM-IF1 and atom features for the same candidates."""
    rows = []
    for r in recs:
        f2 = PH.CACHE / ("%s.npz" % r["id"])
        fi = CACHE / ("%s.npz" % r["id"])
        fa = A.CACHE / ("%s.npz" % r["id"])
        if not (f2.exists() and fi.exists() and fa.exists()):
            continue
        z2, zi, za = np.load(f2), np.load(fi), np.load(fa)
        e2, ei = z2["emb"].astype(np.float32), zi["emb"].astype(np.float32)
        p2 = {int(n): k for k, n in enumerate(z2["nums"])}
        pi = {int(n): k for k, n in enumerate(zi["nums"])}
        atom = {int(rk): v for rk, v in zip(za["rank"], za["X"])}

        for rank, (res, jac) in enumerate(zip(r["residues_by_rank"],
                                              r["jac_by_rank"])):
            if rank not in atom:
                continue
            nums = PH._resnums(res, r["chain"])
            i2 = [p2[n] for n in nums if n in p2]
            ii = [pi[n] for n in nums if n in pi]
            if not i2 or not ii:
                continue
            rows.append({"id": r["id"], "fold": r["fold"],
                         "seq": e2[i2].mean(0), "fold_emb": ei[ii].mean(0),
                         "atom": atom[rank],
                         "y": int(float(jac) >= T)})
    return rows


def per_target(rows, key, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([key(r) for r in rows])
    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    tid = np.array([r["id"] for r in rows])
    s = np.zeros(len(rows))
    for h in sorted(set(fold)):
        tr, te = fold != h, fold == h
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=C))
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    by = {}
    for t, sc, yy in zip(tid, s, y):
        by.setdefault(t, [[], []])
        by[t][0].append(sc)
        by[t][1].append(yy)
    return by, X.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    folds = dataio.fold_map()
    recs = []
    with open(A.CAND, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tool") != "lacuna":
                continue
            f = folds.get(dataio.pdb_id_of(r["id"]))
            if f is None or not f.startswith("train"):
                continue
            if not (r.get("residues_by_rank") and r.get("centroid_by_rank")):
                continue
            r["fold"] = f
            recs.append(r)
    recs.sort(key=lambda r: r["id"])
    if a.limit:
        recs = recs[:a.limit]
    print("structures: %d\n" % len(recs))

    build_cache(recs)
    rows = assemble(recs)
    print("\ncandidates: %d  (%d qualifying)" % (len(rows), sum(r["y"] for r in rows)))
    if len(rows) < 50:
        raise SystemExit("too few candidates assembled to mean anything")

    variants = [
        ("ESM-2 only (baseline)", lambda r: r["seq"], 0.05),
        ("ESM-IF1 only", lambda r: r["fold_emb"], 0.05),
        ("ESM-2 + ESM-IF1", lambda r: np.concatenate([r["seq"], r["fold_emb"]]), 0.05),
        ("ESM-IF1 + atoms", lambda r: np.concatenate([r["fold_emb"], r["atom"]]), 0.05),
        ("all three", lambda r: np.concatenate([r["seq"], r["fold_emb"], r["atom"]]), 0.05),
    ]
    cells, dims = {}, {}
    for name, key, C in variants:
        cells[name], dims[name] = per_target(rows, key, C)

    common = sorted(set.intersection(*(set(v) for v in cells.values())))
    common = [t for t in common if any(cells["ESM-2 only (baseline)"][t][1])]

    def hit(by, t, k):
        sc, ys = by[t]
        o = np.argsort(-np.asarray(sc), kind="stable")[:k]
        return int(any(ys[i] for i in o))

    res = {n: {k: np.array([hit(by, t, k) for t in common]) for k in KS}
           for n, by in cells.items()}

    print("\ncross-fitted, %d covered targets\n" % len(common))
    print("%-26s %6s %8s %8s %8s" % ("features", "dims", "conv@1", "conv@5", "conv@10"))
    print("-" * 62)
    for name, _k, _c in variants:
        print("%-26s %6d %7.1f%% %7.1f%% %7.1f%%"
              % (name, dims[name], *(100 * res[name][k].mean() for k in KS)))

    rng = np.random.default_rng(SEED)
    n = len(common)
    bi = rng.integers(0, n, size=(20000, n))
    base = "ESM-2 only (baseline)"
    print("\npaired against ESM-2 alone, 20,000 resamples")
    for name, _k, _c in variants[1:]:
        print("  %s" % name)
        for k in KS:
            d = res[name][k] - res[base][k]
            bm = np.sort(d[bi].mean(axis=1))
            lo, hi = bm[500], bm[19499]
            star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
            print("    top-%-3d %+6.1f  [%+6.1f, %+6.1f]%s"
                  % (k, 100 * d.mean(), 100 * lo, 100 * hi, star))

    out = {"n_covered": n,
           "cells": {nm: {"dims": dims[nm],
                          **{"conv_%d" % k: float(res[nm][k].mean()) for k in KS}}
                     for nm in res}}
    tag = "smoke%d" % a.limit if a.limit else "full"
    (HERE / ("esmif_route2_%s.json" % tag)).write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / ("esmif_route2_%s.json" % tag)))


if __name__ == "__main__":
    main()
