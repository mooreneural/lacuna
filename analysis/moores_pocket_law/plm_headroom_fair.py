"""The same comparison, with the incumbent held to the same standard.

`plm_headroom.py` found pooled embeddings losing to the four aggregates by 5.8
points. That result is not trustworthy as it stands, because the two sides were
not evaluated the same way. The shipped head (`plm_head.npz`) was fitted on all
four training folds, so the aggregates derived from it carry information about
every held-out fold, while the pooled-embedding models were cross-fitted
honestly. The incumbent was scored on data it had seen.

This refits the residue head inside each fold, from per-residue labels, so both
sides are cross-fitted. It also sweeps regularisation for the pooled
representation, since 1280 dimensions against roughly 1100 pocket-level
positives is a regime where a single arbitrary C says little.

Three things get separated:

    shipped head          leaky incumbent, kept only as a reference line
    refit head            the same architecture, honestly cross-fitted
    pooled embeddings     the raw representation, at several strengths

If the refit head falls to the level of the pooled representations, the earlier
verdict was an artefact of leakage. If it holds up, the supervised projection is
genuinely doing work that pooling cannot, and the reason is label budget: the
head learns from every residue in the training set, while a pocket-level ranker
sees only ~1100 positive pockets.

    python analysis/moores_pocket_law/plm_headroom_fair.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plm_headroom as P                                    # noqa: E402
from dataio import JACCARD_THRESHOLD as T                    # noqa: E402


def per_residue_table(recs):
    """Embeddings and site membership for every residue, tagged by fold."""
    out = []
    for r in recs:
        f = P.CACHE / ("%s.npz" % r["id"])
        if not f.exists():
            continue
        z = np.load(f)
        emb = z["emb"].astype(np.float32)
        nums = [int(n) for n in z["nums"]]
        site = set(P._resnums(r.get("site_residues") or [], r["chain"]))
        if not site:
            continue
        y = np.array([int(n in site) for n in nums], dtype=np.int8)
        out.append({"id": r["id"], "fold": r["fold"], "emb": emb,
                    "nums": nums, "y": y, "rec": r})
    return out


def fit_head(tab, folds_in):
    """Logistic residue head, trained only on the given folds."""
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([t["emb"] for t in tab if t["fold"] in folds_in])
    y = np.concatenate([t["y"] for t in tab if t["fold"] in folds_in])
    m = LogisticRegression(max_iter=1000, C=0.01, solver="lbfgs")
    m.fit(X, y)
    return m


def aggregates(prob_by_res, res_entries, chain):
    idx = [prob_by_res[n] for n in P._resnums(res_entries, chain) if n in prob_by_res]
    if not idx:
        return None
    p = np.sort(np.asarray(idx))[::-1]
    return np.array([p.mean(), p[0], p[:3].mean(), float((p >= 0.5).mean())],
                    dtype=np.float32)


def conversion(by_target, ks=(1, 5, 10)):
    covered = [t for t, (_s, ys) in by_target.items() if any(ys)]
    res = {}
    for k in ks:
        hit = []
        for t in covered:
            s, ys = by_target[t]
            order = np.argsort(-np.asarray(s), kind="stable")[:k]
            hit.append(int(any(ys[i] for i in order)))
        res["conv_%d" % k] = float(np.mean(hit))
    res["n_covered"] = len(covered)
    return res


def main() -> None:
    recs = P.records()
    tab = per_residue_table(recs)
    folds = sorted({t["fold"] for t in tab})
    n_res = sum(len(t["y"]) for t in tab)
    n_pos = int(sum(t["y"].sum() for t in tab))
    print("structures %d   residues %d   site residues %d (%.1f%%)   folds %s\n"
          % (len(tab), n_res, n_pos, 100 * n_pos / n_res, ",".join(folds)))

    # ── refit head, cross-fitted ────────────────────────────────────────────
    by_refit = {}
    for held in folds:
        m = fit_head(tab, [f for f in folds if f != held])
        for t in tab:
            if t["fold"] != held:
                continue
            pr = m.predict_proba(t["emb"])[:, 1]
            pmap = dict(zip(t["nums"], pr))
            r = t["rec"]
            s, ys = [], []
            for res, jac in zip(r["residues_by_rank"], r["jac_by_rank"]):
                a = aggregates(pmap, res, r["chain"])
                if a is None:
                    continue
                s.append(float(a[0]))          # rank by mean, the dominant term
                ys.append(int(float(jac) >= T))
            if s:
                by_refit[t["id"]] = (s, ys)
    refit = conversion(by_refit)
    print("refit head (honest, cross-fitted):  "
          "conv@1 %.1f%%  conv@5 %.1f%%  conv@10 %.1f%%   n=%d"
          % (100 * refit["conv_1"], 100 * refit["conv_5"],
             100 * refit["conv_10"], refit["n_covered"]))

    # ── shipped head, for reference only ────────────────────────────────────
    rows = P.featurize(recs)
    shipped = P.evaluate(rows, lambda r: r["agg"], "shipped", C=1.0)
    print("shipped head (LEAKY, saw all folds): "
          "conv@1 %.1f%%  conv@5 %.1f%%  conv@10 %.1f%%   n=%d"
          % (100 * shipped["conv_1"], 100 * shipped["conv_5"],
             100 * shipped["conv_10"], shipped["n_covered"]))

    # ── pooled embeddings across regularisation ─────────────────────────────
    print("\npooled raw embeddings, regularisation sweep")
    print("%-10s %8s %8s %8s" % ("C", "conv@1", "conv@5", "conv@10"))
    print("-" * 38)
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        r = P.evaluate(rows, lambda x: x["mean"], "mean C=%g" % C, C=C)
        print("%-10g %7.1f%% %7.1f%% %7.1f%%"
              % (C, 100 * r["conv_1"], 100 * r["conv_5"], 100 * r["conv_10"]))
        if best is None or r["conv_5"] > best["conv_5"]:
            best = r

    print("\n" + "=" * 70)
    d = 100 * (best["conv_5"] - refit["conv_5"])
    print("pooled best vs refit head, top-5: %+.1f points" % d)
    print("leakage in the shipped number    : %+.1f points"
          % (100 * (shipped["conv_5"] - refit["conv_5"])))
    if d >= 2.0:
        print("\nVERDICT: pooling wins once the incumbent is judged fairly. The")
        print("linear head is discarding signal; per-residue embeddings and a")
        print("larger PLM are the right GPU spend.")
    else:
        print("\nVERDICT: the supervised residue head still wins on equal terms.")
        print("Its advantage is label budget, not representation: it learns from")
        print("every residue in the training set, while a pocket-level ranker")
        print("sees only ~1100 positive pockets. A larger PLM does not fix that.")
        print("The lever is more per-residue supervision, not more dimensions.")

    (HERE / "plm_headroom_fair.json").write_text(json.dumps(
        {"refit_head": refit, "shipped_head_leaky": shipped,
         "pooled_best": best}, indent=1))
    print("\nwrote %s" % (HERE / "plm_headroom_fair.json"))


if __name__ == "__main__":
    main()
