"""The fused pool is better. Can a ranker that has seen it exploit that?

`end_to_end_fusion.py` measured the surface detector through the whole pipeline
on 747 train-fold targets. Fusing it with the alpha detector raises coverage
from 65.7% to 81.7%, +15.9 [+13.3, +18.7], and costs top-five recovery:
-4.3 [-7.8, -0.7], with the interval clear of zero. More sites are found and
fewer are returned.

That is the decoy-injection result again, arriving from the other direction.
Injecting competitors drawn from the wrong-answer score distribution cost 17
points; removing low-scoring candidates at random bought nothing; and now
adding candidates the shipped ranker was never fitted on costs 4.3. In every
case what hurts is wrong answers the ranker scores highly, which is exactly
what a surface candidate is to a model fitted only on alpha-detector features.

So the question is whether the loss is intrinsic to a bigger pool or an
artefact of the ranker never having seen one. This refits on the fused pool and
compares four orderings of candidates already computed:

    alpha pool, shipped ranker     the current product
    fused pool, shipped ranker     what shipping the detector would do today
    fused pool, refit ranker       same candidates, ranker cross-fitted on them
    fused pool, refit + provenance the same, plus which detectors found the site

Provenance is worth testing on its own. A cluster both detectors independently
proposed is a different object from one only the surface model liked, and the
shipped feature set has no way to say so.

Nothing is re-run: `end_to_end_fusion.jsonl` already carries every candidate's
27 ranker features, its Jaccard and its detector sources. Cross-fitting is
leave-one-fold-out over the four homology-separated training folds.

Protocol: train folds only. The designated test fold is never read here.

    python analysis/moores_pocket_law/refit_fused_ranker.py
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

SRC = HERE / "end_to_end_fusion.jsonl"

#: end_to_end_fusion ranked with rank_by="learned", which is the 23-feature
#: geometry ranker plus an intercept, NOT the 27-feature PLM ranker. Scoring the
#: saved features with the PLM weights instead put its four plm_* columns at
#: zero and produced a different, worse ordering, which understated the baseline
#: by 3.6 points and inflated everything measured against it. The check in
#: main() exists so that can never pass silently again.
from lacuna.pockets.clusterer import (                       # noqa: E402
    _RANKER_FEATURES, _RANKER_INTERCEPT, _RANKER_WEIGHTS,
)

FEATS = list(_RANKER_FEATURES)
W = np.asarray(_RANKER_WEIGHTS)
B = float(_RANKER_INTERCEPT)
KS = (1, 5, 10)
N_BOOT = 20000


def load():
    """Per-target candidate matrices for the alpha and fused pools."""
    out = []
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("fused_feat"):
                continue
            row = {"pdb": r["pdb"], "fold": r["fold"]}
            for arm in ("alpha", "fused"):
                feat = r.get(arm + "_feat") or []
                if not feat:
                    row[arm] = None
                    continue
                X = np.array([[float(c.get(f, 0.0)) for f in FEATS] for c in feat])
                jac = np.array([float(x) for x in r[arm + "_jac"]])
                src = r[arm + "_src"]
                prov = np.array([[float("alpha" in s), float("surface" in s),
                                  float("alpha" in s and "surface" in s)]
                                 for s in src])
                row[arm] = {"X": X, "jac": jac, "y": (jac >= T).astype(int),
                            "prov": prov}
            out.append(row)
    return [r for r in out if r["alpha"] is not None and r["fused"] is not None]


def reconstruction_matches(rows) -> float:
    """Fraction of targets where X @ W + B reproduces the pipeline's hit@5.

    Compares against `alpha_jac`, which was written in the clusterer's own rank
    order, so agreeing on hit@5 means the reconstructed score orders candidates
    the same way where it matters.
    """
    ok = 0
    for r in rows:
        d = r["alpha"]
        pipeline = bool((d["jac"][:5] >= T).any())     # already in rank order
        order = np.argsort(-(d["X"] @ W + B), kind="stable")
        if bool((d["jac"][order[:5]] >= T).any()) == pipeline:
            ok += 1
    return ok / len(rows)


def hits_from(scores_by_pdb, rows, arm) -> dict:
    per = {k: [] for k in KS}
    for r in rows:
        d = r[arm]
        s = scores_by_pdb[r["pdb"]]
        order = np.argsort(-s, kind="stable")
        for k in KS:
            per[k].append(int(d["y"][order[:k]].any()))
    return per


def crossfit(rows, arm, with_prov: bool):
    """Leave-one-fold-out logistic ranking over the pooled candidates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def mat(d):
        return np.hstack([d["X"], d["prov"]]) if with_prov else d["X"]

    scores = {}
    for held in sorted({r["fold"] for r in rows}):
        tr = [r for r in rows if r["fold"] != held]
        te = [r for r in rows if r["fold"] == held]
        X = np.vstack([mat(r[arm]) for r in tr])
        y = np.concatenate([r[arm]["y"] for r in tr])
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0))
        m.fit(X, y)
        for r in te:
            scores[r["pdb"]] = m.predict_proba(mat(r[arm]))[:, 1]
    return scores


def ci(d, seed=0):
    rng = np.random.default_rng(seed)
    bm = np.sort(d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1))
    return 100 * d.mean(), 100 * bm[500], 100 * bm[19499]


def main() -> None:
    rows = load()
    if len(rows) < 100:
        raise SystemExit("only %d targets loaded from %s" % (len(rows), SRC))
    n = len(rows)
    cov_a = np.mean([r["alpha"]["y"].any() for r in rows])
    cov_f = np.mean([r["fused"]["y"].any() for r in rows])
    print("targets: %d   coverage  alpha %.1f%%  fused %.1f%%" % (n, 100 * cov_a, 100 * cov_f))
    print("median candidates  alpha %d  fused %d\n"
          % (int(np.median([len(r["alpha"]["jac"]) for r in rows])),
             int(np.median([len(r["fused"]["jac"]) for r in rows]))))

    # The pipeline wrote its own ranked residue lists. Reconstructing its
    # ordering from the saved features must reproduce them, or every number
    # below is measured against a baseline that never ran.
    check = reconstruction_matches(rows)
    print("reconstruction of the shipped ordering matches the pipeline on "
          "%.1f%% of targets" % (100 * check))
    if check < 0.98:
        raise SystemExit(
            "reconstruction disagrees with the pipeline on %.1f%% of targets. "
            "Fix that before reading anything below: the baseline would be a "
            "ranking that was never actually run." % (100 * (1 - check)))

    arms = {}
    arms["alpha pool, shipped ranker"] = (
        "alpha", {r["pdb"]: r["alpha"]["X"] @ W + B for r in rows})
    arms["fused pool, shipped ranker"] = (
        "fused", {r["pdb"]: r["fused"]["X"] @ W + B for r in rows})
    arms["fused pool, refit ranker"] = (
        "fused", crossfit(rows, "fused", with_prov=False))
    arms["fused pool, refit + provenance"] = (
        "fused", crossfit(rows, "fused", with_prov=True))
    # The ceiling: the fused pool ranked perfectly. Not achievable, but it says
    # how much of the coverage gain any ranker could ever convert.
    arms["fused pool, oracle ranking"] = (
        "fused", {r["pdb"]: r["fused"]["jac"] for r in rows})

    print("%-34s %8s %8s %8s" % ("arm", "top-1", "top-5", "top-10"))
    print("-" * 62)
    per = {}
    for name, (arm, sc) in arms.items():
        per[name] = hits_from(sc, rows, arm)
        print("%-34s %7.1f%% %7.1f%% %7.1f%%"
              % (name, 100 * np.mean(per[name][1]), 100 * np.mean(per[name][5]),
                 100 * np.mean(per[name][10])))

    base = "alpha pool, shipped ranker"
    print("\npaired against the current product, %d-resample CI" % N_BOOT)
    print("-" * 62)
    res = {}
    for name in arms:
        if name == base:
            continue
        for k in (1, 5):
            d = np.array(per[name][k], float) - np.array(per[base][k], float)
            m, lo, hi = ci(d)
            mark = "  resolved" if (lo > 0 or hi < 0) else ""
            print("  top-%-2d %-30s %+6.2f  [%+6.2f, %+6.2f]%s"
                  % (k, name, m, lo, hi, mark))
            res["%s@%d" % (name, k)] = (m, lo, hi)

    print("\n%s" % ("=" * 62))
    best = max(("fused pool, refit ranker", "fused pool, refit + provenance"),
               key=lambda nm: res["%s@5" % nm][1])
    m, lo, hi = res["%s@5" % best]
    if lo > 0:
        print("A ranker fitted on the fused pool converts the coverage gain.")
        print("The regression was the ranker, not the detector: %s" % best)
    elif hi < 0:
        print("Even a refit ranker loses on the fused pool. The extra candidates")
        print("are genuinely harder to order, so coverage bought at this cost is")
        print("not worth shipping as a default.")
    else:
        print("A refit recovers the regression but does not beat the baseline.")
        print("Fusion is then neutral at top five and positive at top ten, which")
        print("is a real gain only for callers who look past five.")

    (HERE / "refit_fused_ranker.json").write_text(json.dumps(
        {"n": n, "coverage": {"alpha": float(cov_a), "fused": float(cov_f)},
         "arms": {nm: {"hit_%d" % k: float(np.mean(v[k])) for k in KS}
                  for nm, v in per.items()},
         "deltas": {k: {"mean": v[0], "lo": v[1], "hi": v[2]}
                    for k, v in res.items()}}, indent=1))
    print("\nwrote %s" % (HERE / "refit_fused_ranker.json"))


if __name__ == "__main__":
    main()
