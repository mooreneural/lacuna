"""If a crowded pool hurts ranking, does thinning the pool help?

`phase10_decoy_injection.py` showed that adding synthetic competitors, drawn
from each target's own wrong-answer score distribution and holding the true
site, the candidate pool and the ranker fixed, costs about 17 points of top-5
recovery. Candidate quality did not change. Only the density did.

Two experiments have since closed the obvious routes out of the ranking
bottleneck. `ranker_headroom.py` found no model over the existing 27 features
beating the shipped linear one, and `plm_headroom_fair.py` found 1280-dimension
pooled embeddings matching four scalars once both sides are cross-fitted
honestly. Neither more capacity nor more sequence signal is the answer.

So this asks the question the injection result actually implies. If crowding
costs recovery, removing candidates before ranking should buy some of it back.

The design is deliberately built to fail if the effect is not real:

    baseline          every candidate, shipped ranker. What ships today.
    random keep f     drop a uniform random fraction, ranker unchanged. This is
                      the decisive arm. It manipulates density and nothing else,
                      so a gain here is a pure density effect.
    feature floors    drop the bottom quantile by one cheap feature that is not
                      the ranker's output. A gain here but not under random
                      means quality, not density, was doing the work.
    oracle            keep the qualifying candidate, drop a fraction of the
                      rest. Not achievable, and not meant to be: it bounds what
                      any filter could ever buy.

There is a reason to expect the random arm to come out flat, and it is worth
stating before the numbers rather than after. The injected decoys were drawn
from the wrong-answer *score* distribution, so they landed above the true site
by construction. Real candidates removed at random are mostly low-scoring and
already ranked below it, so deleting them moves nothing. If that is what
happens, the finding is not "density does not matter" but "only high-scoring
competitors matter", which is a different and harder filtering problem.

The diagnostic printed first says whether filtering can help at all, by
counting how many non-qualifying candidates outrank the true site. If that
number is usually zero or huge, no filter will change the answer.

Protocol: train folds only. The designated test fold is never read here.

    python analysis/moores_pocket_law/pool_density.py
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

FEATS = list(dataio._PLM_RANKER_FEATURES)
W = np.asarray(dataio._PLM_RANKER_WEIGHTS)
KS = (1, 5, 10)
N_SEEDS = 20          # random arms are stochastic; average over draws
N_BOOT = 20000


def load():
    """Per-target feature matrix, labels and shipped-ranker scores, train folds."""
    folds = dataio.fold_map()
    out = []
    for rec in dataio._jsonl(dataio.SWEEP[20]):
        # folds.json is keyed by PDB id, not by the pdb+chain structure id.
        fold = folds.get(dataio.pdb_id_of(rec["id"]))
        if fold is None or not fold.startswith("train"):
            continue
        cl = rec.get("clusters") or []
        if not cl:
            continue
        X = np.array([[float(c.get(f, 0.0)) for f in FEATS] for c in cl])
        jac = np.array([float(c.get("jac", 0.0)) for c in cl])
        out.append({"id": rec["id"], "X": X, "jac": jac, "y": jac >= T,
                    "s": X @ W,
                    "raw": {f: np.array([float(c.get(f, 0.0)) for c in cl])
                            for f in ("vol", "pers", "drug", "enc")}})
    return out


def hits(t, keep) -> dict:
    """hit@k for one target under a boolean keep-mask over its candidates."""
    if not keep.any():
        return {k: 0 for k in KS}
    s, y = t["s"][keep], t["y"][keep]
    order = np.argsort(-s, kind="stable")
    return {k: int(y[order[:k]].any()) for k in KS}


def arm(targets, mask_fn, seeds=1) -> dict:
    """Mean hit@k and coverage over targets, averaged across random draws."""
    acc = {k: [] for k in KS}
    cov = []
    for si in range(seeds):
        rng = np.random.default_rng(si)
        per = {k: [] for k in KS}
        c = []
        for t in targets:
            keep = mask_fn(t, rng)
            h = hits(t, keep)
            for k in KS:
                per[k].append(h[k])
            c.append(int(t["y"][keep].any()) if keep.any() else 0)
        for k in KS:
            acc[k].append(np.mean(per[k]))
        cov.append(np.mean(c))
    return {"hit": {k: float(np.mean(acc[k])) for k in KS},
            "coverage": float(np.mean(cov))}


def paired_delta(targets, mask_fn, k=5, seeds=N_SEEDS):
    """Per-target hit@k difference against the unfiltered baseline."""
    base = np.array([hits(t, np.ones(len(t["jac"]), bool))[k] for t in targets], float)
    got = np.zeros(len(targets))
    for si in range(seeds):
        rng = np.random.default_rng(si)
        got += [hits(t, mask_fn(t, rng))[k] for t in targets]
    return got / seeds - base


def ci(d, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    bm = np.sort(d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1))
    return 100 * d.mean(), 100 * bm[int(0.025 * n_boot)], 100 * bm[int(0.975 * n_boot) - 1]


# ────────────────────────────────── filters ──────────────────────────────────

def keep_all(t, rng):
    return np.ones(len(t["jac"]), bool)


def random_keep(frac):
    def f(t, rng):
        n = len(t["jac"])
        m = np.zeros(n, bool)
        m[rng.permutation(n)[:max(1, int(round(frac * n)))]] = True
        return m
    return f


def feature_floor(name, q):
    """Drop the bottom q quantile by one raw feature. Never sees jac."""
    def f(t, rng):
        v = t["raw"][name]
        if len(v) < 3:
            return np.ones(len(v), bool)
        return v >= np.quantile(v, q)
    return f


def oracle_keep(frac):
    """Keep every qualifying candidate, drop a fraction of the rest.

    Unachievable by construction: it uses the labels. Reported only to bound
    what a perfect filter of that strength could ever be worth.
    """
    def f(t, rng):
        m = t["y"].copy()
        wrong = np.flatnonzero(~t["y"])
        if len(wrong):
            m[rng.permutation(wrong)[:int(round(frac * len(wrong)))]] = True
        return m
    return f


def main() -> None:
    targets = load()
    if len(targets) < 50:
        raise SystemExit("only %d train-fold targets loaded; expected ~740. "
                         "Check the fold key format before trusting this."
                         % len(targets))

    covered = [t for t in targets if t["y"].any()]
    print("train-fold targets: %d   covered: %d   median candidates: %d\n"
          % (len(targets), len(covered),
             int(np.median([len(t["jac"]) for t in targets]))))

    # ── diagnostic: can filtering possibly help? ─────────────────────────────
    # For covered targets, count non-qualifying candidates the ranker puts above
    # the best qualifying one. That is the burden a filter would have to remove.
    burden = []
    for t in covered:
        order = np.argsort(-t["s"], kind="stable")
        y = t["y"][order]
        first = int(np.flatnonzero(y)[0])
        burden.append(first)
    burden = np.array(burden)
    print("competitors ranked above the true site (covered targets)")
    print("  zero (already rank 1) : %5.1f%%" % (100 * (burden == 0).mean()))
    print("  1 to 4  (in top 5)    : %5.1f%%" % (100 * ((burden >= 1) & (burden < 5)).mean()))
    print("  5 to 9  (near miss)   : %5.1f%%" % (100 * ((burden >= 5) & (burden < 10)).mean()))
    print("  10 or more            : %5.1f%%" % (100 * (burden >= 10).mean()))
    print("  median %d, 90th percentile %d\n"
          % (int(np.median(burden)), int(np.quantile(burden, 0.9))))

    # ── arms ─────────────────────────────────────────────────────────────────
    arms = [("baseline (no filter)", keep_all, 1)]
    arms += [("random keep %.0f%%" % (100 * f), random_keep(f), N_SEEDS)
             for f in (0.9, 0.75, 0.5, 0.25)]
    arms += [("floor: drop bottom %.0f%% %s" % (100 * q, n), feature_floor(n, q), 1)
             for n in ("vol", "pers", "drug", "enc") for q in (0.25, 0.5)]
    arms += [("ORACLE keep %.0f%% of wrong" % (100 * f), oracle_keep(f), N_SEEDS)
             for f in (0.5, 0.25)]

    print("%-34s %9s %8s %8s %8s" % ("arm", "coverage", "top-1", "top-5", "top-10"))
    print("-" * 72)
    rows = []
    for name, fn, seeds in arms:
        r = arm(targets, fn, seeds)
        rows.append((name, r))
        print("%-34s %8.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (name, 100 * r["coverage"], 100 * r["hit"][1],
                 100 * r["hit"][5], 100 * r["hit"][10]))

    print("\npaired difference in top-5 against baseline, %d-resample CI" % N_BOOT)
    print("-" * 72)
    verdict = {}
    for name, fn, seeds in arms[1:]:
        d = paired_delta(targets, fn, k=5, seeds=seeds)
        m, lo, hi = ci(d)
        mark = "  resolved" if (lo > 0 or hi < 0) else ""
        print("  %-32s %+6.2f  [%+6.2f, %+6.2f]%s" % (name, m, lo, hi, mark))
        verdict[name] = (m, lo, hi)

    print("\n%s" % ("=" * 72))
    best_rand = max((v for k, v in verdict.items() if k.startswith("random")),
                    key=lambda v: v[1])
    best_floor = max((v for k, v in verdict.items() if k.startswith("floor")),
                     key=lambda v: v[1])
    if best_rand[1] > 0:
        print("Thinning the pool at random helps. Density itself is the lever,")
        print("independent of which candidates are removed, and a cheap filter")
        print("is worth building.")
    elif best_floor[1] > 0:
        print("Random thinning does nothing but a feature floor helps. The gain")
        print("is candidate quality, not pool density: the useful filter removes")
        print("specific bad candidates rather than simply fewer of them.")
    else:
        print("Neither helps. The decoys in the injection experiment hurt because")
        print("they were drawn to score highly, and real candidates removed at")
        print("random sit below the true site already. The filtering problem is")
        print("to find high-scoring wrong answers, which is the ranking problem")
        print("again rather than a way around it.")
    print("The oracle rows bound what any filter of that strength could buy.")

    (HERE / "pool_density.json").write_text(json.dumps(
        {"n_targets": len(targets), "n_covered": len(covered),
         "burden_median": int(np.median(burden)),
         "arms": {n: r for n, r in rows},
         "delta_top5": {k: {"mean": v[0], "lo": v[1], "hi": v[2]}
                        for k, v in verdict.items()}}, indent=1))
    print("\nwrote %s" % (HERE / "pool_density.json"))


if __name__ == "__main__":
    main()
