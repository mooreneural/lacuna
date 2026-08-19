"""Phase 10b: the same injection, with decoys that cannot be miscalibrated.

Phase 10 injected clusters from other proteins. The production ranker is fitted
on within-structure pairs, so it is only ever asked to order pockets inside one
structure and nothing constrains its scores to be comparable across structures.
A foreign pocket can therefore outscore the true site for reasons that have
nothing to do with candidate burden, which would inflate the measured effect.

This variant removes that route entirely. Injected decoys are given scores drawn
from the target's *own* non-qualifying candidates, so a decoy is statistically
indistinguishable from a wrong answer the target already produced. Any remaining
drop is burden and nothing else.

Comparing the two gives the decomposition: how much of Phase 10's effect is
genuine competition, and how much is cross-structure score miscalibration.

    python analysis/moores_pocket_law/phase10b_within_target_decoys.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import JACCARD_THRESHOLD as T
from dataio import KS

HERE = Path(__file__).resolve().parent
SEED = 0
N_REPEATS = 20
M_GRID = [0, 5, 10, 20, 40]

W = np.asarray(dataio._PLM_RANKER_WEIGHTS)
FEATS = list(dataio._PLM_RANKER_FEATURES)


def score(c) -> float:
    return float(np.dot(W, [c.get(f, 0.0) for f in FEATS]))


def main() -> None:
    folds = dataio.fold_map()
    clusters = {}
    with open(dataio.DATA / dataio.SWEEP[20], encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if dataio.split_of(rec["id"], folds) == "train" and rec.get("clusters"):
                clusters[rec["id"]] = rec["clusters"]

    targets = []
    own_scores, own_wrong = {}, {}
    for s, cl in clusters.items():
        scored = [(score(c), float(c.get("jac", 0.0))) for c in cl]
        if not any(j >= T for _sc, j in scored):
            continue
        wrong = [sc for sc, j in scored if j < T]
        if len(wrong) < 2:          # need a within-target null distribution
            continue
        targets.append(s)
        own_scores[s] = sorted(scored, key=lambda t: -t[0])
        own_wrong[s] = np.array(wrong)

    print("targets usable (qualifying candidate + >=2 wrong candidates): %d\n"
          % len(targets))

    rng = np.random.default_rng(SEED)
    results = {}
    print("=" * 74)
    print("decoys drawn from each target's OWN wrong-answer score distribution")
    print("=" * 74)
    print("%-5s %8s %10s %10s %10s" % ("m", "mean N", "hit@1", "hit@5", "hit@10"))

    for m in M_GRID:
        per = {k: [] for k in KS}
        for _rep in range(N_REPEATS if m else 1):
            hits = {k: [] for k in KS}
            for s in targets:
                merged = list(own_scores[s])
                if m:
                    draw = rng.choice(own_wrong[s], size=m, replace=True)
                    merged = merged + [(float(x), 0.0) for x in draw]
                    merged.sort(key=lambda t: -t[0])
                r = dataio.first_qualifying_rank([j for _sc, j in merged])
                for k in KS:
                    hits[k].append(int(r is not None and r <= k))
            for k in KS:
                per[k].append(float(np.mean(hits[k])))
        row = {"mean_N": float(np.mean([len(own_scores[s]) for s in targets])) + m}
        for k in KS:
            row["hit_at_%d" % k] = float(np.mean(per[k]))
        results[str(m)] = row
        print("%-5d %8.1f %9.1f%% %9.1f%% %9.1f%%"
              % (m, row["mean_N"], 100 * row["hit_at_1"],
                 100 * row["hit_at_5"], 100 * row["hit_at_10"]))

    # Paired, largest injection.
    m = M_GRID[-1]
    rng2 = np.random.default_rng(SEED + 1)
    deltas = {k: [] for k in KS}
    for s in targets:
        base = own_scores[s]
        r0 = dataio.first_qualifying_rank([j for _sc, j in base])
        acc = {k: [] for k in KS}
        for _ in range(N_REPEATS):
            draw = rng2.choice(own_wrong[s], size=m, replace=True)
            merged = base + [(float(x), 0.0) for x in draw]
            merged.sort(key=lambda t: -t[0])
            r1 = dataio.first_qualifying_rank([j for _sc, j in merged])
            for k in KS:
                acc[k].append(int(r1 is not None and r1 <= k))
        for k in KS:
            deltas[k].append(np.mean(acc[k]) - int(r0 is not None and r0 <= k))

    print("\npaired effect of m=%d, within-target decoys" % m)
    print("  %-5s %26s" % ("k", "paired delta (95% CI)"))
    paired = {}
    for k in KS:
        d = np.array(deltas[k])
        rb = np.random.default_rng(SEED)
        bi = rb.integers(0, len(d), size=(20000, len(d)))
        bm = np.sort(d[bi].mean(axis=1))
        lo, hi = float(bm[500]), float(bm[19499])
        paired["hit_at_%d" % k] = {"delta": float(d.mean()), "ci": [lo, hi]}
        star = "  <-- excludes 0" if (lo > 0 or hi < 0) else ""
        print("  k=%-3d %+11.1f [%+6.1f, %+6.1f]%s"
              % (k, 100 * d.mean(), 100 * lo, 100 * hi, star))

    # Decomposition against the foreign-decoy run.
    print("\n" + "=" * 74)
    print("how much of Phase 10 was competition, and how much miscalibration?")
    print("=" * 74)
    foreign = json.loads((HERE / "phase10_decoy.json").read_text())
    print("  %-5s %14s %14s %14s" %
          ("k", "foreign", "within-target", "miscalibration"))
    decomp = {}
    for k in KS:
        f = foreign["paired_max_injection"]["hit_at_%d" % k]["delta"]
        w = paired["hit_at_%d" % k]["delta"]
        decomp["hit_at_%d" % k] = {"foreign": f, "within_target": w,
                                   "attributable_to_miscalibration": f - w}
        print("  k=%-3d %+13.1f %+13.1f %+13.1f"
              % (k, 100 * f, 100 * w, 100 * (f - w)))

    out = {"seed": SEED, "repeats": N_REPEATS, "n_targets": len(targets),
           "by_m": results, "paired_max_injection": paired,
           "decomposition_vs_foreign": decomp}
    (HERE / "phase10b_within_target_decoy.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase10b_within_target_decoy.json"))


if __name__ == "__main__":
    main()
