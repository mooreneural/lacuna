"""Phase 10: burden manipulation that holds candidate quality fixed.

The conformer sweep raises burden by sampling more conformations, so it changes
what the candidates are as well as how many. This isolates the burden.

For each target that already has a qualifying candidate, its own clusters are
left untouched and m extra clusters are injected, drawn from the non-qualifying
clusters of *other* targets. Everything is then scored by the production
PLM-assisted ranker and re-ranked. The true site is unchanged, the target's own
candidates are unchanged, the ranker is unchanged. Only N moves.

Under candidate competition, hit@k falls with m. Under no competition it does
not move, because the injected clusters are by construction wrong answers.

The obvious way for this test to be toothless is for injected decoys to score so
low that they never reach the top of the list, so the diagnostics report where
they actually land. A decoy pool that never enters the top five would make a null
result uninformative rather than evidence of absence.

Train fold only. Seeded.

    python analysis/moores_pocket_law/phase10_decoy_injection.py
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


def score(cluster) -> float:
    return float(np.dot(W, [cluster.get(f, 0.0) for f in FEATS]))


def load_train_clusters():
    """{target: [cluster dicts]} for train-fold targets, 20-conformer run."""
    folds = dataio.fold_map()
    out = {}
    with open(dataio.DATA / dataio.SWEEP[20], encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if dataio.split_of(rec["id"], folds) != "train":
                continue
            cl = rec.get("clusters") or []
            if cl:
                out[rec["id"]] = cl
    return out


def main() -> None:
    clusters = load_train_clusters()

    # Decoy pool: every non-qualifying cluster, tagged with its home target so a
    # target is never handed its own pockets back as distractors.
    pool = [(sid, c) for sid, cl in clusters.items()
            for c in cl if c.get("jac", 0.0) < T]
    pool_scores = np.array([score(c) for _s, c in pool])
    print("decoy pool: %d non-qualifying clusters from %d targets"
          % (len(pool), len(clusters)))

    targets = [s for s, cl in clusters.items()
               if any(c.get("jac", 0.0) >= T for c in cl)]
    print("targets with a qualifying candidate: %d\n" % len(targets))

    base = {}
    for s in targets:
        cl = clusters[s]
        base[s] = sorted(((score(c), float(c.get("jac", 0.0))) for c in cl),
                         key=lambda t: -t[0])

    rng = np.random.default_rng(SEED)
    results = {}
    diag = {}

    print("=" * 76)
    print("injecting m real-but-wrong candidates into a fixed candidate set")
    print("=" * 76)
    print("%-5s %8s %10s %10s %10s %12s" %
          ("m", "mean N", "hit@1", "hit@5", "hit@10", "decoys in top5"))

    for m in M_GRID:
        per_repeat = {k: [] for k in KS}
        decoy_top5, decoy_rank = [], []
        for rep in range(N_REPEATS if m else 1):
            hits = {k: [] for k in KS}
            for s in targets:
                own = base[s]
                if m:
                    idx = rng.integers(0, len(pool), size=m * 2)
                    picks, seen = [], 0
                    for i in idx:
                        if pool[i][0] != s:
                            picks.append(pool_scores[i])
                            seen += 1
                        if seen == m:
                            break
                    merged = [(sc, jc, 0) for sc, jc in own] + \
                             [(sc, 0.0, 1) for sc in picks]
                else:
                    merged = [(sc, jc, 0) for sc, jc in own]
                merged.sort(key=lambda t: -t[0])
                jacs = [t[1] for t in merged]
                r = dataio.first_qualifying_rank(jacs)
                for k in KS:
                    hits[k].append(int(r is not None and r <= k))
                if m and rep == 0:
                    flags = [t[2] for t in merged]
                    decoy_top5.append(sum(flags[:5]))
                    pos = [i + 1 for i, f in enumerate(flags) if f]
                    if pos:
                        decoy_rank.append(float(np.median(pos)))
            for k in KS:
                per_repeat[k].append(float(np.mean(hits[k])))

        mean_N = float(np.mean([len(base[s]) for s in targets])) + m
        row = {"mean_N": mean_N, "n_targets": len(targets),
               "repeats": N_REPEATS if m else 1}
        for k in KS:
            v = np.array(per_repeat[k])
            row["hit_at_%d" % k] = float(v.mean())
            row["hit_at_%d_sd" % k] = float(v.std())
        results[str(m)] = row
        d5 = float(np.mean(decoy_top5)) if decoy_top5 else 0.0
        diag[str(m)] = {"mean_decoys_in_top5": d5,
                        "median_decoy_rank": (float(np.median(decoy_rank))
                                              if decoy_rank else None)}
        print("%-5d %8.1f %9.1f%% %9.1f%% %9.1f%% %11.2f"
              % (m, mean_N, 100 * row["hit_at_1"], 100 * row["hit_at_5"],
                 100 * row["hit_at_10"], d5))

    # Paired effect of the largest injection, target by target.
    print("\n" + "=" * 76)
    print("paired effect of m=%d against m=0, same targets" % M_GRID[-1])
    print("=" * 76)
    m = M_GRID[-1]
    rng2 = np.random.default_rng(SEED + 1)
    deltas = {k: [] for k in KS}
    for s in targets:
        own = base[s]
        r0 = dataio.first_qualifying_rank([jc for _sc, jc in own])
        acc = {k: [] for k in KS}
        for _ in range(N_REPEATS):
            idx = rng2.integers(0, len(pool), size=m * 2)
            picks, seen = [], 0
            for i in idx:
                if pool[i][0] != s:
                    picks.append(pool_scores[i]); seen += 1
                if seen == m:
                    break
            merged = [(sc, jc) for sc, jc in own] + [(sc, 0.0) for sc in picks]
            merged.sort(key=lambda t: -t[0])
            r1 = dataio.first_qualifying_rank([t[1] for t in merged])
            for k in KS:
                acc[k].append(int(r1 is not None and r1 <= k))
        for k in KS:
            deltas[k].append(np.mean(acc[k]) - int(r0 is not None and r0 <= k))

    print("  %-5s %26s" % ("k", "paired delta (95% CI)"))
    paired = {}
    for k in KS:
        d = np.array(deltas[k])
        rngb = np.random.default_rng(SEED)
        bi = rngb.integers(0, len(d), size=(20000, len(d)))
        bm = np.sort(d[bi].mean(axis=1))
        lo, hi = float(bm[500]), float(bm[19499])
        paired["hit_at_%d" % k] = {"delta": float(d.mean()), "ci": [lo, hi]}
        star = "  <-- excludes 0" if (lo > 0 or hi < 0) else ""
        print("  k=%-3d %+11.1f [%+6.1f, %+6.1f]%s"
              % (k, 100 * d.mean(), 100 * lo, 100 * hi, star))

    out = {"seed": SEED, "repeats": N_REPEATS, "m_grid": M_GRID,
           "n_targets": len(targets), "pool_size": len(pool),
           "by_m": results, "decoy_diagnostics": diag,
           "paired_max_injection": paired}
    (HERE / "phase10_decoy.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase10_decoy.json"))


if __name__ == "__main__":
    main()
