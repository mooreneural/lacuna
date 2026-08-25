"""Phase 15: the injection experiment again, on the designated test fold.

Phase 10b established the causal result: adding competing entries to a ranked
list, drawn so that they carry no information the target did not already
produce, pushes findable sites out of a fixed budget. It ran on the pooled train
folds, for sample size.

That leaves one opening. The PLM-assisted ranker was fitted on CryptoBench
training data, so the strongest causal claim in the paper was measured on
targets that helped fit the scorer. The intervention is still controlled, and
the likely direction of the bias is conservative: a ranker that overfits its
training targets separates the true site from the wrong-candidate distribution
more cleanly there, which makes the site harder to displace, not easier. But
that is an argument, and a reviewer is not obliged to accept an argument.

This runs the identical procedure on the 179-target designated test fold, which
the ranker never saw. Fewer targets qualify, so the intervals are wider. The
question is only whether the effect survives on held-out data.

Nothing here replaces Phase 10b. Train is the high-power estimate; this is the
validation.

    python analysis/moores_pocket_law/phase15_heldout_injection.py
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


def cohort(dump: Path, want_split: str):
    """Eligible targets, identically to Phase 10b but on a chosen split."""
    folds = dataio.fold_map()
    own_scores, own_wrong = {}, {}
    n_seen = n_covered = 0
    for rec in dataio._jsonl(dump):
        if dataio.split_of(rec["id"], folds) != want_split:
            continue
        cl = rec.get("clusters") or []
        n_seen += 1
        scored = [(score(c), float(c.get("jac", 0.0))) for c in cl]
        if not any(j >= T for _s, j in scored):
            continue
        n_covered += 1
        wrong = [s for s, j in scored if j < T]
        if len(wrong) < 2:            # need a within-target null to draw from
            continue
        own_scores[rec["id"]] = sorted(scored, key=lambda t: -t[0])
        own_wrong[rec["id"]] = np.array(wrong)
    return own_scores, own_wrong, n_seen, n_covered


def run(own_scores, own_wrong, label):
    targets = sorted(own_scores)
    rng = np.random.default_rng(SEED)
    by_m = {}
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
        by_m[str(m)] = row
        print("%-5d %8.1f %9.1f%% %9.1f%% %9.1f%%"
              % (m, row["mean_N"], 100 * row["hit_at_1"],
                 100 * row["hit_at_5"], 100 * row["hit_at_10"]))

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

    print("\npaired effect of m=%d (%s)" % (m, label))
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
    return by_m, paired, targets


def main() -> None:
    print("=" * 74)
    print("held-out replication: designated test fold, ranker never saw these")
    print("=" * 74)
    os_, ow, seen, cov = cohort(dataio.TEST_DUMP, "test")
    print("test structures in dump          : %d" % seen)
    print("  with a qualifying candidate    : %d" % cov)
    print("  and >=2 non-qualifying (usable): %d\n" % len(os_))
    by_m, paired, targets = run(os_, ow, "test fold")

    train = json.loads((HERE / "phase10b_within_target_decoy.json").read_text())
    print("\n" + "=" * 74)
    print("train (n=%d) against held-out test (n=%d)"
          % (train["n_targets"], len(targets)))
    print("=" * 74)
    print("  %-5s %24s %26s" % ("k", "train delta [95% CI]", "test delta [95% CI]"))
    for k in KS:
        a = train["paired_max_injection"]["hit_at_%d" % k]
        b = paired["hit_at_%d" % k]
        print("  k=%-3d %+7.1f [%+6.1f, %+6.1f]   %+8.1f [%+6.1f, %+6.1f]"
              % (k, 100 * a["delta"], 100 * a["ci"][0], 100 * a["ci"][1],
                 100 * b["delta"], 100 * b["ci"][0], 100 * b["ci"][1]))

    out = {"seed": SEED, "repeats": N_REPEATS, "split": "test",
           "n_structures": seen, "n_covered": cov, "n_targets": len(targets),
           "by_m": by_m, "paired_max_injection": paired,
           "train_comparison": train["paired_max_injection"]}
    (HERE / "phase15_heldout_injection.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase15_heldout_injection.json"))


if __name__ == "__main__":
    main()
