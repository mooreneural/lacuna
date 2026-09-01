"""Does the learned ranker beat sorting the candidates by size?

The manuscript compares the learned ranker against the analytic crypticity rule
it replaced, 55.6% against 17.8%, a factor of three. That comparison is true and
the baseline is weak. The obvious question, raised by a reader and not answered
anywhere in the paper, is what happens against no model at all: take the
candidates Lacuna already produced and order them by size.

Two orderings, both computed from quantities already on every cluster:

  largest_in_input      the biggest cavity in the uploaded structure. The only
                        one obtainable without an ensemble at all.
  largest_mean_volume   the biggest mean volume across the ensemble. Needs the
                        conformers, the per-conformer detection and the
                        cross-conformer clustering, so it is not an alternative
                        to Lacuna, only an alternative ordering of its output.

Result: the answer depends on the budget, and on how much power the fold has.

    python analysis/moores_pocket_law/trivial_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "benchmarks"))

import dataio
from dataio import JACCARD_THRESHOLD as T
from lacuna.pockets.clusterer import (_RANKER_FEATURES, _RANKER_WEIGHTS,
                                      _PLM_RANKER_FEATURES, _PLM_RANKER_WEIGHTS)

F, W = list(_RANKER_FEATURES), np.asarray(_RANKER_WEIGHTS)
FP, WP = list(_PLM_RANKER_FEATURES), np.asarray(_PLM_RANKER_WEIGHTS)
SEED = 0


def load(dump, split):
    folds = dataio.fold_map()
    out = []
    for rec in dataio._jsonl(dump):
        if split and dataio.split_of(rec["id"], folds) != split:
            continue
        cl = rec.get("clusters") or []
        if not cl:
            continue
        out.append({
            "id": rec["id"],
            "fold": folds.get(dataio.pdb_id_of(rec["id"])),
            "jac": np.array([float(c.get("jac", 0.0)) for c in cl]),
            "apo_vol": np.array([float(c.get("apo_vol", 0.0)) for c in cl]),
            "vol": np.array([float(c.get("vol", 0.0)) for c in cl]),
            "X": np.array([[c.get(k, 0.0) for k in F] for c in cl], float),
            "XP": np.array([[c.get(k, 0.0) for k in FP] for c in cl], float)})
    return out


def hits(targets, fn, k):
    return np.array([int((t["jac"][np.argsort(-fn(t), kind="stable")[:k]] >= T).any())
                     for t in targets])


def report(targets, label):
    print("\n%s  (n = %d)" % (label, len(targets)))
    rows = [("largest cavity in input", lambda t: t["apo_vol"]),
            ("largest mean volume",     lambda t: t["vol"]),
            ("learned, 23 features",    lambda t: t["X"] @ W),
            ("learned-plm, 27 features", lambda t: t["XP"] @ WP)]
    got = {}
    print("  %-28s %8s %9s" % ("ranking", "top-1", "top-5"))
    for name, fn in rows:
        got[name] = {k: hits(targets, fn, k) for k in (1, 5)}
        print("  %-28s %7.1f%% %8.1f%%"
              % (name, 100 * got[name][1].mean(), 100 * got[name][5].mean()))

    rng = np.random.default_rng(SEED)
    n = len(targets)
    bi = rng.integers(0, n, size=(20000, n))
    base = "largest mean volume"
    print("  against %s:" % base)
    for name in ("learned, 23 features", "learned-plm, 27 features"):
        for k in (1, 5):
            d = got[name][k] - got[base][k]
            b = np.sort(d[bi].mean(axis=1))
            lo, hi = 100 * b[500], 100 * b[19499]
            flag = "resolved" if (lo > 0 or hi < 0) else "not resolved"
            print("    top-%-2d %-26s %+6.1f  [%+5.1f, %+5.1f]  %s"
                  % (k, name, 100 * d.mean(), lo, hi, flag))
    return got


def main() -> None:
    report(load(dataio.TEST_DUMP, None), "designated test fold")
    report(load(dataio.SWEEP[20], "train"), "pooled train folds")
    print("\nThe train folds have four times the targets and resolve a difference"
          "\nthe test fold cannot. Both estimates are consistent; only one has the"
          "\npower to exclude zero at top-5.")


if __name__ == "__main__":
    main()
