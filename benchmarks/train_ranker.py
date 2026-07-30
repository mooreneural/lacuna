# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Fit and validate the learned pocket ranker used by ``rank_by="learned"``.

Why a learned ranker exists
---------------------------
Measured on shuffled CryptoBench structures (NMA, 20 conformers, top-5,
size-robust Jaccard >= 0.25), recovery is lost at the ranking stage, not at
sampling or detection:

    best over top-5 ranked clusters   12.7%   <- what Lacuna reported
    best over ANY cluster (oracle)    75.3%
    best over ANY raw pocket          81.3%
    picking 5 clusters at random      13.0%

There is a median of 64 clusters per structure and typically exactly 1 of them is
correct, so the analytic scores were performing at chance. This script fits a
linear model to tell the correct cluster from the rest.

Protocol
--------
The split follows CryptoBench's OWN folds: fitted on train-0..train-3, reported
on the designated test fold. Those folds exist to separate homologous proteins,
so a naive random or hash split can place homologs on both sides and inflate the
held-out estimate. Every reported number carries a target-level bootstrap CI, and
a shuffled-label control is fitted identically to confirm the gain is signal
rather than an artifact of the evaluation.

Standardization is folded into the exported weights so Lacuna scores pockets with
a plain dot product and needs no scikit-learn at runtime; scikit-learn is required
only to run this script.

Usage
-----
    python benchmarks/train_ranker.py --dump features.jsonl --limit 400   # collect
    python benchmarks/train_ranker.py --dump features.jsonl --fit         # fit + report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import run_lacuna  # noqa: E402
from cryptobench_benchmark import (  # noqa: E402
    _fetch, download_cif, main_pocket, MAX_RESIDUES,
)
from compare_fpocket import residue_jaccard  # noqa: E402
from lacuna.io.structure import load_structure  # noqa: E402
from lacuna.pockets.clusterer import _RANKER_FEATURES, ranker_features  # noqa: E402

JACCARD_THRESHOLD = 0.25

_FOLD_OF: dict[str, str] = {}


def _fold_lookup() -> dict[str, str]:
    """Map lowercase PDB id -> CryptoBench fold name."""
    global _FOLD_OF
    if not _FOLD_OF:
        folds = json.loads(_fetch("folds.json").read_text())
        _FOLD_OF = {pid.lower(): name for name, ids in folds.items() for pid in ids}
    return _FOLD_OF


def split_of(structure_id: str) -> str:
    """Train/test assignment from CryptoBench's own folds.

    The dataset's folds separate homologous proteins; splitting on them keeps
    homologs out of the held-out set, which a random or hash split would not.
    ``structure_id`` is ``<pdb><chain>``.
    """
    fold = _fold_lookup().get(structure_id[:-1].lower())
    if fold is None:
        return "unmapped"
    return "test" if fold == "test" else "train"


def collect(dump_path: Path, limit: int, conformers: int) -> None:
    """Run the pipeline and record every cluster's features plus whether it is the
    true site. Resumable: structures already in the dump are skipped."""
    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    ids = [pid for f in folds.values() for pid in f]
    random.Random(0).shuffle(ids)

    done = set()
    if dump_path.exists():
        for line in dump_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    n = len(done)
    t0 = time.perf_counter()

    with open(dump_path, "a", encoding="utf-8") as out:
        for apo in ids:
            if n >= limit:
                break
            assocs = dataset.get(apo)
            if not assocs:
                continue
            chain, known = main_pocket(assocs)
            if not known:
                continue
            sid = f"{apo}{chain}"
            if sid in done:
                continue
            try:
                cif = download_cif(apo)
                s = load_structure(cif, chain=chain)
                if not (10 <= len(s.residues) <= MAX_RESIDUES):
                    continue
                clusters, _ = run_lacuna(cif, conformers, chain=chain,
                                         backend_name="nma", rank_by="crypticity")
            except Exception:
                continue
            if not clusters:
                continue
            out.write(json.dumps({
                "id": sid, "split": split_of(sid),
                "clusters": [
                    {**ranker_features(c), "rank": c.rank,
                     "jac": round(residue_jaccard(c.lining_residues, known), 3)}
                    for c in clusters
                ],
            }) + "\n")
            out.flush()
            n += 1
            if n % 25 == 0:
                print(f"  [{n}/{limit}] ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    print(f"collected {n} structures -> {dump_path}")


def _load(dump_path: Path) -> list[dict]:
    rows = []
    for line in dump_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["clusters"]:
                rows.append(r)
    return rows


def _matrix(structs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for s in structs:
        for c in s["clusters"]:
            X.append([c[f] for f in _RANKER_FEATURES])
            y.append(1 if c["jac"] >= JACCARD_THRESHOLD else 0)
    return np.asarray(X, dtype=float), np.asarray(y)


def _bootstrap_ci(hits, n_boot: int = 5000, seed: int = 0):
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(hits) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def _top5_hits(structs, weights, intercept):
    """Whether a top-5 cluster under this linear score is the true site."""
    out = []
    for s in structs:
        scored = sorted(
            s["clusters"],
            key=lambda c: intercept + float(np.dot(
                weights, [c[f] for f in _RANKER_FEATURES])),
            reverse=True,
        )
        out.append(any(c["jac"] >= JACCARD_THRESHOLD for c in scored[:5]))
    return out


def _random_null(structs) -> float:
    """Expected top-5 recovery from picking 5 clusters uniformly at random."""
    total = 0.0
    for s in structs:
        n = len(s["clusters"])
        k = sum(1 for c in s["clusters"] if c["jac"] >= JACCARD_THRESHOLD)
        if k:
            total += 1.0 - (comb(n - k, 5) / comb(n, 5) if n - k >= 5 else 0.0)
    return total / max(len(structs), 1)


def fit(dump_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = _load(dump_path)
    # Re-derive the split from the fold table rather than trusting the dump, so an
    # older dump written under a different split rule is still evaluated correctly.
    for s in data:
        s["split"] = split_of(s["id"])
    train = [s for s in data if s["split"] == "train"]
    test = [s for s in data if s["split"] == "test"]
    Xtr, ytr = _matrix(train)

    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(scaler.transform(Xtr), ytr)

    # Fold standardization into the weights: score = intercept + w . x on raw features.
    weights = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_))

    print(f"structures {len(data)} ({len(train)} train / {len(test)} test), "
          f"train clusters {len(ytr)}, positives {int(ytr.sum())}")

    oracle = sum(1 for s in test
                 if any(c["jac"] >= JACCARD_THRESHOLD for c in s["clusters"])) / len(test)
    current = [any(c["jac"] >= JACCARD_THRESHOLD
                   for c in sorted(s["clusters"], key=lambda c: c["rank"])[:5])
               for s in test]
    m_cur, lo_cur, hi_cur = _bootstrap_ci(current)
    m_new, lo_new, hi_new = _bootstrap_ci(_top5_hits(test, weights, intercept))

    # Identical fit on shuffled labels: must collapse toward the random null.
    y_shuffled = np.random.RandomState(0).permutation(ytr)
    control = LogisticRegression(max_iter=5000, class_weight="balanced")
    control.fit(scaler.transform(Xtr), y_shuffled)
    w_ctrl = control.coef_[0] / scaler.scale_
    b_ctrl = float(control.intercept_[0] - np.sum(control.coef_[0] * scaler.mean_ / scaler.scale_))
    m_ctl, lo_ctl, hi_ctl = _bootstrap_ci(_top5_hits(test, w_ctrl, b_ctrl))

    print("\nheld-out top-5 recovery (size-robust Jaccard >= 0.25):")
    print(f"  oracle (any cluster)     {oracle:.1%}")
    print(f"  random top-5 null        {_random_null(test):.1%}")
    print(f"  crypticity (current)     {m_cur:.1%}  CI[{lo_cur:.1%},{hi_cur:.1%}]")
    print(f"  learned ranker           {m_new:.1%}  CI[{lo_new:.1%},{hi_new:.1%}]")
    print(f"  shuffled-label control   {m_ctl:.1%}  CI[{lo_ctl:.1%},{hi_ctl:.1%}]")

    print("\npaste into lacuna/pockets/clusterer.py:\n")
    print("_RANKER_WEIGHTS = (")
    for name, value in zip(_RANKER_FEATURES, weights):
        print(f"    {value!r},  # {name}")
    print(")")
    print(f"_RANKER_INTERCEPT = {intercept!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="ranker_features.jsonl",
                    help="feature dump to write (collect) or read (fit)")
    ap.add_argument("--limit", type=int, default=400, help="structures to collect")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--fit", action="store_true",
                    help="fit and report from an existing dump instead of collecting")
    args = ap.parse_args()

    path = Path(args.dump)
    if args.fit:
        fit(path)
    else:
        collect(path, args.limit, args.conformers)


if __name__ == "__main__":
    main()
