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


def collect(dump_path: Path, limit: int, conformers: int,
            fold_filter: str | None = None) -> None:
    """Run the pipeline and record every cluster's features plus whether it is the
    true site. Resumable: structures already in the dump are skipped."""
    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    if fold_filter:
        ids = [pid for f in fold_filter.split(",") for pid in folds[f.strip()]]
    else:
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


def _matrix(structs: list[dict], feats=None) -> tuple[np.ndarray, np.ndarray]:
    feats = list(feats or _RANKER_FEATURES)
    X, y = [], []
    for s in structs:
        for c in s["clusters"]:
            X.append([c[f] for f in feats])
            y.append(1 if c["jac"] >= JACCARD_THRESHOLD else 0)
    return np.asarray(X, dtype=float), np.asarray(y)


def _train_fold_of(structure_id: str) -> str | None:
    """Which train fold a structure belongs to (None for the test fold)."""
    fold = _fold_lookup().get(structure_id[:-1].lower())
    return None if fold in (None, "test") else fold


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


# ── model variants ─────────────────────────────────────────────────────────────
# Each returns a scorer: a callable mapping a cluster dict to a float, higher is
# better. Keeping them behind one interface lets the cross-validation loop compare
# feature sets, training objectives, and model classes on equal footing.

def _fit_pointwise(train, feats):
    """Logistic regression over individual clusters (the shipped approach).

    Optimizes "is this cluster the site?" independently per cluster, so it is
    dominated by the ~98% negatives and has to learn a globally calibrated
    threshold even though the real decision is within-protein.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = _matrix(train, feats)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(scaler.transform(X), y)
    w = model.coef_[0] / scaler.scale_
    b = float(model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_))
    return lambda c: b + float(np.dot(w, [c[f] for f in feats])), ("linear", w, b)


def _fit_pairwise(train, feats, max_pairs_per_structure: int = 40, seed: int = 0):
    """Linear RankNet: logistic regression on within-structure feature differences.

    The task is to order clusters inside one protein, not to classify them against
    a global threshold. Training on (positive - negative) differences from the same
    structure cancels every structure-level nuisance term (protein size, cluster
    count, overall scale), and targets ordering directly. A model fitted this way
    scores clusters with the same dot product, so nothing downstream changes.
    """
    from sklearn.linear_model import LogisticRegression

    rng = random.Random(seed)
    diffs, labels = [], []
    for s in train:
        pos = [c for c in s["clusters"] if c["jac"] >= JACCARD_THRESHOLD]
        neg = [c for c in s["clusters"] if c["jac"] < JACCARD_THRESHOLD]
        if not pos or not neg:
            continue
        for p in pos:
            picks = neg if len(neg) <= max_pairs_per_structure else rng.sample(
                neg, max_pairs_per_structure)
            pv = np.array([p[f] for f in feats], dtype=float)
            for q in picks:
                qv = np.array([q[f] for f in feats], dtype=float)
                # Both directions keep the problem balanced and the fit unbiased.
                diffs.append(pv - qv); labels.append(1)
                diffs.append(qv - pv); labels.append(0)
    if not diffs:
        raise RuntimeError("no usable positive/negative pairs")
    D = np.asarray(diffs, dtype=float)
    scale = D.std(axis=0)
    scale[scale < 1e-9] = 1.0
    model = LogisticRegression(max_iter=5000, fit_intercept=False)
    model.fit(D / scale, np.asarray(labels))
    w = model.coef_[0] / scale
    return lambda c: float(np.dot(w, [c[f] for f in feats])), ("linear", w, 0.0)


def _fit_gbm(train, feats, seed: int = 0, **kw):
    """Gradient-boosted trees, regularized for this sample size.

    Shallow trees plus subsampling; an earlier unregularized fit on far less data
    overfitted badly (66% train against 27% test), so depth and learning rate are
    deliberately conservative and the honest check is the cross-validated score.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _matrix(train, feats)
    params = dict(max_depth=3, learning_rate=0.06, max_iter=300,
                  min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15,
                  random_state=seed)
    params.update(kw)
    model = HistGradientBoostingClassifier(**params)
    # Class imbalance is ~2% positives; weight them up so the trees do not simply
    # predict "not a pocket" everywhere.
    n_pos = max(int(y.sum()), 1)
    sw = np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * max(len(y) - n_pos, 1)))
    model.fit(X, y, sample_weight=sw)

    def score(c, m=model, f=feats):
        return float(m.predict_proba(np.asarray([[c[k] for k in f]], dtype=float))[0, 1])
    return score, ("gbm", model, None)


def _score_hits(structs, scorer, top_k: int = 5):
    """Whether a top-k cluster under `scorer` is the true site, per structure."""
    out = []
    for s in structs:
        ranked = sorted(s["clusters"], key=scorer, reverse=True)[:top_k]
        out.append(any(c["jac"] >= JACCARD_THRESHOLD for c in ranked))
    return out


def _paired_ci(a, b, n_boot: int = 10000, seed: int = 0):
    """Bootstrap CI on the paired difference in hit rate (a - b)."""
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append((sum(a[i] for i in idx) - sum(b[i] for i in idx)) / n)
    diffs.sort()
    return ((sum(a) - sum(b)) / n,
            diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1])


#: Feature set selected by cross-validation (see cross_validate). The geometry
#: block is what lifted recovery; the ordering objective then added more on top.
#: Deduplicated because the geometry columns became part of _RANKER_FEATURES once
#: they shipped, and repeating a feature would give it a split, meaningless weight.
FIT_FEATURES = list(dict.fromkeys(
    list(_RANKER_FEATURES) + [
        "bur_raw", "depth", "depth_max", "mouth", "elong", "flat", "dcen",
        "centroid_std", "vol_cv",
    ]
))


def fit(dump_path: Path, test_dump: Path | None = None) -> None:
    """Fit the shipped ranker and report it on genuinely held-out structures.

    Uses the configuration cross-validation selected: a linear model trained on
    within-structure pairs, over the geometry-augmented feature set. Pairwise
    training beat pointwise by +2.4 points (CI[+0.7,+4.2]) once geometry was
    available, and gradient boosting was not separable from it, so the shipped
    model stays a readable dot product with no runtime dependency.

    ``test_dump`` supplies test-fold structures when they live in a separate file.
    """
    data = _load(dump_path)
    if test_dump is not None and test_dump.exists():
        data += _load(test_dump)
    for s in data:
        s["split"] = split_of(s["id"])
    train = [s for s in data if s["split"] == "train"]
    test = [s for s in data if s["split"] == "test"]

    have = set(data[0]["clusters"][0])
    feats = [f for f in FIT_FEATURES if f in have]
    missing = [f for f in FIT_FEATURES if f not in have]
    if missing:
        print(f"WARNING: dump lacks {missing}; fitting on {len(feats)} features")

    scorer, (_, weights, _) = _fit_pairwise(train, feats)
    n_clusters = sum(len(s["clusters"]) for s in train)
    n_pos = sum(1 for s in train for c in s["clusters"] if c["jac"] >= JACCARD_THRESHOLD)
    print(f"structures {len(data)} ({len(train)} train-fold / {len(test)} test-fold), "
          f"train clusters {n_clusters}, positives {n_pos}")

    if not test:
        print("no test-fold structures available; skipping held-out report")
    else:
        oracle = sum(1 for s in test
                     if any(c["jac"] >= JACCARD_THRESHOLD for c in s["clusters"])) / len(test)
        current = [any(c["jac"] >= JACCARD_THRESHOLD
                       for c in sorted(s["clusters"], key=lambda c: c["rank"])[:5])
                   for s in test]
        new = _score_hits(test, scorer)
        # Identical fit on shuffled labels: must collapse toward the random null.
        shuffled = [dict(s, clusters=list(s["clusters"])) for s in train]
        rng = random.Random(1)
        for s in shuffled:
            jacs = [c["jac"] for c in s["clusters"]]
            rng.shuffle(jacs)
            s["clusters"] = [dict(c, jac=j) for c, j in zip(s["clusters"], jacs)]
        try:
            ctrl_scorer, _ = _fit_pairwise(shuffled, feats, seed=1)
            ctrl = _score_hits(test, ctrl_scorer)
        except RuntimeError:
            ctrl = [False] * len(test)

        m_cur, lo_cur, hi_cur = _bootstrap_ci(current)
        m_new, lo_new, hi_new = _bootstrap_ci(new)
        m_ctl, lo_ctl, hi_ctl = _bootstrap_ci(ctrl)
        d, dlo, dhi = _paired_ci(new, current)

        print("\nheld-out test-fold top-5 recovery (size-robust Jaccard >= 0.25):")
        print(f"  oracle (any cluster)     {oracle:.1%}")
        print(f"  random top-5 null        {_random_null(test):.1%}")
        print(f"  dump ranking             {m_cur:.1%}  CI[{lo_cur:.1%},{hi_cur:.1%}]")
        print(f"  learned ranker           {m_new:.1%}  CI[{lo_new:.1%},{hi_new:.1%}]")
        print(f"  shuffled-label control   {m_ctl:.1%}  CI[{lo_ctl:.1%},{hi_ctl:.1%}]")
        print(f"  paired difference        {d:+.1%}  CI[{dlo:+.1%},{dhi:+.1%}]")

    print("\npaste into lacuna/pockets/clusterer.py:\n")
    print("_RANKER_FEATURES = (")
    for name in feats:
        print(f'    "{name}",')
    print(")")
    print("_RANKER_WEIGHTS = (")
    for name, value in zip(feats, weights):
        print(f"    {value!r},  # {name}")
    print(")")
    # Ranking is invariant to a constant offset, and the pairwise fit has no
    # intercept, so it is fixed at zero.
    print("_RANKER_INTERCEPT = 0.0")


def cross_validate(dump_path: Path) -> None:
    """Leave-one-fold-out CV across the CryptoBench train folds.

    Model selection must not touch the designated test fold: iterating against it
    would overfit the one number reserved for the final claim. The train folds are
    themselves homology-separated, so rotating over them controls homology while
    pooling ~4x more held-out structures than the test fold alone, which is the
    difference between resolving a 5-point effect and only a 10-point one.
    """
    data = _load(dump_path)
    by_fold: dict[str, list[dict]] = {}
    for s in data:
        f = _train_fold_of(s["id"])
        if f:
            by_fold.setdefault(f, []).append(s)
    folds = sorted(by_fold)
    if len(folds) < 2:
        print(f"need >=2 train folds in the dump, found {folds}")
        return
    n_total = sum(len(v) for v in by_fold.values())
    print(f"cross-validation over {len(folds)} train folds, {n_total} structures")
    print("  " + ", ".join(f"{f}={len(by_fold[f])}" for f in folds))

    base = list(_RANKER_FEATURES)
    extra = ["bur_raw", "depth", "depth_max", "mouth", "elong", "flat", "dcen",
             "centroid_std", "vol_cv"]
    available = set(data[0]["clusters"][0])
    missing = [f for f in extra if f not in available]
    if missing:
        print(f"  note: dump lacks {missing}; re-dump to evaluate them")
        extra = [f for f in extra if f in available]
    full = base + extra

    configs = [
        ("pointwise linear, base feats", lambda tr: _fit_pointwise(tr, base)),
        ("pairwise linear, base feats", lambda tr: _fit_pairwise(tr, base)),
        ("gbm, base feats", lambda tr: _fit_gbm(tr, base)),
    ]
    if extra:  # only meaningful once the dump carries the geometry features
        configs += [
            ("pointwise linear, +geometry", lambda tr: _fit_pointwise(tr, full)),
            ("pairwise linear, +geometry", lambda tr: _fit_pairwise(tr, full)),
            ("gbm, +geometry", lambda tr: _fit_gbm(tr, full)),
            ("gbm, geometry only", lambda tr: _fit_gbm(tr, extra)),
        ]

    # Reference: the currently shipped ranking, read off the dump's stored order.
    pooled_ref, pooled_null, pooled_oracle = [], 0.0, []
    for f in folds:
        held = by_fold[f]
        pooled_ref += [any(c["jac"] >= JACCARD_THRESHOLD
                           for c in sorted(s["clusters"], key=lambda c: c["rank"])[:5])
                       for s in held]
        pooled_null += _random_null(held) * len(held)
        pooled_oracle += [any(c["jac"] >= JACCARD_THRESHOLD for c in s["clusters"])
                          for s in held]
    n = len(pooled_ref)
    print(f"\n  {'oracle (any cluster)':<32} {sum(pooled_oracle)/n:.1%}")
    print(f"  {'random top-5 null':<32} {pooled_null/n:.1%}")
    m, lo, hi = _bootstrap_ci(pooled_ref)
    print(f"  {'dump ranking (as collected)':<32} {m:.1%}  CI[{lo:.1%},{hi:.1%}]")

    print(f"\n  {'config':<32} {'CV recovery':<22} {'vs dump ranking'}")
    results = {}
    for label, fit in configs:
        pooled = []
        try:
            for f in folds:
                train = [s for g in folds if g != f for s in by_fold[g]]
                scorer, _ = fit(train)
                pooled += _score_hits(by_fold[f], scorer)
        except Exception as e:
            print(f"  {label:<32} failed: {type(e).__name__}: {str(e)[:60]}")
            continue
        results[label] = pooled
        m, lo, hi = _bootstrap_ci(pooled)
        d, dlo, dhi = _paired_ci(pooled, pooled_ref)
        flag = "  *" if dlo > 0 else ""
        print(f"  {label:<32} {m:.1%} CI[{lo:.1%},{hi:.1%}]   "
              f"{d:+.1%} CI[{dlo:+.1%},{dhi:+.1%}]{flag}")

    if len(results) > 1:
        best = max(results, key=lambda k: sum(results[k]))
        print(f"\n  best by CV: {best}")
        for label, hits in results.items():
            if label == best:
                continue
            d, dlo, dhi = _paired_ci(results[best], hits)
            sep = "separated" if dlo > 0 else "not separated"
            print(f"    vs {label:<32} {d:+.1%} CI[{dlo:+.1%},{dhi:+.1%}]  {sep}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="ranker_features.jsonl",
                    help="feature dump to write (collect) or read (fit)")
    ap.add_argument("--limit", type=int, default=400, help="structures to collect")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--fit", action="store_true",
                    help="fit and report from an existing dump instead of collecting")
    ap.add_argument("--cv", action="store_true",
                    help="cross-validate model variants over the train folds "
                         "(model selection; leaves the test fold untouched)")
    ap.add_argument("--folds", default=None,
                    help="restrict collection to these folds, e.g. 'test' or "
                         "'train-0,train-1' (default: all, shuffled)")
    ap.add_argument("--test-dump", default=None,
                    help="extra dump holding the test-fold structures, when they "
                         "were collected into a separate file")
    args = ap.parse_args()

    path = Path(args.dump)
    if args.cv:
        cross_validate(path)
    elif args.fit:
        fit(path, Path(args.test_dump) if args.test_dump else None)
    else:
        collect(path, args.limit, args.conformers, args.folds)


if __name__ == "__main__":
    main()
