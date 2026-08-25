"""Load the six-detector, single-platform candidate collection.

dataio.py reads the dumps behind the v1 manuscript: four detectors, no residues,
no centroids, and Lacuna measured on a different machine from the comparators.
This reads what benchmarks/collect_candidates.py produces instead, which differs
in four ways that matter for v2.

Six detectors rather than four, adding MDpocket and separating Lacuna's two
rankers, so the coverage and conversion spread is measured across more of the
architectural range.

Per-candidate residue sets and centroids, which is what makes cross-detector
work possible at all: with only a Jaccard per candidate there is no way to tell
which fpocket pocket is the same site as which P2Rank one.

One platform throughout. Lacuna's published numbers came from Windows and every
comparator from WSL, because that is where each tool runs. Collecting all six in
WSL costs about 1.7 points of Lacuna top-5, well inside its confidence interval,
and buys a set where every row was produced by the same linear algebra.

And MDpocket receives the identical ensemble Lacuna does, generated once and
shared, so the gap between them isolates detection and clustering from
conformational sampling rather than assuming it does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

JACCARD_THRESHOLD = 0.25
KS = (1, 3, 5, 10, 20)

#: Display order: single-structure detectors, then ensemble ones.
METHODS = ("fpocket", "p2rank", "ifsitepred", "mdpocket", "lacuna", "lacuna_plm")

LABEL = {"fpocket": "fpocket", "p2rank": "P2Rank", "ifsitepred": "IF-SitePred",
         "mdpocket": "MDpocket", "lacuna": "Lacuna", "lacuna_plm": "Lacuna-PLM"}

#: Where the collection lands. WSL paths seen from Windows.
DEFAULT_DIRS = [Path(r"\\wsl$\Ubuntu\root\candidates"),
                Path("/root/candidates"),
                HERE / "data" / "candidates"]


def _find(fold: str) -> Path:
    name = "candidates_%s.jsonl" % fold
    for d in DEFAULT_DIRS:
        p = d / name
        try:
            if p.exists():
                return p
        except OSError:
            continue
    raise SystemExit(
        "cannot find %s. Copy it out of WSL, e.g.\n"
        "  wsl cp /root/candidates/%s <repo>/analysis/moores_pocket_law/data/candidates/"
        % (name, name))


def load(fold: str) -> dict:
    """{structure_id: {method: record}} for one fold."""
    out: dict = {}
    with open(_find(fold), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["id"], {})[r["tool"]] = r
    return out


def paired(fold: str, methods=METHODS) -> dict:
    """Only structures every named method ran on, so comparisons are paired."""
    d = load(fold)
    return {s: v for s, v in d.items() if all(m in v for m in methods)}


def jac(rec) -> list:
    return rec["jac_by_rank"]


def covered(rec) -> bool:
    return any(j >= JACCARD_THRESHOLD for j in jac(rec))


def hit(rec, k: int) -> bool:
    return any(j >= JACCARD_THRESHOLD for j in jac(rec)[:k])


def centroids(rec):
    """Per-candidate centroids as an (n, 3) array, NaN where unresolvable."""
    out = []
    for c in rec["centroid_by_rank"]:
        out.append([np.nan] * 3 if c is None else c)
    return np.asarray(out, dtype=float)


def residue_sets(rec) -> list:
    """Per-candidate residue numbers, as sets of int."""
    sets = []
    for labels in rec["residues_by_rank"]:
        s = set()
        for lab in labels:
            digits = "".join(ch for ch in lab.split(":")[0] if ch.isdigit() or ch == "-")
            if digits.lstrip("-").isdigit():
                s.add(int(digits))
        sets.append(s)
    return sets


def boot_ci(values, n_boot=20000, seed=0, alpha=0.05):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = np.sort(rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1))
    return (float(v.mean()), float(m[int((alpha / 2) * n_boot)]),
            float(m[int((1 - alpha / 2) * n_boot) - 1]))


def paired_delta_ci(a, b, n_boot=20000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = np.sort(d[idx].mean(axis=1))
    return float(d.mean()), float(m[int(0.025 * n_boot)]), float(m[int(0.975 * n_boot) - 1])
