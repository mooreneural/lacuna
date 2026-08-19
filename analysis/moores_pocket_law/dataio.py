"""Load per-candidate detector output into one tidy target x method table.

Read-only with respect to everything in benchmarks/ and paper/. Nothing here
refits a ranker or regenerates a benchmark; it reads stored candidate lists and
recomputes only the accounting.

Two things this module is deliberately strict about.

The fpocket rows inside detectors_cb_paper_baselines.jsonl predate the
rank-ordering fix and understate its top-5 by about fifteen points. They are
always overridden from detectors_cb_paper_fpocket_fixed.jsonl, and a partial
override raises rather than silently mixing corrected and uncorrected rankings.

The hit criterion is the paper's, Jaccard >= 0.25 with no centroid clause, taken
from paper/make_data.py:_hit. The repository's benchmark scripts use a wider
"size-robust" rule that also accepts a centroid within 4 A, but no per-candidate
centroid distance is stored in any artifact, so that rule cannot be evaluated
here and no column pretends otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data"

sys.path.insert(0, str(REPO / "benchmarks"))
from lacuna.pockets.clusterer import (  # noqa: E402
    _PLM_RANKER_FEATURES, _PLM_RANKER_WEIGHTS,
)

JACCARD_THRESHOLD = 0.25
KS = (1, 3, 5, 10, 20)
METHODS = ("fpocket", "p2rank", "ifsitepred", "lacuna")

#: The 20-conformer level is the run the paper reports, and it now ships with
#: the paper itself, so it is read from there rather than duplicated here. The
#: values are absolute, and `DATA / value` still resolves correctly because
#: pathlib returns the right-hand side when it is absolute.
PAPER_DATA = REPO / "paper" / "data"
SWEEP = {1: DATA / "conf_1.jsonl",
         3: DATA / "conf_3.jsonl",
         5: DATA / "conf_5.jsonl",
         10: DATA / "conf_10.jsonl",
         20: PAPER_DATA / "lacuna_candidates_train.jsonl"}
TEST_DUMP = PAPER_DATA / "lacuna_candidates_test.jsonl"


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def pdb_id_of(structure_id: str) -> str:
    """PDB id from a '<pdb><chain>' structure id.

    Chains are not always one character, so index on the four-character PDB id,
    which is what benchmarks/train_ranker.py does.
    """
    return structure_id[:4].lower()


def fold_map() -> dict:
    folds = json.loads((REPO / "benchmarks/cb_data/folds.json").read_text())
    return {p.lower(): name for name, ids in folds.items() for p in ids}


def split_of(structure_id: str, folds: dict):
    fold = folds.get(pdb_id_of(structure_id))
    if fold is None:
        return None
    return "test" if fold == "test" else "train"


def lacuna_ranked(dump: Path) -> dict:
    """Lacuna's candidate list, ordered by the PLM-assisted ranker.

    Reproduces paper/make_data.py exactly. Lacuna stores unordered clusters plus
    features; the published ordering is produced at analysis time by this dot
    product, so it has to be redone here rather than read off the dump.
    """
    w = np.asarray(_PLM_RANKER_WEIGHTS)
    feats = list(_PLM_RANKER_FEATURES)
    out = {}
    for rec in _jsonl(dump):
        clusters = rec.get("clusters") or []
        if not clusters:
            continue
        ranked = sorted(
            clusters,
            key=lambda c: float(np.dot(w, [c.get(f, 0.0) for f in feats])),
            reverse=True)
        out[rec["id"]] = [float(c["jac"]) for c in ranked]
    return out


def load_candidates() -> dict:
    """{structure_id: {method: [jaccard by rank]}} for all four methods."""
    out = {}

    for r in _jsonl(REPO / "benchmarks/detectors_cb_paper_baselines.jsonl"):
        out.setdefault(r["id"], {})[r["tool"]] = list(r["jac_by_rank"])

    for r in _jsonl(REPO / "benchmarks/detectors_cb_paper_ifsitepred.jsonl"):
        if r["id"] in out:
            out[r["id"]]["ifsitepred"] = list(r["jac_by_rank"])

    n_fixed = 0
    for r in _jsonl(REPO / "benchmarks/detectors_cb_paper_fpocket_fixed.jsonl"):
        if r["id"] in out:
            out[r["id"]]["fpocket"] = list(r["jac_by_rank"])
            n_fixed += 1
    n_fp = sum(1 for d in out.values() if "fpocket" in d)
    if n_fixed < n_fp:
        raise SystemExit(
            "fpocket override covers %d of %d structures. Refusing to mix "
            "corrected and uncorrected rankings in one column." % (n_fixed, n_fp))

    for sid, jacs in lacuna_ranked(SWEEP[20]).items():
        out.setdefault(sid, {})["lacuna"] = jacs
    for sid, jacs in lacuna_ranked(TEST_DUMP).items():
        out.setdefault(sid, {})["lacuna"] = jacs

    return out


def site_sizes() -> dict:
    """|known site| per structure, from whichever dump carries n_known."""
    out = {}
    for name in ("detectors_cb_paper_baselines.jsonl",
                 "detectors_cb_paper_ifsitepred.jsonl"):
        for r in _jsonl(REPO / "benchmarks" / name):
            if r.get("n_known"):
                out.setdefault(r["id"], int(r["n_known"]))
    return out


def protein_lengths() -> dict:
    """Resolved residue count of the scored chain, from the benchmark CIFs.

    Cached, because parsing ~900 mmCIF files takes far longer than the rest of
    the analysis and the answer never changes.
    """
    cache = DATA / "protein_lengths.json"
    if cache.exists():
        return {k: int(v) for k, v in json.loads(cache.read_text()).items()}
    import gemmi
    lengths = {}
    cif_dir = REPO / "benchmarks/cb_data/cif"
    for sid in sorted(load_candidates()):
        path = cif_dir / ("%s.cif" % pdb_id_of(sid).upper())
        if not path.exists():
            continue
        try:
            st = gemmi.read_structure(str(path))
            st.setup_entities()
            st.remove_ligands_and_waters()
        except Exception:
            continue
        ch = next((c for c in st[0] if c.name == sid[4:]), None)
        if ch is None:
            continue
        n = 0
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info is not None and info.is_amino_acid():
                n += 1
        if n:
            lengths[sid] = n
    cache.write_text(json.dumps(lengths, indent=0))
    return lengths


def first_qualifying_rank(jacs):
    """1-based rank of the first qualifying candidate, or None."""
    for i, j in enumerate(jacs, start=1):
        if j >= JACCARD_THRESHOLD:
            return i
    return None


def build_table() -> list:
    """One row per target x method, paired across all four methods."""
    cands = load_candidates()
    folds = fold_map()
    sizes = site_sizes()
    lengths = protein_lengths()

    paired = {sid: d for sid, d in cands.items()
              if all(m in d for m in METHODS)}
    rows = []
    for sid in sorted(paired):
        per_method = paired[sid]
        split = split_of(sid, folds)
        for method in METHODS:
            jacs = per_method[method]
            rank = first_qualifying_rank(jacs)
            n_qual = sum(1 for j in jacs if j >= JACCARD_THRESHOLD)
            row = {
                "target_id": sid,
                "pdb": pdb_id_of(sid),
                "chain": sid[4:],
                "method": method,
                "split": split,
                "protein_length": lengths.get(sid),
                "site_size": sizes.get(sid),
                "total_candidate_count": len(jacs),
                "qualifying_candidate_exists": int(rank is not None),
                "best_qualifying_rank": rank,
                "number_of_qualifying_candidates": n_qual,
                "best_jaccard": max(jacs) if jacs else 0.0,
                "best_centroid_distance": None,   # not stored in any artifact
            }
            for k in KS:
                row["hit_at_%d" % k] = int(rank is not None and rank <= k)
            row["candidate_density"] = (
                len(jacs) / lengths[sid] if lengths.get(sid) else None)
            rows.append(row)
    return rows


def boot_ci(values, n_boot=20000, seed=0, alpha=0.05):
    """Percentile bootstrap over targets. Seeded, so reruns agree exactly."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.sort(rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return float(v.mean()), float(lo), float(hi)
