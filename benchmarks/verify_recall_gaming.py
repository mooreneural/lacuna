"""Re-derive the claim that ranking on volume alone games the recall metric.

The dumps store Jaccard, not recall, but recall is recoverable exactly. With
A the pocket lining, B the known site, and I their intersection:

    J = I / (|A| + |B| - I)   =>   I = J(|A| + |B|) / (1 + J)
    R = I / |B|

|A| is n_lin in the dump, J is jac, and |B| comes from CryptoBench's own
annotation via the same main_pocket rule the benchmarks use. So the only new
input is the known site size.

The legacy criterion was recall >= 0.30 or a centroid within 4 A. The centroid
term is not recoverable from these dumps, so what this computes is the recall
term alone, which is a lower bound on the legacy number.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\clayt\Documents\GitHub\lacuna")
OLD = Path(
    r"C:\Users\clayt\AppData\Local\Temp\claude"
    r"\C--Users-clayt-Documents-GitHub-lacuna"
    r"\2030379c-f563-47aa-80e1-8e5b229f64df\scratchpad"
)
RECALL_MIN = 0.30
JACCARD_MIN = 0.25
TOP_K = 5

import sys
sys.path.insert(0, str(REPO / "benchmarks"))
from lacuna.pockets.clusterer import _PLM_RANKER_FEATURES, _PLM_RANKER_WEIGHTS  # noqa
from lacuna.pockets.clusterer import _RANKER_FEATURES, _RANKER_WEIGHTS  # noqa

dataset = json.loads((REPO / "benchmarks/cb_data/dataset.json").read_text())


def _resnum(part):
    d = "".join(c for c in part if c.isdigit())
    return None if not d else (-int(d) if part.lstrip().startswith("-") else int(d))


def site_size(sid):
    """|B|, following benchmarks/cryptobench_benchmark.py:main_pocket."""
    assocs = dataset.get(sid[:4].lower())
    if not assocs:
        return None
    main = next((a for a in assocs if a.get("is_main_holo_structure")), None)
    if main is None:
        main = max(assocs, key=lambda a: a.get("pRMSD", 0.0))
    chain = main["apo_chain"]
    res = {n for s in main["apo_pocket_selection"]
           for parts in [s.split("_")]
           if len(parts) >= 2 and parts[0] == chain
           for n in [_resnum(parts[1])] if n is not None}
    return len(res) or None


def recall_of(jac, n_lin, b):
    if b is None or b <= 0 or jac <= 0:
        return 0.0
    inter = jac * (n_lin + b) / (1.0 + jac)
    return min(inter / b, 1.0)


def score(clusters, feats, weights):
    w = np.asarray(weights)
    return [float(np.dot(w, [c.get(f, 0.0) for f in feats])) for c in clusters]


rows = []
for line in (OLD / "test_off.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    cl = r["clusters"]
    if not cl:
        continue
    b = site_size(r["id"])
    if b is None:
        continue
    for c in cl:
        c["_recall"] = recall_of(c.get("jac", 0.0), c.get("n_lin", 0.0), b)
    rows.append((r["id"], cl, b))

print(f"test-fold structures scored: {len(rows)}\n")

ORDERINGS = {
    "volume only (vol, desc)":      lambda cl: [-c.get("vol", 0.0) for c in cl],
    "volume only (vol_max, desc)":  lambda cl: [-c.get("vol_max", 0.0) for c in cl],
    "learned ranker (default)":     lambda cl: [-s for s in score(cl, _RANKER_FEATURES, _RANKER_WEIGHTS)],
    "learned-plm ranker":           lambda cl: [-s for s in score(cl, _PLM_RANKER_FEATURES, _PLM_RANKER_WEIGHTS)],
    "as-detected (dump order)":     lambda cl: list(range(len(cl))),
}

print(f"{'ordering':<28} {'recall>=.30':>12} {'jaccard>=.25':>14}")
print("-" * 56)
results = {}
for name, keyfn in ORDERINGS.items():
    hit_r = hit_j = 0
    for _sid, cl, _b in rows:
        order = np.argsort(keyfn(cl), kind="stable")[:TOP_K]
        top = [cl[i] for i in order]
        if any(c["_recall"] >= RECALL_MIN for c in top):
            hit_r += 1
        if any(c.get("jac", 0.0) >= JACCARD_MIN for c in top):
            hit_j += 1
    n = len(rows)
    results[name] = (hit_r / n, hit_j / n)
    print(f"{name:<28} {100*hit_r/n:>10.1f}% {100*hit_j/n:>13.1f}%")

print()
vr, vj = results["volume only (vol, desc)"]
lr, lj = results["learned ranker (default)"]
print(f"Volume-only ordering scores {100*vr:.1f}% under the recall criterion "
      f"but only {100*vj:.1f}% under Jaccard.")
print(f"The learned ranker scores {100*lr:.1f}% recall / {100*lj:.1f}% Jaccard.")
print(f"\nGap between what recall credits and what Jaccard credits, for a "
      f"ranking that uses nothing but size: {100*(vr-vj):.1f} points.")
