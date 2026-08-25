"""Three experiments that follow from the label-scarcity diagnosis.

Earlier runs ruled two things out. Model capacity is not the limit: LightGBM is
worse than logistic regression on the same 27 features. Representation is not
the limit either: 1280-dimensional pooled embeddings tie four scalars. What is
left is supervision. A pocket-level ranker sees roughly 1,092 positive pockets,
while the residue head underneath it sees 10,492 positive residues.

So all three experiments here try to buy supervision rather than capacity.

  E1  Refit the residue head per fold, linear against a small MLP. This is the
      one place labels are plentiful, and the shipped head is both fitted on all
      four folds (worth about +0.9 points of leaked conversion) and only a
      single linear layer.

  E2  Train the pocket scorer on all six detectors' candidates instead of
      Lacuna's alone. Same structures, same annotations, roughly six times the
      labelled pockets, because a pocket is a pocket regardless of which tool
      proposed it. Evaluated on Lacuna's candidates either way, so the only
      thing that changes is how much the model was allowed to learn from.

  E3  Normalise features within target before fitting. Ranking is per target,
      but the model is fitted across targets, so a pocket that is large for its
      protein and one that is large in absolute terms look identical to it.

Geometry cannot be combined with embeddings here: the released feature file has
no residue lists and the collection file has no features, and the two runs agree
on only 3.3% of Jaccard vectors so they must not be joined. That combination
needs a re-collection emitting both, which is weekend work.

Protocol: train folds only, leave-one-fold-out over CryptoBench's four
homology-separated train folds. The designated test fold is never read.

    python analysis/moores_pocket_law/ranker_next.py --limit 60   # smoke
    python analysis/moores_pocket_law/ranker_next.py              # full
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
from dataio import JACCARD_THRESHOLD as T
import plm_headroom as PH

warnings.filterwarnings("ignore")

CAND = HERE / "data" / "candidates" / "candidates_train.jsonl"
KS = (1, 5, 10)
SEED = 0


def load_all(limit=None):
    """Every tool's candidates, train folds only, grouped by structure."""
    folds = dataio.fold_map()
    by_id = {}
    with open(CAND, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            fold = folds.get(dataio.pdb_id_of(r["id"]))
            if fold is None or not fold.startswith("train"):
                continue
            if not (r.get("residues_by_rank") and r.get("jac_by_rank")):
                continue
            r["fold"] = fold
            by_id.setdefault(r["id"], []).append(r)
    ids = sorted(by_id)
    if limit:
        ids = ids[:limit]
    return {i: by_id[i] for i in ids}


def conversion(by_target):
    """Conversion among covered targets, from {id: (scores, labels)}."""
    out = {}
    covered = [t for t, (_s, y) in by_target.items() if any(y)]
    for k in KS:
        hit = []
        for t in covered:
            s, y = by_target[t]
            order = np.argsort(-np.asarray(s), kind="stable")[:k]
            hit.append(int(any(y[i] for i in order)))
        out["conv_%d" % k] = float(np.mean(hit)) if hit else float("nan")
    out["n_covered"] = len(covered)
    return out


# ───────────────────────────────── E1: residue head ──────────────────────────

def residue_head(data, arch):
    """Cross-fitted residue scorer; returns {id: per-residue probability}."""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    for sid, recs in data.items():
        f = PH.CACHE / ("%s.npz" % sid)
        if not f.exists():
            continue
        z = np.load(f)
        site = set(int(x) for x in recs[0].get("site_residues") or [])
        nums = z["nums"].astype(int)
        cache[sid] = (z["emb"].astype(np.float32), nums,
                      np.isin(nums, list(site)).astype(np.float32),
                      recs[0]["fold"])

    probs = {}
    for held in sorted({v[3] for v in cache.values()}):
        tr = [k for k, v in cache.items() if v[3] != held]
        te = [k for k, v in cache.items() if v[3] == held]
        if not tr or not te:
            continue
        X = torch.tensor(np.vstack([cache[k][0] for k in tr]), device=dev)
        y = torch.tensor(np.concatenate([cache[k][2] for k in tr]), device=dev)
        pw = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], device=dev)

        torch.manual_seed(SEED)
        net = (nn.Linear(1280, 1) if arch == "linear" else
               nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(256, 1))).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        n = len(X)
        for _ep in range(25):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 8192):
                b = perm[i:i + 8192]
                opt.zero_grad()
                lossf(net(X[b]).squeeze(-1), y[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            for k in te:
                e = torch.tensor(cache[k][0], device=dev)
                p = torch.sigmoid(net(e).squeeze(-1)).cpu().numpy()
                probs[k] = (p, cache[k][1])
    return probs


def pocket_from_residues(data, probs):
    """Score Lacuna candidates by aggregating residue probabilities."""
    by_target = {}
    for sid, recs in data.items():
        if sid not in probs:
            continue
        p, nums = probs[sid]
        pos = {int(n): i for i, n in enumerate(nums)}
        lac = next((r for r in recs if r["tool"] == "lacuna"), None)
        if lac is None:
            continue
        s, y = [], []
        for res, jac in zip(lac["residues_by_rank"], lac["jac_by_rank"]):
            idx = [pos[n] for n in PH._resnums(res, lac["chain"]) if n in pos]
            if not idx:
                continue
            v = np.sort(p[idx])[::-1]
            s.append(0.6 * v[:3].mean() + 0.4 * v.mean())
            y.append(int(float(jac) >= T))
        if s:
            by_target[sid] = (s, y)
    return by_target


# ─────────────────────────── E2/E3: pocket-level pooling ─────────────────────

def pocket_rows(data, tools, within_target=False):
    rows = []
    for sid, recs in data.items():
        f = PH.CACHE / ("%s.npz" % sid)
        if not f.exists():
            continue
        z = np.load(f)
        emb = z["emb"].astype(np.float32)
        pos = {int(n): i for i, n in enumerate(z["nums"])}
        local = []
        for r in recs:
            if r["tool"] not in tools:
                continue
            for res, jac in zip(r["residues_by_rank"], r["jac_by_rank"]):
                idx = [pos[n] for n in PH._resnums(res, r["chain"]) if n in pos]
                if not idx:
                    continue
                local.append({"id": sid, "fold": r["fold"], "tool": r["tool"],
                              "x": emb[idx].mean(0),
                              "y": int(float(jac) >= T)})
        if within_target and local:
            M = np.vstack([r["x"] for r in local])
            M = (M - M.mean(0)) / (M.std(0) + 1e-6)
            for r, v in zip(local, M):
                r["x"] = v
        rows.extend(local)
    return rows


def eval_pocket(rows, train_tools, C=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    by_target = {}
    for held in sorted({r["fold"] for r in rows}):
        tr = [r for r in rows if r["fold"] != held and r["tool"] in train_tools]
        te = [r for r in rows if r["fold"] == held and r["tool"] == "lacuna"]
        if not tr or not te:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=C))
        m.fit(np.vstack([r["x"] for r in tr]), [r["y"] for r in tr])
        p = m.predict_proba(np.vstack([r["x"] for r in te]))[:, 1]
        for r, s in zip(te, p):
            by_target.setdefault(r["id"], [[], []])
            by_target[r["id"]][0].append(s)
            by_target[r["id"]][1].append(r["y"])
    return conversion({k: (v[0], v[1]) for k, v in by_target.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    data = load_all(a.limit)
    print("structures: %d   (%s)\n"
          % (len(data), "SMOKE, limit=%d" % a.limit if a.limit else "full train folds"))

    flat = [r for recs in data.values() for r in recs]
    PH.build_cache(flat)

    out = []
    print("\n--- E1: residue head, refit per fold ---")
    for arch in ("linear", "mlp"):
        r = conversion(pocket_from_residues(data, residue_head(data, arch)))
        r["name"] = "E1 residue head, %s" % arch
        out.append(r)
        print("  %-28s conv@5 %.1f%%" % (arch, 100 * r["conv_5"]))

    print("\n--- E2: pocket labels from one tool vs six ---")
    rows = pocket_rows(data, {"lacuna", "fpocket", "p2rank", "mdpocket",
                              "ifsitepred", "lacuna_plm"})
    npos = sum(r["y"] for r in rows)
    print("  pooled candidates: %d (%d qualifying)" % (len(rows), npos))
    for tools, label in (({"lacuna"}, "trained on lacuna only"),
                         ({"lacuna", "fpocket", "p2rank", "mdpocket",
                           "ifsitepred", "lacuna_plm"}, "trained on all six")):
        r = eval_pocket(rows, tools)
        r["name"] = "E2 %s" % label
        out.append(r)
        print("  %-28s conv@5 %.1f%%" % (label, 100 * r["conv_5"]))

    print("\n--- E3: within-target feature normalisation ---")
    rows_n = pocket_rows(data, {"lacuna"}, within_target=True)
    r = eval_pocket(rows_n, {"lacuna"})
    r["name"] = "E3 within-target normalised"
    out.append(r)
    print("  %-28s conv@5 %.1f%%" % ("normalised", 100 * r["conv_5"]))

    print("\n" + "=" * 62)
    print("%-34s %8s %8s %8s" % ("experiment", "conv@1", "conv@5", "conv@10"))
    print("-" * 62)
    for r in out:
        print("%-34s %7.1f%% %7.1f%% %7.1f%%"
              % (r["name"], 100 * r["conv_1"], 100 * r["conv_5"], 100 * r["conv_10"]))
    print("\ncovered targets: %d" % out[0]["n_covered"])

    tag = "smoke%d" % a.limit if a.limit else "full"
    p = HERE / ("ranker_next_%s.json" % tag)
    p.write_text(json.dumps(out, indent=1))
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
