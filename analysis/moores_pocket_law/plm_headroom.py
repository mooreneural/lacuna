"""Is Lacuna's sequence signal limited by ESM-2, or by how it is compressed?

`ranker_headroom.py` established that no model over the existing 27 features
beats the shipped linear ranker: gradient boosting is 1.9 points worse than
cross-fitted logistic regression, and 13.7 points of top-5 conversion remain
unreachable. It also showed that stripping the four `plm_*` features costs 8.7
points, so almost all the ranker's discriminative power is sequence-derived.

That makes the compression path worth examining. Today a residue's ESM-2
embedding reaches the ranker through two lossy steps:

    emb (1280 dims)  ->  sigmoid(emb @ w + b)  ->  one scalar per residue
    N residue scalars  ->  mean, max, top3, frac  ->  four numbers per pocket

Both are fixed and linear. The question is whether they discard signal that a
ranker could use, which decides where GPU time is worth spending.

The comparison here is deliberately sequence-only, with no geometric features
on either side, so the single manipulated variable is how much of ESM-2 the
model sees:

    (a) the four aggregates production uses
    (b) mean-pooled raw embeddings over the pocket's lining residues
    (c) mean and max pooled, concatenated

Cross-fitting is leave-one-fold-out over CryptoBench's four homology-separated
train folds. The designated test fold is never read.

Note on data: candidate residue lists come from the six-detector collection,
which is a different run from the released feature file. Those two runs agree
on only 3.3% of Jaccard vectors, a consequence of the platform-dependent mode
ordering documented in the manuscript, so they must never be joined. Everything
here comes from the one file that carries residues and labels together.

    python analysis/moores_pocket_law/plm_headroom.py
"""
from __future__ import annotations

import json
import re
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

warnings.filterwarnings("ignore")

CAND = HERE / "data" / "candidates" / "candidates_train.jsonl"
CIF = REPO / "benchmarks" / "cb_data" / "cif"
CACHE = HERE / "data" / "plm_emb"          # ~575 MB; keep out of git
KS = (1, 5, 10)
SEED = 0


# ────────────────────────────── embedding cache ──────────────────────────────

def records():
    """Lacuna's train-fold candidates, with residues and labels together."""
    folds = dataio.fold_map()
    out = []
    with open(CAND, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tool") != "lacuna":
                continue
            fold = folds.get(dataio.pdb_id_of(r["id"]))
            if fold is None or not fold.startswith("train"):
                continue
            if not (r.get("residues_by_rank") and r.get("jac_by_rank")):
                continue
            r["fold"] = fold
            out.append(r)
    return out


def build_cache(recs) -> None:
    """One ESM-2 forward pass per structure, cached to disk. Resumable."""
    todo = [r for r in recs if not (CACHE / ("%s.npz" % r["id"])).exists()]
    print("embedding cache: %d present, %d to compute"
          % (len(recs) - len(todo), len(todo)))
    if not todo:
        return

    import torch
    from transformers import AutoTokenizer, AutoModel
    from lacuna.io.structure import load_structure
    from lacuna.pockets import plm

    CACHE.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("  device: %s" % (torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"))
    tok = AutoTokenizer.from_pretrained(plm.MODEL_NAME)
    model = AutoModel.from_pretrained(plm.MODEL_NAME).eval().to(dev)
    if dev == "cuda":
        model = model.half()

    ok = fail = 0
    for i, r in enumerate(todo, 1):
        path = CIF / ("%s.cif" % r["pdb"].upper())
        try:
            st = load_structure(path, chain=r["chain"])
            seq, nums = plm.sequence_of(st)
            if not seq:
                raise ValueError("empty sequence")
            with torch.no_grad():
                enc = tok(seq, return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                emb = model(**enc).last_hidden_state[0, 1:-1].float().cpu().numpy()
            if len(emb) != len(nums):
                raise ValueError("tokenizer/residue mismatch %d vs %d"
                                 % (len(emb), len(nums)))
            np.savez_compressed(CACHE / ("%s.npz" % r["id"]),
                                emb=emb.astype(np.float16),
                                nums=np.asarray(nums, dtype=np.int32))
            ok += 1
        except Exception as exc:                       # noqa: BLE001
            fail += 1
            if fail <= 5:
                print("  skip %s: %s" % (r["id"], exc))
        if i % 100 == 0 or i == len(todo):
            print("  %4d/%d  ok=%d fail=%d" % (i, len(todo), ok, fail))


# ───────────────────────────── feature construction ──────────────────────────

#: Residue entries are written 'ALA92:B': three-letter name, sequence number,
#: then chain. Parsing them as 'number:chain' yields nothing at all, and because
#: candidates with no resolvable residues are skipped, that failure is silent.
_RES = re.compile(r"^[A-Za-z]*(-?\d+)[A-Za-z]?$")


def _resnums(entries, chain):
    """'ALA92:B' -> 92, keeping only residues on this chain."""
    out = []
    for e in entries:
        s = str(e)
        if ":" in s:
            s, ch = s.rsplit(":", 1)
            if ch != chain:
                continue
        m = _RES.match(s)
        if m:
            out.append(int(m.group(1)))
    return out


def featurize(recs):
    """Per-candidate: the four aggregates, and pooled raw embeddings."""
    w = np.load(REPO / "lacuna" / "pockets" / "plm_head.npz")
    W, B = w["w"].astype(np.float32), float(w["b"])

    rows = []
    miss = 0
    for r in recs:
        f = CACHE / ("%s.npz" % r["id"])
        if not f.exists():
            miss += 1
            continue
        z = np.load(f)
        emb = z["emb"].astype(np.float32)
        pos = {int(n): i for i, n in enumerate(z["nums"])}
        prob = 1.0 / (1.0 + np.exp(-(emb @ W + B)))

        for rank, (res, jac) in enumerate(zip(r["residues_by_rank"],
                                              r["jac_by_rank"])):
            idx = [pos[n] for n in _resnums(res, r["chain"]) if n in pos]
            if not idx:
                continue
            p = np.sort(prob[idx])[::-1]
            agg = np.array([p.mean(), p[0], p[:3].mean(), float((p >= 0.5).mean())],
                           dtype=np.float32)
            E = emb[idx]
            rows.append({"id": r["id"], "fold": r["fold"], "rank": rank,
                         "y": int(float(jac) >= T),
                         "agg": agg, "mean": E.mean(0), "max": E.max(0)})
    if miss:
        print("  (%d structures had no cached embedding)" % miss)
    return rows


# ──────────────────────────────── evaluation ─────────────────────────────────

def evaluate(rows, key_fn, name, C=1.0):
    """Cross-fitted logistic ranking; conversion among covered targets."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([key_fn(r) for r in rows])
    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    tid = np.array([r["id"] for r in rows])
    score = np.zeros(len(rows), dtype=np.float64)

    for held in sorted(set(fold)):
        tr, te = fold != held, fold == held
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        score[te] = m.predict_proba(X[te])[:, 1]

    res = {"name": name, "dims": X.shape[1]}
    by = {}
    for t, s, yy in zip(tid, score, y):
        by.setdefault(t, [[], []])
        by[t][0].append(s)
        by[t][1].append(yy)
    covered = [t for t, (_s, ys) in by.items() if any(ys)]
    for k in KS:
        conv = []
        for t in covered:
            s, ys = by[t]
            order = np.argsort(-np.asarray(s), kind="stable")[:k]
            conv.append(int(any(ys[i] for i in order)))
        res["conv_%d" % k] = float(np.mean(conv))
    res["n_targets"] = len(by)
    res["n_covered"] = len(covered)
    return res


def main() -> None:
    recs = records()
    print("train-fold structures with residues+labels: %d\n" % len(recs))
    build_cache(recs)

    print("\nbuilding candidate features")
    rows = featurize(recs)
    print("  %d candidates, %d qualifying" % (len(rows), sum(r["y"] for r in rows)))

    variants = [
        (lambda r: r["agg"], "four aggregates (production path)", 1.0),
        (lambda r: r["mean"], "mean-pooled raw embeddings", 0.05),
        (lambda r: np.concatenate([r["mean"], r["max"]]), "mean + max pooled", 0.05),
        (lambda r: np.concatenate([r["agg"], r["mean"]]), "aggregates + mean-pooled", 0.05),
    ]
    out = [evaluate(rows, f, n, C) for f, n, C in variants]

    print("\nsequence-only ranking, cross-fitted over 4 homology-separated folds")
    print("n_targets=%d  covered=%d\n" % (out[0]["n_targets"], out[0]["n_covered"]))
    print("%-36s %6s %8s %8s %8s" % ("representation", "dims", "conv@1", "conv@5", "conv@10"))
    print("-" * 70)
    for r in out:
        print("%-36s %6d %7.1f%% %7.1f%% %7.1f%%"
              % (r["name"], r["dims"], 100 * r["conv_1"],
                 100 * r["conv_5"], 100 * r["conv_10"]))

    base, best = out[0], max(out[1:], key=lambda r: r["conv_5"])
    d = 100 * (best["conv_5"] - base["conv_5"])
    print("\n" + "=" * 70)
    print("best pooled representation beats the four aggregates by %+.1f points "
          "at top-5" % d)
    if d >= 2.0:
        print("\nVERDICT: the 1280->1 linear head is discarding usable signal.")
        print("Per-residue embeddings are the right GPU spend; scaling to ESM-2 3B")
        print("on the shared box is justified.")
    elif d <= 0.5:
        print("\nVERDICT: the four aggregates are not the bottleneck. ESM-2 650M's")
        print("representation is already exhausted for this task, so a larger PLM")
        print("is unlikely to pay. Spend the weekend on a second benchmark instead.")
    else:
        print("\nVERDICT: marginal. Not enough to justify a larger PLM on its own.")

    (HERE / "plm_headroom.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "plm_headroom.json"))


if __name__ == "__main__":
    main()
