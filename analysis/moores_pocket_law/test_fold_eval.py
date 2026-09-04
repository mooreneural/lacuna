"""The one-shot test-fold measurement. Read this before running it again.

Everything else in this directory runs on the training folds and can be
iterated on freely. This cannot. CryptoBench's test fold is the only held-out
estimate this project has, and it is worth exactly one measurement per claim.
Running it, changing something, and running it again turns it into a validation
set and the number stops meaning anything.

Two arms, both through the real pipeline, differing only in what is being
claimed:

    current    alpha detector alone, ranked by the shipped `learned` ranker.
               This is what `pip install lacuna-pockets` does today.
    proposed   alpha and surface detectors fused, ranked by fused_ranker.npz.

Every model involved was fitted on training folds only and is frozen before
this runs:

    lacuna/pockets/plm_head.npz       per-residue sequence probabilities
    lacuna/pockets/surface_head.npz   the learned surface detector
    fused_ranker.npz                  the ranker fitted on fused candidates

Their SHA-256 digests are written into the result, so a later run against
different artifacts is visible rather than silent.

What the training folds predicted, for comparison after the fact rather than
before: coverage 65.7% to 81.7%, top-five 55.0% to 67.6% (+12.6 [+9.3, +16.0]),
top-one 28.7% to 36.2%. The test fold is a different and slightly larger cohort
(222 targets against 747), so the baseline will not land exactly on the
published 55.6%, which came from a cohort filtered to structures where every
compared detector ran.

    python analysis/moores_pocket_law/test_fold_eval.py --limit 20   # smoke test
    python analysis/moores_pocket_law/test_fold_eval.py              # the shot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
from dataio import JACCARD_THRESHOLD as T
from lacuna.pockets import surface_detector as sd

CIF = REPO / "benchmarks" / "cb_data" / "cif"
DATASET = REPO / "benchmarks" / "cb_data" / "dataset.json"
RANKER = HERE / "fused_ranker.npz"
CACHE = HERE / "test_fold_eval.jsonl"

ARTIFACTS = (REPO / "lacuna" / "pockets" / "plm_head.npz",
             REPO / "lacuna" / "pockets" / "surface_head.npz",
             RANKER)
KS = (1, 5, 10)
SEED = 42
N_BOOT = 20000


def digests() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in ARTIFACTS}


def load_ranker():
    z = np.load(RANKER, allow_pickle=False)
    feats = [str(x) for x in np.load(RANKER)["features"]]
    return feats, z["mean"], z["scale"], z["coef"], float(z["intercept"])


def run_target(pdb, entry, n_conf, ranker) -> dict:
    from lacuna.ensemble.nma_backend import NMABackend
    from lacuna.io.structure import coords_array, load_structure
    from lacuna.pockets import plm
    from lacuna.pockets.clusterer import cluster_pockets, ranker_features
    from lacuna.pockets.detector import detect_pockets

    feats, mean, scale, coef, b = ranker
    path = CIF / ("%s.cif" % pdb.upper())
    st = load_structure(path, chain=entry["apo_chain"])
    base = coords_array(st)
    confs = [base]
    if n_conf > 0:
        confs += NMABackend(seed=SEED, max_rmsd=2.0, n_modes=10).generate(
            path, n_conformers=n_conf, chain=entry["apo_chain"])

    # Sequence probabilities come from the shipped head, computed once per
    # structure and reused across conformers, exactly as production would.
    probs = plm.residue_probabilities(st) if plm.available() else None

    alpha = [detect_pockets(c, st) for c in confs]
    surf = [sd.detect_pockets_surface(c, st, plm_residue_probs=probs) for c in confs]
    fused = [a + s for a, s in zip(alpha, surf)]

    truth = dataio.cb_residues(entry["apo_pocket_selection"])
    out = {"pdb": pdb}

    cur = cluster_pockets(alpha, n_conformers=len(confs), rank_by="learned")
    out["current"] = [dataio.jaccard(dataio.lacuna_residues(c.lining_residues), truth)
                      for c in cur]

    prop = cluster_pockets(fused, n_conformers=len(confs), rank_by="learned")
    X = np.array([[float(ranker_features(c).get(f, 0.0)) for f in feats] for c in prop])
    if len(X):
        s = ((X - mean) / scale) @ coef + b
        order = np.argsort(-s, kind="stable")
        prop = [prop[i] for i in order]
    out["proposed"] = [dataio.jaccard(dataio.lacuna_residues(c.lining_residues), truth)
                       for c in prop]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conformers", type=int, default=5)
    a = ap.parse_args()

    print("frozen artifacts:")
    for k, v in digests().items():
        print("  %-20s %s" % (k, v))
    print()

    folds = dataio.fold_map()
    ds = json.loads(DATASET.read_text())
    targets = []
    for pdb, entries in ds.items():
        f = folds.get(pdb.lower())
        if f is None or f.startswith("train"):
            continue
        e = entries[0] if isinstance(entries, list) else entries
        if not (e.get("apo_chain") and e.get("apo_pocket_selection")):
            continue
        if not (CIF / ("%s.cif" % pdb.upper())).exists():
            continue
        targets.append((pdb, e))
    targets.sort()
    if a.limit:
        targets = targets[:a.limit]
    print("TEST-FOLD targets: %d   conformers: %d\n" % (len(targets), a.conformers))

    ranker = load_ranker()
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["pdb"]] = r
    todo = [t for t in targets if t[0] not in done]
    print("cached: %d   to run: %d" % (len(targets) - len(todo), len(todo)))

    fail, t0 = 0, time.time()
    with CACHE.open("a", encoding="utf-8") as fh:
        for i, (pdb, e) in enumerate(todo, 1):
            try:
                r = run_target(pdb, e, a.conformers, ranker)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done[pdb] = r
            except Exception as exc:                    # noqa: BLE001
                fail += 1
                if fail <= 5:
                    print("  skip %s: %s" % (pdb, exc), flush=True)
            if i % 20 == 0 or i == len(todo):
                rt = (time.time() - t0) / i
                print("  %4d/%d  ok=%d fail=%d  %.1f s/target  eta %.0f min"
                      % (i, len(todo), i - fail, fail, rt,
                         rt * (len(todo) - i) / 60), flush=True)

    rows = [done[p] for p, _e in targets if p in done]
    if len(rows) < 100:
        raise SystemExit("only %d targets scored; not a usable held-out estimate."
                         % len(rows))

    per = {}
    for arm in ("current", "proposed"):
        d = {"cov": [], **{k: [] for k in KS}}
        for r in rows:
            j = r[arm]
            d["cov"].append(int(any(x >= T for x in j)))
            for k in KS:
                d[k].append(int(any(x >= T for x in j[:k])))
        per[arm] = d

    n = len(rows)
    print("\n%d test-fold targets scored, %d skipped\n" % (n, fail))
    print("%-12s %9s %8s %8s %8s" % ("arm", "coverage", "top-1", "top-5", "top-10"))
    print("-" * 50)
    for arm in ("current", "proposed"):
        d = per[arm]
        print("%-12s %8.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (arm, 100 * np.mean(d["cov"]), 100 * np.mean(d[1]),
                 100 * np.mean(d[5]), 100 * np.mean(d[10])))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    print("\nproposed minus current, paired, %d-resample CI" % N_BOOT)
    print("-" * 50)
    res = {}
    for label, key in (("coverage", "cov"), ("top-1", 1), ("top-5", 5), ("top-10", 10)):
        dd = np.array(per["proposed"][key], float) - np.array(per["current"][key], float)
        bm = np.sort(dd[idx].mean(axis=1))
        lo, hi = 100 * bm[500], 100 * bm[19499]
        mark = "  resolved" if (lo > 0 or hi < 0) else "  not resolved"
        print("  %-9s %+6.2f  [%+6.2f, %+6.2f]%s" % (label, 100 * dd.mean(), lo, hi, mark))
        res[label] = (100 * dd.mean(), lo, hi)

    print("\n%s" % ("=" * 50))
    m, lo, hi = res["top-5"]
    if lo > 0:
        print("Held out, the fused detector and refit ranker beat the shipped")
        print("pipeline at top five by %+.1f points [%+.1f, %+.1f]." % (m, lo, hi))
    elif hi < 0:
        print("Held out, this is worse than what ships. The training-fold gain")
        print("did not transfer, and that is the answer.")
    else:
        print("Held out, the difference does not resolve. The training-fold gain")
        print("did not reproduce at this n, and it should not be claimed.")

    (HERE / "test_fold_eval.json").write_text(json.dumps(
        {"n": n, "conformers": a.conformers, "artifacts": digests(),
         "arms": {arm: {"coverage": float(np.mean(per[arm]["cov"])),
                        **{"hit_%d" % k: float(np.mean(per[arm][k])) for k in KS}}
                  for arm in per},
         "deltas": {k: {"mean": v[0], "lo": v[1], "hi": v[2]}
                    for k, v in res.items()}}, indent=1))
    print("\nwrote %s" % (HERE / "test_fold_eval.json"))


if __name__ == "__main__":
    main()
