"""Is the ensemble machinery worth anything independent of the detector?

Every number Lacuna reports confounds two things it does at once: it proposes
candidates with its own alpha-sphere detector, and it runs those candidates
through an ensemble, a cross-conformer clusterer and a fitted ranker. Nothing
published separates them, so "Lacuna recovers 55.6%" leaves the interesting
question unanswered. If the second half is a general amplifier, it should lift
someone else's detector too. If it only works on candidates its own detector
produces, that is a much narrower claim and worth knowing before someone else
finds out first.

The design is 2x2, detector crossed with mode:

    p2rank-static      P2Rank on the input structure, P2Rank's own ranking.
                       This is the published tool, used exactly as shipped.
    p2rank-ensemble    P2Rank run per conformer, then Lacuna's clusterer and
                       fitted ranker. Same candidates, Lacuna's machinery.
    lacuna-static      Alpha detector on the input structure only, n=1.
    lacuna-ensemble    The shipped configuration.

The comparison that answers the question is the first pair. The second pair is
the control: it says how much the machinery is worth on the detector it was
built around, so the two lifts can be read side by side.

One honest caveat, stated here because the numbers cannot be read without it.
The fitted ranker's features include volume dynamics, persistence and
crypticity, all of which degenerate when there is one conformer. The static
arms are therefore ranked by what their own method offers: P2Rank by its
learned probability, and the alpha detector by analytic druggability. That is
the fair comparison (each method as it would actually be used) but it is not a
clean single-variable manipulation, and lacuna-static should not be read as
"the ranker without the ensemble".

Protocol: train folds only. The designated test fold is never read here.

    python analysis/moores_pocket_law/detector_ablation.py --limit 20
    python analysis/moores_pocket_law/detector_ablation.py --conformers 20
"""
from __future__ import annotations

import argparse
import json
import re
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

CIF = REPO / "benchmarks" / "cb_data" / "cif"
DATASET = REPO / "benchmarks" / "cb_data" / "dataset.json"
CACHE = HERE / "detector_ablation.jsonl"

KS = (1, 5, 10)
SEED = 42

ARMS = ("p2rank-static", "p2rank-ensemble", "lacuna-static", "lacuna-ensemble")


# ─────────────────────────────── residue keys ────────────────────────────────

#: Lacuna writes lining residues as 'G12:A' or 'ALA92:B': residue name, sequence
#: number, then chain. CryptoBench writes its annotations as 'B_12'. Both are
#: normalised to (chain, resnum) so a Jaccard between them means something. A
#: parser that silently drops what it cannot read would make every arm look
#: equally bad, so anything unparseable raises rather than being skipped.
_LACUNA_RES = re.compile(r"^[A-Za-z]*(-?\d+)[A-Za-z]?:(\S+)$")
_CB_RES = re.compile(r"^(\S+)_(-?\d+)$")


def _lacuna_residues(entries) -> set:
    out = set()
    for e in entries:
        m = _LACUNA_RES.match(str(e))
        if not m:
            raise ValueError("unparseable Lacuna residue %r" % (e,))
        out.add((m.group(2), int(m.group(1))))
    return out


def _truth_residues(selection) -> set:
    out = set()
    for e in selection:
        m = _CB_RES.match(str(e))
        if not m:
            raise ValueError("unparseable CryptoBench residue %r" % (e,))
        out.add((m.group(1), int(m.group(2))))
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ──────────────────────────────── the four arms ──────────────────────────────

def _conformers(path: Path, chain: str, n: int, base):
    """Input structure first, then n sampled conformers."""
    from lacuna.ensemble.nma_backend import NMABackend
    if n <= 0:
        return [base]
    sampled = NMABackend(seed=SEED, max_rmsd=2.0, n_modes=10).generate(
        path, n_conformers=n, chain=chain)
    return [base] + sampled


def _rank_clusters(pocket_lists, n_conf, rank_by):
    from lacuna.pockets.clusterer import cluster_pockets
    clusters = cluster_pockets(pocket_lists, n_conformers=n_conf, rank_by=rank_by)
    return [_lacuna_residues(c.lining_residues) for c in clusters]


def run_target(pdb: str, entry: dict, n_conf: int, p2rank_exe: str | None) -> dict:
    """Ranked residue sets for each arm, plus the annotated truth."""
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.p2rank_detector import detect_pockets_p2rank, run_p2rank

    chain = entry["apo_chain"]
    path = CIF / ("%s.cif" % pdb.upper())
    structure = load_structure(path, chain=chain)
    base = coords_array(structure)
    confs = _conformers(path, chain, n_conf, base)

    out: dict = {"pdb": pdb, "chain": chain,
                 "truth": sorted("%s_%d" % r for r in
                                 _truth_residues(entry["apo_pocket_selection"]))}

    # Arm 1. P2Rank exactly as shipped: input structure, its own ranking. Its
    # predictions come back already ordered by learned probability, so the rank
    # order is P2Rank's, not ours. Residues are recharacterised by Lacuna's
    # geometry at each P2Rank centre, which is what detect_pockets_p2rank does,
    # so the residue sets being compared are on one scale across all four arms.
    static_p2 = detect_pockets_p2rank(base, structure, executable=p2rank_exe)
    out["p2rank-static"] = [sorted("%s_%d" % r for r in _lacuna_residues(p.lining_residues))
                            for p in static_p2]

    # Arm 2. The substitution that answers the question: P2Rank supplies every
    # candidate, Lacuna supplies the ensemble, the clustering and the ranker.
    per_conf = [detect_pockets_p2rank(c, structure, executable=p2rank_exe)
                for c in confs]
    out["p2rank-ensemble"] = [sorted("%s_%d" % r for r in s)
                              for s in _rank_clusters(per_conf, len(confs), "learned")]

    # Arm 3. Alpha detector, one conformer. Ranked by analytic druggability
    # because the fitted ranker's ensemble features are undefined at n=1.
    out["lacuna-static"] = [sorted("%s_%d" % r for r in s)
                            for s in _rank_clusters([detect_pockets(base, structure)],
                                                    1, "druggability")]

    # Arm 4. The shipped configuration.
    out["lacuna-ensemble"] = [sorted("%s_%d" % r for r in s)
                              for s in _rank_clusters(
                                  [detect_pockets(c, structure) for c in confs],
                                  len(confs), "learned")]
    return out


# ──────────────────────────────── evaluation ─────────────────────────────────

def score(rows: list) -> dict:
    """Per-arm hit@k and the coverage/conversion split, per target."""
    per = {a: {"hit": {k: [] for k in KS}, "cov": []} for a in ARMS}
    for r in rows:
        truth = _truth_residues(r["truth"])
        for arm in ARMS:
            jacs = [jaccard(_truth_residues(s), truth) for s in r.get(arm, [])]
            covered = any(j >= T for j in jacs)
            per[arm]["cov"].append(int(covered))
            for k in KS:
                per[arm]["hit"][k].append(int(any(j >= T for j in jacs[:k])))
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conformers", type=int, default=10)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the cache and recompute every target")
    a = ap.parse_args()

    from lacuna.pockets.p2rank_detector import p2rank_available, p2rank_executable
    if not p2rank_available():
        raise SystemExit(
            "P2Rank not found. It is a JVM tool and not a Python dependency: "
            "install it (needs Java 11+) and either put 'prank' on PATH or set "
            "LACUNA_P2RANK to the launcher. Without it three of the four arms "
            "cannot run, and a two-arm result would not answer the question.")
    exe = p2rank_executable()

    folds = dataio.fold_map()
    ds = json.loads(DATASET.read_text())
    targets = []
    for pdb, entries in ds.items():
        f = folds.get(pdb.lower())
        if f is None or not f.startswith("train"):
            continue
        e = entries[0] if isinstance(entries, list) else entries
        if not (e.get("apo_chain") and e.get("apo_pocket_selection")):
            continue
        targets.append((pdb, e))
    targets.sort()
    if a.limit:
        targets = targets[:a.limit]

    done = {}
    if CACHE.exists() and not a.refresh:
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["pdb"]] = r
    todo = [(p, e) for p, e in targets if p not in done]
    print("train-fold targets: %d   cached: %d   to run: %d   conformers: %d\n"
          % (len(targets), len(targets) - len(todo), len(todo), a.conformers))

    fail = 0
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as fh:
        for i, (pdb, e) in enumerate(todo, 1):
            try:
                r = run_target(pdb, e, a.conformers, exe)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[pdb] = r
            except Exception as exc:                    # noqa: BLE001
                fail += 1
                if fail <= 6:
                    print("  skip %s: %s" % (pdb, exc), flush=True)
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print("  %4d/%d  ok=%d fail=%d  %.1f s/target  eta %.0f min"
                      % (i, len(todo), i - fail, fail, rate,
                         rate * (len(todo) - i) / 60), flush=True)

    rows = [done[p] for p, _e in targets if p in done]
    # A verdict from a handful of targets is worse than no verdict: the
    # intervals will straddle zero no matter what is true, and printing
    # "not resolved" would read as evidence of no effect.
    if len(rows) < 30:
        raise SystemExit("only %d targets scored; run more before reading "
                         "anything into this." % len(rows))

    per = score(rows)
    n = len(rows)
    print("\n%d targets scored, %d skipped\n" % (n, fail))
    print("%-18s %8s %8s %8s %8s" % ("arm", "cover", "top-1", "top-5", "top-10"))
    print("-" * 56)
    for arm in ARMS:
        d = per[arm]
        print("%-18s %7.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (arm, 100 * np.mean(d["cov"]),
                 100 * np.mean(d["hit"][1]), 100 * np.mean(d["hit"][5]),
                 100 * np.mean(d["hit"][10])))

    # Paired over targets: the same protein contributes to both arms, so the
    # bootstrap resamples targets, not observations.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(20000, n))
    print("\npaired differences, 20,000-resample percentile CI")
    print("-" * 56)
    for lab, hi_arm, lo_arm in (
            ("ensemble lift, P2Rank ", "p2rank-ensemble", "p2rank-static"),
            ("ensemble lift, Lacuna ", "lacuna-ensemble", "lacuna-static"),
            ("detector gap, ensemble", "lacuna-ensemble", "p2rank-ensemble")):
        for k in (5,):
            d = (np.array(per[hi_arm]["hit"][k], float)
                 - np.array(per[lo_arm]["hit"][k], float))
            bm = np.sort(d[idx].mean(axis=1))
            lo, hi = 100 * bm[500], 100 * bm[19499]
            mark = "  resolved" if (lo > 0 or hi < 0) else "  not resolved"
            print("  top-%-2d %s  %+6.1f  [%+6.1f, %+6.1f]%s"
                  % (k, lab, 100 * d.mean(), lo, hi, mark))

    print("\n%s" % ("=" * 56))
    d = (np.array(per["p2rank-ensemble"]["hit"][5], float)
         - np.array(per["p2rank-static"]["hit"][5], float))
    bm = np.sort(d[idx].mean(axis=1))
    if bm[500] > 0:
        print("The ensemble machinery lifts a detector it was not built around.")
        print("That is a general claim, and a stronger one than the paper makes.")
    elif bm[19499] < 0:
        print("The machinery HURTS P2Rank. Worth knowing, and worth saying: it")
        print("would mean the gains are specific to the alpha detector's output.")
    else:
        print("Not resolved at this n. Add targets before concluding either way.")

    (HERE / "detector_ablation.json").write_text(json.dumps(
        {"n": n, "conformers": a.conformers,
         "arms": {arm: {"coverage": float(np.mean(per[arm]["cov"])),
                        **{"hit_%d" % k: float(np.mean(per[arm]["hit"][k]))
                           for k in KS}} for arm in ARMS}}, indent=1))
    print("\nwrote %s" % (HERE / "detector_ablation.json"))


if __name__ == "__main__":
    main()
