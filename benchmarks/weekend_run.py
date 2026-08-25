#!/usr/bin/env python
"""One command for the weekend box: preflight, smoke, then the full collection.

What this produces is the one artefact the ranking investigation could not
build: ranker features and lining-residue lists for the same candidate, in the
same row. Everything measured so far had to choose between them, because the
released feature file carries no residues and the collection file carries no
features, and the two runs agree on only 3.3% of Jaccard vectors so they cannot
be joined after the fact.

With both in one file these become answerable:

  * does ESM-IF1 beat the four plm_* aggregates *alongside* geometry, which is
    the only question whose answer would change the shipped ranker
  * can the 23 atomistic descriptors replace the language model entirely, and
    drop a 2.5 GB download and the torch dependency with it
  * do ensemble dynamics discriminate once sequence is represented properly
    rather than through four scalars

Stages, in order, each resumable and each skipped if already complete:

  1  collect   Lacuna and Lacuna-PLM over the CryptoBench train folds, emitting
               features, residues, centroids and Jaccard together. No external
               binaries: this is Lacuna's own pipeline and nothing else.
  2  esmif     Per-residue ESM-IF1 encoder output for the same structures.
  3  verify    Join the two and confirm every candidate resolves, so a failure
               surfaces here rather than during analysis.

Runtime is not the constraint. Lacuna is about 1.3 s per structure and ESM-IF1
about 1 s, so the full train fold is well under an hour on one core and one GPU.
Setup is the constraint, which is what --preflight exists for.

Usage, from the repository root:

    python benchmarks/weekend_run.py --preflight
    python benchmarks/weekend_run.py --smoke 5
    python benchmarks/weekend_run.py --full

Interrupting is safe. Every stage appends per structure and skips what is
already on disk, so rerunning continues rather than restarting.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "analysis" / "moores_pocket_law" / "data"
COLLECT = OUT / "candidates_v3"
ESMIF = OUT / "esmif_emb"
STAMP = OUT / "weekend_platform.json"

GREEN, RED, YEL, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    GREEN = RED = YEL = OFF = ""          # older consoles render the codes literally


def _ok(msg):   print("  %s[ ok ]%s %s" % (GREEN, OFF, msg))
def _warn(msg): print("  %s[warn]%s %s" % (YEL, OFF, msg))
def _bad(msg):  print("  %s[FAIL]%s %s" % (RED, OFF, msg))


def preflight(need_gpu=False) -> bool:
    """Check everything before any work starts. Returns False on a blocker."""
    print("\npreflight\n" + "-" * 60)
    fatal = []

    v = sys.version_info
    (_ok if v >= (3, 10) else _bad)("python %d.%d.%d" % v[:3])
    if v < (3, 10):
        fatal.append("python >= 3.10")

    for mod, why in (("numpy", None), ("scipy", None), ("sklearn", None),
                     ("torch", None), ("gemmi", None), ("Bio", "biopython"),
                     ("esm", "fair-esm, for ESM-IF1"),
                     ("transformers", "for ESM-2"),
                     ("lightgbm", "analysis only, not collection")):
        try:
            m = __import__(mod)
            _ok("%-14s %s" % (mod, getattr(m, "__version__", "")))
        except ImportError:
            if mod == "lightgbm":
                _warn("%-14s missing (%s)" % (mod, why))
            else:
                _bad("%-14s MISSING %s" % (mod, why or ""))
                fatal.append(mod)

    try:
        import torch
        if torch.cuda.is_available():
            _ok("cuda           %s" % torch.cuda.get_device_name(0))
        else:
            (_bad if need_gpu else _warn)("cuda           not available; ESM-IF1 "
                                          "falls back to CPU (slower, still works)")
    except Exception:
        pass

    try:
        from lacuna.pockets.clusterer import ranker_features  # noqa: F401
        _ok("lacuna importable, ranker_features present")
    except Exception as e:                                   # noqa: BLE001
        _bad("lacuna import failed: %s" % e)
        fatal.append("lacuna")

    cif = REPO / "benchmarks" / "cb_data" / "cif"
    n = len(list(cif.glob("*.cif"))) if cif.is_dir() else 0
    if n > 100:
        _ok("cif cache      %d structures" % n)
    else:
        _warn("cif cache      %d present; missing ones download on demand" % n)

    folds = REPO / "benchmarks" / "cb_data" / "folds.json"
    (_ok if folds.exists() else _bad)("folds.json     %s" %
                                      ("present" if folds.exists() else "MISSING"))
    if not folds.exists():
        fatal.append("folds.json")

    free = shutil.disk_usage(REPO).free / 1e9
    (_ok if free > 10 else _bad)("disk free      %.1f GB (need ~5)" % free)
    if free < 5:
        fatal.append("disk")

    # Not required, but the difference between five detectors and two.
    for exe, note in (("fpocket", "comparators"), ("prank", "P2Rank"),
                      ("mdpocket", "ensemble grid")):
        p = shutil.which(exe)
        (_ok if p else _warn)("%-14s %s" % (exe, p or "not on PATH (%s only)" % note))

    print("-" * 60)
    if fatal:
        print("%sblocked:%s %s" % (RED, OFF, ", ".join(sorted(set(fatal)))))
        return False
    print("%sready%s" % (GREEN, OFF))
    return True


def stamp_platform() -> None:
    """Results are not comparable across platforms; record which this is."""
    import numpy as np
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "?")
    except Exception:                                        # noqa: BLE001
        blas = "?"
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps({
        "node": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "blas": blas, "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime()),
    }, indent=1))
    print("platform recorded -> %s" % STAMP)
    print("  NOTE: eigendecomposition mode ordering is resolved by the linear")
    print("  algebra backend, so candidate lists from this machine are not")
    print("  interchangeable with those collected elsewhere. Do not mix them.")


def stage_collect(limit: int) -> bool:
    cmd = [sys.executable, str(HERE / "collect_candidates.py"),
           "--tools", "lacuna,lacuna_plm", "--folds", "train",
           "--out", str(COLLECT)]
    if limit:
        cmd += ["--limit", str(limit)]
    print("\nstage 1/3  collect   %s\n" % " ".join(cmd[1:]))
    return subprocess.run(cmd, cwd=str(REPO)).returncode == 0


def stage_esmif(limit: int) -> bool:
    print("\nstage 2/3  esmif     per-residue ESM-IF1 encoder output\n")
    sys.path.insert(0, str(REPO / "analysis" / "moores_pocket_law"))
    import esmif_route2 as E
    recs = []
    path = COLLECT / "candidates_train.jsonl"
    if not path.exists():
        _bad("stage 1 produced no output")
        return False
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            recs.append(r)
    if limit:
        recs = recs[:limit]
    E.CACHE = ESMIF
    E.build_cache(recs)
    return True


def stage_verify(limit: int) -> bool:
    print("\nstage 3/3  verify    joining features to residues\n")
    import numpy as np
    path = COLLECT / "candidates_train.jsonl"
    sys.path.insert(0, str(REPO / "analysis" / "moores_pocket_law"))
    import plm_headroom as PH

    n_rows = n_cand = n_feat = n_emb = n_qual = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tool") != "lacuna":
                continue
            n_rows += 1
            feats = r.get("features_by_rank") or []
            res = r.get("residues_by_rank") or []
            jac = r.get("jac_by_rank") or []
            n_cand += len(res)
            n_feat += sum(1 for f in feats if f)
            n_qual += sum(1 for j in jac if float(j) >= 0.25)
            f = ESMIF / ("%s.npz" % r["id"])
            if f.exists():
                z = np.load(f)
                pos = {int(x) for x in z["nums"]}
                for rl in res:
                    if any(n in pos for n in PH._resnums(rl, r["chain"])):
                        n_emb += 1

    print("  structures          %d" % n_rows)
    print("  candidates          %d" % n_cand)
    print("  with 27 features    %d  (%.1f%%)" % (n_feat, 100 * n_feat / max(n_cand, 1)))
    print("  with ESM-IF1 vector %d  (%.1f%%)" % (n_emb, 100 * n_emb / max(n_cand, 1)))
    print("  qualifying          %d" % n_qual)
    good = n_cand and n_feat / n_cand > 0.95 and n_emb / n_cand > 0.90
    (_ok if good else _bad)("join is %susable" % ("" if good else "NOT "))
    return bool(good)


def main() -> None:
    # Subprocess output is unbuffered but the parent's is not, so without this
    # the stage banners appear after the work they announce, which is confusing
    # to watch over a long run.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true", help="check and exit")
    g.add_argument("--smoke", type=int, metavar="N", help="run N structures end to end")
    g.add_argument("--full", action="store_true", help="the whole train fold, resumable")
    a = ap.parse_args()

    if not preflight(need_gpu=False):
        raise SystemExit(2)
    if a.preflight:
        print("\nnothing run. Next: python benchmarks/weekend_run.py --smoke 5")
        return

    limit = a.smoke or 0
    if limit:
        print("\n%sSMOKE: %d structures. Numbers here prove plumbing, not science.%s"
              % (YEL, limit, OFF))
    stamp_platform()

    t0 = time.perf_counter()
    for name, fn in (("collect", stage_collect), ("esmif", stage_esmif),
                     ("verify", stage_verify)):
        if not fn(limit):
            print("\n%sstage '%s' failed. Nothing after it ran; rerun to resume.%s"
                  % (RED, name, OFF))
            raise SystemExit(1)

    mins = (time.perf_counter() - t0) / 60
    print("\n%sdone in %.1f min%s" % (GREEN, mins, OFF))
    print("  candidates -> %s" % (COLLECT / "candidates_train.jsonl"))
    print("  embeddings -> %s" % ESMIF)
    if limit:
        print("\nSmoke passed. Now: python benchmarks/weekend_run.py --full")


if __name__ == "__main__":
    main()
