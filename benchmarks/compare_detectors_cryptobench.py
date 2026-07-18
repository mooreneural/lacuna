# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Multi-detector complementarity benchmark on the full CryptoBench set.

Scores up to four detectors per CryptoBench structure under one identical
size-robust protocol (top-5; Jaccard >= 0.25 size-robust, recall >= 0.30 legacy),
so their recovered-pocket sets can be compared and their union measured:

    lacuna_md   Lacuna ensemble, OpenMM implicit-solvent MD backend (GPU)
    lacuna_nma  Lacuna ensemble, NMA backend (CPU) - the control for lacuna_md
    fpocket     fpocket single-structure geometric detector (WSL/Linux)
    p2rank      P2Rank single-structure ML surface detector (JVM, Java 11+)

Design notes
------------
The four detectors do not all run on the same machine: OpenMM's GPU platform is
available under Windows here, while fpocket is built under WSL. This script is
therefore split by ``--tools`` and ``--tag``: run the Lacuna detectors on the
GPU host and the single-structure baselines on the fpocket host, each writing its
own newline-delimited JSONL (``detectors_cb_<tag>.jsonl``). ``--analyze`` merges
every ``detectors_cb_*.jsonl`` by structure id, so the two legs never write the
same file concurrently and either can resume independently.

Results are keyed per (structure, detector); a row already present in the tag
file is skipped, so a crash or reboot mid-run loses at most one structure. The
dataset, known-site resolution and size gate are reused verbatim from
cryptobench_benchmark.py so every leg scores an identical id set.

Usage
-----
    # GPU host (Windows): Lacuna MD + NMA control, all folds
    python benchmarks/compare_detectors_cryptobench.py \
        --tools lacuna_md,lacuna_nma --tag lacuna --folds test,train-0,train-1,train-2,train-3

    # fpocket host (WSL): single-structure baselines, all folds
    python benchmarks/compare_detectors_cryptobench.py \
        --tools fpocket,p2rank --tag baselines --folds test,train-0,train-1,train-2,train-3

    # merge + report (either host, once both legs are done or in progress)
    python benchmarks/compare_detectors_cryptobench.py --analyze
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import (  # noqa: E402
    run_lacuna, OVERLAP_THRESHOLD, JACCARD_THRESHOLD,
)
from cryptobench_benchmark import (  # noqa: E402
    _fetch, download_cif, main_pocket, MAX_RESIDUES,
)
from compare_fpocket import run_fpocket, residue_overlap, residue_jaccard  # noqa: E402
from metrics import paired_bootstrap_ci  # noqa: E402
from lacuna.io.structure import load_structure  # noqa: E402

ALL_TOOLS = ("lacuna_md", "lacuna_nma", "fpocket", "p2rank")


def _best_over_topk(residue_lists, known, k=5):
    """Best legacy recall and size-robust Jaccard over the top-k proposals.

    ``residue_lists`` is a ranked list of residue-label lists (rank 1 first).
    """
    best_ov = max((residue_overlap(r, known) for r in residue_lists[:k]), default=0.0)
    best_jac = max((residue_jaccard(r, known) for r in residue_lists[:k]), default=0.0)
    return best_ov, best_jac


def _run_tool(tool, cif, chain, n_conformers):
    """Run one detector on one structure; return (recall, jaccard, n_prop, elapsed)."""
    t0 = time.perf_counter()
    if tool in ("lacuna_md", "lacuna_nma"):
        backend = "openmm" if tool == "lacuna_md" else "nma"
        clusters, _ = run_lacuna(cif, n_conformers, chain=chain,
                                 backend_name=backend, rank_by="crypticity")
        residue_lists = [c.lining_residues for c in clusters[:5]]
        n_prop = len(clusters)
    elif tool == "fpocket":
        pockets = run_fpocket(cif, chain)
        residue_lists = [p["residues"] for p in pockets[:5]]
        n_prop = len(pockets)
    elif tool == "p2rank":
        from lacuna.pockets.p2rank_detector import run_p2rank
        pockets = run_p2rank(cif, chain)
        residue_lists = [p["residues"] for p in pockets[:5]]
        n_prop = len(pockets)
    else:
        raise ValueError(f"unknown tool {tool}")
    return residue_lists, n_prop, time.perf_counter() - t0


def _load_done(out_path):
    """Return the set of (id, tool) already scored in the tag file."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["id"], r["tool"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run(args):
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    for t in tools:
        if t not in ALL_TOOLS:
            print(f"unknown tool '{t}'; valid: {', '.join(ALL_TOOLS)}")
            return

    # Preflight external-tool availability so a long run fails fast, not midway.
    if "fpocket" in tools and shutil.which("fpocket") is None:
        print("WARNING: fpocket not on PATH; see compare_fpocket.py for build notes.")
        return
    if "p2rank" in tools:
        from lacuna.pockets.p2rank_detector import p2rank_available, p2rank_executable
        if not p2rank_available():
            print("WARNING: P2Rank not found. Install Java 11+ and put 'prank' on PATH "
                  "or set LACUNA_P2RANK. See compare_p2rank.py.")
            return
        print(f"Using P2Rank: {p2rank_executable()}")

    out_path = Path(__file__).parent / f"detectors_cb_{args.tag}.jsonl"
    done = _load_done(out_path)

    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    ids = [pid for f in args.folds.split(",") for pid in folds[f.strip()]]
    if args.shuffle:
        import random
        random.Random(0).shuffle(ids)
    if args.limit:
        ids = ids[:args.limit]

    print("=" * 70)
    print(f"  DETECTORS on CryptoBench [{args.folds}]  tools={tools}  tag={args.tag}")
    print(f"  {len(ids)} candidate ids, {len(done)} (id,tool) rows already done")
    print("=" * 70, flush=True)

    n_skip = 0
    running = {t: [0, 0] for t in tools}  # tool -> [n_pass_robust, n_scored]
    t_start = time.perf_counter()

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, apo in enumerate(ids, 1):
            assocs = dataset.get(apo)
            if not assocs:
                n_skip += 1
                continue
            chain, known = main_pocket(assocs)
            if not known:
                n_skip += 1
                continue
            tag_id = f"{apo}{chain}"

            todo = [t for t in tools if (tag_id, t) not in done]
            if not todo:
                continue

            try:
                cif = download_cif(apo)
                s = load_structure(cif, chain=chain)
                if len(s.residues) > args.max_residues or len(s.residues) < 10:
                    n_skip += 1
                    continue
            except Exception as e:
                n_skip += 1
                print(f"  [{i}/{len(ids)}] [skip] {tag_id}: {type(e).__name__}", flush=True)
                continue

            for tool in todo:
                try:
                    residue_lists, n_prop, elapsed = _run_tool(
                        tool, cif, chain, args.conformers)
                    ov, jac = _best_over_topk(residue_lists, known)
                except Exception as e:
                    print(f"  [{i}/{len(ids)}] {tag_id} {tool}: ERROR "
                          f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                    continue
                row = {
                    "id": tag_id, "pdb": apo, "chain": chain, "n_known": len(known),
                    "tool": tool, "recall": round(ov, 3), "jaccard": round(jac, 3),
                    "n_prop": n_prop, "elapsed_s": round(elapsed, 2),
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                done.add((tag_id, tool))
                running[tool][1] += 1
                running[tool][0] += int(jac >= JACCARD_THRESHOLD)

            tally = "  ".join(
                f"{t}={running[t][0]}/{running[t][1]}" for t in tools if running[t][1])
            print(f"  [{i}/{len(ids)}] {tag_id} ({len(known)} res)  {tally}", flush=True)

    dt = (time.perf_counter() - t_start) / 60
    print("-" * 70)
    print(f"  done leg tools={tools}: {n_skip} skipped, {dt:.1f} min")
    print(f"  rows -> {out_path}")


def analyze(args):
    """Merge every detectors_cb_*.jsonl by id and report the complementarity matrix."""
    rows_by_id = {}  # id -> {tool -> {recall, jaccard, ...}}
    files = sorted(Path(__file__).parent.glob("detectors_cb_*.jsonl"))
    if not files:
        print("No detectors_cb_*.jsonl files found; run some legs first.")
        return
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows_by_id.setdefault(r["id"], {})[r["tool"]] = r

    print("=" * 70)
    print("  COMPLEMENTARITY ANALYSIS (size-robust: Jaccard >= "
          f"{JACCARD_THRESHOLD:.0%}, legacy recall >= {OVERLAP_THRESHOLD:.0%})")
    print(f"  merged from: {', '.join(f.name for f in files)}")
    print("=" * 70)

    present = {t: sum(1 for d in rows_by_id.values() if t in d) for t in ALL_TOOLS}
    print("  coverage (structures scored per tool):")
    for t in ALL_TOOLS:
        if present[t]:
            print(f"    {t:12s} {present[t]}")

    def robust(d, t):
        return t in d and d[t]["jaccard"] >= JACCARD_THRESHOLD

    def legacy(d, t):
        return t in d and d[t]["recall"] >= OVERLAP_THRESHOLD

    # Per-tool rate + CI over the structures where that tool ran.
    print("\n  per-tool recovery:")
    for t in ALL_TOOLS:
        hits = [robust(d, t) for d in rows_by_id.values() if t in d]
        leg = [legacy(d, t) for d in rows_by_id.values() if t in d]
        if not hits:
            continue
        m, lo, hi = paired_bootstrap_ci(hits)
        print(f"    {t:12s} size-robust {sum(hits)}/{len(hits)} "
              f"({m:.0%}, CI[{lo:.0%},{hi:.0%}])   legacy {sum(leg)}/{len(leg)} "
              f"({sum(leg)/len(leg):.0%})")

    # MD vs NMA control, on structures where BOTH Lacuna backends ran.
    both_lac = [d for d in rows_by_id.values()
                if "lacuna_md" in d and "lacuna_nma" in d]
    if both_lac:
        md = [robust(d, "lacuna_md") for d in both_lac]
        nma = [robust(d, "lacuna_nma") for d in both_lac]
        print(f"\n  GPU-MD vs NMA control (n={len(both_lac)} both ran):")
        print(f"    lacuna_md  {sum(md)}/{len(md)} ({sum(md)/len(md):.0%})   "
              f"lacuna_nma {sum(nma)}/{len(nma)} ({sum(nma)/len(nma):.0%})")
        md_only = sum(1 for a, b in zip(md, nma) if a and not b)
        nma_only = sum(1 for a, b in zip(md, nma) if b and not a)
        print(f"    MD-only catches {md_only}   NMA-only catches {nma_only}")

    # Three-way complementarity: Lacuna (primary) vs fpocket vs P2Rank, on the
    # structures where all three ran.
    primary = "lacuna_md" if present.get("lacuna_md") else "lacuna_nma"
    triple = [d for d in rows_by_id.values()
              if primary in d and "fpocket" in d and "p2rank" in d]
    if triple:
        _complementarity(triple, primary, robust)


def _complementarity(rows, primary, robust):
    n = len(rows)
    lac = [robust(d, primary) for d in rows]
    fp = [robust(d, "fpocket") for d in rows]
    p2 = [robust(d, "p2rank") for d in rows]

    union = [a or b or c for a, b, c in zip(lac, fp, p2)]
    lac_only = sum(1 for a, b, c in zip(lac, fp, p2) if a and not b and not c)
    fp_only = sum(1 for a, b, c in zip(lac, fp, p2) if b and not a and not c)
    p2_only = sum(1 for a, b, c in zip(lac, fp, p2) if c and not a and not b)
    none = sum(1 for u in union if not u)

    um, ulo, uhi = paired_bootstrap_ci(union)
    best_single = max(sum(lac), sum(fp), sum(p2))

    print(f"\n  three-way complementarity ({primary} vs fpocket vs p2rank, n={n} all ran):")
    print(f"    {primary:12s} {sum(lac)}/{n} ({sum(lac)/n:.0%})")
    print(f"    {'fpocket':12s} {sum(fp)}/{n} ({sum(fp)/n:.0%})")
    print(f"    {'p2rank':12s} {sum(p2)}/{n} ({sum(p2)/n:.0%})")
    print(f"    unique catches:  {primary}={lac_only}  fpocket={fp_only}  p2rank={p2_only}")
    print(f"    none of three:   {none}/{n}")
    print(f"    UNION (any):     {sum(union)}/{n} ({um:.0%}, CI[{ulo:.0%},{uhi:.0%}])")
    print(f"    vs best single:  {best_single}/{n} ({best_single/n:.0%})  "
          f"-> union adds {sum(union) - best_single}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", default="lacuna_md,lacuna_nma",
                    help=f"comma list from: {', '.join(ALL_TOOLS)}")
    ap.add_argument("--tag", default="run",
                    help="output file tag: detectors_cb_<tag>.jsonl (keep the two "
                         "platform legs on different tags)")
    ap.add_argument("--folds", default="test",
                    help="comma-separated fold names (test,train-0,train-1,train-2,train-3)")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="only first N ids (0=all)")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle id order (seed 0) so --limit is representative")
    ap.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    ap.add_argument("--analyze", action="store_true",
                    help="merge all detectors_cb_*.jsonl and print the report")
    args = ap.parse_args()

    if args.analyze:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
