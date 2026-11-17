# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""CryptoBench cryptic-binding-site benchmark (Vavra et al. 2024, Bioinformatics).

CryptoBench (https://osf.io/pz4a9/) is the largest cryptic-pocket dataset: 1107
apo structures with holo-derived pocket residue annotations by auth_seq_id. This
script evaluates Lacuna on the held-out **test fold** (222 apo structures).

For each apo structure the main (highest-pRMSD) cryptic pocket is the target; the
apo residues are taken directly from ``apo_pocket_selection`` (e.g. ``"B_12"`` =
chain B, residue 12 - no order-based mapping needed). An interface site spans
more than one chain and its ``apo_chain`` is the hyphen-joined set of them
("G-H", "M-O-P"), which :func:`_selection_residues` reads. Lacuna runs with its
default configuration (NMA backend, crypticity ranking); a top-5 pocket counts
as a hit if it centres within 4 A of the site's Ca centroid or reaches a
Jaccard of 0.25 against its residues.

Every residue comparison here is chain aware: residues are (chain_id, seq_num)
pairs on both sides, so a candidate pocket on one chain of an interface
structure cannot be credited against site residues on another chain that happen
to reuse a sequence number. The chain-blind values are recorded alongside, under
``jaccard_chain_blind`` and ``recall_chain_blind``, for comparison only.

Every structure in the fold is either scored or listed as unscored with a
reason, in the printed summary and in the result JSON.

    python benchmarks/cryptobench_benchmark.py            # full test fold (~1-2 h)
    python benchmarks/cryptobench_benchmark.py --limit 30 # quick subset

Dataset files (dataset.json, folds.json) are auto-downloaded from OSF.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from cryptic_benchmark import (  # noqa: E402
    run_lacuna, residue_overlap, residue_jaccard,
    residue_overlap_pairs, residue_jaccard_pairs, pocket_min_centroid_dist,
    CENTROID_THRESHOLD, OVERLAP_THRESHOLD, JACCARD_THRESHOLD,
)
from lacuna.io.structure import load_structure, chain_selector  # noqa: E402
from lacuna.pockets.clusterer import DEFAULT_RANK_BY, RANK_STRATEGIES  # noqa: E402

CB_DIR = Path(__file__).parent / "cb_data"
CIF_DIR = CB_DIR / "cif"
MIN_RESIDUES = 10
# Legacy ceiling. Kept at its original value because the comparison scripts in
# this directory import it and their stored baselines were collected under it;
# this benchmark uses DEFAULT_MAX_RESIDUES instead.
MAX_RESIDUES = 700
# This benchmark's ceiling. 700 excluded four test structures (7f4y B 901,
# 2aka A 764, 7o1i A 726, 4zoe B 724) and, once the interface sites below are
# scored, eight more that span two chains (8hc1 d-h is the largest at 1138).
# The ceiling was a bare constant with no guard behind it in the library, so it
# was measured rather than guessed: on this machine 8hc1 runs in 14.7 s at
# 458 MB peak RSS and 7f4y in 8.8 s at 200 MB, against ~2-3 s for a median
# entry. Nothing in the fold needs excluding, so the ceiling is raised to
# cover all of it and kept only as a guard against a pathological input.
DEFAULT_MAX_RESIDUES = 1200
_OSF = {
    "dataset.json": "https://osf.io/download/ta2ju/",
    "folds.json": "https://osf.io/download/5s93p/",
}


def _fetch(name: str) -> Path:
    CB_DIR.mkdir(exist_ok=True)
    dst = CB_DIR / name
    if not dst.exists():
        print(f"  Downloading CryptoBench {name} ...", flush=True)
        urllib.request.urlretrieve(_OSF[name], dst)
    return dst


def download_cif(pdb: str) -> Path:
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    out = CIF_DIR / f"{pdb.upper()}.cif"
    if not out.exists():
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb.upper()}.cif", out)
    return out


def _resnum(part: str) -> int | None:
    """Parse a residue number, tolerating a trailing insertion code (e.g. '60A' ->
    60). Lacuna's residue model drops insertion codes, so this matches its
    numbering. Handles negative residue numbers too."""
    digits = "".join(c for c in part if c.isdigit())
    if not digits:
        return None
    return -int(digits) if part.lstrip().startswith("-") else int(digits)


def main_association(assocs: list) -> dict:
    """The main (highest-pRMSD) holo association of an apo structure."""
    main = next((a for a in assocs if a.get("is_main_holo_structure")), None)
    if main is None:
        main = max(assocs, key=lambda a: a.get("pRMSD", 0.0))
    return main


def _selection_residues(main: dict) -> set[tuple[str, int]]:
    """Site residues of one association as (chain_id, seq_num) pairs.

    ``apo_chain`` names a single chain ("A") or, for an interface site, every
    chain the site spans, hyphen-joined: "G-H", "M-O-P", "AAA-CCC", "B-D-F-H".
    Each entry of ``apo_pocket_selection`` is always prefixed with one single
    chain id drawn from that set, never with the composite string. 4x19 has
    apo_chain "M-O-P" and selections ``['M_1', 'M_2', 'M_37', 'M_39', 'P_39',
    'P_41', 'O_6', ...]``; 7nbc has apo_chain "AAA-CCC" and ``['AAA_30', ...,
    'CCC_24', ...]``. Across all 222 test entries no selection prefix lies
    outside the chain set its apo_chain names, so membership in that set is the
    whole rule.

    The old test was ``parts[0] == chain``, an equality against the composite
    string. For the 38 test entries with a composite apo_chain it matched
    nothing, the residue set came back empty, and ``main()`` skipped them
    without printing anything.
    """
    want = chain_selector(main["apo_chain"])
    res: set[tuple[str, int]] = set()
    for s in main["apo_pocket_selection"]:
        parts = s.split("_")
        if len(parts) >= 2 and parts[0] in want:
            n = _resnum(parts[1])
            if n is not None:
                res.add((parts[0], n))
    return res


def main_pocket(assocs: list) -> tuple[str, set[int]]:
    """Return (apo_chain, residue_numbers) for the main cryptic pocket.

    The residue numbers are chain-blind, to match the chain-blind metrics
    ``residue_overlap``/``residue_jaccard``, which parse Lacuna's lining-residue
    labels into bare sequence numbers. This benchmark no longer scores with
    either; it uses :func:`main_pocket_pairs` with ``residue_overlap_pairs`` and
    ``residue_jaccard_pairs`` and keeps the chain-blind values only as a recorded
    comparison. The function stays because seven other scripts in this directory
    import it and their stored baselines were collected under it.
    """
    main = main_association(assocs)
    return main["apo_chain"], {n for _, n in _selection_residues(main)}


def main_pocket_pairs(assocs: list) -> tuple[str, set[tuple[str, int]]]:
    """Same site as :func:`main_pocket`, keeping each residue's chain.

    The reference centroid must not be chain-blind. 1fe6 has apo_chain "A-D"
    and a site of A_19, A_20, A_22, A_23, D_19, D_22, D_23, D_26: a lookup on
    bare numbers would also pull in A_26, which is not in the site, and move
    the centroid. For a single-chain site the two are identical, since numbers
    cannot collide inside one chain.
    """
    main = main_association(assocs)
    return main["apo_chain"], _selection_residues(main)


def known_centroid(structure, chain, known):
    """Ca centroid of the known site.

    ``known`` holds (chain_id, seq_num) pairs, or bare sequence numbers for a
    single-chain site (the older calling convention, still used by the feature
    extraction and detector-comparison scripts here).
    """
    want = chain_selector(chain)
    pairs = any(isinstance(k, tuple) for k in known)
    ca = [a.coords for a in structure.atoms
          if a.name == "CA" and (
              (a.chain_id, a.res_seq) in known if pairs
              else ((want is None or a.chain_id in want) and a.res_seq in known))]
    return tuple(np.mean(ca, axis=0).tolist()) if ca else None


def site_radius(structure, known_pairs) -> float | None:
    """Rms distance of the site's Ca atoms from their centroid, in Angstrom.

    Recorded so that a composite entry whose "site" were really two separate
    patches would be visible rather than scored as one pocket. Measured over
    the test fold it is not: the 38 interface sites have a median radius of
    8.5 A and a maximum of 11.0 A, against 8.4 A and 12.3 A for the 184
    single-chain sites, so every interface site here is as compact as an
    ordinary one and none needs separate treatment.
    """
    ca = np.array([a.coords for a in structure.atoms
                   if a.name == "CA" and (a.chain_id, a.res_seq) in known_pairs])
    if len(ca) < 2:
        return None
    return float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only run first N (0=all)")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--folds", default="test",
                    help="comma-separated fold names to run (default: test). Use "
                         "e.g. train-0,train-1,train-2,train-3 to score brand-new "
                         "pockets held out from all tuning.")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the id order (seed 0) so a --limit subset is a "
                         "representative sample rather than the first N.")
    ap.add_argument("--max-residues", type=int, default=DEFAULT_MAX_RESIDUES,
                    help="skip structures larger than this (default %d; the old "
                         "value was %d, which silently excluded four test "
                         "structures)" % (DEFAULT_MAX_RESIDUES, MAX_RESIDUES))
    ap.add_argument("--rank-by", dest="rank_by", choices=list(RANK_STRATEGIES),
                    default=DEFAULT_RANK_BY,
                    help="pocket ranking strategy (default: %s, the shipped "
                         "default). Only the ordering of the candidate list "
                         "changes; detection, backend and clustering are "
                         "untouched, so this is an ablation on one candidate "
                         "set." % DEFAULT_RANK_BY)
    ap.add_argument("--out", default=str(Path(__file__).parent / "cryptobench_results.json"),
                    help="write per-structure results, and every unscored "
                         "structure with its reason, to this JSON file")
    args = ap.parse_args()

    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    test_ids = [pid for f in args.folds.split(",") for pid in folds[f.strip()]]
    if args.shuffle:
        import random
        random.Random(0).shuffle(test_ids)
    if args.limit:
        test_ids = test_ids[:args.limit]

    print("=" * 70)
    print(f"  CRYPTOBENCH [{args.folds}]  ({len(test_ids)} apo structures, "
          f"NMA + {args.rank_by})")
    print("=" * 70, flush=True)

    records: list[dict] = []
    unscored: list[dict] = []

    def drop(apo, chain, reason, detail=""):
        """Record an unscored structure. Nothing leaves this loop silently: the
        old harness dropped an entry with an empty residue set without a word,
        which is how 38 composite-chain structures disappeared from every
        number measured here."""
        unscored.append({"id": apo, "chain": chain, "reason": reason,
                         "detail": detail})
        print(f"  [unscored] {apo}{chain or ''}: {reason}"
              f"{': ' + detail if detail else ''}", flush=True)

    n_pass = n_run = n_pass_legacy = 0
    t_start = time.perf_counter()
    for i, apo in enumerate(test_ids, 1):
        assocs = dataset.get(apo)
        if not assocs:
            drop(apo, None, "no_dataset_entry")
            continue
        chain, known = main_pocket(assocs)
        _, known_pairs = main_pocket_pairs(assocs)
        if not known:
            drop(apo, chain, "empty_known_residues",
                 "no apo_pocket_selection entry prefixed with a chain of "
                 f"{sorted(chain_selector(chain))}")
            continue
        tag = f"{apo}{chain}"
        try:
            cif = download_cif(apo)
            s = load_structure(cif, chain=chain)
            n_res = len(s.residues)
            if n_res > args.max_residues:
                drop(apo, chain, "over_residue_ceiling",
                     f"{n_res} residues > --max-residues {args.max_residues}")
                continue
            if n_res < MIN_RESIDUES:
                drop(apo, chain, "under_residue_floor", f"{n_res} residues")
                continue
            ref = known_centroid(s, chain, known_pairs)
            rg = site_radius(s, known_pairs)
            clusters, elapsed = run_lacuna(cif, args.conformers, chain=chain,
                                           rank_by=args.rank_by)
        except Exception as e:
            drop(apo, chain, "exception", f"{type(e).__name__}: {str(e)[:120]}")
            continue

        # Residue terms are chain aware: they score (chain, seq) pairs against
        # ``known_pairs``, so a candidate pocket on one chain of an interface
        # structure can no longer be credited against site residues that live on
        # another chain. The chain-blind values are measured alongside and kept in
        # the record, so the size of the correction stays auditable per structure.
        def best(top_n):
            sub = clusters if top_n is None else clusters[:top_n]
            ov = max((residue_overlap_pairs(c.lining_residues, known_pairs)
                      for c in sub), default=0.0)
            jac = max((residue_jaccard_pairs(c.lining_residues, known_pairs)
                       for c in sub), default=0.0)
            ov_b = max((residue_overlap(c.lining_residues, known)
                        for c in sub), default=0.0)
            jac_b = max((residue_jaccard(c.lining_residues, known)
                         for c in sub), default=0.0)
            dist, rank = pocket_min_centroid_dist(
                clusters, ref, top_n=len(clusters) if top_n is None else top_n)
            return ov, jac, dist, rank, ov_b, jac_b

        best_ov, best_jac, best_dist, best_rank, best_ov_b, best_jac_b = best(5)
        ov8, jac8, dist8, _, ov8_b, jac8_b = best(8)
        ovA, jacA, distA, rankA, ovA_b, jacA_b = best(None)
        # Size-robust headline: centroid OR Jaccard. Legacy (recall) tracked alongside.
        found = best_dist <= CENTROID_THRESHOLD or best_jac >= JACCARD_THRESHOLD
        found_legacy = best_dist <= CENTROID_THRESHOLD or best_ov >= OVERLAP_THRESHOLD
        found8 = dist8 <= CENTROID_THRESHOLD or jac8 >= JACCARD_THRESHOLD
        found_any = distA <= CENTROID_THRESHOLD or jacA >= JACCARD_THRESHOLD
        n_run += 1
        n_pass += int(found)
        n_pass_legacy += int(found_legacy)
        records.append({
            "id": apo, "chain": chain, "composite": len(chain_selector(chain)) > 1,
            # n_known is the size of the site actually scored, the (chain, seq)
            # set. n_known_chain_blind is the bare-number set it used to be: on
            # eight interface entries the two chains reuse a sequence number and
            # the blind count is short by one to three residues.
            "n_residues": n_res, "n_known": len(known_pairs),
            "n_known_chain_blind": len(known), "site_radius_a": rg,
            "n_clusters": len(clusters), "elapsed_s": round(elapsed, 2),
            "top5": {"jaccard": round(best_jac, 4), "recall": round(best_ov, 4),
                     "jaccard_chain_blind": round(best_jac_b, 4),
                     "recall_chain_blind": round(best_ov_b, 4),
                     "centroid_dist_a": None if best_dist == float("inf") else round(best_dist, 3),
                     "rank": best_rank, "found": found, "found_legacy": found_legacy},
            "top8": {"jaccard": round(jac8, 4), "recall": round(ov8, 4),
                     "jaccard_chain_blind": round(jac8_b, 4),
                     "recall_chain_blind": round(ov8_b, 4),
                     "centroid_dist_a": None if dist8 == float("inf") else round(dist8, 3),
                     "found": found8},
            "any_rank": {"jaccard": round(jacA, 4), "recall": round(ovA, 4),
                         "jaccard_chain_blind": round(jacA_b, 4),
                         "recall_chain_blind": round(ovA_b, 4),
                         "centroid_dist_a": None if distA == float("inf") else round(distA, 3),
                         "rank": rankA, "found": found_any},
        })
        mark = "PASS" if found else "miss"
        dist_s = f"{best_dist:.1f}A" if best_dist < float("inf") else "n/a"
        rate = n_pass / max(n_run, 1)
        print(f"  [{i}/{len(test_ids)}] {mark} {tag}  jac={best_jac:.0%} recall={best_ov:.0%} "
              f"dist={dist_s} ({len(known)} res, {elapsed:.1f}s)  "
              f"running={n_pass}/{n_run} ({rate:.0%})",
              flush=True)

    dt = time.perf_counter() - t_start
    n_attempt = len(test_ids)
    n_pass8 = sum(r["top8"]["found"] for r in records)
    n_ceiling = sum(r["any_rank"]["found"] for r in records)

    def pct(k, n):
        return f"{k}/{n} ({k / max(n, 1):.1%})"

    print("-" * 70)
    print(f"  CRYPTOBENCH [{args.folds}] (size-robust, cen<={CENTROID_THRESHOLD:.0f}A "
          f"OR jac>={JACCARD_THRESHOLD:.0%}), top 5: {pct(n_pass, n_run)}")
    print(f"  legacy recall-based, top 5:            {pct(n_pass_legacy, n_run)}")
    print(f"  top 8:                                 {pct(n_pass8, n_run)}")
    print(f"  coverage ceiling (any rank):           {pct(n_ceiling, n_run)}")
    print(f"  scored {n_run} of {n_attempt} attempted; {len(unscored)} unscored")
    # An unscored structure is a structure the tool failed to deliver a pocket
    # for, so the strict rate counts it as a miss instead of dropping it.
    print(f"  strict top 5 (unscored counted as misses): {pct(n_pass, n_attempt)}")
    if unscored:
        print("  unscored structures:")
        for u in unscored:
            print(f"    {u['id']}{u['chain'] or ''}  {u['reason']}"
                  f"{': ' + u['detail'] if u['detail'] else ''}")
    print(f"  [{dt/60:.1f} min]")

    summary = {
        "folds": args.folds,
        "conformers": args.conformers,
        "rank_by": args.rank_by,
        "backend": "nma",
        "max_residues": args.max_residues,
        "residue_metric": "chain_aware",
        "centroid_threshold_a": CENTROID_THRESHOLD,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "n_attempted": n_attempt,
        "n_scored": n_run,
        "n_unscored": len(unscored),
        "top5_found": n_pass,
        "top5_found_legacy": n_pass_legacy,
        "top8_found": n_pass8,
        "coverage_ceiling_found": n_ceiling,
        "top5_strict_found_over_attempted": [n_pass, n_attempt],
        "wall_minutes": round(dt / 60, 2),
    }
    out = Path(args.out)
    out.write_text(json.dumps(
        {"summary": summary, "unscored": unscored, "structures": records},
        indent=1), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
