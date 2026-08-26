#!/usr/bin/env python
"""Lacuna candidates on PocketMiner and COACH420, in the CryptoBench schema.

The failure anatomy left one prediction standing. Conversion is capped near 84%,
the criterion is not what caps it, and the failures are confident errors rather
than ties: median score gap 0.368, with the answer sitting at rank 11 of 28 and
nothing above 0.10 overlap in the top five. If that residual is experimental
contingency, which cavity a ligand happened to be crystallised in, rather than
something a better model would capture, then the same kind of target should fail
on benchmarks built by other people from other structures.

That prediction needs Lacuna only. The comparator detectors matter for union and
consensus, which is a different question, so nothing here needs fpocket, P2Rank,
MDpocket or the WSL IF-SitePred environment.

The two sets test different things and should not be pooled:

  pocketminer  61 apo structures with per-residue cryptic labels. Same regime as
               CryptoBench, so this is the actual replication.
  coach420     150 holo structures with ligand-defined binding sites. These are
               already-open sites, which Lacuna is not built for and where
               P2Rank is better. Useful as a contrast, not as a replication.

Written as a separate entry point rather than a flag on collect_candidates,
which is CryptoBench-fold-driven and currently correct. The detector functions,
context object and scoring are imported from it, so both paths run the same
code.

    python benchmarks/collect_external.py --dataset pocketminer --limit 3
    python benchmarks/collect_external.py --dataset pocketminer
    python benchmarks/collect_external.py --dataset coach420
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from collect_candidates import (  # noqa: E402
    Ctx, DETECTORS, MAX_RESIDUES, _ca_index, _centroid, _load_done,
)
from compare_fpocket import residue_jaccard, residue_overlap  # noqa: E402
from lacuna.io.structure import load_structure  # noqa: E402

OUT = REPO / "analysis" / "moores_pocket_law" / "data" / "candidates_ext"
TOOLS = ("lacuna", "lacuna_plm")


def targets_pocketminer():
    """Apo structures with per-residue cryptic labels, both splits."""
    import pocketminer_benchmark as PM
    from cryptic_benchmark import download_pdb

    PM._fetch_pm_data()
    cache = HERE / "pdb_cache"
    cache.mkdir(exist_ok=True)
    out = []
    for split in ("test", "val"):
        for pdb, chain, labels in PM.load_split(split):
            try:
                path = download_pdb(pdb.upper(), cache)
                known, n_res, n_lab = PM.cryptic_residues(path, chain, labels)
            except Exception as e:                       # noqa: BLE001
                print("  [skip] %s%s: %s" % (pdb, chain, e), flush=True)
                continue
            if not known:
                continue
            out.append({"pdb": pdb, "chain": chain, "known": set(known),
                        "path": path, "split": split})
    return out


def targets_coach420():
    """Holo structures with ligand-defined binding sites."""
    import coach420_benchmark as C

    out = []
    for pdb, chain, codes in C.fetch_mlig():
        try:
            path = C.download_pdb(pdb)
            site = C.binding_site(path, chain, codes)
        except Exception as e:                           # noqa: BLE001
            print("  [skip] %s%s: %s" % (pdb, chain, e), flush=True)
            continue
        known = site[0] if isinstance(site, tuple) else site
        known = {int(x) for x in known} if known else set()
        if not known:
            continue
        out.append({"pdb": pdb, "chain": chain, "known": known,
                    "path": path, "split": "coach420"})
    return out


SOURCES = {"pocketminer": targets_pocketminer, "coach420": targets_coach420}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, help="42 is the shipped value")
    ap.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    a = ap.parse_args()

    print("loading %s targets ..." % a.dataset, flush=True)
    targets = SOURCES[a.dataset]()
    if a.limit:
        targets = targets[:a.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / ("candidates_%s.jsonl" % a.dataset)
    done = _load_done(out_path)
    print("%d targets, %d rows already collected -> %s\n"
          % (len(targets), len(done), out_path), flush=True)

    n_skip = 0
    tally = {t: [0, 0] for t in TOOLS}
    t0 = time.perf_counter()

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, tg in enumerate(targets, 1):
            sid = "%s%s" % (tg["pdb"], tg["chain"])
            todo = [t for t in TOOLS if (sid, t) not in done]
            if not todo:
                continue
            try:
                structure = load_structure(tg["path"], chain=tg["chain"])
                if not (10 <= len(structure.residues) <= a.max_residues):
                    n_skip += 1
                    continue
            except Exception as e:                       # noqa: BLE001
                n_skip += 1
                print("  [%d/%d] [skip] %s: %s"
                      % (i, len(targets), sid, type(e).__name__), flush=True)
                continue

            ca = _ca_index(structure)
            known = tg["known"]
            site_centroid = _centroid(["%d:%s" % (n, tg["chain"]) for n in known], ca)
            ctx = Ctx(structure=structure, conformers=a.conformers, seed=a.seed,
                      cif=tg["path"], chain=tg["chain"])

            line = "  [%d/%d] %s (%d res)" % (i, len(targets), sid, len(known))
            for tool in todo:
                ctx.last_features = None
                t_tool = time.perf_counter()
                try:
                    residue_lists = DETECTORS[tool][0](tg["path"], tg["chain"], ctx)
                except Exception as e:                   # noqa: BLE001
                    print("  [%d/%d] %s %s: ERROR %s"
                          % (i, len(targets), sid, tool, type(e).__name__), flush=True)
                    continue
                elapsed = time.perf_counter() - t_tool
                jacs = [round(residue_jaccard(r, known), 4) for r in residue_lists]
                best5 = max(jacs[:5], default=0.0)
                fout.write(json.dumps({
                    "id": sid, "pdb": tg["pdb"], "chain": tg["chain"],
                    "dataset": a.dataset, "split": tg["split"],
                    "n_known": len(known), "tool": tool,
                    "recall": round(max((residue_overlap(r, known)
                                         for r in residue_lists[:5]), default=0.0), 3),
                    "jaccard": round(best5, 3),
                    "n_prop": len(residue_lists),
                    "elapsed_s": round(elapsed, 2),
                    "jac_by_rank": jacs,
                    "residues_by_rank": residue_lists,
                    "centroid_by_rank": [_centroid(r, ca) for r in residue_lists],
                    "site_centroid": site_centroid,
                    "site_residues": sorted(known),
                    "features_by_rank": ctx.last_features,
                }) + "\n")
                fout.flush()
                tally[tool][1] += 1
                tally[tool][0] += int(best5 >= 0.25)
                line += "  %s=%d/%d" % (tool, tally[tool][0], tally[tool][1])
            print(line, flush=True)

    print("\n" + "-" * 66)
    for t in TOOLS:
        hit, n = tally[t]
        if n:
            print("  %-12s coverage-at-5 %d/%d = %.1f%%" % (t, hit, n, 100 * hit / n))
    print("  %d skipped, %.1f min" % (n_skip, (time.perf_counter() - t0) / 60))
    print("  rows -> %s" % out_path)


if __name__ == "__main__":
    main()
