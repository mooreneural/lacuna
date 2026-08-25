#!/usr/bin/env python
"""Collect per-candidate detector output, keeping what the old collector threw away.

compare_detectors_cryptobench.py computes each candidate's Jaccard against the
annotated site and then discards the candidate itself. That is enough to answer
coverage, conversion and top-k for one detector at a time, and not enough for
anything that has to reason about candidates *across* detectors, because there is
no way to tell which fpocket pocket is the same site as which P2Rank one.

This collector keeps three things per candidate:

    residues    the lining residue labels, "<resnum>:<chain>"
    centroid    mean CA coordinate of those residues, in the input frame
    jaccard     overlap with the annotated site

Centroids are computed here from the lining residues rather than read from each
tool, so they mean the same thing for every detector. Two consequences beyond
cross-detector matching: the repository's "size-robust" criterion, which accepts
a centroid within 4 A of the site centroid, becomes evaluable from a dump for the
first time, and so does any distance-based analysis.

The output is a superset of the existing schema. Every field the old dumps carry
is present and computed identically, so an existing analysis can read these files
unchanged.

Adding a detector is one entry in DETECTORS. The contract is small on purpose:

    fn(cif: Path, chain: str, ctx: Ctx) -> list[list[str]]

a rank-ordered list of residue-label lists, best first. Anything that can produce
that can be compared against everything else here.

    python benchmarks/collect_candidates.py --tools fpocket,p2rank --folds test
    python benchmarks/collect_candidates.py --tools lacuna --conformers 20
    python benchmarks/collect_candidates.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from cryptobench_benchmark import (  # noqa: E402
    _fetch, download_cif, main_pocket, MAX_RESIDUES,
)
from compare_fpocket import (  # noqa: E402
    run_fpocket, residue_jaccard, residue_overlap, _parse_resnums,
)
from lacuna.io.structure import load_structure, coords_array  # noqa: E402
from lacuna.io.writers import write_structure_pdb  # noqa: E402
from lacuna.ensemble.nma_backend import NMABackend  # noqa: E402
from compare_mdpocket import (  # noqa: E402
    run_mdpocket, parse_dx, grid_pockets, lining_index, lining_residues,
)

DEFAULT_OUT = HERE / "candidates"
JACCARD_THRESHOLD = 0.25


#: MDpocket emits a frequency grid rather than a ranked list, so thresholding it
#: and ranking the survivors is an adaptation, not MDpocket's own method. To
#: avoid handicapping it, compare_mdpocket.py sweeps five isovalues by two
#: ranking rules and reports the best; that sweep picked 0.7 with volume ranking
#: at 44.1%, against 40.2% at the 0.5 default. The winner is used here so the
#: comparison stays generous, and it is a constant rather than a re-derivation.
MDPOCKET_ISOVALUE = 0.7
MDPOCKET_RANKING = "volume"


@dataclass
class Ctx:
    """Everything a detector might need beyond the structure path.

    The NMA ensemble is generated once and shared. That is not only a saving:
    the point of comparing against MDpocket is that it analyses the *same*
    conformers Lacuna does, so any difference is the analysis pipeline rather
    than the sampler. Generating it twice would leave that on trust.
    """
    structure: object
    conformers: int
    seed: int
    cif: object = None
    chain: object = None
    _ensemble: object = None
    #: Set by the Lacuna detectors so the writer can record ranker
    #: features in the same row as the residue lists they describe.
    last_features: object = None

    def ensemble(self):
        """[input coords] + N generated conformers, cached per structure.

        Built exactly as cryptic_benchmark._make_backend builds it for the
        shipped configuration, and from the same input: the original file with
        the chain named, not a written-out single-chain PDB. Those two paths do
        not produce the same ensemble, which is how an earlier version of this
        file managed to hand Lacuna and MDpocket different conformers while a
        comment claimed they shared one.
        """
        if self._ensemble is None:
            sets = NMABackend(seed=self.seed, max_rmsd=2.0, n_modes=10).generate(
                self.cif, n_conformers=self.conformers, chain=self.chain)
            self._ensemble = [coords_array(self.structure)] + sets
        return self._ensemble


# ── detectors ──────────────────────────────────────────────────────────────────
# Each returns a rank-ordered list of residue-label lists, best first.

def _fpocket(cif: Path, chain: str, ctx: Ctx):
    return [p["residues"] for p in run_fpocket(cif, chain)]


def _p2rank(cif: Path, chain: str, ctx: Ctx):
    from lacuna.pockets.p2rank_detector import run_p2rank
    return [p["residues"] for p in run_p2rank(cif, chain)]


def _lacuna(cif: Path, chain: str, ctx: Ctx, rank_by="learned"):
    """Lacuna's pipeline, driven from the shared ensemble.

    This is cryptic_benchmark.run_lacuna's body with the ensemble injected
    rather than generated internally, so MDpocket provably sees the same
    conformers. Detection, clustering and ranking are the shipped code.
    """
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets

    all_coords = ctx.ensemble()
    pocket_lists = []
    for ci, coords in enumerate(all_coords):
        pockets = detect_pockets(coords, ctx.structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)

    plm_probs = None
    if rank_by == "learned-plm":
        from lacuna.pockets import plm as _plm
        plm_probs = _plm.residue_probabilities(ctx.structure)

    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords),
                               rank_by=rank_by, plm_residue_probs=plm_probs)
    # Side channel rather than a changed return type: the other five detectors
    # have no features to give, and widening the registry contract for them
    # would mean touching code that currently works. Features and residue lists
    # must land in the same row or they cannot be joined later: the two
    # collection runs agree on only 3.3% of Jaccard vectors, so pairing a
    # feature file against a residue file after the fact is not sound.
    from lacuna.pockets.clusterer import ranker_features
    ctx.last_features = [ranker_features(c) for c in clusters]
    return [c.lining_residues for c in clusters]


def _lacuna_plm(cif: Path, chain: str, ctx: Ctx):
    return _lacuna(cif, chain, ctx, rank_by="learned-plm")


def _mdpocket(cif: Path, chain: str, ctx: Ctx):
    """MDpocket over the same NMA ensemble Lacuna receives."""
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        dx = run_mdpocket(ctx.structure, ctx.ensemble(), wd)
        vals, origin, spacing = parse_dx(dx)
    pockets = grid_pockets(vals, origin, spacing, MDPOCKET_ISOVALUE)
    key = ((lambda p: len(p[1])) if MDPOCKET_RANKING == "volume"
           else (lambda p: len(p[1]) * p[2]))
    index = lining_index(ctx.structure)
    ranked = sorted(pockets, key=key, reverse=True)
    return [["%d:%s" % (n, chain) for n in sorted(lining_residues(index, p[1]))]
            for p in ranked]


def _ifsitepred(cif: Path, chain: str, ctx: Ctx):
    """IF-SitePred, via its own two interpreters. See ifsitepred_runner."""
    from ifsitepred_runner import run_ifsitepred, IFSitePredUnavailable
    target = "%s_%s" % (Path(cif).stem, chain)
    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / ("%s.pdb" % target)
        write_structure_pdb(ctx.structure, pdb)
        try:
            return run_ifsitepred(pdb, chain, ctx.structure, target)
        except IFSitePredUnavailable as e:
            raise NotImplementedError(str(e))


#: name -> (callable, note). Extend here.
DETECTORS = {
    "fpocket":     (_fpocket,    "geometric, alpha spheres on a Voronoi tessellation"),
    "p2rank":      (_p2rank,     "random forest over solvent-accessible surface points"),
    "lacuna":      (_lacuna,     "ensemble, default learned ranker"),
    "lacuna_plm":  (_lacuna_plm, "ensemble, PLM-assisted ranker"),
    "mdpocket":    (_mdpocket,   "ensemble density grid, same conformers as Lacuna"),
    "ifsitepred":  (_ifsitepred, "ESM-IF1 residue scoring, WSL only"),
}


# ── geometry ───────────────────────────────────────────────────────────────────

def _ca_index(structure):
    """resnum -> CA coordinate, for the chain the structure was loaded with."""
    out = {}
    for atom in structure.atoms:
        if atom.name == "CA":
            out.setdefault(atom.res_seq, atom.coords)
    return out


def _centroid(residue_labels, ca):
    """Mean CA coordinate of a candidate's lining residues.

    Returns None when none of the residues resolve, which happens for a detector
    that reports a residue the loaded chain does not contain. Recording None is
    deliberate: a silently wrong centroid would corrupt every distance downstream.
    """
    pts = [ca[n] for n in _parse_resnums(residue_labels) if n in ca]
    if not pts:
        return None
    return [round(float(v), 3) for v in np.mean(np.asarray(pts), axis=0)]


def _load_done(path: Path):
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["tool"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", default="fpocket,p2rank",
                    help="comma-separated; see --list")
    ap.add_argument("--folds", default="test",
                    help="'test', 'train', or comma-separated CryptoBench folds")
    ap.add_argument("--conformers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42,
                    help="NMA seed; 42 is the shipped configuration")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--list", action="store_true", help="list detectors and exit")
    a = ap.parse_args()

    if a.list:
        print("detectors:")
        for name, (_fn, note) in DETECTORS.items():
            print("  %-12s %s" % (name, note))
        return

    tools = [t.strip() for t in a.tools.split(",") if t.strip()]
    unknown = [t for t in tools if t not in DETECTORS]
    if unknown:
        raise SystemExit("unknown detector(s): %s. Try --list." % ", ".join(unknown))

    dataset = json.loads(_fetch("dataset.json").read_text())
    folds = json.loads(_fetch("folds.json").read_text())
    if a.folds == "train":
        want = [f for f in folds if f.startswith("train-")]
    elif a.folds == "test":
        want = ["test"]
    else:
        want = [f.strip() for f in a.folds.split(",")]
    ids = sorted({p for f in want for p in folds[f]})
    if a.limit:
        ids = ids[:a.limit]

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("candidates_%s.jsonl" % a.folds.replace(",", "_"))
    done = _load_done(out_path)
    print("%d structures, tools=%s, %d rows already collected -> %s\n"
          % (len(ids), ",".join(tools), len(done), out_path))

    n_skip = 0
    tally = {t: [0, 0] for t in tools}
    t0 = time.perf_counter()

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
            sid = "%s%s" % (apo, chain)
            todo = [t for t in tools if (sid, t) not in done]
            if not todo:
                continue

            try:
                cif = download_cif(apo)
                structure = load_structure(cif, chain=chain)
                if not (10 <= len(structure.residues) <= a.max_residues):
                    n_skip += 1
                    continue
            except Exception as e:
                n_skip += 1
                print("  [%d/%d] [skip] %s: %s" % (i, len(ids), sid, type(e).__name__),
                      flush=True)
                continue

            ca = _ca_index(structure)
            site_centroid = _centroid(["%d:%s" % (n, chain) for n in known], ca)
            ctx = Ctx(structure=structure, conformers=a.conformers, seed=a.seed,
                      cif=cif, chain=chain)

            for tool in todo:
                fn = DETECTORS[tool][0]
                t_tool = time.perf_counter()
                ctx.last_features = None      # stale features would misattribute
                try:
                    residue_lists = fn(cif, chain, ctx)
                except NotImplementedError as e:
                    print("  %s: %s" % (tool, e), flush=True)
                    tools = [t for t in tools if t != tool]
                    continue
                except Exception as e:
                    print("  [%d/%d] %s %s: ERROR %s: %s"
                          % (i, len(ids), sid, tool, type(e).__name__, str(e)[:80]),
                          flush=True)
                    continue
                elapsed = time.perf_counter() - t_tool

                jacs = [round(residue_jaccard(r, known), 4) for r in residue_lists]
                best5 = max(jacs[:5], default=0.0)
                row = {
                    # identical to the existing schema
                    "id": sid, "pdb": apo, "chain": chain,
                    "n_known": len(known), "tool": tool,
                    "recall": round(max((residue_overlap(r, known)
                                         for r in residue_lists[:5]), default=0.0), 3),
                    "jaccard": round(best5, 3),
                    "n_prop": len(residue_lists),
                    "elapsed_s": round(elapsed, 2),
                    "jac_by_rank": jacs,
                    # new: what makes cross-detector work possible
                    "residues_by_rank": residue_lists,
                    "centroid_by_rank": [_centroid(r, ca) for r in residue_lists],
                    "site_centroid": site_centroid,
                    "site_residues": sorted(known),
                    "features_by_rank": ctx.last_features,
                    "schema": 2,
                    "conformers": a.conformers if tool.startswith("lacuna") else None,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                done.add((sid, tool))
                tally[tool][1] += 1
                tally[tool][0] += int(best5 >= JACCARD_THRESHOLD)

            summary = "  ".join("%s=%d/%d" % (t, tally[t][0], tally[t][1])
                                for t in tools if tally[t][1])
            print("  [%d/%d] %s (%d res)  %s" % (i, len(ids), sid, len(known), summary),
                  flush=True)

    print("-" * 70)
    print("  %d skipped, %.1f min" % (n_skip, (time.perf_counter() - t0) / 60))
    print("  rows -> %s" % out_path)


if __name__ == "__main__":
    main()
