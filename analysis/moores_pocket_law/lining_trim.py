"""Would trimming a candidate's lining residues recover the small-site gap?

The detection diagnosis says Lacuna is finding these sites and failing the
overlap test rather than missing them. Of the recoverable misses with a proposal
within 8 A of the annotation, 87% score Jaccard between 0.15 and 0.25, just
under the bar, and the missed sites have a median of 7 residues against 16 for
covered ones.

Jaccard is intersection over union, so residue-set size is not neutral. A
20-residue proposal that captures all 7 residues of a small site scores 7/20 =
0.35; capture 5 of them and it scores 5/22 = 0.23. Over-proposing sinks a
correctly located pocket on a small site, and only on a small site.

If that is the mechanism, trimming is close to free. It changes no candidate
counts, so the candidate-burden effect that dominates the ranking results does
not apply, and conversion cannot fall for that reason.

But trimming cannot be evaluated only where it helps. A tighter lining set will
also break targets that currently qualify with a large, generous proposal
against a large site. So every rule here is applied globally, to every candidate
of every target, and scored on net coverage. A rule that gains 40 small sites
and loses 45 large ones is a loss.

Two families, both keyed on the pocket centroid Lacuna already computes:

    top-k     keep the k residues closest to the centroid
    radius    keep residues whose CA sits within d angstroms of it

Coverage is the fraction of targets with any candidate at Jaccard >= 0.25.
Intervals are paired bootstrap over targets against the untrimmed baseline.

    python analysis/moores_pocket_law/lining_trim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
from dataio import JACCARD_THRESHOLD as T

CAND = HERE / "data" / "candidates" / "candidates_train.jsonl"
CIF = REPO / "benchmarks" / "cb_data" / "cif"
CACHE = HERE / "data" / "ca_coords"
SEED = 0
TOPK = (6, 8, 10, 12, 15, 20)
RADII = (5.0, 6.0, 7.0, 8.0, 10.0)


def ca_cache(recs):
    """resnum -> CA coordinate per structure, cached. Resumable."""
    from lacuna.io.structure import load_structure
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recs if not (CACHE / ("%s.npz" % r["id"])).exists()]
    print("CA cache: %d present, %d to compute" % (len(recs) - len(todo), len(todo)))
    for i, r in enumerate(todo, 1):
        try:
            st = load_structure(CIF / ("%s.cif" % r["pdb"].upper()), chain=r["chain"])
            nums, xyz = [], []
            for a in st.atoms:
                if a.name == "CA":
                    nums.append(a.res_seq)
                    xyz.append(a.coords)
            np.savez_compressed(CACHE / ("%s.npz" % r["id"]),
                                nums=np.asarray(nums, np.int32),
                                xyz=np.asarray(xyz, np.float32))
        except Exception as exc:                          # noqa: BLE001
            if i <= 3:
                print("  skip %s: %s" % (r["id"], exc))
        if i % 150 == 0 or i == len(todo):
            print("  %4d/%d" % (i, len(todo)))


def _resnum(label):
    s = str(label)
    if ":" in s:
        s = s.rsplit(":", 1)[0]
    d = "".join(ch for ch in s if ch.isdigit() or ch == "-")
    return int(d) if d else None


def jac(pred, known):
    if not pred:
        return 0.0
    p, k = set(pred), set(known)
    return len(p & k) / len(p | k)


def main() -> None:
    folds = dataio.fold_map()
    recs = []
    with open(CAND, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("tool") != "lacuna":
                continue
            f = folds.get(dataio.pdb_id_of(r["id"]))
            if f is None or not f.startswith("train"):
                continue
            if not (r.get("residues_by_rank") and r.get("site_residues")):
                continue
            recs.append(r)
    print("targets: %d" % len(recs))
    ca_cache(recs)

    # Per target: candidate residue lists as numbers, with their CA distances
    # to the candidate's own centroid, plus the annotated site.
    targets = []
    for r in recs:
        f = CACHE / ("%s.npz" % r["id"])
        if not f.exists():
            continue
        z = np.load(f)
        pos = {int(n): xyz for n, xyz in zip(z["nums"], z["xyz"])}
        known = {int(x) for x in r["site_residues"]}
        cands = []
        for res, cent in zip(r["residues_by_rank"], r["centroid_by_rank"]):
            nums = [n for n in (_resnum(x) for x in res) if n is not None and n in pos]
            if not nums or not cent:
                continue
            d = np.linalg.norm(np.asarray([pos[n] for n in nums], float)
                               - np.asarray(cent, float), axis=1)
            order = np.argsort(d)
            cands.append(([nums[i] for i in order], d[order]))
        if cands:
            targets.append({"id": r["id"], "cands": cands, "known": known,
                            "n_site": len(known)})
    print("usable targets: %d\n" % len(targets))

    def coverage(rule):
        hit = []
        for t in targets:
            best = 0.0
            for nums, d in t["cands"]:
                keep = rule(nums, d)
                if keep:
                    best = max(best, jac(keep, t["known"]))
            hit.append(int(best >= T))
        return np.array(hit)

    base = coverage(lambda nums, d: nums)
    rules = [("untrimmed", base)]
    for k in TOPK:
        rules.append(("top-%d closest" % k,
                      coverage(lambda nums, d, k=k: nums[:k])))
    for rr in RADII:
        rules.append(("within %.0f A" % rr,
                      coverage(lambda nums, d, rr=rr: [n for n, dd in zip(nums, d) if dd <= rr])))

    rng = np.random.default_rng(SEED)
    n = len(targets)
    bi = rng.integers(0, n, size=(20000, n))

    print("%-18s %9s %8s %-22s" % ("rule", "coverage", "delta", "95% CI"))
    print("-" * 62)
    out = {}
    for name, v in rules:
        d = v - base
        if name == "untrimmed":
            print("%-18s %8.1f%%        -" % (name, 100 * v.mean()))
            out[name] = {"coverage": float(v.mean())}
            continue
        bm = np.sort(d[bi].mean(axis=1))
        lo, hi = 100 * bm[500], 100 * bm[19499]
        star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
        print("%-18s %8.1f%% %+7.1f  [%+5.1f, %+5.1f]%s"
              % (name, 100 * v.mean(), 100 * d.mean(), lo, hi, star))
        out[name] = {"coverage": float(v.mean()), "delta": float(d.mean()),
                     "ci": [lo / 100, hi / 100]}

    # Where any gain comes from, split by annotated site size.
    best = max((r for r in rules[1:]), key=lambda r: r[1].mean())
    small = np.array([t["n_site"] <= 8 for t in targets])
    print("\nbest rule: %s" % best[0])
    for lab, m in (("sites <= 8 residues", small), ("sites > 8 residues", ~small)):
        d = (best[1] - base)[m]
        print("  %-22s n=%3d  %+.1f points" % (lab, int(m.sum()), 100 * d.mean()))

    (HERE / "lining_trim.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "lining_trim.json"))


if __name__ == "__main__":
    main()
