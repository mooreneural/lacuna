"""Can a learned surface scorer find the sites the geometry filter throws away?

`pool_density.py` moved the problem. Among covered targets the ranker already
puts the true site first 57.3% of the time and inside the top five 86.3% of the
time, so ranking is not where Lacuna's remaining loss lives. Coverage is: the
alpha detector proposes nothing qualifying on 27.7% of train-fold targets,
while union coverage across detectors reaches 92.2%. Roughly twenty points sit
in candidates other detectors find and this one does not.

The alpha detector only proposes at concavities. Anything shallow, flat, or not
yet open is discarded before ranking ever sees it. A learned scorer over
surface points has no such filter: it proposes wherever the model says
ligandable. That is what P2Rank does, and porting P2Rank is not the interesting
version of this, because the geometry engine here already computes most of what
such a scorer needs. `_GridContext` gives distance to nearest atom, local
density, and bulk_depth (how deeply a voxel is recessed, measured against a
rolling-probe definition of bulk solvent) for every voxel, once per conformer.

So this trains a point classifier on those fields plus physicochemical
aggregates, and asks one question, on the population that matters:

    On targets where the alpha detector currently has ZERO qualifying
    candidate, does a learned scorer put a cluster at the annotated site?

Only that subset is scored. Improving targets that already work is not the
question, and averaging over all targets would hide the answer.

Two things keep this honest. Cross-fitting is leave-one-fold-out over
CryptoBench's four homology-separated train folds, so no structure is scored by
a model trained on its homologues. And proposed clusters are characterised by
``characterize_pockets``, Lacuna's own code, so the residue sets compared
against the annotation are on exactly the scale every other number uses.

Protocol: train folds only. The designated test fold is never read here.

    python analysis/moores_pocket_law/surface_scorer.py --limit 60
    python analysis/moores_pocket_law/surface_scorer.py
"""
from __future__ import annotations

import argparse
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

CIF = REPO / "benchmarks" / "cb_data" / "cif"
DATASET = REPO / "benchmarks" / "cb_data" / "dataset.json"
CACHE = HERE / "data" / "surface_feat"

#: Voxels this far from the nearest heavy atom form the probe-accessible shell.
#: Below 1.4 A is inside the van der Waals surface; beyond 4.0 A is bulk solvent.
SHELL_LO, SHELL_HI = 1.4, 4.0
#: A point counts as positive if it is this close to any atom of an annotated
#: site residue. Ligand heavy atoms sit 3 to 4 A off the atoms lining them.
POS_RADIUS = 5.0
#: Neighbourhood for the physicochemical aggregates.
NBR_RADIUS = 8.0
MAX_POINTS = 4000        # per structure, subsampled; keeps the run tractable
NEG_PER_POS = 8          # training-set balance, all positives kept

FEATS = ("dist", "density", "bulk_depth", "radial", "n_nbr", "n_res",
         "f_hydrophobic", "f_aromatic", "f_polar", "f_carbon", "atom_density")


def _point_features(coords, structure, ctx, pts_xyz):
    """Per-point geometry from the grid context, plus atom-neighbourhood chemistry."""
    from lacuna.io.structure import is_hydrophobic, is_aromatic

    idx = np.round((pts_xyz - ctx.lo) / ctx.grid_spacing).astype(int)
    idx = np.clip(idx, 0, np.asarray(ctx.shape) - 1)
    g = (idx[:, 0], idx[:, 1], idx[:, 2])

    radial = (np.linalg.norm(pts_xyz - ctx.protein_centroid, axis=1)
              / max(ctx.radius_gyration, 1e-6))

    # One KD-tree query for every point, then aggregate atom properties per
    # neighbourhood. Building the per-atom property arrays once outside the loop
    # is what keeps this from dominating the runtime.
    atoms = structure.atoms
    hyd = np.array([is_hydrophobic(a.res_name) for a in atoms], bool)
    aro = np.array([is_aromatic(a.res_name) for a in atoms], bool)
    pol = np.array([a.element.upper() in ("N", "O") for a in atoms], bool)
    car = np.array([a.element.upper() == "C" for a in atoms], bool)
    res_key = np.array([hash((a.chain_id, a.res_seq)) for a in atoms])

    nbrs = ctx.atom_tree.query_ball_point(pts_xyz, NBR_RADIUS)
    n_nbr = np.zeros(len(pts_xyz)); n_res = np.zeros(len(pts_xyz))
    f_h = np.zeros(len(pts_xyz)); f_a = np.zeros(len(pts_xyz))
    f_p = np.zeros(len(pts_xyz)); f_c = np.zeros(len(pts_xyz))
    for i, nb in enumerate(nbrs):
        if not nb:
            continue
        nb = np.asarray(nb)
        n = len(nb)
        n_nbr[i] = n
        n_res[i] = len(np.unique(res_key[nb]))
        f_h[i] = hyd[nb].mean(); f_a[i] = aro[nb].mean()
        f_p[i] = pol[nb].mean(); f_c[i] = car[nb].mean()

    return np.column_stack([
        ctx.dist[g], ctx.local_density[g], ctx.bulk_depth[g], radial,
        n_nbr, n_res, f_h, f_a, f_p, f_c,
        n_nbr / (4.0 / 3.0 * np.pi * NBR_RADIUS ** 3),
    ]).astype(np.float32)


def build_target(pdb: str, entry: dict, rng) -> dict | None:
    """Surface points, features and labels for one structure. Cached to disk."""
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.pockets.detector import _build_grid_context, GRID_SPACING

    chain = entry["apo_chain"]
    structure = load_structure(CIF / ("%s.cif" % pdb.upper()), chain=chain)
    coords = coords_array(structure)
    ctx = _build_grid_context(coords, structure, GRID_SPACING)

    shell = (ctx.dist >= SHELL_LO) & (ctx.dist <= SHELL_HI)
    vox = np.argwhere(shell)
    if len(vox) < 50:
        return None
    if len(vox) > MAX_POINTS:
        vox = vox[rng.permutation(len(vox))[:MAX_POINTS]]
    xyz = ctx.lo + vox * ctx.grid_spacing

    # Positive = near an atom of an annotated site residue.
    want = set()
    for e in entry["apo_pocket_selection"]:
        ch, _, num = str(e).partition("_")
        want.add((ch, int(num)))
    site_xyz = np.array([a.coords for a in structure.atoms
                         if (a.chain_id, a.res_seq) in want], float)
    if len(site_xyz) < 3:
        return None
    from scipy.spatial import cKDTree
    d, _ = cKDTree(site_xyz).query(xyz, k=1)
    y = (d <= POS_RADIUS).astype(np.int8)
    if y.sum() < 5:
        return None

    return {"X": _point_features(coords, structure, ctx, xyz),
            "y": y, "xyz": xyz.astype(np.float32)}


def cached(pdb, entry, rng):
    f = CACHE / ("%s.npz" % pdb)
    if f.exists():
        z = np.load(f)
        return {"X": z["X"], "y": z["y"], "xyz": z["xyz"]}
    d = build_target(pdb, entry, rng)
    if d is None:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **d)
    return d


def zero_coverage_ids() -> set:
    """Train-fold targets where the alpha detector proposes nothing qualifying."""
    folds = dataio.fold_map()
    out = set()
    for rec in dataio._jsonl(dataio.SWEEP[20]):
        fold = folds.get(dataio.pdb_id_of(rec["id"]))
        if fold is None or not fold.startswith("train"):
            continue
        cl = rec.get("clusters") or []
        if cl and max(float(c.get("jac", 0.0)) for c in cl) < T:
            out.add(dataio.pdb_id_of(rec["id"]))
    return out


def recover_zero_coverage(data, scores, zero, ent):
    """Does a proposal land on the annotated site, on targets with no coverage?

    Kept as one function so any arm scoring the same question scores it the same
    way: cluster the top 2% of points, keep five proposals by summed score, and
    characterise them with Lacuna's own code so the residue sets are on the scale
    every other number in this project uses.
    """
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.pockets.detector import characterize_pockets
    from lacuna.pockets.clusterer import _greedy_cluster

    hits, checked = [], []
    for p in sorted(scores):
        if p.lower() not in zero:
            continue
        d, s = data[p], scores[p]
        top = np.flatnonzero(s >= np.quantile(s, 0.98))
        if len(top) < 3:
            continue
        lab = _greedy_cluster(d["xyz"][top].astype(float), eps=5.0)
        centres = [d["xyz"][top][lab == c].mean(axis=0)
                   for c in range(lab.max() + 1)]
        strength = [float(s[top][lab == c].sum()) for c in range(lab.max() + 1)]
        centres = [c for _v, c in sorted(zip(strength, centres),
                                         key=lambda kv: -kv[0])[:5]]
        e = ent[p]
        st = load_structure(CIF / ("%s.cif" % p.upper()), chain=e["apo_chain"])
        pocks = characterize_pockets(coords_array(st), st, centres)
        want = {(str(x).partition("_")[0], int(str(x).partition("_")[2]))
                for x in e["apo_pocket_selection"]}
        best = 0.0
        for pk in pocks:
            if pk is None:
                continue
            got = set()
            for r in pk.lining_residues:
                head, _, ch = str(r).rpartition(":")
                num = "".join(c for c in head if c.isdigit() or c == "-")
                if num:
                    got.add((ch, int(num)))
            if got:
                best = max(best, len(got & want) / len(got | want))
        checked.append(p)
        hits.append(int(best >= T))
    return hits, checked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import lightgbm as lgb

    folds = dataio.fold_map()
    ds = json.loads(DATASET.read_text())
    zero = zero_coverage_ids()

    targets = []
    for pdb, entries in ds.items():
        f = folds.get(pdb.lower())
        if f is None or not f.startswith("train"):
            continue
        e = entries[0] if isinstance(entries, list) else entries
        if not (e.get("apo_chain") and e.get("apo_pocket_selection")):
            continue
        if not (CIF / ("%s.cif" % pdb.upper())).exists():
            continue
        targets.append((pdb, e, f))
    targets.sort()
    if a.limit:
        # Keep the zero-coverage cases in a pilot: they are what is being scored.
        z = [t for t in targets if t[0].lower() in zero][:a.limit // 2]
        o = [t for t in targets if t[0].lower() not in zero][:a.limit - len(z)]
        targets = sorted(z + o)

    n_zero = sum(1 for p, _e, _f in targets if p.lower() in zero)
    print("train-fold targets: %d   zero-coverage: %d (%.1f%%)\n"
          % (len(targets), n_zero, 100 * n_zero / max(len(targets), 1)))

    rng = np.random.default_rng(0)
    data, fail, t0 = {}, 0, time.time()
    for i, (pdb, e, f) in enumerate(targets, 1):
        try:
            d = cached(pdb, e, rng)
            if d is not None:
                d["fold"] = f
                data[pdb] = d
        except Exception as exc:                        # noqa: BLE001
            fail += 1
            if fail <= 5:
                print("  skip %s: %s" % (pdb, exc), flush=True)
        if i % 25 == 0 or i == len(targets):
            r = (time.time() - t0) / i
            print("  %4d/%d  ok=%d fail=%d  %.2f s/target  eta %.0f min"
                  % (i, len(targets), len(data), fail, r,
                     r * (len(targets) - i) / 60), flush=True)

    if len(data) < 40:
        raise SystemExit("only %d structures featurised; too few to conclude "
                         "anything." % len(data))

    # ── cross-fitted point classifier ────────────────────────────────────────
    scores = {}
    for held in sorted({d["fold"] for d in data.values()}):
        tr = [p for p, d in data.items() if d["fold"] != held]
        te = [p for p, d in data.items() if d["fold"] == held]
        Xs, ys = [], []
        for p in tr:
            d = data[p]
            pos = np.flatnonzero(d["y"] == 1)
            neg = np.flatnonzero(d["y"] == 0)
            # All positives, a bounded sample of negatives: the shell is
            # overwhelmingly negative and an unbalanced fit predicts zero.
            take = rng.permutation(neg)[:NEG_PER_POS * max(len(pos), 1)]
            keep = np.concatenate([pos, take])
            Xs.append(d["X"][keep]); ys.append(d["y"][keep])
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=63, min_child_samples=50,
                               subsample=0.9, colsample_bytree=0.8,
                               random_state=0, verbose=-1)
        m.fit(np.vstack(Xs), np.concatenate(ys))
        for p in te:
            scores[p] = m.predict_proba(data[p]["X"])[:, 1]

    from sklearn.metrics import roc_auc_score, average_precision_score
    ally = np.concatenate([data[p]["y"] for p in scores])
    alls = np.concatenate([scores[p] for p in scores])
    print("\nper-point classifier, cross-fitted over %d folds"
          % len({d["fold"] for d in data.values()}))
    print("  AUC %.3f   average precision %.3f   (base rate %.3f)"
          % (roc_auc_score(ally, alls), average_precision_score(ally, alls),
             ally.mean()))

    ent = {p: e for p, e, _f in targets}
    hits, checked = recover_zero_coverage(data, scores, zero, ent)

    if not checked:
        raise SystemExit("no zero-coverage targets scored; nothing to conclude.")

    hits = np.array(hits, float)
    n = len(hits)
    rng2 = np.random.default_rng(0)
    bm = np.sort(hits[rng2.integers(0, n, size=(20000, n))].mean(axis=1))
    lo, hi = 100 * bm[500], 100 * bm[19499]
    print("\nzero-coverage targets scored: %d" % n)
    print("recovered at top-5 (Jaccard >= %.2f): %d  (%.1f%%  [%.1f, %.1f])"
          % (T, int(hits.sum()), 100 * hits.mean(), lo, hi))

    print("\n%s" % ("=" * 66))
    if lo > 5.0:
        print("The learned scorer reaches sites the geometry filter discards.")
        print("These targets contribute zero coverage today, so this is")
        print("headroom the current detector cannot reach at any ranking depth.")
    elif hi < 5.0:
        print("It does not find them either. The misses are not a matter of the")
        print("concavity filter being too strict, so a surface scorer is not")
        print("the way to close the coverage gap.")
    else:
        print("Not resolved at this n. Run without --limit before deciding.")

    (HERE / "surface_scorer.json").write_text(json.dumps(
        {"n_structures": len(data), "n_zero_scored": n,
         "recovered": float(hits.mean()), "ci": [lo, hi],
         "auc": float(roc_auc_score(ally, alls)),
         "ap": float(average_precision_score(ally, alls))}, indent=1))
    print("\nwrote %s" % (HERE / "surface_scorer.json"))


if __name__ == "__main__":
    main()
