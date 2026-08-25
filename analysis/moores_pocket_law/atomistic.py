"""Route 1: atom-level descriptors of the pocket lining, no ML.

Nothing atomic reaches Lacuna's ranker today. The 23 geometric features are
pocket-level scalars, and the two chemistry features among them, `hyd` and
`aro`, are counts of residue *names*: a leucine contributes the same whether its
sidechain faces the cavity or points away, and a tyrosine counts as aromatic
whether or not its ring is anywhere near the pocket.

This computes the descriptors that distinction implies, from the atoms of the
lining residues rather than their names: element composition, hydrogen bond
donors and acceptors, formal charge, aromatic ring atoms, sidechain versus
backbone, per-atom burial, and the shape of the atom cloud itself.

Burial is a neighbour count within 8 A rather than a solvent accessible surface.
Lacuna carries no SASA code, and a count is monotone in accessibility, which is
all a ranker needs from it.

The question is not whether these features predict anything on their own, but
whether they add to what the language model already knows. So three models are
compared under one cross-fitting protocol: atoms alone, sequence alone, and both.
If atoms carry nothing ESM-2 has not already inferred, the third will not beat
the second, and route 1 is finished cheaply.

Protocol: train folds only, leave-one-fold-out over the four homology-separated
train folds. The designated test fold is never read.

    python analysis/moores_pocket_law/atomistic.py --limit 60   # smoke
    python analysis/moores_pocket_law/atomistic.py              # full
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
from dataio import JACCARD_THRESHOLD as T
import plm_headroom as PH

warnings.filterwarnings("ignore")

CAND = HERE / "data" / "candidates" / "candidates_train.jsonl"
CIF = REPO / "benchmarks" / "cb_data" / "cif"
CACHE = HERE / "data" / "atom_feat"
KS = (1, 5, 10)
SEED = 0

BACKBONE = {"N", "CA", "C", "O", "OXT"}
#: Sidechain atoms that donate a hydrogen bond.
DONORS = {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
          ("ASN", "ND2"), ("GLN", "NE2"), ("TRP", "NE1"), ("LYS", "NZ"),
          ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
          ("HIS", "ND1"), ("HIS", "NE2")}
#: Sidechain atoms that accept one.
ACCEPTORS = {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("ASN", "OD1"),
             ("GLN", "OE1"), ("ASP", "OD1"), ("ASP", "OD2"),
             ("GLU", "OE1"), ("GLU", "OE2"), ("HIS", "ND1"), ("HIS", "NE2")}
POS = {("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ"),
       ("HIS", "ND1"), ("HIS", "NE2")}
NEG = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
RING = {("PHE", n) for n in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")} | \
       {("TYR", n) for n in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")} | \
       {("TRP", n) for n in ("CG", "CD1", "CD2", "NE1", "CE2", "CE3",
                             "CZ2", "CZ3", "CH2")} | \
       {("HIS", n) for n in ("CG", "ND1", "CD2", "CE1", "NE2")}

NAMES = ["n_atoms", "f_C", "f_N", "f_O", "f_S", "polar_frac", "donors",
         "acceptors", "hb_density", "charge_pos", "charge_neg", "net_charge",
         "arom_atoms", "arom_frac", "sidechain_frac", "burial_mean",
         "burial_p10", "burial_p90", "rg", "d_cent_mean", "d_cent_std",
         "aniso", "planarity"]


def _feats(xyz, elems, keys, burial, centroid):
    """Atom-level descriptors for one pocket lining."""
    n = len(xyz)
    if n < 3:
        return None
    el = np.asarray(elems)
    f = {"n_atoms": float(n)}
    for e in ("C", "N", "O", "S"):
        f["f_%s" % e] = float((el == e).mean())
    f["polar_frac"] = float(np.isin(el, ["N", "O"]).mean())

    don = sum(1 for k in keys if k in DONORS)
    acc = sum(1 for k in keys if k in ACCEPTORS)
    f["donors"], f["acceptors"] = float(don), float(acc)
    f["hb_density"] = float((don + acc) / n)
    p = sum(1 for k in keys if k in POS)
    q = sum(1 for k in keys if k in NEG)
    f["charge_pos"], f["charge_neg"], f["net_charge"] = float(p), float(q), float(p - q)
    ar = sum(1 for k in keys if k in RING)
    f["arom_atoms"], f["arom_frac"] = float(ar), float(ar / n)
    f["sidechain_frac"] = float(np.mean([k[1] not in BACKBONE for k in keys]))

    f["burial_mean"] = float(np.mean(burial))
    f["burial_p10"] = float(np.percentile(burial, 10))
    f["burial_p90"] = float(np.percentile(burial, 90))

    c = xyz.mean(0)
    f["rg"] = float(np.sqrt(((xyz - c) ** 2).sum(1).mean()))
    d = np.linalg.norm(xyz - np.asarray(centroid), axis=1)
    f["d_cent_mean"], f["d_cent_std"] = float(d.mean()), float(d.std())
    # Shape of the lining cloud: eigenvalues of its covariance.
    ev = np.sort(np.linalg.eigvalsh(np.cov((xyz - c).T)))[::-1]
    ev = np.maximum(ev, 1e-9)
    f["aniso"] = float(ev[0] / ev[2])
    f["planarity"] = float(ev[2] / ev.sum())
    return [f[k] for k in NAMES]


def build(recs) -> None:
    """Atom features per structure, cached. Resumable."""
    from lacuna.io.structure import load_structure

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recs if not (CACHE / ("%s.npz" % r["id"])).exists()]
    print("atom-feature cache: %d present, %d to compute"
          % (len(recs) - len(todo), len(todo)))
    ok = fail = 0
    for i, r in enumerate(todo, 1):
        try:
            st = load_structure(CIF / ("%s.cif" % r["pdb"].upper()), chain=r["chain"])
            xyz = np.array([a.coords for a in st.atoms], dtype=float)
            tree = cKDTree(xyz)
            burial = np.array([len(tree.query_ball_point(p, 8.0)) - 1 for p in xyz],
                              dtype=float)
            by_label = {res.label: res for res in st.residues}

            rows, keep = [], []
            for rank, (res_list, cent) in enumerate(zip(r["residues_by_rank"],
                                                        r["centroid_by_rank"])):
                idx = []
                for lab in res_list:
                    res = by_label.get(str(lab))
                    if res is not None:
                        idx.extend(res.atom_indices)
                if len(idx) < 3:
                    continue
                idx = np.asarray(idx, dtype=int)
                keys = [(st.atoms[j].res_name, st.atoms[j].name) for j in idx]
                v = _feats(xyz[idx], [st.atoms[j].element for j in idx],
                           keys, burial[idx], cent)
                if v is not None:
                    rows.append(v)
                    keep.append(rank)
            if not rows:
                raise ValueError("no candidate had >=3 resolvable atoms")
            np.savez_compressed(CACHE / ("%s.npz" % r["id"]),
                                X=np.asarray(rows, dtype=np.float32),
                                rank=np.asarray(keep, dtype=np.int32))
            ok += 1
        except Exception as exc:                        # noqa: BLE001
            fail += 1
            if fail <= 5:
                print("  skip %s: %s" % (r["id"], exc))
        if i % 100 == 0 or i == len(todo):
            print("  %4d/%d  ok=%d fail=%d" % (i, len(todo), ok, fail))


def assemble(recs):
    """Rows carrying atom features, pooled embedding, label and fold."""
    rows = []
    for r in recs:
        fa, fe = CACHE / ("%s.npz" % r["id"]), PH.CACHE / ("%s.npz" % r["id"])
        if not (fa.exists() and fe.exists()):
            continue
        za, ze = np.load(fa), np.load(fe)
        emb = ze["emb"].astype(np.float32)
        pos = {int(n): i for i, n in enumerate(ze["nums"])}
        A, ranks = za["X"], za["rank"]
        for v, rank in zip(A, ranks):
            res = r["residues_by_rank"][rank]
            idx = [pos[n] for n in PH._resnums(res, r["chain"]) if n in pos]
            if not idx:
                continue
            rows.append({"id": r["id"], "fold": r["fold"], "atom": v,
                         "emb": emb[idx].mean(0),
                         "y": int(float(r["jac_by_rank"][rank]) >= T)})
    return rows


def evaluate(rows, key, name, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([key(r) for r in rows])
    y = np.array([r["y"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    tid = np.array([r["id"] for r in rows])
    s = np.zeros(len(rows))
    for held in sorted(set(fold)):
        tr, te = fold != held, fold == held
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]

    by = {}
    for t, sc, yy in zip(tid, s, y):
        by.setdefault(t, [[], []])
        by[t][0].append(sc)
        by[t][1].append(yy)
    covered = [t for t, (_a, ys) in by.items() if any(ys)]
    out = {"name": name, "dims": X.shape[1], "n_covered": len(covered)}
    for k in KS:
        hit = []
        for t in covered:
            sc, ys = by[t]
            order = np.argsort(-np.asarray(sc), kind="stable")[:k]
            hit.append(int(any(ys[i] for i in order)))
        out["conv_%d" % k] = float(np.mean(hit))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

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
            if not (r.get("residues_by_rank") and r.get("centroid_by_rank")):
                continue
            r["fold"] = f
            recs.append(r)
    recs.sort(key=lambda r: r["id"])
    if a.limit:
        recs = recs[:a.limit]
    print("structures: %d  (%s)\n"
          % (len(recs), "SMOKE limit=%d" % a.limit if a.limit else "full train folds"))

    PH.build_cache(recs)
    build(recs)

    rows = assemble(recs)
    print("\ncandidates: %d  (%d qualifying)" % (len(rows), sum(r["y"] for r in rows)))
    if len(rows) < 50:
        raise SystemExit("too few candidates assembled to mean anything")

    out = [
        evaluate(rows, lambda r: r["atom"], "atoms only", 1.0),
        evaluate(rows, lambda r: r["emb"], "sequence only (ESM-2 pooled)", 0.05),
        evaluate(rows, lambda r: np.concatenate([r["atom"], r["emb"]]),
                 "atoms + sequence", 0.05),
    ]

    print("\ncross-fitted over 4 homology-separated folds, %d covered targets\n"
          % out[0]["n_covered"])
    print("%-32s %6s %8s %8s %8s" % ("features", "dims", "conv@1", "conv@5", "conv@10"))
    print("-" * 66)
    for r in out:
        print("%-32s %6d %7.1f%% %7.1f%% %7.1f%%"
              % (r["name"], r["dims"], 100 * r["conv_1"],
                 100 * r["conv_5"], 100 * r["conv_10"]))

    d = 100 * (out[2]["conv_5"] - out[1]["conv_5"])
    print("\natoms add %+.1f points at top-5 over sequence alone" % d)
    tag = "smoke%d" % a.limit if a.limit else "full"
    (HERE / ("atomistic_%s.json" % tag)).write_text(json.dumps(out, indent=1))
    print("wrote %s" % (HERE / ("atomistic_%s.json" % tag)))


if __name__ == "__main__":
    main()
