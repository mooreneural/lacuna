"""Does Lacuna's ensemble open the pocket enough to hold the real ligand?

Every number in both papers measures localisation: Jaccard overlap between
proposed lining residues and annotated site residues. That answers "did you find
the right place" and says nothing about whether the cavity produced is shaped
like the real one. For search that is enough. For structure-based design, where
the conformer becomes the receptor a molecule is docked or generated against, it
is the only thing that matters, and it has never been measured.

CryptoBench carries what the question needs: for each apo structure, the holo
PDB id and chain, the bound ligand, and matched apo/holo pocket residue
selections. So the test is direct rather than a proxy.

  1  superpose the holo chain onto the apo chain on their common CA atoms
  2  carry the ligand into the apo frame
  3  count steric clashes between the ligand and the protein, first in the apo
     structure, then in each conformer of Lacuna's ensemble

In the apo structure the pocket is closed by definition, so the ligand should
clash badly. If the ensemble genuinely opens the site, some conformer should
relieve those clashes. If none does, Lacuna is telling you where to look and not
giving you a receptor to design against, which is a materially different claim
from the one a Pocket2Mol-style integration would need.

Two controls keep this honest. Superposition uses all common CA atoms rather
than the pocket residues, which would bias the fit toward making the pocket
agree. And a random conformer is scored alongside the best one, because taking
the minimum over 20 conformers will relieve some clashes by chance.

Protocol: train folds only. The designated test fold is never read.

    python analysis/moores_pocket_law/holo_geometry.py --limit 5
    python analysis/moores_pocket_law/holo_geometry.py
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import dataio
from lacuna.io.structure import load_structure, coords_array
from lacuna.ensemble.nma_backend import NMABackend

CIF = REPO / "benchmarks" / "cb_data" / "cif"
HOLO = REPO / "benchmarks" / "holo_cache"
DATASET = REPO / "benchmarks" / "cb_data" / "dataset.json"

CLASH = 2.2        # A between heavy atoms; below this is a severe steric overlap
SOFT = 3.0         # A; below this is a mild clash, above it heavy atoms are comfortable
CONTACT = 4.5      # A; above this the ligand atom is not in contact at all
SEED = 42
N_CONF = 20


def fetch_holo(pdb: str) -> Path:
    HOLO.mkdir(exist_ok=True)
    dst = HOLO / ("%s.pdb" % pdb.upper())
    if not dst.exists():
        urllib.request.urlretrieve(
            "https://files.rcsb.org/download/%s.pdb" % pdb.upper(), dst)
    return dst


def holo_parts(path: Path, chain: str, lig: str, lig_chain: str, lig_idx: str):
    """CA atoms by residue number, and the named ligand's heavy atoms."""
    ca, lig_xyz = {}, []
    want = str(lig_idx).strip()
    for line in path.read_text(errors="replace").splitlines():
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        el = line[76:78].strip().upper()
        if el == "H":
            continue
        name, res, ch, num = (line[12:16].strip(), line[17:20].strip(),
                              line[21], line[22:26].strip())
        if rec == "ATOM  " and ch == chain and name == "CA":
            try:
                ca[int(num)] = xyz
            except ValueError:
                pass
        elif rec == "HETATM" and res.upper() == lig.upper() and ch == lig_chain \
                and num == want:
            lig_xyz.append(xyz)
    return ca, np.asarray(lig_xyz, float)


def kabsch(P, Q):
    """Rotation and translation taking P onto Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def clashes(lig, prot):
    """Ligand heavy atoms in steric overlap, severe and mild, and out of contact.

    Two thresholds because one is not enough to see the effect. A cryptic site is
    rarely filled solid; it is usually too small or the wrong shape, so the
    ligand's worst atoms sit at 2.5 to 3.0 A rather than on top of a backbone.
    Counting only severe overlaps therefore reports a pocket as nearly fitting
    when in fact nothing could bind it.
    """
    from scipy.spatial import cKDTree
    t = cKDTree(prot)
    d, _ = t.query(lig, k=1)
    return (int((d < CLASH).sum()), int((d < SOFT).sum()),
            int((d > CONTACT).sum()), float(d.min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    folds = dataio.fold_map()
    ds = json.loads(DATASET.read_text())
    targets = []
    for pdb, entries in ds.items():
        f = folds.get(pdb.lower())
        if f is None or not f.startswith("train"):
            continue
        e = entries[0] if isinstance(entries, list) else entries
        if not all(e.get(k) for k in ("holo_pdb_id", "holo_chain", "apo_chain",
                                      "ligand", "ligand_chain")):
            continue
        targets.append((pdb, e))
    targets.sort()
    if a.limit:
        targets = targets[:a.limit]
    print("train-fold targets with a holo pair: %d\n" % len(targets))

    rng = np.random.default_rng(SEED)
    rows, fail = [], 0
    for i, (pdb, e) in enumerate(targets, 1):
        try:
            apo = load_structure(CIF / ("%s.cif" % pdb.upper()), chain=e["apo_chain"])
            heavy = [k for k, at in enumerate(apo.atoms)
                     if at.element.upper() != "H"]
            apo_ca = {at.res_seq: at.coords for at in apo.atoms if at.name == "CA"}

            hpath = fetch_holo(e["holo_pdb_id"])
            hca, lig = holo_parts(hpath, e["holo_chain"], e["ligand"],
                                  e["ligand_chain"], e.get("ligand_index", ""))
            if len(lig) < 5:
                raise ValueError("ligand %s not found (%d atoms)" % (e["ligand"], len(lig)))
            common = sorted(set(apo_ca) & set(hca))
            if len(common) < 30:
                raise ValueError("only %d common CA" % len(common))

            # Superpose on all common CA, not the pocket, which would bias the fit.
            P = np.array([hca[r] for r in common], float)
            Q = np.array([apo_ca[r] for r in common], float)
            R, t = kabsch(P, Q)
            rmsd = float(np.sqrt((((P @ R.T) + t - Q) ** 2).sum(1).mean()))
            lig_apo = (lig @ R.T) + t

            base = coords_array(apo)
            confs = [base] + NMABackend(seed=SEED, max_rmsd=2.0,
                                        n_modes=10).generate(
                CIF / ("%s.cif" % pdb.upper()), n_conformers=N_CONF,
                chain=e["apo_chain"])

            per = [clashes(lig_apo, c[heavy]) for c in confs]
            apo_c, apo_soft, apo_nc, _apo_min = per[0]
            # Rank conformers on the mild threshold: it is the one that reflects
            # whether the cavity actually admits the ligand.
            best = min(range(1, len(per)), key=lambda k: per[k][1])
            rand = int(rng.integers(1, len(per)))
            rows.append({
                "pdb": pdb, "holo": e["holo_pdb_id"], "ligand": e["ligand"],
                "n_lig": int(len(lig)), "super_rmsd": rmsd,
                "apo_clash": apo_c, "best_clash": per[best][0],
                "rand_clash": per[rand][0],
                "apo_soft": apo_soft, "best_soft": per[best][1],
                "rand_soft": per[rand][1],
                "apo_nocontact": apo_nc, "best_nocontact": per[best][2],
                "best_conf": best,
            })
            if i % 25 == 0 or i == len(targets):
                print("  %4d/%d  ok=%d fail=%d" % (i, len(targets), len(rows), fail),
                      flush=True)
        except Exception as exc:                        # noqa: BLE001
            fail += 1
            if fail <= 6:
                print("  skip %s: %s" % (pdb, exc), flush=True)

    if not rows:
        raise SystemExit("nothing measured")

    A = np.array([r["apo_clash"] for r in rows], float)
    B = np.array([r["best_clash"] for r in rows], float)
    C = np.array([r["rand_clash"] for r in rows], float)
    L = np.array([r["n_lig"] for r in rows], float)

    print("\n%d targets measured, %d skipped" % (len(rows), fail))
    print("median superposition RMSD: %.2f A"
          % float(np.median([r["super_rmsd"] for r in rows])))
    print("median ligand heavy atoms: %d\n" % int(np.median(L)))

    S = np.array([r["apo_soft"] for r in rows], float)
    Sb = np.array([r["best_soft"] for r in rows], float)
    Sr = np.array([r["rand_soft"] for r in rows], float)
    print("ligand heavy atoms within %.1f A of protein (mild clash)" % SOFT)
    print("  apo structure          median %5.1f   (%.0f%% of ligand)"
          % (np.median(S), 100 * np.median(S / L)))
    print("  best of %d conformers   median %5.1f   (%.0f%% of ligand)"
          % (N_CONF, np.median(Sb), 100 * np.median(Sb / L)))
    print("  a random conformer     median %5.1f   (%.0f%% of ligand)"
          % (np.median(Sr), 100 * np.median(Sr / L)))
    for lab, d in (("best vs apo", Sb - S), ("best vs random", Sb - Sr)):
        bmm = np.sort(d[np.random.default_rng(0).integers(
            0, len(rows), size=(20000, len(rows)))].mean(axis=1))
        lo, hi = bmm[500], bmm[19499]
        star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
        print("  delta %-26s %+6.2f  [%+6.2f, %+6.2f]%s" % (lab, d.mean(), lo, hi, star))

    print("\nligand heavy atoms in severe clash (< %.1f A)" % CLASH)
    print("  apo structure          median %5.1f   (%.0f%% of ligand)"
          % (np.median(A), 100 * np.median(A / L)))
    print("  best of %d conformers   median %5.1f   (%.0f%% of ligand)"
          % (N_CONF, np.median(B), 100 * np.median(B / L)))
    print("  a random conformer     median %5.1f   (%.0f%% of ligand)"
          % (np.median(C), 100 * np.median(C / L)))

    rng2 = np.random.default_rng(0)
    bi = rng2.integers(0, len(rows), size=(20000, len(rows)))
    for lab, d in (("best vs apo", B - A), ("best vs random conformer", B - C)):
        bm = np.sort(d[bi].mean(axis=1))
        lo, hi = bm[500], bm[19499]
        star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
        print("  delta %-26s %+6.2f  [%+6.2f, %+6.2f]%s"
              % (lab, d.mean(), lo, hi, star))

    # Report both thresholds, each named. An earlier version printed only the
    # severe-threshold counts under a bare "clash-free" label, which reads as the
    # mild-threshold result the rest of this output argues from and overstates
    # how often the ligand is actually accommodated: 30.0% against 11.7%.
    for lab, apo_v, best_v in (("severe   (< %.1f A)" % CLASH, A, B),
                               ("any contact (< %.1f A)" % SOFT, S, Sb)):
        print("\n  clash-free at %s" % lab)
        print("    in some conformer : %d of %d  (%.1f%%)"
              % (int((best_v == 0).sum()), len(rows), 100 * (best_v == 0).mean()))
        print("    already in apo    : %d  (%.1f%%)"
              % (int((apo_v == 0).sum()), 100 * (apo_v == 0).mean()))

    (HERE / "holo_geometry.json").write_text(json.dumps(
        {"n": len(rows), "rows": rows}, indent=1))
    print("\nwrote %s" % (HERE / "holo_geometry.json"))


if __name__ == "__main__":
    main()
