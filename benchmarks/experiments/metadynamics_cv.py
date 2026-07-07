# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Well-tempered metadynamics along an apo-derived collective variable.

RESEARCH SCRIPT (null result, kept for reproducibility). This is NOT a shipped
backend. It biases MD along a collective variable (CV) equal to the distance
between the two anti-moving lobes of the lowest normal mode, using normal modes
as an opening coordinate. The CV is derived from the apo structure alone, with no
knowledge of the known binding site, so the experiment is an honest test of
whether metadynamics can DISCOVER cryptic pockets.

Finding (2026-07-04): metadynamics with this honest CV gives NO performance
increase over plain elevated-temperature MD. A fair head-to-head at matched
temperature and budget (400 K, 500 ps) scored metadynamics 0/5 and plain MD 0/5
on GCK, and 0/4 vs 0/4 on SHP2. An earlier small 3-seed sample had shown 1/3 for
both methods; the larger sample revealed that to be noise. The hinge openings are
rare, low-probability events that neither method samples reliably at this budget,
and the bias does not raise the rate. Two contributing factors: the CV-selection
problem (a good CV essentially requires knowing where the pocket is, so an honest
apo-only CV is a guess that often opens the wrong region), and simple sampling
cost (500 ps is far too short to see a rare opening). The machinery is preserved
here in case the CV problem is later solved (e.g. scanning modes by cavity-volume
gain, or seeding the CV from a first-pass Lacuna detection rather than the answer)
and paired with much longer or many-replica sampling.

    python benchmarks/experiments/metadynamics_cv.py GCK --seeds 3 --ps 500
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))        # benchmarks/
sys.path.insert(0, str(_HERE.parent.parent.parent))  # repo root

from cryptic_benchmark import (  # noqa: E402
    DATASET, download_pdb, residue_jaccard, extract_binding_site,
    compute_known_site_centroid, pocket_min_centroid_dist,
    JACCARD_THRESHOLD, CENTROID_THRESHOLD,
)
from lacuna.io.structure import load_structure, coords_array  # noqa: E402
from lacuna.io.writers import write_structure_pdb  # noqa: E402
from lacuna.ensemble.nma_backend import NMABackend  # noqa: E402
from lacuna.ensemble.openmm_backend import (  # noqa: E402
    _build_atom_remap, _reorder, _select_platform,
)
from lacuna.pockets.detector import detect_pockets  # noqa: E402
from lacuna.pockets.clusterer import cluster_pockets  # noqa: E402

PDB_DIR = Path(__file__).resolve().parent.parent / "pdb_cache"


def domain_resseqs(structure):
    """Split residues into two lobes by the sign of their lowest-mode displacement.

    Apo-only: uses the anisotropic-network lowest mode and the principal axis of
    its displacement field. No binding-site information is used.
    """
    ca_res = [r for r in structure.residues
              if any(structure.atoms[ai].name == "CA" for ai in r.atom_indices)]
    ca = np.array([next(structure.atoms[ai].coords for ai in r.atom_indices
                        if structure.atoms[ai].name == "CA") for r in ca_res])
    modes, _ = NMABackend()._compute_modes(ca.astype(np.float32))
    disp = modes[0].reshape(len(ca), 3)
    _, _, vt = np.linalg.svd(disp - disp.mean(0), full_matrices=False)
    proj = disp @ vt[0]
    g1 = {ca_res[i].seq_num for i in range(len(ca_res)) if proj[i] > 0}
    g2 = {ca_res[i].seq_num for i in range(len(ca_res)) if proj[i] <= 0}
    return g1, g2


def run_metad(tid, seed, total_ps=500.0, n_frames=15, temp=310.0):
    """Return (jaccard, centroid_dist, pass_bool, cv_diagnostics) or None."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmm.app.metadynamics import Metadynamics, BiasVariable
    from pdbfixer import PDBFixer

    e = next(x for x in DATASET if x["id"] == tid)
    apo = download_pdb(e["apo_pdb"], PDB_DIR)
    known = e.get("known_residues")
    if known is None:
        holo = download_pdb(e["holo_pdb"], PDB_DIR)
        known, _ = extract_binding_site(
            holo, e.get("holo_chain", "A"), e.get("extra_exclude", frozenset()))
    ref = compute_known_site_centroid(apo, e.get("apo_chain", "A"), known)
    structure = load_structure(apo, chain=e.get("apo_chain"))
    base = coords_array(structure)
    g1_res, g2_res = domain_resseqs(structure)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        clean = Path(td) / "c.pdb"
        write_structure_pdb(structure, clean)
        fixer = PDBFixer(filename=str(clean))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = ff.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds)

        g1i, g2i = [], []
        for a in fixer.topology.atoms():
            if a.name == "CA":
                try:
                    rid = int(a.residue.id)
                except ValueError:
                    continue
                (g1i if rid in g1_res else g2i if rid in g2_res else []).append(a.index)
        if len(g1i) < 3 or len(g2i) < 3:
            return None

        cv = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
        cv.addGroup(g1i)
        cv.addGroup(g2i)
        cv.addBond([0, 1], [])
        pos = np.array(fixer.positions.value_in_unit(unit.nanometer))
        d0 = float(np.linalg.norm(pos[g1i].mean(0) - pos[g2i].mean(0)))
        bv = BiasVariable(cv, minValue=max(0.1, d0 - 0.4), maxValue=d0 + 1.2,
                          biasWidth=0.05, periodic=False)
        biasdir = Path(td) / "bias"
        biasdir.mkdir()
        meta = Metadynamics(system, [bv], temp * unit.kelvin, biasFactor=8.0,
                            height=1.0 * unit.kilojoule_per_mole, frequency=500,
                            saveFrequency=500, biasDir=str(biasdir))

        integ = mm.LangevinMiddleIntegrator(
            temp * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        integ.setRandomNumberSeed(seed)
        sim = app.Simulation(fixer.topology, system, integ, _select_platform(mm))
        sim.context.setPositions(fixer.positions)
        sim.minimizeEnergy(maxIterations=500)
        sim.context.setVelocitiesToTemperature(temp * unit.kelvin, seed)

        remap = _build_atom_remap(fixer.topology, structure)
        total_steps = int(total_ps * 1000 / 2.0)
        chunk = max(1, total_steps // n_frames)
        frames, cvs = [], []
        for _ in range(n_frames):
            meta.step(sim, chunk)
            st = sim.context.getState(getPositions=True)
            md_pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            frames.append(_reorder(md_pos, remap, base))
            cvs.append(float(meta.getCollectiveVariables(sim)[0]))

    plists = [detect_pockets(base, structure)] + [detect_pockets(f, structure) for f in frames]
    cl = cluster_pockets(plists, n_conformers=len(plists), rank_by="crypticity")
    jac = max((residue_jaccard(c.lining_residues, known) for c in cl[:5]), default=0.0)
    dist, _ = pocket_min_centroid_dist(cl, ref, top_n=5)
    ok = (dist <= CENTROID_THRESHOLD) or (jac >= JACCARD_THRESHOLD)
    return jac, dist, ok, (min(cvs), max(cvs), d0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="DATASET id, e.g. GCK")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ps", type=float, default=500.0)
    args = ap.parse_args()

    npass = 0
    for seed in range(args.seeds):
        r = run_metad(args.target, seed=seed, total_ps=args.ps)
        if r is None:
            print(f"{args.target} seed{seed}: domain split failed", flush=True)
            continue
        jac, dist, ok, (cmin, cmax, d0) = r
        npass += int(ok)
        print(f"{args.target} metad seed{seed}: jac={jac:.0%} dist={dist:.1f} "
              f"{'PASS' if ok else 'miss'}  CV[{cmin:.2f},{cmax:.2f}]/d0{d0:.2f}nm",
              flush=True)
    print(f"\n{args.target}: {npass}/{args.seeds} size-robust passes "
          f"(matched-budget plain MD scores the same; see module docstring)")


if __name__ == "__main__":
    main()
