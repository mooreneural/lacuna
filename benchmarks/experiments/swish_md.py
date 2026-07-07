# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""SWISH: scaled protein-water interactions for hydrophobic cryptic-pocket opening.

RESEARCH SCRIPT (null result, kept for reproducibility). This is NOT a shipped
backend. SWISH (Oleinikovas / Gervasio) is a cosolvent-family enhanced-sampling
method: it scales the protein-water Lennard-Jones epsilon upward so water wets
hydrophobic surfaces more readily, which helps buried hydrophobic cavities open.
It needs no cosolvent parameterization (unlike MixMD), only explicit solvent, so
it is the feasible member of the family on a workstation without the openff /
openmmforcefields stack.

Implementation: the LJ is moved into a CustomNonbondedForce that scales epsilon by
lambda for protein-water pairs only; electrostatics stay in the original
NonbondedForce. The force surgery is verified: at lambda=1.0 the potential energy
reproduces the unmodified system to ~0.03 kJ/mol out of 250000 (see --check).

Finding (2026-07-04): on T4L L99A (the textbook buried hydrophobic cavity, which
NMA and implicit MD both miss), 3 ns runs gave plain explicit solvent 0/3
(mean Jaccard 11%) and SWISH lambda=1.4 0/3 (mean Jaccard 13%). No performance
increase, and neither opens the cavity. Caveat: 3 ns is short for cosolvent MD
(published SWISH/MixMD studies run tens to hundreds of ns); this shows SWISH does
not open the cavity at the sampling feasible here, not that the method is useless.
The frontier remains compute-limited: rare openings need much longer sampling than
a single workstation can afford at benchmark scale.

    python benchmarks/experiments/swish_md.py --check          # verify force surgery
    python benchmarks/experiments/swish_md.py T4L_L99A --lam 1.4 --seeds 3 --ns 3
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer

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
from lacuna.ensemble.openmm_backend import (  # noqa: E402
    _select_platform, _build_atom_remap, _reorder,
)
from lacuna.pockets.detector import detect_pockets  # noqa: E402
from lacuna.pockets.clusterer import cluster_pockets  # noqa: E402

PDB_DIR = _HERE.parent.parent / "pdb_cache"

_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR "
          "TRP TYR VAL HID HIE HIP CYX ASH GLH LYN".split())
_WATER = {"HOH", "WAT", "TIP3", "T3P"}


def build_solvated(structure, padding_nm=0.9):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        cp = Path(td) / "c.pdb"
        write_structure_pdb(structure, cp)
        fx = PDBFixer(filename=str(cp))
        fx.findMissingResidues()
        fx.missingResidues = {}
        fx.findMissingAtoms()
        fx.addMissingAtoms()
        fx.addMissingHydrogens(7.0)
        ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        mod = app.Modeller(fx.topology, fx.positions)
        mod.addSolvent(ff, model="tip3p", padding=padding_nm * unit.nanometer,
                       ionicStrength=0.15 * unit.molar)
        system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.0 * unit.nanometer,
                                 constraints=app.HBonds)
    return system, mod.topology, mod.positions


def apply_swish(system, topology, lam):
    """Scale protein-water LJ epsilon by lam, in place. lam=1.0 is a no-op."""
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    n = topology.getNumAtoms()
    isprot = np.zeros(n)
    iswat = np.zeros(n)
    for res in topology.residues():
        fp = res.name in _AA
        fw = res.name in _WATER
        for a in res.atoms():
            isprot[a.index] = fp
            iswat[a.index] = fw

    custom = mm.CustomNonbondedForce(
        "4*eps_eff*((sig/r)^12-(sig/r)^6);"
        "eps_eff=eps*(1+(lambda-1)*(isprot1*iswat2+iswat1*isprot2));"
        "sig=0.5*(sigma1+sigma2); eps=sqrt(epsilon1*epsilon2)")
    custom.addGlobalParameter("lambda", lam)
    for p in ("sigma", "epsilon", "isprot", "iswat"):
        custom.addPerParticleParameter(p)
    custom.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    custom.setCutoffDistance(nb.getCutoffDistance())
    custom.setUseSwitchingFunction(nb.getUseSwitchingFunction())
    if nb.getUseSwitchingFunction():
        custom.setSwitchingDistance(nb.getSwitchingDistance())
    # Replicate the LJ dispersion correction the NonbondedForce provided before we
    # zeroed its LJ; without it energy/pressure is off by a constant (~0.8%).
    custom.setUseLongRangeCorrection(nb.getUseDispersionCorrection())

    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        custom.addParticle([sig, eps, float(isprot[i]), float(iswat[i])])
        nb.setParticleParameters(i, q, sig, 0.0 * unit.kilojoule_per_mole)
    for e in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(e)
        custom.addExclusion(i, j)
    system.addForce(custom)
    return system


def _energy(system, positions, lam=None):
    integ = mm.LangevinMiddleIntegrator(310 * unit.kelvin, 1 / unit.picosecond,
                                        0.002 * unit.picosecond)
    ctx = mm.Context(system, integ, _select_platform(mm))
    ctx.setPositions(positions)
    if lam is not None:
        ctx.setParameter("lambda", lam)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx, integ
    return e


def run(structure, lam, seed, prod_ns=3.0, equil_ps=200.0, n_frames=15, temp=310.0):
    system, top, pos = build_solvated(structure)
    apply_swish(system, top, lam)
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, temp * unit.kelvin))
    integ = mm.LangevinMiddleIntegrator(temp * unit.kelvin, 1 / unit.picosecond,
                                        0.002 * unit.picosecond)
    integ.setRandomNumberSeed(seed)
    sim = app.Simulation(top, system, integ, _select_platform(mm))
    sim.context.setPositions(pos)
    sim.context.setParameter("lambda", lam)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(temp * unit.kelvin, seed)
    sim.step(int(equil_ps * 1000 / 2))
    remap = _build_atom_remap(top, structure)
    base = coords_array(structure)
    total = int(prod_ns * 1e6 / 2)
    chunk = max(1, total // n_frames)
    frames = []
    for _ in range(n_frames):
        sim.step(chunk)
        st = sim.context.getState(getPositions=True)
        md = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        frames.append(_reorder(md, remap, base))
    return frames


def score(structure, frames, known, ref):
    base = coords_array(structure)
    plists = [detect_pockets(base, structure)] + [detect_pockets(f, structure) for f in frames]
    cl = cluster_pockets(plists, n_conformers=len(plists), rank_by="crypticity")
    jac = max((residue_jaccard(c.lining_residues, known) for c in cl[:5]), default=0.0)
    d, _ = pocket_min_centroid_dist(cl, ref, top_n=5)
    return jac, d, (d <= CENTROID_THRESHOLD or jac >= JACCARD_THRESHOLD)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", nargs="?", default="T4L_L99A", help="DATASET id")
    ap.add_argument("--lam", type=float, default=1.4, help="SWISH protein-water scale")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ns", type=float, default=3.0)
    ap.add_argument("--check", action="store_true",
                    help="verify the force surgery (lambda=1.0 == unmodified energy)")
    args = ap.parse_args()

    e = next(x for x in DATASET if x["id"] == args.target)
    apo = download_pdb(e["apo_pdb"], PDB_DIR)
    known = e.get("known_residues")
    if known is None:
        holo = download_pdb(e["holo_pdb"], PDB_DIR)
        known, _ = extract_binding_site(
            holo, e.get("holo_chain", "A"), e.get("extra_exclude", frozenset()))
    ref = compute_known_site_centroid(apo, e.get("apo_chain", "A"), known)
    s = load_structure(apo, chain=e.get("apo_chain"))

    if args.check:
        system, top, pos = build_solvated(s)
        e0 = _energy(system, pos)
        apply_swish(system, top, 1.0)
        e1 = _energy(system, pos, lam=1.0)
        print(f"E_plain={e0:.1f}  E_swish(1.0)={e1:.1f}  diff={abs(e0-e1):.2f} kJ/mol "
              f"(should be ~0 of |E|~{abs(e0):.0f})")
        return

    for lam, name in ((1.0, "plain-explicit"), (args.lam, f"SWISH-{args.lam}")):
        npass = 0
        js = []
        for seed in range(args.seeds):
            frames = run(s, lam, seed, prod_ns=args.ns)
            jac, d, ok = score(s, frames, known, ref)
            npass += int(ok)
            js.append(jac)
            print(f"  {name} seed{seed}: jac={jac:.0%} dist={d:.1f} "
                  f"{'PASS' if ok else 'miss'}", flush=True)
        print(f"=> {name}: {npass}/{args.seeds}  mean jac {sum(js)/len(js):.0%}\n", flush=True)


if __name__ == "__main__":
    main()
