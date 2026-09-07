---
name: "lacuna"
description: "Find cryptic binding pockets in a protein structure: sites that are closed or shallow in the unbound form and only open on ligand binding. Use when someone has a PDB or mmCIF file and asks where a ligand could bind, whether a target is druggable, where an allosteric or hidden pocket might be, or wants AutoDock Vina boxes or Boltz constraints prepared for a target. Also use to prepare those docking artifacts from an existing pocket report."
---

# Lacuna: cryptic binding pocket discovery

Lacuna generates a conformational ensemble from one structure, detects pockets in
each conformer, clusters them across the ensemble, and ranks the sites. The point
is sites a single static structure does not show: a cryptic pocket is closed or
shallow in the apo form and opens on binding, so a detector looking at one
snapshot finds nothing there.

This skill describes **v1.0.3**, the current release on PyPI. Check
`lacuna --version` if behaviour differs from what is written here.

## Install

```bash
pip install lacuna-pockets          # core, no GPU needed
pip install "lacuna-pockets[plm]"   # adds the sequence-aware ranker (torch)
```

Core dependencies are numpy, scipy, biopython, click, rich. MIT licensed.

## The basic command

```bash
lacuna discover protein.pdb
```

Writes `pocket_report.json` into `<input_stem>_lacuna/`, ranked best first.
Roughly 2 seconds per structure on CPU at the default 20 conformers.

## Choosing options

**Ranking** (`--rank-by`, default `learned`)

`learned` is the fitted ranker and roughly doubles recovery over the analytic
rules. Use `learned-plm` if the `plm` extra is installed: it adds protein language
model features and is the strongest option, reaching 66.3% top-five recovery on
the CryptoBench test fold against 55.9% for `learned`. Use `crypticity` only when
you specifically want sites that open a lot relative to the input, and
`druggability` for always-open orthosteric sites, which is a different question
from the one Lacuna is built for.

**Conformers** (`--conformers`, default 20)

Twenty is the measured sweet spot. More does not help: at forty and eighty the
ensemble finds more sites and surfaces none of them, because extra conformers
multiply candidates and the ranking problem gets harder at the same rate. Ten is
statistically indistinguishable from twenty at roughly half the wall clock.

**Backend** (`--backend`, default `auto`)

`nma` (normal mode analysis) is the fast, dependency-free default and is what
`auto` selects on a base install. `openmm` runs implicit-solvent MD and `boltz`
uses diffusion sampling, both needing their extras. Measured on CryptoBench, MD
did **not** beat NMA (10% against 12% at n=924), so prefer `nma` unless there is a
specific reason.

**Detector** (`--detector`, default `alpha`)

`alpha` is the built-in geometric detector. `p2rank` and `fusion` use P2Rank
instead and require it installed separately (Java 11+, `prank` on PATH or
`LACUNA_P2RANK` set). Without P2Rank, leave this alone.

**Other flags worth knowing**

- `--homodimer` is required for pockets at a dimer interface. It builds the
  biological assembly from BIOMT / `_pdbx_struct_oper_list`. Without it those
  sites cannot be detected at all. Prefer RCSB's biological assembly download.
- `--ensemble PATH` uses conformers you already have (a multi-model file or a
  directory) instead of generating them. Frames are matched by residue number and
  atom name, so a frame missing a loop is still usable.
- `--seed-from-sequence` proposes pockets where the sequence model scores highly
  but geometry sees nothing. The measured gain is confined to small ensembles:
  +8.5% (95% CI +4.0 to +13.6) at one conformer, and nothing separable from zero
  at twenty. Only worth it in fast single-structure mode.

## Docking preparation

```bash
lacuna discover protein.pdb --emit-vina-boxes --emit-boltz-constraints
lacuna dock-prep pocket_report.json --top 3    # or from an existing report
```

`--emit-pocket-pdbs` writes pseudoatom PDBs for visualising pockets in PyMOL or
ChimeraX.

## Reading the output

`pocket_report.json` lists pockets ranked best first. Per pocket: lining residues,
volume, druggability, crypticity (how much it opens relative to the input),
persistence (fraction of conformers it appears in), and centroid.

**Report a ranked shortlist, not a single answer.** Lacuna's own benchmarking is
the reason: on CryptoBench the site is usually proposed somewhere and then
out-ranked. Top-five recovery is 66.3% with `learned-plm`, so the top hit alone is
frequently not the answer even when the answer is in the list. Show the top three
to five with their scores.

## What to tell the user honestly

- These are **predictions on one structure**, not experimental evidence. A high
  rank is a hypothesis worth testing, not a validated site.
- **Persistence matters.** A pocket appearing in two of twenty conformers is
  weaker evidence than one appearing in eighteen, even at a similar rank.
- Lacuna is at **statistical parity with P2Rank**, not better: +2.8% on the
  CryptoBench test fold with a paired interval of [-4.4, +9.4] that includes zero.
  Do not claim superiority.
- On general open binding sites, rather than cryptic ones, **P2Rank is a strong
  and faster alternative**. Lacuna is built for the closed-in-apo case.
- Combining detectors beats any single one: across four detectors, the union
  proposes a qualifying candidate for 92.1% of CryptoBench test targets against
  74.2% for the best single tool. If a result matters, cross-check with another
  method.

## When not to use this

Lacuna needs a 3D structure. It cannot work from sequence alone; fold the sequence
first (AlphaFold, Boltz, Chai, ESMFold) and pass the model in. It also does not
predict binding affinity, dock ligands, or tell you what to put in a pocket.

## Reference

- Docs: `docs/USAGE.md`, benchmarks in `docs/BENCHMARKS.md`
- Repo: https://github.com/mooreneural/lacuna
- Archive: https://doi.org/10.5281/zenodo.20533638
