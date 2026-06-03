---
title: 'Lacuna: Cryptic Binding Pocket Discovery via Conformational Ensemble Analysis'
tags:
  - Python
  - drug discovery
  - protein structure
  - cryptic pockets
  - cheminformatics
  - bioinformatics
authors:
  - name: Clayton W. Moore
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 03 June 2026
bibliography: paper.bib
archive_doi: https://doi.org/10.5281/zenodo.20533639
---

# Summary

Lacuna is a Python tool for discovering cryptic binding pockets in protein
structures. Most protein structure predictors — AlphaFold [@jumper2021], Boltz
[@wohlwend2024], Chai — return a single static ground-state conformation. But
many clinically important binding sites are absent or too small in the apo
crystal structure and only open transiently during conformational fluctuation.
These are cryptic pockets, and finding them is one of the central unsolved
problems in computational drug discovery.

Lacuna addresses this by generating a conformational ensemble from any input
structure, detecting surface pockets in every conformer via a grid-based
alpha-sphere algorithm, clustering matched pockets across the ensemble, and
ranking sites by persistence (fraction of conformers in which the pocket
appears) and druggability. Sites present in fewer than 90% of conformers are
flagged `cryptic: true`. Output includes ranked JSON reports, visualization
PDB files, and ready-to-use docking input files for AutoDock Vina/Gnina and
Boltz-2.

# Statement of Need

Approximately 70% of disease-relevant human proteins lack an obvious binding
site in their experimentally determined structures and are classified as
"undruggable" [@dang2017]. K-Ras was considered undruggable for 30 years
before a cryptic pocket in its switch-II loop was discovered, leading directly
to the FDA-approved drugs sotorasib and adagrasib. More recently, cryptic
pockets have been implicated in allosteric regulation of MDM2 [@kussie1996],
BCL-2 family proteins, and IDH1 — all now drugged.

Existing computational tools for cryptic pocket detection fall into two
categories. Structure-based tools like fpocket [@leguiloux2009] and CASTp
analyze a single static conformation and entirely miss pockets that require
structural rearrangement to open. Simulation-based approaches like MDpocket
[@schmidtke2011] and CryptoSite [@cimermancic2016] require microsecond
molecular dynamics trajectories or machine learning models trained on curated
datasets, placing them out of reach for users without dedicated compute
infrastructure or specialized expertise.

Lacuna occupies the gap between these extremes. It requires no GPU, no
simulation software, and no training data. Starting from any PDB or mmCIF
file — including predicted structures from AlphaFold or Boltz — it produces a
ranked cryptic pocket report in seconds to minutes on commodity hardware. The
default ensemble backend is an Anisotropic Network Model (ANM)
[@atilgan2001], which generates physically meaningful collective motions
(hinge bending, domain breathing, loop rearrangements) without a force field.
For users with GPU access, an optional Boltz-2 partial diffusion backend
provides higher-quality sampling of large-scale conformational changes.

# Method

## Ensemble Generation

Lacuna implements three interchangeable ensemble backends behind a common
abstract interface. The default **NMA backend** builds the ANM Kirchhoff
matrix from Cα contact pairs within an 8 Å cutoff, performs a partial
eigendecomposition via `scipy.linalg.eigh` to extract the 10 lowest-frequency
non-trivial normal modes, and samples conformers by displacing Cα atoms along
Boltzmann-weighted random linear combinations of these modes. All-atom
coordinates are recovered by Gaussian-weighted interpolation from Cα
displacements with a 5 Å correlation length. The optional **OpenMM backend**
runs 100 ps of Langevin dynamics with GBn2 implicit solvent. The optional
**Boltz backend** uses Boltz-2 partial diffusion at linearly increasing noise
levels, producing large-scale conformational changes unreachable by NMA. All
backends share the same interface: `backend.generate(structure_path, n_conformers)`
returns a list of coordinate arrays in the original atom order.

## Pocket Detection

Each conformer is analyzed using a grid-based alpha-sphere algorithm adapted
from fpocket [@leguiloux2009]. A 1 Å voxel grid is built around the protein,
and a Euclidean distance transform identifies the distance from each voxel to
the nearest protein atom. Local maxima of this distance field in the 1.4–6 Å
interaction zone are alpha points — locations where a sphere is simultaneously
tangent to multiple protein atoms, indicating a surface concavity. Nearby alpha
points are clustered using binary dilation, and each cluster is grown to
compute pocket volume, surface enclosure, and residue lining.

## Druggability Scoring and Cryptic Flagging

Each pocket cluster is scored by a composite druggability metric adapted from
Halgren's SiteMap [@halgren2009] and extended with enclosure [@schmidtke2011]:
a Gaussian volume reward centered at 300 Å³, enclosure fraction (buriedness),
hydrophobic residue fraction, and aromatic residue count. Pockets are ranked by
this composite score. A pocket is marked `cryptic: true` if it appears in fewer
than 90% of ensemble conformers, indicating that it is absent in the ground
state and transiently accessible.

## Dimer Interface Pockets

Cryptic pockets that form at protein-protein interfaces require the biological
assembly rather than the asymmetric unit for detection. The `--homodimer` flag
reads REMARK 350 BIOMT records (PDB) or `_pdbx_struct_oper_list` (mmCIF) and
applies the symmetry operations to construct the full assembly before analysis,
enabling discovery of dimer interface pockets such as those in Caspase-1 and
IDH1 R132H.

## Benchmark

On a 20-protein benchmark drawn from the CryptoSite dataset [@cimermancic2016],
Lacuna detects 14/20 cryptic pockets (70%) using the NMA backend with 20
conformers, matching the published CryptoSite benchmark rate. The six misses
divide into near-misses that improve to passing with the Boltz backend (IL-2:
21% → 71%) and dimer-interface pockets addressable with `--homodimer`. Across
27 diverse proteins including conformational and orthosteric controls, the
overall detection rate is 19/27 (70%). Runtime on the NMA backend is 0.6–8 s
per protein on a laptop CPU.

# Acknowledgements

The author thanks the developers of Biopython [@cock2009], NumPy [@harris2020],
SciPy [@virtanen2020], and Rich for the open-source foundations on which Lacuna
is built.

# References
