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
    orcid: 0009-0001-1033-6320
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 03 June 2026
bibliography: paper.bib
archive_doi: https://doi.org/10.5281/zenodo.20533638
---

# Summary

Lacuna is a Python tool for discovering cryptic binding pockets in protein
structures. Most protein structure predictors - AlphaFold [@jumper2021], Boltz
[@wohlwend2024], Chai - return a single static ground-state conformation. But
many clinically important binding sites are absent or too small in the apo
crystal structure and only open transiently during conformational fluctuation.
These are cryptic pockets, and finding them is one of the central unsolved
problems in computational drug discovery.

Lacuna addresses this by generating a conformational ensemble from any input
structure, detecting surface pockets in every conformer via a grid-based
alpha-sphere algorithm, clustering matched pockets across the ensemble, and
ranking sites by their peak open-state druggability. Each site additionally
receives a continuous *crypticity* score capturing how much it opens relative to
the apo state and how druggable it is once open, and is flagged `cryptic: true`
if it is present in fewer than 90% of conformers. Output includes ranked JSON
reports, visualization PDB files, and ready-to-use docking input files for
AutoDock Vina/Gnina and Boltz-2.

# Statement of Need

Approximately 70% of disease-relevant human proteins lack an obvious binding
site in their experimentally determined structures and are classified as
"undruggable" [@dang2017]. K-Ras was considered undruggable for 30 years
before a cryptic pocket in its switch-II loop was discovered, leading directly
to the FDA-approved drugs sotorasib and adagrasib. More recently, cryptic
pockets have been implicated in allosteric regulation of MDM2 [@kussie1996],
BCL-2 family proteins, and IDH1 - all now drugged.

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
file - including predicted structures from AlphaFold or Boltz - it produces a
ranked cryptic pocket report in seconds to minutes on commodity hardware. The
default ensemble backend is an Anisotropic Network Model (ANM)
[@atilgan2001], which generates physically meaningful collective motions
(hinge bending, domain breathing, loop rearrangements) without a force field.
An optional, experimental Boltz-2 backend (GPU) samples structures via diffusion;
in current benchmarking it does not yet reliably improve cryptic detection over
the NMA backend (see Method).

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
runs 100 ps of Langevin dynamics with GBn2 implicit solvent. The optional,
experimental **Boltz backend** runs Boltz-2 diffusion sampling; it currently
predicts each conformer de novo from sequence rather than diffusing from the
input structure, which yields high but noisy structural diversity and does not
yet reliably improve cryptic detection over NMA (a structure-templated
integration is future work). All backends share the same interface:
`backend.generate(structure_path, n_conformers)`
returns a list of coordinate arrays in the original atom order.

## Pocket Detection

Each conformer is analyzed using a grid-based alpha-sphere algorithm adapted
from fpocket [@leguiloux2009]. A 1 Å voxel grid is built around the protein,
and a Euclidean distance transform identifies the distance from each voxel to
the nearest protein atom. Local maxima of this distance field in the 1.4–6 Å
interaction zone are alpha points - locations where a sphere is simultaneously
tangent to multiple protein atoms, indicating a surface concavity. Nearby alpha
points are clustered using binary dilation, and each cluster is grown to
compute pocket volume, surface enclosure, and residue lining.

## Druggability Scoring and Cryptic Flagging

Each pocket cluster is scored by a composite druggability metric adapted from
Halgren's SiteMap [@halgren2009] and extended with enclosure [@schmidtke2011]:
a Gaussian volume reward centered at 300 Å³, enclosure fraction (buriedness),
hydrophobic residue fraction, and aromatic residue count. By default pockets are
ranked by their peak open-state composite score - the druggability evaluated in
the most-open conformer, which is the relevant figure for a transiently-open
cryptic site (alternative rankings by persistence, a balanced combination, or
crypticity are available). A pocket is marked `cryptic: true` if it appears in
fewer than 90% of ensemble conformers, and is assigned a continuous crypticity
score in [0, 1], `((max_volume − apo_volume) / max_volume) × max_druggability`,
which is ≈ 0 for a constitutive site already formed in the apo structure and near
1 for a site that is absent in the apo state but opens into a druggable cavity.

## Dimer Interface Pockets

Cryptic pockets that form at protein-protein interfaces require the biological
assembly rather than the asymmetric unit for detection. The `--homodimer` flag
reads REMARK 350 BIOMT records (PDB) or `_pdbx_struct_oper_list` (mmCIF) and
applies the symmetry operations to construct the full assembly before analysis.
In practice, however, the oligomeric-interface targets in the benchmark
(Caspase-1, IDH1 R132H, PKM2) remain misses under the size-robust criterion:
building the assembly increases the pocket count without localizing the interface
site, so these classes are limited by conformational sampling rather than by
assembly construction (see Benchmark).

## Benchmark

On a 22-protein benchmark of apo/holo cryptic-pocket pairs (targets drawn from
the cryptic-pocket literature, including CryptoSite examples [@cimermancic2016]),
Lacuna localizes 7/22 cryptic pockets (32%) using the NMA backend with 20
conformers and crypticity ranking, under a size-robust success criterion. A pocket
counts as detected if, among the top five ranked clusters, one's lining residues
(true atomic contact, ≤5 Å from the detected cavity) reach a Jaccard overlap
(intersection over union) of at least 0.25 with the known ligand-contact residues,
or its center lies within 4 Å of the binding-site centroid. We use Jaccard rather
than plain recall (|found ∩ known| / |known|) because recall is size-gameable: a
large pocket engulfs most of a small known site without being localized on it, and
we verified that a learned re-ranker can reach 84% on the recall metric purely by
ranking pockets on volume. Under the legacy recall criterion (≥30% recall or ≤4 Å
centroid) this benchmark scores 13/22 (59%); the size-robust number is roughly half
that, and `cryptic_benchmark.py` prints both alongside a Jaccard threshold sweep
(2/22 satisfy the strict centroid test alone). A diagnostic at a top-20 cutoff
lifts the size-robust score only from 7/22 to 10/22, so the residual gap is
dominated by sampling and localization rather than ranking. The remaining misses
fall into oligomeric-interface pockets (Caspase-1, IDH1 R132H, PKM2) that form
between subunits and large-rearrangement sites (p38 DFG-out, c-ABL myristate)
beyond elastic-network sampling. Runtime on the NMA backend is 0.6–8 s per protein
on a laptop CPU.

Independent validation on two further datasets is consistent: under the size-robust
criterion Lacuna scores 14/45 (31%) on the PocketMiner cryptic-pocket set (Meller
et al. 2023) and 32/180 (18%) on the held-out test fold of CryptoBench (Vavra et
al. 2024), the largest and most diverse cryptic-site dataset (legacy recall: 60%
and 49% respectively). The two curated/field-standard sets converge at ~31-32% and
the hardest comprehensive set floors at 18%.

The benchmark also uses a stricter atomic-contact lining definition than earlier
revisions (which used a ~13 Å sphere around the pocket center); the stricter,
more conservative criterion lowers the reported overlap, and the figures above
reflect it.

# Acknowledgements

The author thanks the developers of Biopython [@cock2009], NumPy [@harris2020],
SciPy [@virtanen2020], and Rich for the open-source foundations on which Lacuna
is built.

# References
