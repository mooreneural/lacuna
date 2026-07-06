# Changelog

All notable changes to Lacuna are documented here. The project follows
[semantic versioning](https://semver.org/) and the honesty-first principle that
governs its benchmarks: reported numbers are the ones we can defend on held-out
data, never the most flattering ones available.

## [0.3.0] — 2026-07-04

A **rigor + first real ceiling-raise** release. It makes the benchmark
trustworthy (size-robust metrics), diagnoses exactly where the tool fails
(per-mechanism stratification), and — for the first time — cracks a failure
class the elastic-network sampler structurally cannot: the working OpenMM MD
backend opens **oligomeric-interface pockets** that NMA misses entirely. On the
curated cryptic set NMA and MD each pass 7/22 but on *complementary* targets, so
their **union reaches 9/22 (41%)** vs 7/22 (32%) for either alone. Several other
sampling and modelling ideas were tried and honestly shelved as negatives (below).

### Added
- **Size-robust benchmark metric (Jaccard).** All three benchmarks now report a
  size-robust headline — Jaccard overlap (|found ∩ known| / |found ∪ known|) ≥
  0.25 **or** centroid ≤ 4 Å — beside the legacy recall metric. Recall
  (|found ∩ known| / |known|) is size-gameable: a large pocket engulfs a small
  known site without being localized on it. Under the size-robust criterion the
  honest numbers roughly halve: curated **32%** (was 59% recall), PocketMiner
  **31%** (60%), CryptoBench test fold **18%** (49%). Both criteria are printed
  side by side.
- **Hotspot-core metric.** A second size-robust, hotspot-anchored measure
  (fraction of known-site Cα within 8 Å of a pocket's buriedness-weighted
  hotspot), reported as a diagnostic column.
- **Opening-mechanism stratification.** Curated cryptic entries are labelled by
  dominant opening mechanism (sidechain / loop / helix / hinge / interface) with
  a per-mechanism pass-rate breakdown. This exposes the failure structure
  cleanly: NMA handles side-chain openings (3/4) but fails on the large-motion
  classes — hinge (0/2) and interface (0/3) — that an elastic network cannot
  sample.
- **Working OpenMM implicit-MD backend.** The `openmm` backend was previously
  broken (it crashed on any structure containing a HETATM and never aligned MD
  atoms to the detection structure). It now reuses `load_structure` (which drops
  ligands/ions and selects the chain), maps MD positions back onto the original
  heavy-atom order, and selects the fastest available platform (CUDA→OpenCL→CPU).
  A `temperature_k` knob enables simple enhanced sampling (e.g. 400 K opens the
  buried T4-lysozyme L99A cavity that 310 K and NMA both miss). This is the first
  backend to open interface pockets (PKM2, Caspase-1) — complementary to NMA.
  Benchmark flags: `--backend openmm --openmm-temp --openmm-time`.
- **Per-pair spring-constant hook** in the ANM backend (`_compute_modes(gamma=…)`),
  a reusable extension point for spring-perturbation experiments. Default
  behaviour is byte-for-byte unchanged.

### Fixed
- **`write_structure_pdb` column alignment.** The residue-name column was written
  one position too far left (missing the column-17 altLoc blank). Biopython
  tolerated it but strict parsers (OpenMM) rejected the file. Affected every
  consumer, including the homodimer biological-assembly writer.

### Changed
- README, `paper.md`, and `CITATION.cff` now lead with the size-robust numbers,
  keeping the legacy recall figures shown transparently alongside.

### Investigated and shelved (honest negatives)
- **Counterfactual spring-softening NMA** — softening the local "cage" of
  contacts around a candidate cavity did not raise the detection ceiling
  (set-overlap unchanged); only a small localization gain at a conformer-budget
  cost. Backend removed; the γ hook it needed was kept.
- **Interface-first / biological-assembly analysis** — building the assembly
  made things worse (cluster counts balloon, interface pockets rank lower);
  single-chain analysis already partially sees these sites but cannot localize
  them past the bar. The bottleneck is sampling precision, not chain handling.
- **Mode-guided branching** — biasing second-generation sampling toward
  cavity-opening modes beat uniform branching slightly but did not beat the
  plain-NMA baseline and did not touch the hinge/interface failures.
- **Interface-first / biological-assembly analysis** — building the assembly made
  things worse (cluster counts balloon, interface pockets rank lower); the
  bottleneck is sampling precision, not chain handling. (OpenMM MD, above, is what
  actually opened the interface pockets.)
- **Per-residue cryptic-propensity model** — a small feature model reached
  0.834 held-out AUC on PocketMiner labels, but a single geometric feature (depth)
  alone reached 0.849 — the model adds nothing over one trivial feature, and
  neither approaches PocketMiner's 0.87 GNN. A competitive per-residue model needs
  a GNN/PLM (a larger research effort), so nothing was shipped.

The lesson from the NMA-family attempts (spring-softening, mode-guided branching):
elastic-network tricks are exhausted for large-motion cryptic sites. Physical MD
(OpenMM) is what finally opened them, and NMA+MD are complementary.

## [0.2.1] and earlier

Curated 22-pair cryptic benchmark, PocketMiner and CryptoBench cross-validation,
crypticity ranking (default), contact-based lining residues, hotspot-centered
pocket localization, NMA/OpenMM/Boltz/random ensemble backends, and the
`lacuna discover` CLI with Boltz-constraint and Vina-box emission. AGPL-3.0;
published on PyPI as `lacuna-pockets`; Zenodo concept DOI
[10.5281/zenodo.20533638](https://doi.org/10.5281/zenodo.20533638).
