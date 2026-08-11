# Changelog

All notable changes to Lacuna are documented here. The project follows
[semantic versioning](https://semver.org/) and the honesty-first principle that
governs its benchmarks: reported numbers are the ones we can defend on held-out
data, never the most flattering ones available.

## [1.0.0] - 2026-08-07

Relicensed to MIT, and the first release whose headline numbers come from a
four-detector comparison rather than Lacuna alone.

### Changed
- **License: AGPL-3.0-or-later to MIT.** The dual-licensing model is retired.
  A commercial license only has value when the buyer cannot get equivalent
  capability free, and Lacuna sits at statistical parity with Apache-2.0 P2Rank.
  More importantly it contradicted the finding below: the useful recommendation
  is to run several detectors together, and copyleft makes that combination
  awkward for the people best placed to act on it. `LICENSE_COMMERCIAL` removed.
- Corrected fpocket's benchmark ranking. Its proposals were being read in the
  order its output directory listed them, which sorts `pocket10` before
  `pocket2`, rather than by fpocket's own rank. This understated its test-fold
  top-5 by roughly twelve points, from a corrected 43.8% to 28.3%. Only fpocket
  was affected; no other tool was read from a directory listing.

### Added
- `--seed-from-sequence`: proposes pockets at locations the sequence model scores
  highly where geometry finds no concavity. Raises single-structure top-5 by
  8.5% (95% CI +4.0 to +13.6) on the test fold, for at most three extra
  candidates. Opt-in, because the gain is confined to small ensembles: at twenty
  conformers it is not separable from zero.
- `paper/`: the coverage-and-conversion study, its per-candidate data, and the
  scripts that regenerate every number, figure and document from source.

### Fixed
- `train_ranker.py --cv` aborted on its own drifting-feature guard and had never
  run on any dump carrying `depth_max`. It is the protocol's model-selection
  instrument, so it needed to work.
- The benchmark fold lookup stripped one trailing character to find a PDB id,
  which silently dropped six structures with multi-character chain IDs from both
  sides of every split. Nothing leaked into the test fold; the affected
  structures were simply excluded from fitting and cross-validation.

## [Unreleased]

Two changes account for essentially all of the accuracy gain in this release:
the ranking function and the pocket-clustering radius. Together they take
CryptoBench test-fold recovery from 17.8% to 56.7%, which moves Lacuna from
behind fpocket to roughly twice its score and level with P2Rank. Neither came
from better sampling.

### Changed
- **`CLUSTER_RADIUS_A` is now 2.0 Å** (was 4.0). Alpha points were dilated by
  4 Å before connected components were labelled, so points ~8 Å apart fused;
  across connected surface grooves that cascaded into mega-pockets. On
  CryptoBench 21% of structures had the true site *fully covered* (median recall
  100%) by a pocket carrying ~59 lining residues against ~8 known, too diffuse
  to count as localized, while only 1% missed the site outright.

  Splitting those blobs also cuts some genuine sites apart, but candidates per
  structure fall from ~59 to ~16 and the ranking gain more than compensates.
  Every variant that restored the lost coverage by adding candidates scored
  worse at top-5 (pooling scales, lowering the volume floor); see
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#why-the-clustering-radius-is-2-å).

  **The ranker weights are specific to the detector geometry.** After this
  change the previous weights scored at the random null on the new pockets, so
  any change to detection constants requires refitting with
  `benchmarks/train_ranker.py --fit`.

- **The default ranking strategy is now `learned`** (was `crypticity`). This is
  the largest accuracy change in the project's history and it came from fixing
  ranking, not sampling.

  The diagnosis: on CryptoBench the detector was already producing a
  well-localized pocket for about three quarters of structures, but top-5
  recovery was 12.7% against a **13.0% random-selection null**. With a median of
  64 candidate clusters per structure and typically exactly one correct, the
  analytic scores were ordering at chance. The recorded conclusion that Lacuna
  was sampling-limited came from a 22-target set too small to resolve anything
  below roughly a 20-point effect.

  On CryptoBench's held-out test fold (n=180, split on the dataset's own
  homology-separated folds):

  | | top-5 recovery |
  |---|---|
  | oracle over all clusters | 73.3% |
  | random top-5 null | 14.4% |
  | `crypticity` (previous default) | 17.8% (95% CI 12.2-23.3) |
  | **`learned` (new default)** | **57.0% (95% CI 49.7-64.2)** |

  Paired gain +24.0 points (95% CI +16.2 to +32.4). An identical fit on shuffled
  labels scores at the random null, so the gain is signal rather than an artifact
  of the evaluation.

  Every independent benchmark improved at the same time (each with the shipped
  2 A clustering radius):

  | dataset | `crypticity` | `learned` |
  |---|---|---|
  | CryptoBench test fold (n=180) | 17.8% | **56.7%** |
  | PocketMiner (n=45) | 14/45 (31%) | **33/45 (73%)** |
  | Curated apo/holo set (n=22) | 7/22 (32%) | **9/22 (41%)** |

  On the curated set the analytic `persistence` and `balanced` strategies reach
  13/22, above `learned`. At n=22 those intervals overlap heavily, but the honest
  reading is that the learned ranker is tuned to CryptoBench's distribution;
  see [docs/BENCHMARKS.md](docs/BENCHMARKS.md#ranking-strategies).

  **Against other detectors** on the same held-out test fold:

  | detector | kind | size-robust recovery |
  |---|---|---|
  | P2Rank | single-structure, learned | 63.3% (56.1-70.6) |
  | **Lacuna** | **ensemble** | **56.7% (48.9-63.9)** |
  | MDpocket | ensemble, same input ensemble | 44.1% (79/179) |
  | fpocket | single-structure, geometric | 28.3% (21.7-34.4) |

  Paired: **+28.3 vs fpocket (CI +20.0 to +36.7)**, **+11.7 vs MDpocket
  (CI +3.9 to +19.6)**, and **-6.7 vs P2Rank (CI -14.4 to +0.6, includes zero)**.
  Lacuna now beats both open baselines by interval-separated margins and is
  statistically indistinguishable from P2Rank. The three-way union is 76.7%, so
  the tools remain complementary.

  `crypticity` remains available via `--rank-by crypticity`.

### Added
- **`rank_by="learned"` / `--rank-by learned`** - a linear ranker over 23 cluster
  features, trained on within-structure pairs to identify the true binding site.
  Scoring is a dot product on raw features: no new runtime dependency, and the
  coefficients are readable in source.

  Two choices were made by cross-validation over the train folds, never on the
  test fold. Training on **pairs** rather than on individual clusters is worth
  +2.4 points (CI +0.7 to +4.2): the decision is "which of this protein's ~64
  candidates is the site", and differencing two clusters from the same structure
  cancels every protein-level nuisance term. Gradient boosting was **not**
  separable from the linear model (+0.7, CI -1.6 to +3.0), so the simpler scorer
  ships.
- **Geometric pocket descriptors** on `Pocket`, computed from grids the detector
  already builds (about +15% detection time), worth +6.9 points (CI +4.0 to +9.7):
  - `depth_a`, burial depth measured against a 5 Å rolling probe. Flood-filling
    empty space from the grid boundary instead would report depth 0 for every
    surface groove, since grooves are open to solvent; only fully enclosed
    cavities would register. A probe too large to enter a binding groove gives a
    discriminative 1.4-11.1 Å range.
  - `buriedness_raw`, the unclipped local density. `enclosure` is
    `min(raw / 0.4, 1.0)` and saturates at exactly 1.00 for most real pockets,
    discarding resolution in the deeply buried regime where binding sites live.
  - `mouth_frac`, `elongation`, `flatness`, `dist_center_frac` for cavity shape
    and placement.
- **Ensemble-native cluster features**: `centroid_std` (a genuine site holds the
  same position across conformers while a spurious blob wanders) and volume
  dispersion. Single-structure detectors have no equivalent.
- **`--cv` on `benchmarks/train_ranker.py`** - leave-one-fold-out
  cross-validation across the train folds for model selection, keeping the
  designated test fold untouched for the final claim and pooling roughly four
  times more held-out structures than that fold alone.
- **`benchmarks/train_ranker.py`** - collects features, fits the ranker, and
  reports held-out recovery with bootstrap CIs and a shuffled-label control. It
  regenerates the shipped constants exactly. Splits on CryptoBench's own folds so
  homologous proteins stay out of the evaluation.
- **`benchmarks/compare_mdpocket.py`** - ensemble-to-ensemble comparison against
  MDpocket, the closest prior art. Both tools receive the identical NMA ensemble
  and the same lining-residue rule, so the comparison isolates detection,
  aggregation and ranking rather than the sampler. MDpocket emits an occupancy
  grid rather than a ranked list, so its isovalue and ranking rule are swept and
  its best configuration is reported.
- **`benchmarks/compare_detectors_cryptobench.py`** - scores Lacuna, fpocket and
  P2Rank per structure under one criterion, with per-(structure, tool)
  checkpointing and resume, and `--exclude-trained` so no tool is scored on its
  own training data.
- **`--rank-by` on the PocketMiner benchmark**, previously pinned to crypticity.

### Fixed
- Benchmark scripts pinned `rank_by="crypticity"` and so silently reproduced the
  old ranker after the default changed. They now take both the default and the
  strategy list from the package, so a reproduction cannot drift from what users
  actually get.
- `paper.md` described MDpocket as requiring "microsecond molecular dynamics
  trajectories" and being "out of reach for users without dedicated compute
  infrastructure". Neither is true: it accepts any supplied conformational
  ensemble, including crystal structures, and processed a 21-conformer ensemble
  in 1.6 s on CPU here. The Statement of Need now separates single-structure,
  learned and ensemble-based tools and states the actual distinction, that
  MDpocket characterizes a site across an ensemble the user supplies while Lacuna
  generates the ensemble and proposes which sites to look at.

## [0.3.1] - 2026-07-04

**Honesty correction to v0.3.0.** The v0.3.0 notes claimed the OpenMM MD backend
opens oligomeric-interface pockets NMA cannot, and that NMA ∪ MD reaches 9/22
(41%) on the curated cryptic set. That came from a single, non-reproducible MD
run. Short implicit-solvent MD is high-variance run to run, and the backend was
not seeding its integrator. On four repeats per target at the same settings, MD
opens Caspase-1 **0/4**, IDH1 **0/4**, PKM2 **1/4**, and 400 K opens the
T4-lysozyme L99A cavity **0/4**. The honest result: the (now working) MD backend
roughly matches NMA on the easy pockets and does **not** reliably open the
hinge/interface classes; the robust union is ~7/22, the same as NMA alone. The
v0.3.0 entry below has been edited to remove the inflated claims.

### Added
- **`benchmarks/metrics.py`** - a shared, canonical size-robust metric module
  (Jaccard, centroid distance, hotspot-core, headline/strict-localized hits) with
  `paired_bootstrap_ci` for target-level confidence intervals and an explicit
  volume-gaming unit test.
- **Top-k detection curve with 95% CIs** in `cryptic_benchmark.py`: the
  size-robust hit rate at k = 1, 3, 5, 10, 20, reported as a range over targets
  rather than a single point. A flat curve (top-5 ≈ top-20) shows the ceiling is
  detection/sampling, not ranking. This tooling is the direct guard against the
  single-number over-claim this release corrects.

### Changed
- **OpenMM backend now seeds its integrator** (default 42) as best-effort
  determinism. Note this does **not** make it bitwise reproducible on GPU
  platforms (OpenCL/CUDA), where floating-point non-determinism plus chaotic
  dynamics still diverge short trajectories run to run. The honest takeaway is
  methodological: short MD is high-variance, so evaluate it with variance across
  runs, never a single number (which is how the v0.3.0 error happened).

## [0.3.0] - 2026-07-04

A **rigor and diagnostics** release. It makes the benchmark trustworthy
(size-robust metrics), diagnoses exactly where the tool fails (per-mechanism
stratification), and repairs the previously-broken OpenMM MD backend so it runs
end to end. (An early version of this entry claimed MD opened interface pockets
and lifted the curated set to 9/22; that did not reproduce, see [0.3.1] above.)
Several sampling and modelling ideas were tried and honestly shelved as negatives
(below).

### Added
- **Size-robust benchmark metric (Jaccard).** All three benchmarks now report a
  size-robust headline - Jaccard overlap (|found ∩ known| / |found ∪ known|) ≥
  0.25 **or** centroid ≤ 4 Å - beside the legacy recall metric. Recall
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
  classes - hinge (0/2) and interface (0/3) - that an elastic network cannot
  sample.
- **Working OpenMM implicit-MD backend.** The `openmm` backend was previously
  broken (it crashed on any structure containing a HETATM and never aligned MD
  atoms to the detection structure). It now reuses `load_structure` (which drops
  ligands/ions and selects the chain), maps MD positions back onto the original
  heavy-atom order, and selects the fastest available platform (CUDA→OpenCL→CPU).
  A `temperature_k` knob exposes elevated-temperature sampling. On the honest,
  size-robust metric it roughly matches NMA and does not reliably open the
  hinge/interface classes (see the [0.3.1] correction above).
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
- **Counterfactual spring-softening NMA** - softening the local "cage" of
  contacts around a candidate cavity did not raise the detection ceiling
  (set-overlap unchanged); only a small localization gain at a conformer-budget
  cost. Backend removed; the γ hook it needed was kept.
- **Interface-first / biological-assembly analysis** - building the assembly
  made things worse (cluster counts balloon, interface pockets rank lower);
  single-chain analysis already partially sees these sites but cannot localize
  them past the bar. The bottleneck is sampling precision, not chain handling.
- **Mode-guided branching** - biasing second-generation sampling toward
  cavity-opening modes beat uniform branching slightly but did not beat the
  plain-NMA baseline and did not touch the hinge/interface failures.
- **Per-residue cryptic-propensity model** - a small feature model reached
  0.834 held-out AUC on PocketMiner labels, but a single geometric feature (depth)
  alone reached 0.849 - the model adds nothing over one trivial feature, and
  neither approaches PocketMiner's 0.87 GNN. A competitive per-residue model needs
  a GNN/PLM (a larger research effort), so nothing was shipped.

The lesson so far: NMA-family tricks (spring-softening, mode-guided branching) are
exhausted for large-motion cryptic sites, and short implicit-solvent MD does not
reliably open them either. The hinge and interface classes remain unsolved; the
most plausible next levers are enhanced sampling (metadynamics on a gate CV) and
cosolvent MD, evaluated with variance, not single runs.

## [0.2.1] and earlier

Curated 22-pair cryptic benchmark, PocketMiner and CryptoBench cross-validation,
crypticity ranking (default), contact-based lining residues, hotspot-centered
pocket localization, NMA/OpenMM/Boltz/random ensemble backends, and the
`lacuna discover` CLI with Boltz-constraint and Vina-box emission. AGPL-3.0;
published on PyPI as `lacuna-pockets`; Zenodo concept DOI
[10.5281/zenodo.20533638](https://doi.org/10.5281/zenodo.20533638).
