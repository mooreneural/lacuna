# Changelog

All notable changes to Lacuna are documented here. The project follows
[semantic versioning](https://semver.org/) and the honesty-first principle that
governs its benchmarks: reported numbers are the ones we can defend on held-out
data, never the most flattering ones available.

## [1.5.0] - 2026-11-14

The same input now gives the same pocket report on every machine. With the
default NMA backend, `pocket_report.json` is identical at any BLAS thread count
for protein, RNA, DNA and assemblies built with `--homodimer`, and identical
between a Windows and a Linux build on the 40 proteins compared. The sampler
also uses less than half the memory from 1,300 residues up, about a third at
3,700, and is faster at the default 20 conformers:

    residues   peak memory, 1.4.0 -> 1.5.0   ensemble time, 1.4.0 -> 1.5.0
      339         154 MB ->   105 MB           0.89 s ->  0.23 s
    1,327         558 MB ->   239 MB           6.6 s  ->  2.1 s
    2,056       1,137 MB ->   435 MB          15.0 s  ->  7.7 s
    3,724       3,563 MB -> 1,202 MB          73.4 s  -> 49.5 s

The full pipeline is 7% to 30% faster on the proteins of the speed table in
`docs/BENCHMARKS.md`, where it now takes 0.59 s at 122 residues and 5.9 s at
536.

Reports differ from 1.4.0 for the same input: every ensemble is drawn by the new
sampler, and the lining rule under "Changed" alters which residues a pocket
lists. The benchmark tables were re-measured with 1.5.0. Top-five recovery with
the default ranker is 57.8% (104/180) on the CryptoBench test fold, 69% (31/45)
on PocketMiner, 41% (9/22) on the curated set and 89% (254/286) on COACH420, and
the per-residue propensity reaches AUROC 0.766 and AUPRC 0.340. The ranker
weights are unchanged. The `learned-plm` and `surface-fusion` figures need the
`plm` extra and stand as measured with 1.4.0.

### Changed
- **The NMA sampler is reproducible across thread counts and platforms.** The
  eigensolver is the one step whose output depends on how BLAS is threaded, and
  three things used to carry that into the conformers. Every intermediate is now
  float64 (distances, Hessian, eigendecomposition, mode combination and the
  propagation to all atoms), and a conformer is cast to float32 once, at the
  end. Every mode gets a fixed sign. Modes that share an eigenvalue, which any
  assembly with a three-fold or higher axis has, get a basis computed from the
  space they span instead of the one the solver happened to return. The Hessian
  is assembled a slab of rows at a time and factored in place, and the
  propagation weights are built once per call, which is where the memory and
  time go down.

  For this release the report was compared byte for byte between the default
  thread count and a single thread on 30 inputs (6 proteins of 122 to 3,724
  residues, 17 RNA, 2 DNA, 2 protein and nucleic-acid complexes, 2 `--homodimer`
  assemblies and the K-Ras example): 30 of 30 identical. The conformer
  coordinates themselves agree to float32 rounding. Across 184 structures at 1,
  2, 8 and the default number of threads, and between a Windows and a Linux
  build, at most 40 of 26 million coordinates differed, none by more than
  1.9e-6 Å. See [docs/USAGE.md](docs/USAGE.md#reproducibility).

- **RNA and DNA are sampled through the same loop as protein.** The nucleic
  backend carried its own copy of the sampling loop. It now selects its beads
  and hands them to the shared one, so a nucleic-acid ensemble is reproducible
  on the same terms and at the same cost: on a 1,483-bead complex, peak memory
  for 20 conformers falls from 2,607 MB to 671 MB and the time from 44 s to
  6.4 s.

- **A pocket's lining residues are a strict majority of the conformers it
  appears in.** The lining was built by counting member pockets against half the
  number of conformers, which for a pocket seen in one or two conformers kept
  every residue of every member. The rule is now a strict majority over distinct
  conformers at every count, the way `residue_propensity` already counted.

  This changes rankings as well as the residue lists. A pocket's lining feeds
  the ranker features `n_lin`, `vol_per_lin` and `chain_shares` and the
  sequence terms of `learned-plm`, so ranks and `residue_propensity` can move
  even on an identical ensemble. Measured on 120 CryptoBench test targets, with
  the same detections clustered under both rules: top-five recovery 71 before
  and 73 after (exact McNemar p = 0.688), coverage 89 and 89, median lining 20
  to 18 residues, and proposals closer to the size of the annotated site (1.38
  to 1.27 times its size), which matters for docking boxes and for the
  per-residue output. A cluster whose members reach no majority falls back to
  the residues they agree on most, so no pocket is reported with an empty
  lining.
  `CONSENSUS_STRICT_MAJORITY` in `lacuna/pockets/clusterer.py` keeps the earlier
  rule reachable.

- **`docs/BENCHMARKS.md` names the command behind each headline figure.** The
  COACH420 block lists the full run that gives the 286-structure figure beside
  the `--limit 150` run that gives the paired tables, and the CryptoBench row
  says which script prints it. The speed table reports 1.4.0 and 1.5.0 side by
  side. The K-Ras worked example in `docs/USAGE.md` selects the switch-II pocket
  by its lining residues. The zPocket figures in the README acknowledgement were
  corrected against that paper's Table A7.

### Added
- **A `RuntimeWarning` when the elastic network is under-constrained.** If the
  zero modes of the network fill the whole window of eigenvalues the sampler
  requests, there is no vibration to sample and the ensemble is built from
  rigid-body and free-bead motion. That case is outside the reproducibility
  guarantee, and it is now detected and reported instead of passing silently.
  The report is still written. Of 157 structures in the benchmark cache it fires
  on two, both AlphaFold models with long extended disordered regions.

- **`canonical_sign` and `canonical_basis`** are public functions of
  `lacuna.ensemble.nma_backend`: the sign rule and the degenerate-set basis the
  sampler uses, for anyone building on the modes directly.

- **A warning when an ensemble does not describe the input structure.** Every
  quantity Lacuna reports is relative to the input, so a conformer set that no
  longer resembles it makes every pocket look newly opened. Four measurements
  now catch that whatever produced the conformers: drift from the input relative
  to its radius of gyration, rigid-body offset, the radius-of-gyration ratio,
  and the spread between conformers. It warns and continues, because a genuine
  large hinge motion can exceed the bars legitimately. On input that mixes
  protein and nucleic acid it measures both polymers together. It stayed silent
  on all 58 default-backend runs checked for this release. The check is
  `lacuna.ensemble.sanity.check_ensemble`.

- **`--no-msa-server`.** `--backend boltz` sends the query protein sequence to
  the public ColabFold MSA server at `https://api.colabfold.com` by default,
  because Boltz-2 does not fold usefully without an alignment. `--backend auto`
  tries `boltz` first, so on a machine where the `boltz` extra is installed a
  plain `lacuna discover protein.pdb` makes that call too. The help text now
  says so, and `--no-msa-server` turns it off. An install without the `boltz`
  extra is not affected: `auto` then selects `openmm` if that extra is
  installed and the NMA backend otherwise, and no sequence leaves the machine.

- **`--rank-by learned-fused`** can be selected by name. It was already the
  ranker `--detector surface-fusion` picks automatically.

- **`superpose` and `align_frame`** in `lacuna.ensemble.base`, the Kabsch
  helpers the external and Boltz backends now share.

### Fixed
- **`--ensemble` frames are superposed onto the input before analysis.** An MD
  trajectory exported without fitting, or the output of a generative model that
  folds in its own frame, used to put every pocket in a cluster of its own and
  report the whole protein as cryptic. On 4OBE chain A, with Lacuna's own
  ensemble exported in displaced frames: 28 clusters and 93% flagged cryptic
  before, 7 clusters after, against 6 for the native ensemble.

- **`--ensemble` frames that a tool renumbered are paired by residue order.**
  Structure predictors and samplers commonly renumber their output from 1. Three
  keys are tried in turn (the input's own numbering, the numbering with the
  chain letter ignored, then position in the sequence), the first whose paired
  residues agree with the input on at least 90% of residue names is used, and
  pairing by order prints a warning. A frame that agrees under no key is
  reported and skipped. The chain-letter fallback applies only when both the
  input and the frame have a single chain; a single-chain frame against a
  multi-chain input is refused instead of being mapped onto every chain.

- **`--backend boltz` pairs residues by sequence order**, with a 90%
  residue-identity check, so conformers correspond to the input when the file
  does not start at residue 1, and it requests an alignment by default (see
  `--no-msa-server` above). The backend remains experimental.

- **Warnings go to stderr, so `--quiet` leaves stdout empty.**

- **Bad input is reported in one line instead of a traceback**: a file that is
  not a structure, a structure with no polymer atoms, a directory given as the
  input, `-o` pointing at an existing file, and `dock-prep` given its two
  arguments in the wrong order. `LACUNA_TRACEBACK=1` restores the traceback.

- **`--homodimer` no longer crashes on a Windows console.** On a structure
  without BIOMT records it printed a character the default cp1252 console cannot
  encode, and any redirect or pipe ended the run with no results written.

## [1.4.0] - 2026-10-28

Ranking behaviour is unchanged. Every number in `docs/BENCHMARKS.md` stands, the
fitted weights are untouched, and no existing report field changes type or value.
What is new is a second kind of output and a backend correction.

### Added
- **`residue_propensity`: a per-residue cryptic-pocket score in
  `pocket_report.json`.** For each residue, the inverse-rank-weighted sum over
  clusters of the fraction of that cluster's conformers in which the residue
  lines the pocket.

  This exists because the field scores per residue, not top-k: CryptoBench,
  PocketMiner, UniPocket and zPocket all report ROC-AUC with a precision-recall
  summary beside it. Lacuna reported only top-k and so could not be placed on
  those tables at all. Measured on the CryptoBench held-out test fold, n=231
  targets and 76,633 residues, with the aggregation form selected on the train
  folds:

      scorer                            AUROC   AUPRC
      this, 20 conformers               0.774   0.345
      this, input structure only        0.697   0.249
      burial neighbour count            0.586   0.115
      random expectation                0.500   0.067

  The burial row is the one that matters, because a cavity-lining score could
  easily be a restatement of how buried a residue is. Paired over targets, the
  propensity beats burial by **+0.188 AUROC** (95% CI +0.164 to +0.211) and
  **+0.229 AUPRC** (+0.199 to +0.261).

  Published comparators need the averaging convention stated. The figures above
  average within each protein then over proteins; CryptoBench's table pools all
  residues into one curve. Pooled the same way this scorer gives AUROC 0.813 and
  AUPRC 0.237, against CryptoBench's P2Rank 0.81 / 0.21 and their own sequence
  model 0.86 / 0.36. So: a little above P2Rank, clearly below the sequence model.

  The ensemble earns its place here: 0.345 against 0.249 for the deposited
  structure alone, and the gap holds under a stricter single-partner label
  definition (0.312 against 0.216).

- **`chain_shares` and `interface` per pocket.** Every lining residue already
  carried a chain id and nothing reported whether a pocket sat at a chain
  interface. A chain counts toward the flag at >=10% of the lining and >=3
  residues, on two or more chains.

  Reported with its measured direction, which is the opposite of the obvious
  one: on the CryptoBench test fold **63.6% of proposed pockets are interfacial,
  rising to 73.6% of the top five, while only 14.8% of annotated sites are**. The
  flag is therefore useful for de-prioritising a candidate, not for finding one.

### Fixed
- **The Boltz backend no longer matches atoms by position or pad with zeros, and
  now superposes onto the input frame.** Two defects, both silent:

  Boltz predicts from sequence and may reorder atoms, which its own comment in
  this file acknowledged, yet a count mismatch was handled by taking the first N
  atoms positionally and filling the remainder with `np.zeros`, placing unmatched
  atoms at the origin. Atoms are now matched on `(chain, residue, atom name)`,
  unmatched atoms keep their reference coordinate, and a frame matching under
  half the input is discarded rather than used.

  Separately, nothing superposed Boltz's output onto the input. It folds de novo
  in its own coordinate frame, and cluster identity downstream is absolute
  Cartesian centroid proximity within 5 Å, so pockets from a Boltz conformer
  would rarely co-cluster with the input's: `apo_volume_A3` would read 0 and the
  whole protein would report as cryptic. Conformers are now Kabsch-superposed on
  their matched atoms, with a reflection guard.

  This affects `--backend boltz` only. The default NMA backend is frame-safe by
  construction and is unchanged.

### Changed
- **Documentation corrected where it contradicted the code.** Each of these
  described behaviour the code has not had for some time:
  - `docs/RANKING.md` and the clusterer docstring said a pocket is flagged
    cryptic when "present in fewer than 90% of conformers". The rule is
    `crypticity >= 0.3` and has been since before 1.0.0.
  - The clusterer described its clustering as "DBSCAN (eps=5 Å, min_samples=1)"
    and as using union-find. It is neither: it is an order-dependent greedy
    single pass, which is why pocket identity can depend on the order conformers
    are pooled in.
  - `apo_volume_A3` was documented as the volume "in the apo structure, 0.0 if
    absent there". It is the largest frame-0 member **assigned to that cluster**,
    so it also reads 0.0 when the greedy pass put the site's frame-0 pocket in a
    neighbouring cluster, measured at 4 to 7% of pockets. `docs/RANKING.md` now
    also states that whenever it is 0.0, `crypticity` is numerically identical to
    `max_druggability`, and points at `--conformers 0` as the way to ask whether
    a site is genuinely open in the input.
  - `appears_in_conformers` is now documented as a list of conformer indices
    rather than a count, and `--output` help names the report file it writes.

## [1.3.1] - 2026-10-26

### Fixed
- **The learned ranker's burial term is corrected for nucleic-acid input, where
  it was pointed the wrong way.** Before this release, on RNA sites that are
  already open in the input structure, the ranker was measurably worse than
  shuffling its own output.

  `enc` and `bur_raw` are one measurement entered twice. The detector computes a
  local-density burial value, clips it to `enc = min(raw / 0.4, 1.0)`, and also
  emits it raw. At the fitted weights, +2.003 and -15.465, the pair forms a
  burial PENALTY, and `bur_raw` alone carries 41% of the model's weight mass.

  That penalty is correct for protein and backwards for nucleic acid, because the
  two polymers put their ligand site on opposite sides of the burial
  distribution. Comparing the true site against the other candidates from the
  same structure, paired so that neither pool size nor structure size can drive
  it:

      protein, CryptoBench test fold, 210 structures    -0.053   p = 1.7e-08
      nucleic, sites shut in the input, 60 structures    +0.030   p = 0.015
      nucleic, sites open in the input, 652 structures   +0.117   p = 3.1e-61

  The protein pocket is the shallower candidate; the nucleic one is the more
  buried. A penalty that finds protein sites therefore hides nucleic ones. The
  clip compounds it: protein clusters straddle the 0.4 knee, median burial 0.42
  with 57% saturated, while nucleic clusters sit well below it, median 0.20 with
  11% saturated, so nucleic input was ranked almost entirely on the
  steeper-penalty side of a function fitted for the other polymer.

  The fix sets `bur_raw` to zero on nucleic input, which leaves `enc` at +2.003
  and turns the nucleic burial slope from -10.46 to +5.0 per unit, matching the
  measured direction. **Nothing is refitted and no new parameter is introduced:**
  one shipped weight is set to zero, and only when the input contains no protein.
  Ranking is reported against the random-ordering baseline it has to beat,
  `1 - C(N-k, m)/C(N, m)` averaged over structures, because on a short RNA with a
  handful of clusters a top-5 hit is close to free and an unbaselined percentage
  there says nothing:

      population                        n    baseline    1.3.0    1.3.1
      nucleic, site open              660         47%      39%      67%
      nucleic, site shut               61         50%      64%      61%
      protein (gate never fires)      210         34%      64%      64%

  At 39% against a 47% baseline the old behaviour was significantly below chance
  (p = 1.3e-07); at 67% the corrected behaviour is significantly above it
  (p = 1.4e-36). Top-ten on the same population goes from 59% to 84%, which is
  the figure that matters most in practice, since a report shows ten rows by
  default. On sites that are shut in the input the change is -3 points over 61
  structures and does not reach significance (paired sign test p = 0.75), so the
  correction is justified by the open population and is merely costless on the
  cryptic one.

  Protein ranking is unchanged to the bit, verified by IEEE-754 comparison over
  4,555 clusters from the 210-structure CryptoBench test fold. The gate fires on
  nucleic-ONLY structures, deliberately stricter than the nucleic-or-mixed test
  that selects the elastic-network backend: residue classification calls a
  structure "mixed" with no minimum nucleic fraction, so a protein target
  carrying one incidental nucleotide is "mixed", and 4 of those 210 CryptoBench
  structures are exactly that. Removing `bur_raw` on protein costs 17 points, so
  gating on the looser test would have introduced a silent regression on 1.9% of
  the protein benchmark. Mixed input therefore keeps the protein weighting, which
  is a conservative choice rather than a measured one.

  `--rank-by learned-plm` and the `learned-fused` model reached by
  `--detector surface` are separate fits and are deliberately untouched: the
  measurement above covers the default `learned` weights only, and the fused
  model is not even a scaled copy of the same failure, since it gives both burial
  terms a negative coefficient instead of the +2.003 / -15.465 pairing.

### Changed
- **Nucleic ranking figures published in 1.3.0 are withdrawn and restated against
  a baseline.** The 1.3.0 notes and the nucleic module quoted "top-5 in 80%" over
  120 apo/holo records and "top-10 65%, top-5 39%" over 31 structures. Those are
  open-site populations, measured with the uncorrected burial term and against no
  baseline, and the 39% is the same below-chance number this release fixes. Some
  were also quoted per apo/holo record rather than per structure, which counts one
  apo entry paired with six holo entries as six independent successes.

  1.3.0's DETECTION figures are unaffected and stand: cryptic sites found at all
  rising from 44% to 73% with the ensemble, and the ligand site found at all for
  87% of structures across 31 structures from 12 BGSU classes. Detection does not
  consult the ranker. 1.3.0's cryptic RANKING figures were above baseline and are
  not retracted, but they were published without one.

## [1.3.0] - 2026-10-26

### Added
- **Nucleic-acid support: `lacuna discover rna.cif` works, with no extra flags.**
  Input is detected as nucleic acid and the elastic network is built on
  nucleotide beads (C1', C2 and P at a 9 A cutoff, the three-bead SBP model of
  Pinamonti et al. 2015, *Nucleic Acids Research* 43:7260) instead of on CA
  atoms. Detection geometry, clustering and the learned ranker are Lacuna's
  shipped settings, unchanged, because they already work on RNA: every
  RNA-specific retuning we tried either failed to generalise on held-out data or
  had no measurable effect.

  The point of the ensemble is cryptic sites, so that is what the headline
  measures. On RNA sites with NO detectable cavity in the input structure, over
  62 apo structures spanning 41 independent sequence families, the ensemble is
  what finds them:

      cryptic sites, no cavity in the input      single structure    ensemble
      found at all                                           44%         73%
      ranked in the top five                                 32%         55%

  On sites that are already open in the input, the same ensemble adds 4 points
  rather than 29, because there is nothing to open. That contrast is the claim:
  the machinery earns its cost exactly where the premise says it should. Under
  60 nt, where the benchmark is thickest at 22 independent families, cryptic
  top-five recovery is 81%.

  Two further figures, quoted separately because they are NOT cryptic claims:
  across 31 structures from 12 independent BGSU classes spanning 62 to 459 nt,
  single-structure detection alone finds the ligand site for 87% of structures
  and ranks it top-ten for 65%; on 120 apo/holo records over 37 apo structures,
  top-five is 76% per structure. Both are over all ligand sites, most of which
  are already open, and neither exercises the ensemble.

  Figures are quoted per apo structure, and with a family count where one is
  available, because an apo entry paired with several holo entries would
  otherwise be counted several times, and because conformational states of one
  RNA are not independent measurements of it.

  Hydrogens are dropped from nucleic input before detection, since every
  benchmark above was measured on heavy atoms only and keeping them changes both
  the atom count and the pocket count.

- `--backend rna-nma`, to request the nucleic elastic network explicitly. The
  default `auto` already selects it from the structure, so this is for scripted
  runs that want to be explicit or to fail loudly on unexpected input.

### Fixed
- **Lacuna no longer prints a confident ranked table for RNA built on random
  noise.** The NMA backend keyed on atoms named CA. RNA has none, so the backend
  silently fell back to random perturbation and still produced a full report with
  crypticity and persistence columns. That fallback recovered 0 of 10 held-out
  closed RNA sites and stretched the backbone by 25% on every structure, which
  makes the output meaningless rather than merely weak. Nucleic input without a
  usable bead set is now refused with an explanation instead of answered.

- Guarded the alpha-sphere bounds against ranker geometry drift.
  `_FITTED_GEOMETRY` couples the learned ranker's weights to the carving
  constants, and `ALPHA_DIST_MIN` and `ALPHA_DIST_MAX` were not among the
  constants it pinned. A change to either silently degraded ranking toward the
  random null rather than failing, so both are now pinned at the values the
  ranker was fitted against, 1.6 and 6.0 A.

- Corrected the version note in `skills/lacuna/SKILL.md`, which attributed
  `--detector surface-fusion` and `--no-sequence` to 1.3.0. Both arrived in
  1.1.0.

- Corrected the stale software version in `CITATION.cff`, which still read 1.0.1.

### Changed, from commits that landed after 1.2.0 and ship here for the first time
- **Retired the advice to pass `--rank-by druggability` for always-open or
  orthosteric sites.** Three user-facing places still gave it: the `--rank-by`
  help text, the `RANK_STRATEGIES` comment in `clusterer.py` and
  `skills/lacuna/SKILL.md`. It was measured wrong by 20 points. On COACH420,
  paired n=145, `learned` recovers 86.9% against 66.9% for `druggability`,
  +20.1 [+13.2, +27.1], with 30 structures flipping in favour of `learned` and
  one against. All three now say `learned` is correct for cryptic and
  always-open sites alike, and describe `druggability` as the legacy pre-model
  rule kept for reproducibility. If you scripted `--rank-by druggability` on
  that advice, this is the release to drop it.

- Corrected three ranker figures that had drifted from the CryptoBench test fold
  at n=180 (`learned` 55.6%, `learned-plm` 66.1%, `crypticity` 17.8%). The
  `--rank-by` help said `learned` recovers roughly twice what the analytic rules
  do; the ratio is 3.1x. `SKILL.md` quoted 66.3% against 55.9%, which spliced
  two different cohorts together. The `clusterer.py` comment quoted 57.0%, which
  is not achievable at n=180. Strings and comments only, no behaviour change.

- Pointed the bioRxiv badge at v2 of the preprint.

### Not included
This release makes RNA work on the same footing as protein. It does not add the
RNA-specific cryptic-pocket machinery developed alongside it: there is no
ensemble candidate short list, no per-size recovery calibration and no cavity
hotspot splitting here. Those are measured but held back, because the strongest
figures behind them rest on too few independent RNA families to publish as
rates. See the project roadmap for what is measured and what is still provisional.

## [1.2.0] - 2026-09-15

### Added
- `--emit-conformers`: write the generated conformational ensemble as a
  multi-model PDB (`<input_stem>_ensemble.pdb`), with the input structure as
  model 1 and the generated conformers as subsequent models. Loadable as a
  trajectory in PyMOL/ChimeraX/VMD and splittable per frame for docking or MD.

### Fixed
- Corrected the CryptoBench citation across the docs to Škrhák et al. 2025
  (*Bioinformatics* 41(1):btae745), previously mis-attributed to Vavra et al.
  2024. Thanks to Vít Škrhák for flagging it.

## [1.1.0] - 2026-09-07

### Added
- **A learned surface detector, and `--detector surface-fusion` to pool it with
  the geometric one.** The alpha detector only proposes candidates where there is
  a concavity, and a cryptic site in its apo form frequently has none: that is the
  definition of the problem rather than a fault in the detector. The surface
  detector scores the probe-accessible surface with a learned model and has no
  such filter. Pooling both, held out on CryptoBench, takes coverage from 68.5% to
  86.4% and top-five recovery from 57.1% to 73.9%. It selects the matching ranker
  automatically.

  The default stays `alpha`. Changing it would alter every existing user's results
  silently, and the fused pool is slower. Pass `--detector surface-fusion`
  deliberately.

- **`--no-sequence`**, which runs the surface detector on geometry alone and skips
  the ESM-2 pass. The cost is measured rather than assumed. Refitting on a
  geometry-only fused pool over 746 training targets:

  | pool and ranker | recovery | vs alpha |
  |---|---|---|
  | alpha pool, shipped ranker | 55.0% | reference |
  | fused pool, refit on geometry only | 60.2% | +5.23 [+2.14, +8.31] |
  | fused pool, shipped `fused_ranker.npz` | 60.5% | +5.50 [+2.41, +8.58] |

  Against +12.6 [+9.3, +16.0] with sequence on the same folds, so geometry alone
  holds a little under half the gain and still resolves clear of zero. On 1AKE at
  ten conformers the sequence pass is half the wall clock: 10.4 s without, 20.7 s
  with, against 2.4 s for alpha.

  Notably, `fused_ranker.npz` was fitted on a pool built *with* sequence features
  and an installation without torch feeds it a pool built without them. That
  mismatch costs nothing measurable, +5.50 against +5.23 for a ranker fitted on
  the right pool, so a second ranker would have been complexity for no gain.

- **A local stdio MCP server** exposing pocket discovery to Claude, and
  `skills/lacuna/SKILL.md` so the repository can be imported as a Claude skill.

### Fixed
- **`--backend auto` always chose boltz, so a base install could not run.** Every
  backend module imports cleanly whether or not its heavy dependency is present,
  because `import boltz` happens inside `generate()` rather than at module scope.
  The `except ImportError` around `_resolve_backend` therefore never fired and the
  rest of the chain was dead code. A plain `pip install lacuna-pockets` followed by
  `lacuna discover protein.pdb` failed at generation time despite nma being
  available. Auto now probes the dependency rather than the module.

## [1.0.3] - 2026-09-04

### Fixed
- **Interface sites could not be read at all.** A site spanning two chains names
  them as a hyphenated pair, which is how CryptoBench writes them: `G-H`.
  `load_structure` compared that string to a chain id, matched nothing, and
  returned an empty `Structure`. The failure then surfaced wherever an empty
  coordinate array first reached a reduction, with an error naming neither the
  chain nor the file. Because nothing reported these as errors they were simply
  absent: 123 of 885 CryptoBench training entries and 38 of 222 test entries,
  13.9% and 17.1%. Every benchmark figure published for Lacuna is therefore
  single-chain by construction, which is also why the reported test cohort is
  179 rather than 222. `chain=` now accepts `"A"`, `"G-H"` and `"A,B"`.

### Changed
- A `chain=` selector that matches nothing raises `ValueError` and names the
  chains the file does contain, rather than returning an empty structure.
  Silence is what let the bug above go unnoticed.

## [1.0.2] - 2026-09-02

### Fixed
- **`--backend auto` always chose Boltz, so a base install could not run.** The
  selector walked boltz, openmm, nma, random and caught `ImportError` to skip
  what was not installed. It never fired. Each backend module imports cleanly
  without its heavy dependency, because `import boltz` happens inside
  `generate()` rather than at module scope, so the first candidate always
  resolved and the rest of the chain was unreachable. After `pip install
  lacuna-pockets`, a plain `lacuna discover protein.pdb` therefore spent thirty
  seconds preparing the structure and then failed with "No module named
  'boltz'", despite the nma backend being available and requiring nothing
  beyond numpy and scipy. In a clean virtualenv the same command now finishes
  in under six seconds. Selection probes the dependency rather than the backend
  module; an environment that already has Boltz installed is unaffected.
- The Boltz backend's install hint named the package `lacuna`, not
  `lacuna-pockets`. Following it installed an unrelated project, or nothing.

### Added
- Links to the hosted webservers on Tamarind Bio and Neurosnap, for running
  Lacuna without installing anything.

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
