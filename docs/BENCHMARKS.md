# Benchmarks: full detail

Per-target breakdowns, ablations and timings behind the headline results in the
[README](../README.md#benchmarks). Every number here is reproducible with the
command shown; each benchmark script prints the full per-metric breakdown
(centroid, Jaccard at 0.20/0.25/0.30, and legacy recall) when you run it.

Unless stated otherwise: NMA backend, 20 conformers, the default `learned`
ranker, top-5, and the size-robust criterion (Jaccard ≥ 0.25 **or** centroid
≤ 4 Å).

## Head-to-head: four detectors, one criterion

CryptoBench's held-out test fold. MDpocket receives the **same NMA ensemble**
Lacuna uses, so that row isolates detection, aggregation and ranking rather than
the sampler. fpocket and P2Rank are single-structure tools and see the apo
structure, which is their intended usage.

All rows paired on the same 180 structures, and every Lacuna number
re-measured end to end after the last change to the ranker weights. An earlier
revision of this table mixed measurements taken days apart, which is why the
figures moved slightly.

| Detector | Size-robust top-5 | Legacy recall | Paired vs `learned-plm` |
|----------|:-----------------:|:-------------:|-----------------------------|
| **Lacuna** (`learned-plm`) | **66.1% (119/180)** | 82% | - |
| P2Rank | 63.3% (114/180) | 82% | +2.8% CI[-4.4, +9.4], includes zero |
| **Lacuna** (`learned`, default) | **55.6% (100/180)** | 77% | +10.6% CI[+6.1, +15.0] |
| MDpocket (best of 10 configs) | 43.9% (79/180) | - | +22.2% CI[+14.4, +30.0] |
| fpocket | 28.3% (51/180) | 32% | +37.8% CI[+29.4, +46.1] |

The sequence ranker is level with P2Rank: the interval on the difference spans
zero, so parity is the claim, not a win.

**The default trails P2Rank by 7.8 points (CI -15.0 to -0.6, excluding zero.)**
That is a change from what this file said previously. Before the
conformer-invariant refit the default scored 56.7% with a wider interval that
included zero, and it was described as indistinguishable from P2Rank. Refitting
cost about a point and tightened the interval, so the difference is now
resolvable and the earlier description no longer holds. The default still beats
MDpocket by +11.7% (CI +3.9 to +19.4).

Union of `learned-plm` and P2Rank: 76.1%, with 23 structures recovered by Lacuna
alone. Union of all five detectors: 79.4%. The tools remain complementary.

MDpocket is the closest relative of this work and the fair ensemble baseline. It
emits an occupancy grid rather than a ranked list, so its intended workflow is
visual inspection at a chosen isovalue. Benchmarking it requires thresholding
that grid, grouping the surviving voxels into pockets and ranking them. Because
that adaptation is ours and not the tool's, the isovalue and ranking rule were
swept and its **best** configuration is reported. Its default isovalue of 0.5
scores 40.2%; the most permissive setting tried (0.2) collapses to 33.0%, the
same over-merging failure described under
[clustering radius](#why-the-clustering-radius-is-2-å).

```bash
python benchmarks/compare_detectors_cryptobench.py --tools lacuna_learned --tag learned --folds test
python benchmarks/compare_detectors_cryptobench.py --analyze
python benchmarks/compare_mdpocket.py --folds test     # needs mdpocket on PATH
```

## Curated cryptic set: 10 / 22

Hand-assembled from published cryptic-pocket case studies, and the hardest of
the three benchmarks despite being the smallest: it is deliberately enriched for
the large-motion sites this pipeline handles worst.

| Metric | `learned` (default) | `learned-plm` |
|--------|--------|--------|
| Size-robust (Jaccard ≥ 0.25 or centroid ≤ 4 Å) | **10/22 (45%)** | 9/22 (41%) |
| Legacy recall (≥ 30% or centroid ≤ 4 Å) | 15/22 (68%) | - |

The default edges out the sequence ranker here by a single structure, the
opposite ordering to CryptoBench. One structure at n=22 means nothing by itself,
but it is worth stating that the sequence ranker's advantage is established on
CryptoBench and does not automatically carry over.

### By opening mechanism

Coarse literature labels for the dominant motion that opens each site.

| Mechanism | Recovered | Misses |
|-----------|:---------:|--------|
| sidechain | 2/4 | 1M47, 1HMV |
| loop | 2/6 | 1NB4, 2ERK, 2OZR, 1RTC |
| helix | 3/7 | 3CS9, 1LXL, 1G5M, 1JWP |
| hinge | 1/2 | 1V4S |
| interface | 1/3 | 2HBQ, 1ZJH |

Hinge and interface sites were previously 0/2 and 0/3; both now recover one
target. They remain the weakest classes, which is expected: a harmonic
elastic-network ensemble cannot generate large inter-domain or inter-subunit
motions. Dimer-interface pockets are partly addressable with `--homodimer`
(reads BIOMT records and builds the biological assembly), though this
benchmark's single-chain-referenced scoring does not credit them.

```bash
python benchmarks/cryptic_benchmark.py --category cryptic
python benchmarks/cryptic_benchmark.py --category cryptic --top-n 20   # detection ceiling
```

## Ranking strategies

`--rank-by` on the curated 22-target cryptic set:

| Strategy | Description | Curated 22 |
|----------|-------------|:----------:|
| `persistence` | legacy persistence × druggability | **13/22 (59%)** |
| `balanced` | druggability with a mild persistence bonus | **13/22 (59%)** |
| `druggability` | peak open-state composite druggability | 11/22 (50%) |
| `learned` (default) | fitted linear ranker over 23 features | 10/22 (45%) |
| `crypticity` | most cryptic sites first (previous default) | 7/22 (32%) |

**The default is not the winner here, and that is worth stating plainly.** On
CryptoBench's test fold (n=180) `learned` recovers 55.6% against 17.8% for
`crypticity`, an interval-separated gap on the largest and most diverse
benchmark, which is why it ships as the default. On this 22-target set the
ordering reverses. At n=22 the confidence intervals overlap heavily
(`persistence` [41%, 77%] vs `learned` [23%, 68%]), so the two results are not
formally in conflict, but the honest reading is that the learned ranker is tuned
to CryptoBench's distribution while the analytic rules do better on the classic
literature targets. If your proteins resemble the latter, try
`--rank-by persistence`.

## Orthosteric / conformational controls

| Category | Result | Notes |
|----------|--------|-------|
| Orthosteric | 4/6 | lysozyme, HIV-1 protease, DHFR, HIF-2α (1.0 Å centroid); misses thrombin, trypsin (1S0Q numbering) |
| Conformational | 0/1 | adenylate kinase open→closed |

Orthosteric recovery improved from 3/6. The single conformational target
(adenylate kinase) regressed from 1/1: it is an always-open active site that the
finer clustering radius now splits. With n=1 this is an anecdote rather than a
trend, but it is reported rather than dropped.

## COACH420: general binding sites, and where Lacuna's specialisation shows

COACH420 is a **general** ligand binding-site set of **holo** structures: the
ligand is present and the pocket is already open. It measures a different task
from the rest of this suite, and it is the direct answer to the reasonable
objection that Lacuna had only been measured on CryptoBench.

Ground truth follows P2Rank's own evaluation (relevant ligands from the
`coach420(mlig).ds` MOAD annotation, so ions and buffers do not count).
Everything below is the same size-robust criterion used everywhere else, and both
tools are paired on the 144 structures each of them scored.

| Detector | COACH420 (general, holo) | CryptoBench test fold (cryptic, apo) |
|----------|:------------------------:|:------------------------------------:|
| P2Rank | **93.8%** (135/144) | 63.3% |
| Lacuna (`learned-plm`) | not measured | **66.1%** |
| Lacuna (`learned`) | 86.8% (125/144) | 55.6% |
| Lacuna (`druggability`) | 66.7% (96/144) | - |

The COACH420 column was run with `learned`, the geometry-only strategy, so that
is the row to compare against P2Rank here; `learned-plm` has not been run on this
dataset. Paired on the 144 structures, `learned` trails P2Rank by **-6.9%
(CI -12.5 to -1.4, excludes zero)**, while on cryptic sites `learned-plm` is
nominally ahead of P2Rank (+2.8 points, an interval that spans zero).

**Each tool wins or ties on the task it was built for.** P2Rank is a
general-purpose predictor and is genuinely better at finding sites that are
already open; Lacuna's ensemble machinery buys nothing when nothing needs to
open. The honest reading of "parity with P2Rank" is therefore that parity holds
on cryptic sites specifically, and that a general-purpose detector should be
preferred for general-purpose work. Union of the two is 95.8%.

Absolute recovery is *higher* here than on CryptoBench (86.8% vs 55.6% for the
same strategy) simply because an open pocket is easier to find than a shut one.
Cross-dataset comparisons of the headline number are not meaningful; only the
within-dataset paired differences are.

### The `learned` ranker also wins on always-open sites

`learned` beats `druggability` by **+20.1% (CI +13.2 to +27.1)** on COACH420.
Earlier documentation advised `--rank-by druggability` for orthosteric and
general pocket finding. That advice predated the learned ranker and was wrong by
20 points; `learned` is now the right default for both cryptic and general work.

```bash
python benchmarks/coach420_benchmark.py --limit 150
python benchmarks/compare_p2rank_coach420.py --limit 150   # needs P2Rank on PATH
```

## Why the clustering radius is 2 Å

Alpha points are dilated before connected components are labelled, so points
roughly twice the radius apart fuse into one pocket. At the previous 4 Å setting
that cascaded across connected surface grooves: on CryptoBench 21% of structures
had the true site **fully covered** (median recall 100%) by a pocket carrying
~59 lining residues against ~8 known, far too diffuse to score as localized,
while only 1% of structures missed the site outright.

Halving the radius to 2 Å splits those blobs. It also cuts some genuine sites
apart, lowering the best-achievable overlap, but candidates per structure fall
from ~59 to ~16 and the ranking gain more than compensates. Every variant that
recovered the lost coverage by *adding* candidates lost at top-5:

| Variant | Effect |
|---------|--------|
| 2 Å (shipped) | baseline |
| pooling 4 Å + 2 Å scales | best-achievable overlap back to 100% on solved cases, top-5 worse (~58% projected vs ~61%) |
| lowering `MIN_VOLUME_A3` 80 → 30 | coverage up, top-5 worse (~56-57% projected) |
| adaptive re-split of oversized pockets only | +2.0% CI[+0.4, +4.0]: real but small, and costs a second detection pass |

Candidate count dominates: coverage you cannot rank is worth less than a
smaller, cleaner candidate set.

**The ranker weights are tied to this geometry.** After the radius change the
previous weights scored at the random-selection null on the new pockets. Any
change to detection constants requires refitting via
`benchmarks/train_ranker.py --fit`.

## Where the remaining gap is

Some cluster in the candidate set clears the criterion for 73.7% of test-fold
structures, against the 66.1% that `learned-plm` reaches in the top 5 (55.6%
for the default). The site is usually found
and then out-ranked, and most of the loss sits just outside the cutoff: the
correct cluster is at rank 6-8 for 14 structures, and top-8 recovery is already
64.8%.

These attempts to close it produced no measurable gain, and are recorded so they
are not repeated:

| Attempt | Result |
|---------|--------|
| spatial non-maximum suppression | +0.0% at any radius ≤ 9 Å; -6.0% at 12 Å |
| merging adjacent sub-pockets | 0% of structures are fragmented, so nothing to merge |
| hard-negative mining (added to uniform pairs) | CV +1.9% CI[-0.4, +4.2]; test fold -1.1% |
| gradient boosting instead of linear | +0.7% CI[-1.6, +3.0], not separable |
| P2Rank's per-pocket confidence as a feature | test fold -1.1% CI[-4.5, +2.2] |
| pruning the candidate set before ranking | +0.4% CI[+0.0, +1.1] in-sample on the train folds, best of a full single-feature threshold sweep |
| buriedness-weighted lining (buried core only) | monotonically negative; -9.9% CI[-17.4, -2.5] at the deepest 10% |
| multi-crystal experimental ensembles | candidates per target 11.7 -> 27.2 at equal conformer count |

Pruning deserves its own note, because the arithmetic looks so inviting: the
median structure carries 19 candidates and top-5 is 26% of that, so shrinking
the haystack ought to convert oracle into recovery. The candidate set really
does compress. A `plm_mean >= 0.29` filter drops 45% of all candidates while
losing 1.3% of true positives, and `plm_frac >= 0.15` drops 40% for 1.8%. The
condition everyone hopes for is met, and it still buys nothing.

The reason is that pruning and ranking read the same features. The candidates a
filter removes are the ones the ranker had already pushed down: the median rank
of a pruned candidate is 21, and only 2.7% of them sit in the top 5 at all.
Removing the tail of a list cannot change its head. Of the 463 false positives
that actually outrank a true site under `learned-plm`, that 45% prune removes
14%. Converting every convertible miss would take 295 *specific* removals, 3.1%
of the pool chosen exactly, which is a description of a better ranker rather
than of a filter.

Geometric filters are worse than useless: dropping the smallest 5% of candidates
by volume costs 13.6% of the true positives, because true sites concentrate in
the large-volume tail. This is the same fact that sinks a lower `MIN_VOLUME_A3`,
seen from the other end, and it is consistent with STILL_BIG dominating the
detection gap.

One combination does gain: the geometry-only `learned` ranker improves +3.6%
CI[+1.8, +5.6] under a `plm_mean` prune. That is sequence information reaching a
ranker that lacks it, not a property of pruning, and the result is still no
better than simply using the sequence-aware ranker (-0.4% CI[-2.4, +1.6] against
`learned-plm`). Nothing to adopt.

The P2Rank-confidence row is the informative one. If per-point scoring were the missing
*ranking* ingredient, handing the ranker P2Rank's own opinion of each cluster
would have helped. It did not. P2Rank's advantage lies in proposing different
candidates, not in ordering ours better.

### Trimming a pocket is not the same as dividing it

Four separate attempts have tried to fix the oversized-pocket class by making
each pocket's residue set smaller: a cross-conformer consensus threshold, a
hotspot-core radius, and now weighting lining residues by the burial of the
cavity voxels they touch. All three trim. All three failed. Measured on 121
train-fold structures, keeping only the most buried fraction of each cavity's
voxels when deriving lining residues:

| kept | mean lining | best Jaccard | oracle at Jaccard >= 0.30 |
|------|------------:|-------------:|--------------------------:|
| all (shipped) | 20.3 | 0.337 | 60.3% |
| deepest 75% | 19.4 | 0.338 | -0.8% |
| deepest 50% | 17.8 | 0.341 | +0.8% CI[-3.3, +5.0] |
| deepest 25% | 15.1 | 0.321 | -5.0% |
| deepest 10% | 12.2 | 0.301 | -9.9% CI[-17.4, -2.5] |

Monotone: mild trimming is neutral, aggressive trimming does significant harm.
An annotated site is the set of residues contacting a bound ligand, and a ligand
is not confined to the deepest part of a cavity, so residues at the mouth are
frequently in the true site. Trimming removes them from the intersection faster
than it removes false ones from the union, and the Jaccard falls.

Worth stating because a fourth attempt at the same class *did* work: watershed
splitting, which raised CV recovery +2.1% CI[+0.7, +3.7]. The difference is that
a split keeps every lining residue and reassigns it to the right sub-pocket,
while trimming discards residues outright. The oversized-pocket problem is a
boundary-placement problem, not a size problem, and the distinction predicts
which attempts are worth making.

First implementation of this used `MOUTH_DEPTH_A` as the core threshold, since
that is the depth `mouth_frac` already uses. It is a no-op: 93% of cavity voxels
lie deeper than it, so the "core" is the whole pocket and the lining sets come out
identical to three significant figures. The quantile above is what actually bites.

### Multi-crystal experimental ensembles

Other PDB depositions of the same UniProt, used as the conformational ensemble in
place of normal-mode sampling. This is the only source of genuinely different
conformers available at no compute cost, and NMA's oracle asymptotes regardless of
how many conformers are drawn, so it is the obvious thing to try.

Both arms were run at the *same* conformer count, because adding conformers is
the move that has failed every previous time. At equal budget, on 23 train-fold
targets, real crystal ensembles more than double the candidate set:

| | NMA | multi-crystal |
|---|---:|---:|
| candidates per target | 11.7 | 27.2 |
| mean best Jaccard | 0.236 | 0.259 |
| oracle at Jaccard >= 0.25 | 43.5% | 39.1% |

Candidate inflation at fixed conformer count is the finding. Crystal structures
of one protein differ enough in loop conformation and in which regions are
resolved that their pockets do not cluster across the ensemble the way
normal-mode conformers do, so each frame contributes its own candidates instead
of reinforcing shared ones. Given that coverage bought by adding candidates has
never converted here, that is disqualifying on its own. It survives restricting
frames to those matching at least 90% of the reference atoms, so it is not an
artifact of chimeric frames, which are a real hazard: unmatched atoms keep their
*reference* coordinates, and the backend only warns below 50%.

Two limits on this. n=23 makes the oracle comparison genuinely inconclusive
rather than proven flat. And these are the PDB-richest targets in the train
folds, so it is a best case rather than a typical one.

Coverage is the other problem, independent of any of that. 99% of targets have
another entry for the same UniProt and 90% have one that is not a CryptoBench
holo partner, but excluding entries with anything bound near the annotated site
leaves roughly 30% of targets with the five conformers this needs. For 1a4u it
leaves none: every other deposition of that protein has a ligand at the site,
which is the reason it is in a cryptic-pocket benchmark in the first place.

The leakage screen is the reusable part, and it validates: told only the site
residues, it independently flags all seven other depositions of 1a4u as bound at
the site, rediscovering their holo status without being given it. It rejects
entries whose residue-numbering correspondence cannot be established rather than
trusting them, which costs coverage and is what makes the rest of the claim
sound. Identifying the corresponding chain has to be part of that: for a
ribosomal target, reading site residue numbers from whichever chain comes first
returns rRNA nucleotides and rejects every entry for the wrong reason.

## Speed (NMA backend, no GPU)

Wall clock for the full pipeline (ensemble generation, per-conformer detection,
clustering, scoring, ranking) at 20 conformers on a laptop CPU.

| Protein | Residues | Time |
|---------|---------:|-----:|
| Interleukin-2 (1M47) | 122 | 0.7s |
| T4 lysozyme (1L90) | 162 | 0.9s |
| K-Ras (4OBE) | 339 | 1.1s |
| Glucokinase (1V4S) | 448 | 3.8s |
| PKM2 (1ZJH) | 507 | 4.5s |
| HIV-1 RT (1HMV) | 536 (chain A) | 7.3s |

The geometric descriptors added for the ranker cost roughly 15% of detection
time and reuse grids the detector already builds.

## Training and re-fitting the ranker

```bash
python benchmarks/train_ranker.py --dump features.jsonl --folds train-0,train-1,train-2,train-3
python benchmarks/train_ranker.py --cv  --dump features.jsonl   # model selection
python benchmarks/train_ranker.py --fit --dump features.jsonl --test-dump test.jsonl
```

Model selection uses leave-one-fold-out cross-validation across the train folds,
never the test fold, which is touched once for the final number. Two choices
were made that way: the geometry features are worth +6.9 points
(CI [+4.0, +9.7]), and training on within-structure pairs rather than individual
clusters adds +2.4 (CI [+0.7, +4.2]). A fit on shuffled labels scores at the
random null, confirming the gain is signal rather than an artifact of the
evaluation.
