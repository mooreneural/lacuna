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

| Detector | Size-robust top-5 | Legacy recall | Paired difference vs Lacuna |
|----------|:-----------------:|:-------------:|-----------------------------|
| P2Rank | 63.3% (114/180) | 82% | -6.7% CI[-14.4, +0.6], includes zero |
| **Lacuna** | **56.7% (102/180)** | 68% | - |
| MDpocket (best of 10 configs) | 44.1% (79/179) | - | +11.7% CI[+3.9, +19.6] |
| fpocket | 28.3% (51/180) | 32% | +28.3% CI[+20.0, +36.7] |

Three-way union (Lacuna ∪ fpocket ∪ P2Rank): 76.7%. The tools remain
complementary; 11 structures on this fold are recovered by Lacuna alone.

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

## Curated cryptic set: 9 / 22

Hand-assembled from published cryptic-pocket case studies, and the hardest of
the three benchmarks despite being the smallest: it is deliberately enriched for
the large-motion sites this pipeline handles worst.

| Metric | Result |
|--------|--------|
| Size-robust (Jaccard ≥ 0.25 or centroid ≤ 4 Å) | **9/22 (41%)** |
| Legacy recall (≥ 30% or centroid ≤ 4 Å) | 14/22 (64%) |

Top-k curve (all 22): top-1 4/22, top-3 6/22, top-5 9/22, top-10 13/22,
top-20 15/22.

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
| `learned` (default) | fitted linear ranker over 23 features | 9/22 (41%) |
| `crypticity` | most cryptic sites first (previous default) | 7/22 (32%) |

**The default is not the winner here, and that is worth stating plainly.** On
CryptoBench's test fold (n=180) `learned` recovers 57.0% against 17.8% for
`crypticity`, an interval-separated gap on the largest and most diverse
benchmark, which is why it ships as the default. On this 22-target set the
ordering reverses. At n=22 the confidence intervals overlap heavily
(`persistence` [41%, 77%] vs `learned` [23%, 64%]), so the two results are not
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
structures, against the 56.7% that reach the top 5. The site is usually found
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

That last row is the informative one. If per-point scoring were the missing
*ranking* ingredient, handing the ranker P2Rank's own opinion of each cluster
would have helped. It did not. P2Rank's advantage lies in proposing different
candidates, not in ordering ours better.

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
