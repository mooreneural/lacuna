# CryptoBench test fold: provenance of the full-fold re-measurement

Date: 2026-11-16. Author: Clayton W. Moore, Molecular & Environmental
Plant Sciences, Texas A&M University, College Station, TX, USA.

This records the harness defect that cut the CryptoBench test fold from 222 to 180
structures, the fix, and the before and after numbers. Numbers only; no manuscript,
figure or document was touched.

Two classes of path named below are deliberately not in the repository. The CIF
cache and the console logs under `cb_data/` are local harness artifacts excluded
by `.gitignore`, which keeps only `cb_data/folds.json`, the fold definition the
numbers depend on; the logs are quoted here wherever they carry evidence, so
nothing rests on reading them. The two `paper/software_paper_v3` files are the
companion software paper's own sources, which travel with that paper rather than
with this note. Every other path named below resolves in the repository.

## 1. The defect

`benchmarks/cryptobench_benchmark.py`, `main_pocket()`, built the known-residue set with

    parts = s.split("_")
    if len(parts) >= 2 and parts[0] == chain:

where `chain` came from `main["apo_chain"]`. For an interface site CryptoBench writes
every chain the site spans, hyphen-joined, so `apo_chain` can be `"M-O-P"`, `"G-H"`,
`"AAA-CCC"` or `"B-D-F-H"`, while each `apo_pocket_selection` entry is prefixed with a
single chain id from that set. The equality then never held, the residue set came back
empty, and `main()` dropped the entry with no message.

Data checked before any change (`cb_data/dataset.json`):

    4x19  apo_chain "M-O-P"    selections ['M_1','M_2','M_37','M_39','P_39','P_41','O_6','O_7','O_8','O_45','O_50','O_52']
    7nbc  apo_chain "AAA-CCC"  selections ['AAA_30','AAA_31',...,'CCC_24','CCC_50',...]
    1fd4  apo_chain "G-H"      selections ['G_5','G_6',...,'H_16','H_17',...]
    1fe6  apo_chain "A-D"      selections ['A_19','A_20','A_22','A_23','D_19','D_22','D_23','D_26']

Over all 222 test entries, no selection prefix lies outside the chain set its
`apo_chain` names, so membership in that set is the whole rule. The selections are
never prefixed with the composite string.

Cohort arithmetic: 222 test entries, 38 with a composite `apo_chain`, 4 above the old
700-residue ceiling (`7f4y` B 901, `2aka` A 764, `7o1i` A 726, `4zoe` B 724, printed as
skips in `cb_data/cryptobench_run.log`, which ends `[42 skipped]`). 222 - 38 - 4 = 180.

## 2. The fix

`benchmarks/cryptobench_benchmark.py`:

- `_selection_residues()` matches each selection prefix against
  `chain_selector(apo_chain)`, the set of chain ids, instead of the composite string.
  `lacuna.io.structure.chain_selector` and `load_structure` already accepted the
  hyphenated form, so the whole defect was in the harness.
- `main_pocket()` keeps its two-value signature and bare sequence numbers, because
  seven other scripts in this directory import it and because the residue metrics in
  `cryptic_benchmark.py` parse Lacuna's lining labels into bare numbers.
- `main_pocket_pairs()` is new and keeps each residue's chain. `known_centroid()` uses
  it, so the reference centroid is chain aware: in 1fe6 (`A-D`) a chain-blind lookup
  also picks up A_26 for D_26 and moves the centroid.
- `site_radius()` records the rms Ca spread of each site, so an interface annotation
  that was really two separate patches would be visible rather than scored as one
  pocket. Measured: the 38 interface sites have median radius 8.5 A and maximum
  11.0 A, against 8.4 A and 12.3 A for the 184 single-chain sites. Every interface
  site in this fold is as compact as an ordinary one, so none needed separate
  treatment.
- Residue ceiling. 700 was a bare constant with no guard behind it in the library.
  Measured on this machine: `8hc1` d-h, 1138 residues, 14.7 s, 458 MB peak RSS;
  `7f4y` B, 901 residues, 8.8 s, 200 MB, against about 2 to 3 s for a median entry.
  The machine takes them, so `DEFAULT_MAX_RESIDUES = 1200` covers the whole fold and
  `--max-residues` exposes it. `MAX_RESIDUES = 700` is left at its old value because
  the comparison scripts here import it and their stored baselines were collected
  under it.
- Skip honesty. Every unscored structure is printed as `[unscored] <id><chain>:
  <reason>` and written to the result JSON under `unscored` with a reason and detail.
  Reasons: `no_dataset_entry`, `empty_known_residues`, `over_residue_ceiling`,
  `under_residue_floor`, `exception`. The summary prints `scored N of M attempted`
  and a strict rate that counts every unscored structure as a miss, so this class of
  defect cannot recur without showing up in the headline.
- The run now writes `benchmarks/cryptobench_results.json`: summary, unscored list,
  and per structure the top-5, top-8 and any-rank Jaccard, recall, centroid distance
  and rank.

## 3. Machine

    Windows 11 Home 10.0.26200 (Windows-10-10.0.26200-SP0)
    AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD, 24 logical CPUs
    CPython 3.11.9 (tags/v3.11.9:de54cf5) [MSC v.1938 64 bit (AMD64)]
    numpy 2.4.6
    lacuna imported from the repository working tree, not from an installed copy
    repo HEAD ed21ba8 "docs(code): reword notes in the code tree"
    pyproject version 1.5.0; the installed lacuna-pockets dist-info still reads
    1.4.0, so lacuna.__version__ under-reports. The code exercised is the 1.5.0
    working tree.

All CIFs for the fold were already in `cb_data/cif` (3494 files), so nothing was
downloaded.

## 4. Commands

Before, the pre-fix logic (original `main_pocket`, original chain-blind
`known_centroid`, ceiling 700) copied verbatim into a scratch script with only
per-structure recording added, run over the full fold so it selects its own 180:

    python -I <scratch>/before_harness.py <scratch>/before_180.json

After, the fixed harness with the shipped 1.5.0 defaults (NMA backend, 20 conformers,
`rank_by` = `learned`):

    python -I benchmarks/cryptobench_benchmark.py --out benchmarks/cryptobench_results.json

Wall time: before 7.78 min for 180 structures; after 11.30 min for 222 structures. The
runs were sequential, never concurrent, so thread count was identical for both.

## 5. Regression on the 180 already scored

Per structure, on the 180 ids both runs score, comparing `n_known`, `n_residues`,
`n_clusters` and every field of the top-5, top-8 and any-rank blocks:

    per-structure differences: 0

Aggregates on that cohort are identical:

    cohort            n    top5          legacy        top8          ceiling
    before, 180      180   107 (59.4%)   132 (73.3%)   115 (63.9%)   135 (75.0%)
    after, same 180  180   107 (59.4%)   132 (73.3%)   115 (63.9%)   135 (75.0%)

No id present before is absent after.

## 6. Result, full fold

Criterion: centroid within 4.0 A of the site Ca centroid, or Jaccard at least 0.25.
Legacy is the size-gameable recall criterion at 0.30, reported alongside. Ceiling is
the same criterion at any rank.

    cohort                    n    top5          legacy        top8          ceiling
    full fold, after         222   136 (61.3%)   167 (75.2%)   149 (67.1%)   175 (78.8%)
    old 180, after           180   107 (59.4%)   132 (73.3%)   115 (63.9%)   135 (75.0%)
    42 newly included         42    29 (69.0%)    35 (83.3%)    34 (81.0%)    40 (95.2%)
      of which 38 composite   38    29 (76.3%)    33 (86.8%)    33 (86.8%)    36 (94.7%)
      of which 4 over 700      4     0 (0.0%)      2 (50.0%)     1 (25.0%)     4 (100.0%)

    unscored: 0 of 222.
    strict top 5 (unscored counted as misses): 136/222 (61.3%), equal to the scored
    rate because nothing is unscored.

**Retracted in Section 11, which supersedes the two sentences that follow.** They were
measured with the chain-blind residue metric, which over-credited every interface
structure. Corrected, the headline falls to 56.8% on 222, below the 57.8% on 180, and
the 38 interface sites score 57.9% against 56.5% for the 184 single-chain structures.
That near-tie at top 5 is two opposed abilities cancelling, not a cohort of ordinary
difficulty; see 11.7 for the coverage and conversion decomposition, and for why n = 38
cannot resolve either half of it. The figures in this section are left as measured, as
the record of what the cohort fix alone produced; they are not the final ones.

The headline rose, from 59.4% on 180 to 61.3% on 222. The 38 interface sites are
easier for this pipeline than the average single-chain site, 76.3% against 59.4%, and
they outweigh the four large structures, which all miss at top 5:

    2aka A  764 res  jac 0.15  dist 10.52 A   9.3 s
    4zoe B  724 res  jac 0.20  dist  8.71 A   6.5 s
    7f4y B  901 res  jac 0.00  dist 22.79 A  12.1 s
    7o1i A  726 res  jac 0.00  dist 31.48 A   7.4 s

All four clear the criterion at some rank, so they are ranking failures, not detection
failures.

The published headline, 57.8% (104/180), is the Jaccard-only criterion produced by
`compare_detectors_cryptobench.py`, which drops the centroid term. Recomputing that
criterion from the same per-structure output confirms the chain end to end and gives
the comparable full-fold figure:

    cohort                   jaccard only        centroid or jaccard
    old 180                  104/180 (57.8%)     107/180 (59.4%)
    full fold 222            133/222 (59.9%)     136/222 (61.3%)
    42 newly included         29/42  (69.0%)      29/42  (69.0%)

104/180 and 107/180 reproduce `docs/BENCHMARKS.md` lines 28 and 36 exactly.

## 7. Residual limitation, not introduced here

**Closed in Section 11.** This section is the record of the limitation as it stood
before the metric was fixed; the deferral it ends with no longer holds.

The residue metrics in `cryptic_benchmark.py` parse Lacuna's lining labels into bare
sequence numbers, so a residue number that occurs in two chains of an interface site
collapses to one. Eight entries are affected, all composite: 1fe6 8 pairs to 5 numbers,
4z0y 12 to 9, 5dy9 8 to 6, 5aon 19 to 17, 4x19 12 to 11, 2nt1 16 to 15, 1g1m 26 to 25,
4p32 21 to 20. The centroid criterion is chain aware and unaffected. Making the residue
metrics chain aware would change the 180 as well, which would destroy the like-for-like
comparison above, so it is left for a separate measurement.

## 8. Optional configurations

Superseded in part by Section 10: `cryptobench_benchmark.py` now has a `--rank-by`
flag, and the analytic `crypticity` arm has been measured on the full fold. The rest
of this section still stands.

Neither learned-plm nor surface-fusion can be re-measured from existing per-structure
output.

- `benchmarks/detectors_cb_current.jsonl` holds `lacuna_plm` rows for 180 ids only,
  with `recall`, `jaccard` and `n_prop` but no centroid distance and no `jac_by_rank`.
  The headline criterion needs the centroid, so it cannot be recomputed even for the
  180, and the 42 newly included structures have no row at all.
- No CryptoBench surface-fusion output exists. `benchmarks/surface_runs/` holds
  `cryptic_surface-fusion_results.json` and `coach420_surface-fusion.log`, which are
  the cryptic and COACH420 benchmarks, not this fold.

What a fresh measurement would take, per configuration: one full-fold run of the same
shape as the learned run above, 11.3 min wall for 222 structures on this machine.
`cryptobench_benchmark.py` hardcodes `rank_by=DEFAULT_RANK_BY` and the default alpha
detector, so it would also need `--rank-by` and `--detector` flags, or
`compare_detectors_cryptobench.py`, which accepts both but records no centroid
distance and so cannot produce this criterion. learned-plm additionally needs the
`plm` extra and locally cached ESM weights; `lacuna.pockets.plm.available()` returns
True on this machine.

## 9. Documentation that would need updating

**Superseded by Section 12**, which records what was actually edited and to which
values. This section is the record of what was outstanding when the cohort fix was
measured, and its figures are the chain-blind ones of Section 6.

Not edited at the time this section was written. What they would need:

- `docs/BENCHMARKS.md` line 28, the headline row: cohort 180 becomes 222, and the
  Jaccard-only figure 57.8% (104/180) becomes 59.9% (133/222). Line 36, which quotes
  `cryptobench_benchmark.py` at 107/180 (59.4%), becomes 136/222 (61.3%). Top 8 is
  67.1% (149/222) and the any-rank coverage ceiling 78.8% (175/222).
- `docs/BENCHMARKS.md` lines 164 and 173 to 183, the head-to-head table: every row is
  paired on the same 180 structures, so none of it can move until the baselines are
  re-run on the same cohort. Those rows should keep their 180 cohort and say so, or
  the whole table has to be re-measured. Only the Lacuna default row has a full-fold
  number today.
- `README.md` lines 149, 167, 169 and 172: the same two figures and the cohort size.
- Both should state that the fold is now scored in full with nothing skipped.

One consequence to flag. `compare_detectors_cryptobench.py`, the script behind the
published headline, imports `main_pocket` from `cryptobench_benchmark.py` and so
inherits the fix, but its `--max-residues` still defaults to the legacy 700. Twelve
of the 222 entries exceed 700 residues once the interface chains are loaded, the four
single-chain ones plus eight composite, so at its default it would now score 210, not
222, and give 128/210 (61.0%) Jaccard only. Run it with `--max-residues 1200` to get
the full fold. A separate decision is whether to move its default too; that would
change the cohort of the stored fpocket, P2Rank and MDpocket baselines, which is why
the constant was left alone here.

## 10. The ranking ablation, re-measured on the full fold

Date: 2026-11-16, same machine and same working tree as Sections 3 to 7.

The paper's "+27.2 points (95% CI +20.6 to +34.4)" compared the default `learned`
ranker on 180 structures against the analytic `crypticity` rule on 180 structures.
With the fold now scored in full, both arms were re-measured on all 222.

### 10.1 Harness change

`cryptobench_benchmark.py` gained one flag, `--rank-by`, with `choices` taken from
`lacuna.pockets.clusterer.RANK_STRATEGIES` and `default=DEFAULT_RANK_BY`. Three call
sites now read `args.rank_by` instead of the module constant: the `run_lacuna` call,
the `rank_by` field of the result summary, and the banner line, which had been
printing a hardcoded "crypticity" since the default moved to `learned`. Nothing else
changed, and the default run is identical in configuration to Section 4: running
without the flag reproduces `rank_by: "learned"` in the summary.

### 10.2 How the candidate set was held fixed

An ablation only means something if the two arms rank the same candidates. Three
things make that true here rather than assumed:

1. By construction. In `cryptic_benchmark.run_lacuna` the ensemble backend, the
   detector and `cluster_pockets` all run before any ranking, and in
   `lacuna/pockets/clusterer.py` `cluster_pockets` builds every `PocketCluster` first
   and only then applies `clusters.sort(key=_rank_key(c, rank_by, ...))`. `rank_by`
   enters nowhere else, except that `learned-plm` also asks for sequence
   probabilities, which `crypticity` does not. Both arms therefore used the same NMA
   backend, the same 20 conformers, the same `alpha` detector and the same clustering.
2. Same invocation. Both runs used the shipped defaults and differed only in the new
   flag: `--conformers 20`, `--max-residues 1200`, fold `test`, backend NMA, detector
   `alpha`. The same 3494 cached CIFs in `cb_data/cif` were read, nothing was
   downloaded, and the two runs were sequential and never concurrent, so the thread
   count was identical for both (the 1.5.0 sampler is deterministic, but the pools
   were still collected under identical settings).
3. Measured, per structure, over all 222. Comparing the two result files field by
   field: `n_clusters`, `n_residues`, `n_known`, `site_radius_a` and the whole
   any-rank block, which is the best value over every cluster and so does not depend
   on the ordering:

       per-structure candidate-set differences: 0

   The any-rank coverage ceiling is identical in both arms, 170/222 Jaccard only and
   175/222 under centroid-or-Jaccard. Only the ordering differs, which is what makes
   this an ablation on one candidate set rather than two experiments.

### 10.3 Result

Criterion as elsewhere in this note. Jaccard only is the published headline criterion
(`compare_detectors_cryptobench.py`); centroid-or-Jaccard is the harness criterion.
Confidence intervals are on the paired difference, see 10.4.

    criterion               learned        crypticity      paired difference
    Jaccard only >= 0.25    133/222 59.9%  72/222  32.4%   +27.5  95% CI [+21.2, +33.8]
    centroid<=4A or Jac     136/222 61.3%  73/222  32.9%   +28.4  95% CI [+22.1, +34.7]

Discordance, Jaccard only: learned alone 65, crypticity alone 4, both 68, neither 85.
Ratio learned to crypticity 1.85 (1.86 under centroid-or-Jaccard).

Other columns of the `crypticity` arm on 222, for completeness:

    top 5 legacy recall >= 0.30   122/222 (55.0%)
    top 8, centroid-or-Jaccard     98/222 (44.1%)
    top 8, Jaccard only            97/222
    coverage ceiling, any rank    175/222 (78.8%), Jaccard only 170/222

The gain is larger on the full fold than on the 180, not smaller: +27.5 points
against +27.2 under the published criterion, and the interval is slightly tighter
because the denominator is larger.

### 10.4 The interval, and why this method

The method was not chosen here, it was found. The recomputation of the published
1.5.0 figures produces the ranking gain with `_paired_ci` from
`benchmarks/train_ranker.py`, imported from that file rather than reimplemented:
a bootstrap over structures
of the paired difference in hit rate, `n_boot = 10000`, `seed = 0`, percentile
interval at 2.5 and 97.5. The per-arm intervals in the docs come from
`paired_bootstrap_ci` in `benchmarks/metrics.py` instead, with `n_boot = 2000`; that
one is a one-arm rate rather than a difference, so it is not the published gain
interval.

Confirmed by reproduction before it was applied to anything new. On the stored 180
rows, the `lacuna_learned` rows of `benchmarks/detectors_cb_current.jsonl` and the
`lacuna_nma` rows of `benchmarks/provenance/detectors_cb_r15crypticity_180.jsonl`:

    learned 104/180 (57.8%), crypticity 55/180 (30.6%)
    paired difference +27.2  95% CI [+20.6, +34.4]

which is the published figure to the digit. The same method was then applied to the
222 rows above.

The new `crypticity` run also reproduces the published analytic arm on the old
cohort. Restricting its own 222 per-structure rows to the 180 the old harness scored,
that is single-chain and at most 700 residues:

    criterion        learned     crypticity   paired difference
    Jaccard only     104/180     55/180       +27.2  95% CI [+20.6, +34.4]
    cen-or-Jaccard   107/180     56/180       +28.3  95% CI [+21.1, +35.6]

104/180, 55/180 and +27.2 CI [+20.6, +34.4] are the published numbers, recovered from
a fresh run through the fixed harness. The old-cohort analytic figure is therefore
confirmed, not merely carried over.

Per structure, not just in aggregate. The 180-id sub-cohort of the new run is exactly
the id set of the stored `lacuna_nma` rows in
`benchmarks/provenance/detectors_cb_r15crypticity_180.jsonl`, and
the hit decision agrees on all 180 of 180. The only per-structure differences are
rounding: the stored rows carry three decimals of Jaccard, the new records four, so
116 rows differ in the fourth decimal and none in the decision.

### 10.5 Commands and cost

    python -I benchmarks/cryptobench_benchmark.py --rank-by crypticity \
        --out benchmarks/cryptobench_results_crypticity.json

Wall 10.76 min for 222 structures, against 11.34 min for the default `learned` run of
Section 4. Records:

    benchmarks/cryptobench_results.json                learned, 222 (Section 4)
    benchmarks/cryptobench_results_crypticity.json     crypticity, 222
    benchmarks/cb_data/cryptobench_crypticity_222.log  its console log
    benchmarks/provenance/ablate_ranker_222.py         the comparison, output in
        benchmarks/provenance/ablation_222.out
    benchmarks/provenance/ci_method_check_180.py       the interval reproduction

### 10.6 learned-plm: measurable, and not for want of anything

Checked without installing anything and without touching the network.

    plm extra            torch 2.11.0+cu128 and transformers 5.14.1 are both
                         importable under the CPython 3.11.9 interpreter
                         recorded in Section 3
    fitted head          lacuna/pockets/plm_head.npz present, 5968 bytes
    available()          lacuna.pockets.plm.available() returns True
    ESM-2 weights        facebook/esm2_t33_650M_UR50D cached locally under
                         ~/.cache/huggingface/hub, snapshot 08e4846e5371...,
                         2.5 GB, with config, tokenizer and model.safetensors
    offline load         _load_model() succeeds in 11.4 s with HF_HUB_OFFLINE=1
                         and TRANSFORMERS_OFFLINE=1 set, so nothing is fetched

Nothing is missing, so the paper does not have to label learned-plm as a
180-structure measurement for want of a dependency. What is missing is the run: it
was not performed here, because this step was scoped to the analytic ablation. With
the new flag it is one command, `--rank-by learned-plm`, about 11 min plus the
embedding pass, and the candidate set stays fixed for the same reason as in 10.2,
since `learned-plm` leaves the `alpha` detector alone and the sequence probabilities
reach only the ranker.

Two cautions for whoever runs it. Set `HF_HUB_OFFLINE=1`, or `from_pretrained` makes
a revision check against huggingface.co before falling back to the cache, so an
unqualified run does reach the network even though the weights are local. And
`torch.cuda.is_available()` is True on this machine, where `_load_model` moves the
embedder to the GPU in half precision; the stored 66.1% (119/180) was measured with
1.4.0 and its device is not recorded here, so a new figure should say which it used.
Until that run exists, the published 66.1% remains a 180-structure, 1.4.0 number and
should keep saying so.

## 11. The residue metric made chain aware, and what it costs

Date: 2026-11-16, same machine and same working tree as Sections 3 to 7 and 10.

Section 7 above recorded the chain-blind residue parse as a residual limitation and
deferred it. It is no longer deferred. This section closes it and reports the
corrected numbers.

**The correction is downward.** The paper A headline, top 5 under the Jaccard term
alone on the CryptoBench designated test fold, falls from 133/222 to 126/222, that is
from 59.90990990990991% to 56.75675675675676%, a loss of 7 structures and
3.15315315315315 points. All 7 losses are interface structures. No single-chain
structure moved by any amount.

The corrected full-fold headline is also below the previously published 180-structure
headline, 56.75675675675676% against 57.77777777777778% (104/180). The sentence in
Section 6, "the headline rose, from 59.4% on 180 to 61.3% on 222", inverts once the
interface sites are scored correctly: under the Jaccard term the full fold is 1.02
points below the old cohort, and under centroid-or-Jaccard 1.34 points below.

### 11.1 The defect

`cryptic_benchmark._found_resnums` (L618) parses Lacuna lining labels,
`NAME+seq:chain`, by taking `label.split(":")[0]` and keeping its digits. The chain is
discarded. It feeds `residue_overlap` (L633) and `residue_jaccard` (L645), and that
Jaccard is the paper A criterion. `metrics.found_resnums` (L30) has the same parse.

On a single-chain structure this is harmless. On an interface structure loaded with
two or more chains it is not: a candidate pocket sitting entirely on chain A can be
credited against site residues that live on chain B whenever the two chains reuse a
sequence number, which for a homomultimer is every number.

All 38 interface entries of this fold are homomultimers, so every one was exposed.

### 11.2 The fix

Added beside the old functions rather than folded into them, because the stored
fpocket, P2Rank, MDpocket and IF-SitePred baselines in this directory were collected
under the chain-blind convention and several scripts import the old names directly:

- `cryptic_benchmark._found_respairs` (L674), `residue_overlap_pairs` (L695),
  `residue_jaccard_pairs` (L705). The sequence number is parsed by the identical
  expression, so the only difference between the two families is that one keeps the
  chain.
- `metrics.found_respairs`, the same parse for the shared-metric path.
- `cryptobench_benchmark.py` scores the residue terms with the pair functions against
  `main_pocket_pairs`, and records the chain-blind values alongside in every record,
  so the size of the correction stays auditable per structure.
- `compare_detectors_cryptobench.py` gained `--residue-metric blind|aware`, default
  `blind`, and stamps `residue_metric` into every row; `analyze` refuses a file that
  mixes the two. The default is `blind` because the stored baselines are, so a
  full-fold head-to-head needs `--residue-metric aware` and re-collected baselines.

The centroid term was already chain aware and did not move: comparing the stored
chain-blind run with the chain-aware run field by field, every `centroid_dist_a` in
every block of all 222 records is identical.

### 11.3 Route taken, and why

The learned arm was **re-scored from the existing per-structure records**, not re-run.
`benchmarks/cryptobench_results_chainaware.json` carries, for every structure and for
each of the top-5, top-8 and any-rank blocks, both the chain-aware Jaccard and recall
and the chain-blind `jaccard_chain_blind` and `recall_chain_blind`, plus the centroid
distance. That is enough to compute every figure below under both conventions from one
file.

The re-score is sound because that run reproduces the stored chain-blind run exactly.
Comparing `cryptobench_results.json` (Section 4) with the chain-blind fields of
`cryptobench_results_chainaware.json` over 222 structures, 3 blocks each, 3 fields per
block:

    per-structure field differences: 0

So the candidate sets, the pockets and the ordering are the same objects in both files,
and the entire movement below is the metric, not the run.

The analytic `crypticity` arm had to be **re-run**:
`cryptobench_results_crypticity.json` predates the fix and stores only the chain-blind
Jaccard, with no per-cluster lining residues, so it cannot be re-scored from disk. One
full-fold run with the fixed harness, 9.25 min wall. It reproduces its own predecessor
on the chain-blind fields exactly (72/222 Jaccard only, 73/222 centroid-or-Jaccard,
122/222 legacy recall, matching Section 10.3 to the structure).

### 11.4 Invariance on the single-chain cohort

The single-chain cohort of this fold is 184 structures, not 180. 222 test entries
minus 38 interface entries is 184. The 180 is the old cohort, single-chain **and**
under the legacy 700-residue ceiling, and is a subset of the 184.

Measured, per structure, on both: for every single-chain record, in all three blocks,
the chain-aware Jaccard equals the chain-blind Jaccard and the chain-aware recall
equals the chain-blind recall, in both the learned and the analytic arm.

    cohort        records x blocks x fields x arms      mismatches
    184 single      184 x 3 x 2 x 2 = 2208                  0
    180 old         180 x 3 x 2 x 2 = 2160                  0

The 184 proof is the stronger of the two and contains the 180. Every aggregate on
either cohort is therefore identical under the two conventions, which is what lets the
three-column table below attribute movement cleanly.

### 11.5 Result, full fold, under both criteria

Jaccard only is the paper A criterion (`compare_detectors_cryptobench.py`, Jaccard
>= 0.25). Centroid-or-Jaccard is the harness criterion (centroid <= 4.0 A of the site
CA centroid, or Jaccard >= 0.25). All 222 attempted, all 222 scored, 0 unscored, so
the strict rate equals the scored rate.

    chain-aware, n = 222         Jaccard only              centroid or Jaccard
    top 5                        126 (56.75675675675676%)  129 (58.108108108108105%)
    top 8                        139 (62.612612612612615%) 142 (63.96396396396396%)
    coverage, any rank           166 (74.77477477477477%)  171 (77.02702702702703%)
    no qualifying candidate       56 (25.225225225225227%)  51 (22.972972972972972%)
    first clears at rank 6 to 8   13                        13

The rank 6 to 8 count is 13 under both criteria and is unchanged by the correction:
the 7 structures lost at top 5 are lost at top 8 as well, so none of them reappears in
that band.

### 11.6 Three-column comparison

Jaccard only, the paper A criterion. The middle column is the number the paper
currently states.

    criterion: Jaccard >= 0.25     old 180         chain-blind 222   chain-aware 222
    n                               180             222               222
    top 5                           104 (57.78%)    133 (59.91%)      126 (56.76%)
    top 8                           112 (62.22%)    146 (65.77%)      139 (62.61%)
    coverage, any rank              131 (72.78%)    170 (76.58%)      166 (74.77%)
    no qualifying candidate          49 (27.22%)     52 (23.42%)       56 (25.23%)
    first clears at rank 6 to 8       8              13                13

    criterion: centroid <= 4 A or Jaccard >= 0.25
    top 5                           107 (59.44%)    136 (61.26%)      129 (58.11%)
    top 8                           115 (63.89%)    149 (67.12%)      142 (63.96%)
    coverage, any rank              135 (75.00%)    175 (78.83%)      171 (77.03%)
    no qualifying candidate          45 (25.00%)     47 (21.17%)       51 (22.97%)
    first clears at rank 6 to 8       8              13                13

Which part of the movement is which, Jaccard only, top 5:

    104/180  57.78%   the published 180-structure figure
      +29 hits, +42 structures   including the 42 the old harness dropped
    133/222  59.91%   chain-blind full fold, the figure the paper states
       -7 hits, +0 structures    scoring the interface sites with the chain
    126/222  56.76%   chain-aware full fold, the corrected figure

The first step is cohort, the second is purely the metric. Of the 29 apparent hits the
42 newly included structures contributed, 7 were artifacts of chain-blind scoring and
22 are real: 22/42 = 52.38095238095238%, against 29/42 = 69.04761904761905% as scored
before. The 4 structures above the old 700-residue ceiling contribute 0 hits at top 5
either way, so the whole of both steps sits on the 38 interface entries.

Said plainly: including the interface sites raised the headline by 2.13 points, of
which 3.15 points was credit for pockets on the wrong chain. Net of both, the
full-fold headline is 1.02 points **below** the published 180-structure headline.

### 11.7 The 38 interface structures alone, chain aware

    n = 38                        Jaccard only               centroid or Jaccard
    top 5                         22 (57.89473684210526%)    22 (57.89473684210526%)
    top 8                         26 (68.42105263157895%)    26 (68.42105263157895%)
    coverage, any rank            32 (84.21052631578948%)    32 (84.21052631578948%)
    no qualifying candidate        6 (15.789473684210526%)    6 (15.789473684210526%)
    first clears at rank 6 to 8    4                          4

The two criteria agree exactly on this cohort: not one interface structure is rescued
by the centroid term that the Jaccard term misses. As scored chain-blind the same
cohort read 29 (76.3157894736842%), 33 (86.84210526315789%), 36 (94.73684210526316%)
and 2 (5.2631578947368425%).

The claim in Section 6 that "the 38 interface sites are easier for this pipeline than
the average single-chain site, 76.3% against 59.4%" does not survive. Corrected, the
interface cohort scores 57.89% at top 5 against 56.52% for the 184 single-chain
structures under the Jaccard term, and 57.89% against 58.15% under
centroid-or-Jaccard.

At top 5 alone they therefore look ordinary, but that reading was wrong too, and for
the reason this project keeps measuring: top 5 is a product of two abilities and they
move in opposite directions here. Decomposed, Jaccard only:

    Jaccard only              interface n = 38        single chain n = 184
    coverage, any rank        32 (84.21052631578948%) 134 (72.82608695652174%)
    top 5                     22 (57.89473684210526%) 104 (56.52173913043478%)
    conversion, top5/coverage 68.75%                  77.61194029850746%
    coverage minus top 5      26.31578947368422 pts   16.304347826086957 pts

    centroid or Jaccard       interface n = 38        single chain n = 184
    coverage, any rank        32 (84.21052631578948%) 139 (75.54347826086956%)
    top 5                     22 (57.89473684210526%) 107 (58.15217391304348%)
    conversion, top5/coverage 68.75%                  76.97841726618705%
    coverage minus top 5      26.31578947368422 pts   17.391304347826086 pts

Interface sites are the easiest cohort in the fold to propose and the hardest to
convert: +11.38 points of coverage and -8.86 points of conversion under the Jaccard
term. The two nearly cancel at top 5, which is why the cohort reads as unremarkable
there, and the 38 are what lifts the full-fold coverage-to-top-5 gap from the
single-chain 16.30 points to 18.02.

Power. n = 38 cannot resolve either difference on its own. Neither figure above is an
independent measurement of the size of the effect; they are the shape of it, on the
cohort the old harness was dropping. Any statement built on them should say so.

Top-5 Jaccard moved on 25 of the 38; at some block it moved on 27 of the 38. The 7
whose decision changed, all from hit to miss, none rescued by the centroid:

    id    chain   n_known blind  n_known aware  Jaccard blind  Jaccard aware  centroid A
    5aon  A-B           17             19          0.4400         0.1111          9.401
    3la7  A-B           13             13          0.3824         0.2273          9.876
    3pfp  A-B           11             11          0.3448         0.1860          4.768
    5wm9  B-C           14             14          0.3214         0.2308          5.754
    3x0x  E-G           19             19          0.2903         0.0000         20.107
    5dy9  H-I            6              8          0.2941         0.2000          5.871
    2huw  A-B            7              7          0.2609         0.2083          6.658

`3x0x` is the clearest case: chain-aware Jaccard 0.0000 against 0.2903 chain blind. The
over-credited candidate is rank 3, 61 residues of lining, whose centroid lies 36.417 A
from the site's CA centroid. It shares no residue at all with the site and was credited
entirely through sequence numbers reused on the other chain, at a chain-blind recall of
0.9474, 18 of the site's 19 numbers.

Read the `centroid A` column above as what it is, and do not attribute a row's three
fields to one candidate. In every record this harness writes, the residue Jaccard, both
conventions, is a maximum over the rank block, while `centroid_dist_a` and `rank` come
from `pocket_min_centroid_dist` and belong to that block's nearest-centroid candidate.
The two can be different candidates, and on `3x0x` they are: the stored top-5 `rank` 4
and `centroid_dist_a` 20.107 are a 26-residue candidate scoring 0.0000 under both
conventions, not the candidate carrying the 0.2903. Per-cluster values for `3x0x`,
reproduced 2026-11-16 on the same machine and tree from one 20-conformer run of the
default configuration (7.89 s, the same 84 clusters, and it reproduces every stored
block field of that record exactly):

    rank  n_lining  Jaccard aware  Jaccard blind  recall blind  centroid A
      1        21         0.0000         0.0278        0.0526      28.677
      2        18         0.0000         0.0000        0.0000      22.255
      3        61         0.0000         0.2903        0.9474      36.417
      4        26         0.0000         0.0000        0.0000      20.107
      5        27         0.0000         0.0000        0.0000      22.557
      6        12         0.0000         0.0000        0.0000      26.665
      7        19         0.0000         0.0000        0.0000      18.264
      8        55         0.2131         0.2131        0.6842      10.055
      9        36         0.0000         0.1000        0.2632      34.418
     10        23         0.0000         0.0000        0.0000      28.163

The max of `Jaccard blind` over ranks 1 to 5 is 0.2903 at rank 3; the min of `centroid A`
over the same block is 20.107 at rank 4. That is the whole of the conflation. The same
caution applies to the other six rows of the table above, and to the top-5 columns of
every structure in the four result files.

Note the `n_known` columns. On 8 of the 38 the two chains of the site reuse a sequence
number, so the chain-blind known set is short by 1 to 3 residues as well: 1fe6 8 pairs
to 5 numbers, 4z0y 12 to 9, 5dy9 8 to 6, 5aon 19 to 17, 4x19 12 to 11, 2nt1 16 to 15,
1g1m 26 to 25, 4p32 21 to 20. The corrected metric scores against the full site.

### 11.8 Ranking ablation, re-measured chain aware

Same method as Section 10.4: `_paired_ci` from `benchmarks/train_ranker.py`, a
bootstrap over structures of the paired difference in hit rate, `n_boot = 10000`,
`seed = 0`, percentile interval at 2.5 and 97.5. Imported from that file rather than
reimplemented.

The candidate set is still held fixed across the two arms. Comparing the two
chain-aware result files field by field over all 222, on `n_clusters`, `n_residues`,
`n_known`, `n_known_chain_blind` and the whole any-rank block, which is the best value
over every cluster and so does not depend on ordering:

    per-structure candidate-set differences: 0

Result, full fold:

    chain aware, n = 222     learned          crypticity       paired difference
    Jaccard only >= 0.25     126/222 56.76%   64/222 28.83%    +27.927927927927925
                                                               95% CI [+21.62162162162162,
                                                                       +34.234234234234236]
    centroid <= 4 A or Jac   129/222 58.11%   65/222 29.28%    +28.82882882882883
                                                               95% CI [+22.07207207207207,
                                                                       +35.13513513513514]

Against the chain-blind figures of Section 10.3, which this run reproduces exactly:

    chain blind, n = 222     learned          crypticity       paired difference
    Jaccard only             133/222 59.91%   72/222 32.43%    +27.47747747747748
                                                               95% CI [+21.17117117117117,
                                                                       +33.78378378378378]
    centroid or Jaccard      136/222 61.26%   73/222 32.88%    +28.37837837837838
                                                               95% CI [+22.07207207207207,
                                                                       +34.68468468468468]

The ablation gain is slightly **larger** under the corrected metric, +27.93 against
+27.48 points, because the chain-blind credit was flattering the analytic arm more in
proportion than the learned one: the analytic arm loses 8 of 72 (11.1%) and the
learned arm 7 of 133 (5.3%). The interval moves by about half a point at each end.

Discordance, Jaccard only, chain aware: learned alone 67, crypticity alone 5, both 59,
neither 91. Ratio learned to crypticity 1.96875 (1.9846153846153847 under
centroid-or-Jaccard), so "roughly twice as many" still holds and is in fact closer to
exactly twice than it was.

Other columns of the analytic arm on 222, chain aware:

    top 5, legacy recall >= 0.30   117/222 (52.7%)
    top 8, centroid-or-Jaccard      89/222 (40.1%)
    top 8, Jaccard only             88/222
    coverage ceiling, any rank     171/222 (77.0%), Jaccard only 166/222

On the old 180 cohort both arms are unchanged by the correction, as 11.4 requires:
learned 104/180, crypticity 55/180, paired difference +27.22222222222222, 95% CI
[+20.555555555555554, +34.44444444444444], which is the published +27.2 CI
[+20.6, +34.4].

### 11.9 Every number in the paper that changes

`paper/software_paper_v3.md`. Numbers only; the file was not edited.

    line(s)        now                                  becomes
    22 to 23       59.9%                                56.8%
    28             27.5 points (95% CI +21.2 to +33.8)  27.9 points (95% CI +21.6 to +34.2)
    119 to 120     59.9% ... against 32.4%              56.8% ... against 28.8%
    555            61.3% (136/222)                      58.1% (129/222)
    594            59.9% (133/222)                      56.8% (126/222)
    613            32.4% (72/222)                       28.8% (64/222)
    613 to 614     59.9% (133/222)                      56.8% (126/222)
    614 to 615     27.5 points (95% CI +21.2 to +33.8)  27.9 points (95% CI +21.6 to +34.2)
    649            the full-fold 59.9% above            the full-fold 56.8% above
    679            32.4% against 59.9%                  28.8% against 56.8%
    895            59.9% of all 222                     56.8% of all 222
    909 to 910     76.6% ... (170/222)                  74.8% ... (166/222)
    910            the 59.9% the default ranker         the 56.8% the default ranker
    912            rank 6 to 8 for 13 structures        unchanged, still 13
    913            65.8% (146/222)                      62.6% (139/222)
    917            23.4% ... (52/222)                   25.2% ... (56/222)

Figure 4a and 4b carry the same values as lines 594 and 613 to 615 and move with them.

Unchanged, and worth stating so they are not edited by reflex:

- 66.1% (119/180), the PLM-assisted ranker, lines 29, 587, 594, 615, 679, 900. It is a
  180-structure, 1.4.0 figure, and 11.4 proves that cohort is chain invariant.
- The head-to-head table and every other 180-structure figure, same reason.
- 69% (31/45) on PocketMiner and 41% (9/22) on the curated set, different benchmarks.
- "all 222 structures ... are scored", lines 22, 557, 669, 677, 895. Still true: 222
  attempted, 222 scored, 0 unscored.
- The rank 6 to 8 count, 13.

Not re-scorable here, and flagged rather than changed: the `surface-fusion` figures at
lines 631 and 913, 57.1% to 73.9% top five, 68.5% to 86.4% coverage, paired gain +16.8
points 95% CI [+11.4, +22.8], measured with 1.4.0 at five conformers on a
184-structure cohort. No per-structure record for that evaluation was found in this
tree, so whether its cohort contains interface structures is unverified. If it was
collected through the pre-fix harness it cannot contain any, since that harness dropped
every composite entry, and the figures would then be chain invariant. That should be
confirmed from its own records before the paper repeats it beside corrected numbers.

Also needing the same edit outside the paper: `docs/BENCHMARKS.md` lines 28 and 36, and
`README.md` lines 149, 167, 169 and 172, which carry 59.9% (133/222), 61.3% (136/222)
and 76.6% (170/222). Not edited at the time this section was written; see Section 12.

### 11.10 Commands, records and cost

    python -I benchmarks/cryptobench_benchmark.py \
        --out benchmarks/cryptobench_results_chainaware.json
    python -I benchmarks/cryptobench_benchmark.py --rank-by crypticity \
        --out benchmarks/cryptobench_results_crypticity_chainaware.json

Wall: learned 9.22 min, crypticity 9.25 min, 222 structures each, sequential and never
concurrent. All CIFs were already in `cb_data/cif`; nothing was downloaded.

    benchmarks/cryptobench_results.json                        learned, chain blind (Sec. 4)
    benchmarks/cryptobench_results_chainaware.json             learned, both conventions
    benchmarks/cryptobench_results_crypticity.json             analytic, chain blind (Sec. 10)
    benchmarks/cryptobench_results_crypticity_chainaware.json  analytic, both conventions
    benchmarks/cb_data/cryptobench_crypticity_chainaware_222.log  its console log
    benchmarks/provenance/rescore_chainaware_222.py            every figure in this section,
        including the bootstrap, with its numbers dumped to
        benchmarks/provenance/section11_numbers.json

No git command that writes was run, nothing was installed, and no manuscript, figure,
caption or document was edited.

## 12. Documentation brought to the final state

Date: 2026-11-16, after Section 11. This records what was edited, not a new
measurement. No benchmark was re-run for this section; every figure written into the
documents below was read back out of the two authoritative result files and checked
before it was written:

    benchmarks/cryptobench_results_chainaware.json              learned arm, 222
    benchmarks/cryptobench_results_crypticity_chainaware.json   analytic arm, 222

Independently recomputed from those files, and matching Section 11 to the digit:

    top 5, Jaccard only                126/222  56.8%
    top 5, centroid-or-Jaccard         129/222  58.1%
    top 8, Jaccard only                139/222  62.6%
    coverage, any rank, Jaccard only   166/222  74.8%
    legacy recall column               165/222  74.3%
    no qualifying candidate             56/222  25.2%
    first clears at rank 6 to 8         13 structures
    analytic arm, top 5, Jaccard only   64/222  28.8%
    ranking gain                       +27.927927927927925, 95% CI
                                       [+21.62162162162162, +34.234234234234236]
    38 interface structures, top 5      22/38   57.9%
    184 single-chain, top 5            104/184  56.5%
    old 180 cohort, top 5              104/180  57.8%, identical under both metrics

The bootstrap used `_paired_ci` imported from `benchmarks/train_ranker.py`, the same
function and the same `n_boot = 10000`, `seed = 0` as Section 11.8, not a
reimplementation. The single-chain invariance of 11.4 was re-verified independently
here: 0 mismatches over the 184 single-chain records, three rank blocks, Jaccard and
recall, both arms.

### 12.1 Files edited

- `docs/BENCHMARKS.md`. The opening note now describes both defects, the cohort one
  and the metric one, and states the 184-structure invariance proof. The headline
  table row reads 56.8% (126/222) with the legacy column at 74% (165/222); the
  paragraph under it reads 129/222 (58.1%), 62.6% (139/222), 74.8% (166/222) and
  77.0% (171/222). The head-to-head preamble, the ranking-strategy paragraph and
  "Where the remaining gap is" carry the corrected figures, and the ablation reads
  +27.9 points (95% CI +21.6 to +34.2), 126/222 against 64/222, ceiling 166/222.
  The `compare_detectors_cryptobench.py` caveat now names both legacy defaults,
  `--max-residues 700` and `--residue-metric blind`, and marks 128/210 (61.0%) as a
  chain-blind figure on a partial cohort.
- `README.md`. The Results paragraph, the configuration table and the paragraph under
  it carry 56.8%, 74.8%, 126/222, 166/222 and 58.1% (129/222). The harness note now
  describes two defects rather than one and says which way each moves the headline.
- `CHANGELOG.md`. The Unreleased benchmark-harness entry now opens by naming both
  defects and the direction each moves the headline. The cohort entry keeps its
  description and its table now carries the chain-aware full-fold column, with one
  line recording that the cohort fix alone first read 59.9% (133/222). A separate
  Fixed entry covers the metric, with the `3x0x` case, the seven structures that
  move from hit to miss, the 2,208-comparison invariance check and the withdrawal of
  the "interface sites are easier, 76.3%" claim. `--residue-metric` is added under
  Added, the ablation reads +27.9 points, and the stale "residue metrics are chain
  blind" known limitation is replaced by the two that remain.
- `paper/software_paper_v3_changes.md`, the public revision summary. The harness
  section is rewritten as two findings, both found and fixed before posting, with the
  corrected table, the retraction of the 76.3% claim, and the ablation at +27.9
  points. The abstract-level figure and the unit-correction paragraph read 126/222.

### 12.2 Deliberately not edited

- `paper/software_paper_v3.md`. Not edited here. Section 11.9 remains the list
  of what it needs.
- The head-to-head table in `docs/BENCHMARKS.md` and every other 180-structure row.
  They stay on that cohort, which 11.4 proves is chain invariant, because the stored
  fpocket, P2Rank, MDpocket and IF-SitePred baselines were collected chain blind and
  on 180.
- `66.1% (119/180)` for `learned-plm`, the `surface-fusion` figures, PocketMiner,
  COACH420 and the curated set. Different cohorts or different benchmarks; 11.9
  already lists them as not to be edited by reflex, and the `surface-fusion` caution
  there still stands.
- Sections 1 to 11 of this note, apart from three pointers added at the head of
  Sections 6, 7 and 9 and one at the end of 11.9, so that a reader cannot take a
  superseded figure for a current one. No measured number in them was altered.

No git command that writes was run, nothing was installed, nothing was downloaded, and
no manuscript, figure or `.docx` was touched.
