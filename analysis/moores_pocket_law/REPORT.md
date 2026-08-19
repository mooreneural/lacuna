# Moore's Pocket Law: a falsification-oriented analysis

Analysed at commit `86085d1ba097f742b409d64641d53241aa7a8236`.
Input SHA-256 hashes in `INPUTS.json`. Seed 0 throughout.

## 1. Dataset and version

Per-candidate overlaps for four detectors on CryptoBench.

| source | tool | provenance |
|---|---|---|
| `benchmarks/detectors_cb_paper_fpocket_fixed.jsonl` | fpocket | repo @ HEAD, post rank-ordering fix |
| `benchmarks/detectors_cb_paper_baselines.jsonl` | p2rank | repo @ HEAD |
| `benchmarks/detectors_cb_paper_ifsitepred.jsonl` | ifsitepred | repo @ HEAD |
| `data/split_off.jsonl`, `data/test_off.jsonl` | lacuna | copied from session scratchpad `2030379c` |
| `data/conf_{1,3,5,10}.jsonl` | lacuna sweep | copied from the same scratchpad |

The fpocket rows *inside* `detectors_cb_paper_baselines.jsonl` are stale and are
always overridden; `dataio.py` raises rather than mixing corrected and
uncorrected rankings.

**Reproducibility hazard.** Lacuna's per-candidate data existed only in a
temporary directory that `paper/make_data.py` hardcodes. It is copied into
`analysis/moores_pocket_law/data/` here. Until those files live in the
repository proper, `paper/data/analysis.json` cannot be regenerated on any other
machine.

## 2. Hit criterion

Taken from `paper/make_data.py`, not assumed:

```python
def _hit(jaccards, k):
    return any(j >= JACCARD_THRESHOLD for j in jaccards[:k])   # 0.25
```

Jaccard >= 0.25, **no centroid clause**. The repository's benchmark scripts use a
wider "size-robust" rule that also accepts a centroid within 4 A, but **no
per-candidate centroid distance is stored in any artifact**, so that rule cannot
be evaluated here. `best_centroid_distance` is null in every row for this reason,
and the criterion sweep in Phase 7 varies the Jaccard threshold instead.

## 3. Sample sizes

912 targets paired across all four methods (734 train, 178 test), 3648 rows.
2537 method-target pairs have a qualifying candidate somewhere. The burden sweep
shares 720 train-fold targets across all five conformer levels.

## 4. Reproduction of published values

Exact to six decimal places against `paper/data/analysis.json`.

| method | coverage | top-5 | published |
|---|---|---|---|
| fpocket | 74.2% | 43.8% | 74.2 / 43.8 |
| P2Rank | 66.3% | 63.5% | 66.3 / 63.5 |
| IF-SitePred | 70.8% | 61.8% | 70.8 / 61.8 |
| Lacuna | 73.6% | 66.3% | 73.6 / 66.3 |
| union | 92.1% | — | 92.1 |

n = 178. **PHASE 2: REPRODUCED.**

## 5. Evidence supporting the hypothesis

**Conversion falls as burden rises, and it is not an artifact of one method.**
Among covered targets, Spearman rho(N, hit@5) = **-0.257** pooled, and negative
within every method separately: fpocket -0.287, P2Rank -0.266, Lacuna -0.271,
IF-SitePred -0.149. Binned conversion at k=5 falls monotonically from 100.0%
[100.0, 100.0] at N=1-5 to 69.6% [65.3, 73.9] at N>=31.

**It survives the obvious confounders.** In a logistic model on covered targets
with method fixed effects and clustered bootstrap CIs, standardized
candidate count carries **beta = -0.705 [-0.916, -0.526]**. Protein length is
**not** significant (-0.083 [-0.255, +0.079]). Site size is (+0.592). Method
fixed effects are large (~+2.0 versus fpocket), but burden survives them.

**It is causally demonstrated, not merely correlational.** This is the load-bearing
result. The conformer sweep varies burden on fixed targets with the same
detector, ranker and criterion. Restricting to the 330 targets covered at *both*
1 and 20 conformers, so coverage selection cannot operate, raising burden from
7.1 to 22.8 candidates costs:

| k | conf_1 | conf_20 | paired delta (95% CI) |
|---|---|---|---|
| 1 | 70.6% | 65.8% | **-4.8 [-9.4, -0.6]** |
| 3 | 93.0% | 83.3% | **-9.7 [-13.0, -6.4]** |
| 5 | 97.6% | 89.7% | **-7.9 [-10.9, -5.2]** |
| 10 | 99.7% | 97.0% | **-2.7 [-4.5, -1.2]** |
| 20 | 100.0% | 99.4% | -0.6 [-1.5, +0.0] |

The same findable site sinks **+0.78 ranks [+0.53, +1.05]**. For targets where
the site was reachable at either burden, the larger candidate set demonstrably
buries it. That is candidate competition, and it is real.

## 5b. Decoy injection: burden isolated from candidate quality

The conformer sweep raises burden by sampling more conformations, so it changes
what the candidates are as well as how many. Two injection experiments remove
that confound. In both, a target's own candidates and its true site are left
untouched, m extra candidates are added, and everything is re-ranked by the
production PLM-assisted ranker. Train fold, 20 repeats, seeded.

**Foreign decoys** (clusters taken from other proteins), n=541 targets:

| m | mean N | hit@1 | hit@5 | hit@10 | decoys reaching top-5 |
|---|---|---|---|---|---|
| 0 | 22.9 | 57.3% | 86.3% | 93.5% | 0.00 |
| 5 | 27.9 | 49.6% | 83.6% | 92.4% | 0.87 |
| 10 | 32.9 | 44.4% | 80.9% | 91.1% | 1.34 |
| 20 | 42.9 | 37.3% | 75.5% | 87.9% | 1.98 |
| 40 | 62.9 | 28.3% | 67.8% | 81.6% | 2.70 |

Decoys do reach the top five, so the test is not toothless. Paired at m=40:
**-18.9 points at k=5 [-21.7, -16.1]**.

That result is confounded, however. The production ranker is fitted on
within-structure pairs, so nothing constrains its scores to be comparable across
structures, and a foreign pocket can outscore the true site for reasons unrelated
to burden.

**Within-target decoys** remove that route: injected scores are drawn from the
target's *own* non-qualifying candidates, making a decoy statistically
indistinguishable from a wrong answer the target already produced. n=539:

| m | mean N | hit@1 | hit@5 | hit@10 |
|---|---|---|---|---|
| 0 | 22.9 | 57.1% | 86.3% | 93.5% |
| 5 | 27.9 | 57.1% | 82.5% | 91.7% |
| 10 | 32.9 | 57.1% | 79.5% | 89.9% |
| 20 | 42.9 | 57.1% | 75.0% | 86.0% |
| 40 | 62.9 | 57.1% | 69.4% | 80.0% |

Paired at m=40: **-16.8 points at k=5 [-19.7, -14.0]**, -16.0 at k=3, -13.5 at
k=10, -8.7 at k=20, every interval excluding zero.

**This is the cleanest evidence in the analysis.** Candidate quality is held
exactly fixed, the true site is untouched, the ranker is unchanged, and only N
moves. The competition effect survives at full strength.

**Decomposition** of the foreign-decoy effect into competition and
miscalibration:

| k | foreign | within-target | attributable to miscalibration |
|---|---|---|---|
| 1 | -28.4 | +0.0 | -28.4 |
| 3 | -22.9 | -16.0 | -6.9 |
| 5 | -18.9 | -16.8 | -2.1 |
| 10 | -11.7 | -13.5 | +1.8 |
| 20 | -6.5 | -8.7 | +2.2 |

At k=5, roughly 89% of the effect is genuine burden.

Two caveats. **The k=1 column is structurally immune by construction**: if the
true site is already rank 1 its score exceeds every wrong answer, and decoys
drawn from that wrong-answer distribution cannot displace it. The +0.0 is a
property of the design, not evidence that burden does not matter at k=1.

And the foreign-decoy k=1 result is a separate finding worth its own attention:
a pocket from an unrelated protein displaces the true site from rank 1 in 28% of
cases. **The production ranker has no cross-structure score calibration.** That
is expected from a within-structure pairwise fit, but it means the scores must
not be compared or thresholded across targets, which is exactly what a pooled
re-ranker over multiple detectors would require.

## 6. Evidence contradicting the hypothesis

**The quantitative form fails.** `d_eta/dN < 0` is the claim. Measured directly
on the sweep, eta = dR_5/dC is:

| step | dN | dC | dR_5 | eta (95% CI) |
|---|---|---|---|---|
| 1 to 3 | +5.2 | +15.8 | +12.1 | 0.76 [0.66, 0.85] |
| 3 to 5 | +2.8 | +5.3 | +1.4 | 0.26 [-0.52, 0.60] |
| 5 to 10 | +3.5 | +3.2 | +3.6 | 1.13 [0.50, 2.67] |
| 10 to 20 | +3.7 | +3.3 | +1.5 | 0.46 [-0.50, 1.00] |

**0.76, 0.26, 1.13, 0.46. Not monotone.** Two intervals include zero, one
includes values above 1. There is no declining efficiency function here.

**The competition effect saturates rather than accelerating.** Paired on targets
covered at both ends of each adjacent step, the damage is concentrated at *low*
burden and disappears at high burden:

| step | dN | d hit@5 (95% CI) |
|---|---|---|
| 1 to 3 | +5.3 | **-2.3 [-4.2, -0.7]** |
| 3 to 5 | +2.9 | **-3.6 [-5.8, -1.5]** |
| 5 to 10 | +3.6 | +0.7 [-1.5, +2.9] |
| 10 to 20 | +3.9 | -0.4 [-2.5, +1.6] |

Naive competition predicts each added candidate hurts at least as much as the
last. The opposite happens.

**The ~15 threshold does not reproduce.** Descriptive best split is **N <= 7**,
bootstrap-selected threshold 95% interval [5, 9], modal 7. 23.6% of resamples
select a grid edge, which indicates an unstable optimum. More importantly a
smooth model beats a step: log-linear in N gives **AIC 2342.06** against
**2364.01** for a step at the best threshold. The relationship is better treated
as continuous.

**More candidates still helps overall.** Across the sweep coverage rises
46.5% to 74.2% while conversion falls 97.3% to 86.1%, and net top-5 rises
**45.3% to 63.9%**. The competition cost is real but smaller than the coverage
gain. Any framing implying more candidates is net-harmful is contradicted.

## 7. Confounders

Addressed: protein length (not significant), method identity (fixed effects,
plus every method analysed alone), site size (significant, positive, retained as
a control), candidate density (rho weakens from -0.257 to -0.174 but keeps sign).

Not addressed: candidate counts are **not commensurable across methods**.
fpocket alpha-sphere pockets, P2Rank SAS-point clusters, IF-SitePred probe
clusters and Lacuna cross-conformer clusters are different objects. Pooled
cross-method coefficients should be read as descriptive only. The within-method
and within-target results do not have this problem.

The conformer sweep changes candidate *quality* as well as count, so it is not a
pure burden manipulation either. What it removes is cross-target and
cross-method confounding.

## 8. Robustness

| perturbation | rho(N, hit@5 \| covered) |
|---|---|
| Jaccard threshold 0.15 / 0.20 / 0.25 / 0.30 / 0.40 | -0.207 / -0.235 / -0.257 / -0.282 / -0.268 |
| k = 1 / 3 / 5 / 10 / 20 | -0.219 / -0.242 / -0.257 / -0.248 / -0.207 |
| trim top 0 / 1 / 5 / 10% of N | -0.257 / -0.255 / -0.257 / -0.265 |
| per method | -0.287 / -0.266 / -0.149 / -0.271 |
| normalised by protein length | -0.174 |
| train / test fold | -0.263 / -0.228 |

The sign and rough magnitude survive everything tried.

## 9. Effect sizes

- Pooled conversion, N=1-5 vs N>=31: 100.0% to 69.6%, about **30 points**.
- Standardized logit coefficient on burden: **-0.705 [-0.916, -0.526]**.
- Causal within-target effect at k=5, burden 7.1 to 22.8: **-7.9 points
  [-10.9, -5.2]**.
- Rank displacement of the same site: **+0.78 ranks [+0.53, +1.05]**.

## 10. Is "law" justified?

**No.** Outcome **B/C hybrid**.

A general, robust, causally demonstrated candidate-competition effect exists and
is not method-specific and not explained by protein size. That is a real finding.

But a "law" needs a stable functional form, and there is none. eta does not
decline monotonically, its intervals are wide and overlapping, the effect
saturates instead of accelerating, no defensible breakpoint exists, and a
continuous model beats a thresholded one. Calling this a law overstates what four
noisy efficiency estimates can carry.

Note also that `R_k = C * V_k` is an accounting identity, verified as such in
`test_metrics.py`. It is not a discovery and must not be presented as one.

## 11. Strongest falsifiable formulation currently supported

> For a fixed detector and target, increasing the number of proposed candidates
> reduces the probability that an already-findable site appears within a fixed
> top-k budget. On CryptoBench with Lacuna, raising burden from ~7 to ~23
> candidates lowers top-5 recovery among targets covered at both burdens by
> 7.9 points (95% CI 5.2 to 10.9) and pushes the first qualifying candidate down
> by 0.78 ranks (95% CI 0.53 to 1.05).

Falsifiable, effect-sized, and it does not claim a functional form. It should be
called a **candidate-competition effect**, not a law.

## 12. Experiments required before publication

1. ~~A burden manipulation that does not change candidate quality.~~ **Done**,
   Phase 10 and 10b. Burden isolated; effect confirmed at -16.8 points at k=5.
2. **Replicate the sweep on a second detector.** Everything causal here is
   Lacuna-only; the cross-method evidence is confounded by candidate semantics.
3. **Confirm on the designated test fold.** The sweep is train-fold only. Only
   `conf_1` exists for the test fold, so the other levels must be generated.
4. **Store per-candidate centroid distances** so the size-robust criterion can be
   tested rather than assumed irrelevant.
5. **Pre-register the breakpoint question** if it is pursued, since the current
   threshold search is post-hoc and its optimum is unstable.

## Reproducing

```
python analysis/moores_pocket_law/phase2_reproduce.py
python analysis/moores_pocket_law/phase3_table.py
python analysis/moores_pocket_law/phase4_hypotheses.py
python analysis/moores_pocket_law/phase5_confounders.py
python analysis/moores_pocket_law/phase6_sweep.py
python analysis/moores_pocket_law/phase7_robustness.py
python analysis/moores_pocket_law/phase8_figures.py
python -m pytest analysis/moores_pocket_law/test_metrics.py -q
```

## Status of each claim

| | |
|---|---|
| Confirmatory | Phase 2 reproduction |
| Exploratory | H1-H5, Phase 5 confounders, Phase 6 sweep, Phase 7 robustness |
| Post-hoc | H6 breakpoint search, and any statement about ~15 |

The designated CryptoBench test fold is **not** fully blind: repository history
shows Lacuna was developed over iterations during which test-fold performance was
measured. The primary analysis here is train-fold to avoid compounding that.
