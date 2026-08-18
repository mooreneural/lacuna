# Ranking

`--rank-by` selects how sites are ordered once they have been detected and
clustered. Ordering is where most recoverable performance lives: across the
candidate set Lacuna generates, some cluster clears the criterion for 73.7% of
test-fold structures against the 66.1% the PLM-assisted ranker surfaces in the
top five.

## Strategies

| Strategy | What it orders by |
|---|---|
| `learned` (default) | Fitted linear model over 23 features |
| `learned-plm` | The same model plus four protein-language-model features (optional extra) |
| `crypticity` | How much a site opens relative to the input structure |
| `druggability` | Peak open-state druggability; better for always-open or orthosteric sites |
| `balanced` | Druggability with a mild persistence bonus |
| `persistence` | The legacy persistence x druggability rule |

## The default: a fitted ranker

`learned` is a linear model over 23 features covering pocket geometry,
druggability, and ensemble-derived terms such as how far a site's centroid
wanders between conformers, which a single-structure detector cannot compute. It
is trained on within-structure pairs, so it optimises ordering directly rather
than classifying pockets in isolation.

On CryptoBench's test fold it recovers 55.6% against 17.8% for the analytic
`crypticity` rule it replaced, rising to 66.1% with `learned-plm`.

Because it is fitted on within-structure pairs, **its scores carry no
cross-structure calibration**. They order candidates inside one structure and
should not be compared or thresholded across different targets.

**On the curated 22-target set the ordering reverses.** `persistence` and
`balanced` reach 13/22 where `learned` reaches 10/22, though at n=22 the
intervals overlap heavily. If your targets resemble the classic literature case
studies more than CryptoBench, the analytic strategies are worth trying.
**[Full ablation →](BENCHMARKS.md#ranking-strategies)**

## The PLM-assisted ranker

`learned-plm` adds four features summarising what a protein language model
thinks of each pocket's lining residues. It is an **optional extra**
(`pip install "lacuna-pockets[plm]"`) because it needs PyTorch and downloads an
ESM-2 checkpoint, and it is a separate strategy rather than the default so that
identical commands give identical rankings on every machine, whether or not the
extra is installed.

In this linear form the sequence features carry most of the ordering: `plm_mean`
outweighs every geometric term, and a PLM-only fit ranks almost identically.
Geometry is what proposes the candidates in the first place, which no sequence
model can do; it simply contributes little to ordering them afterwards.

## Crypticity

Independently of rank, every pocket carries a continuous crypticity score
between 0 and 1, the conformational-selection signature of a cryptic site:

```
opening    = (max_volume − apo_volume) / max_volume    # 1.0 if absent in the apo state
crypticity = opening × peak_open_state_druggability
```

A constitutive pocket already formed in the input scores near 0. One that is
absent in the apo structure and opens into a druggable cavity scores near 1. A
site is additionally flagged `cryptic: true` when it is present in fewer than
90% of conformers.

## Refitting

Any change to the detection constants requires refitting the ranker. See
[BENCHMARKS.md](BENCHMARKS.md#training-and-re-fitting-the-ranker).
