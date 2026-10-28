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

On nucleic-acid input one of the 23 weights is zeroed, which makes the effective
feature count 22 there and 23 on protein. The feature is a duplicated burial
measurement whose fitted weight penalises burial: correct for protein, where the
ligand site is the shallower of the candidates a structure yields, and backwards
for RNA and DNA, where it is the more buried one. Nothing is refitted and the
other 22 weights are shared. See `_NUCLEIC_RANKER_WEIGHTS` in
`lacuna/pockets/clusterer.py` for the measurements, and note the gate fires only
on structures with no protein in them, so a protein carrying an incidental
nucleotide keeps the protein weighting.

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
site is flagged `cryptic: true` when `crypticity >= 0.3`.

> An earlier version of this page said the flag was "present in fewer than 90% of
> conformers". That rule was replaced before 1.0.0 and the sentence outlived it.
>
> Two limits of `apo_volume_A3` are worth knowing before reading `crypticity`.
> It is the largest frame-0 pocket **assigned to that cluster**, so it reads 0.0
> both when the input has no cavity there and when the greedy clustering put the
> frame-0 pocket in a neighbouring cluster; the latter accounts for roughly 4-7%
> of pockets. And whenever it is 0.0, `opening` is 1.0 and `crypticity` becomes
> numerically identical to `max_druggability`, which happens for over half the
> pockets in a typical ensemble run. To ask whether a site is genuinely open in
> your input, re-run with `--conformers 0`: that is byte-identical to frame 0 of
> the ensemble run, and every pocket it reports has `crypticity` exactly 0.

## Refitting

Any change to the detection constants requires refitting the ranker. See
[BENCHMARKS.md](BENCHMARKS.md#training-and-re-fitting-the-ranker).
