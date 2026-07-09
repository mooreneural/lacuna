# Benchmarks: full detail

This is the complete per-target breakdown, ranking-strategy ablation, and speed
numbers behind the headline results in the [README](../README.md#benchmarks).
Every number here is reproducible with the commands shown; each benchmark
script also prints the full per-metric breakdown (centroid, Jaccard at
0.20/0.25/0.30, and legacy recall) when you run it.

## Cryptic pockets: full 22-target breakdown

Sorted by Jaccard (size-robust overlap). PASS = clears the size-robust
criterion (Jaccard ≥ 0.25 **or** centroid ≤ 4 Å); recall is the legacy
size-gameable metric, shown for contrast. "Rank" is the position of the
best-matching top-5 pocket.

| Protein | Apo PDB | Drug target | Jaccard | Recall | Rank |
|---------|---------|-------------|--------:|-------:|:----:|
| ✅ BCL-XL BH3 groove | 1LXL | navitoclax | 56% | 68% | 1 |
| ✅ BCL-2 BH3 groove | 1G5M | venetoclax | 48% | 59% | 1 |
| ✅ MDM2 p53-binding cleft | 1Z1M | nutlin-3 | 39% | 47% | 1 |
| ✅ PTP1B allosteric helix site | 1A5Y | benzofurans | 36% | 94% | 5 |
| ✅ IL-2 helix-α1 site | 1M47 | - | 36% | 93% | 1 |
| ✅ HIV-1 RT NNRTI pocket | 1HMV | nevirapine | 33% | 62% | 4 |
| ✅ K-Ras switch-II pocket | 4OBE | sotorasib/adagrasib | 26% | 79% | 3 |
| ❌ Ricin A pterin pocket | 1RTC | - | 18% | 50% | - |
| ❌ T4 Lysozyme L99A cavity | 1L90 | - | 17% | 62% | - |
| ❌ HCV NS5B thumb-site I | 1NB4 | VXR class | 16% | 47% | - |
| ❌ Glucokinase allosteric site | 1V4S | activators | 15% | 39% | - |
| ❌ Src myristate pocket | 2SRC | - | 14% | 36% | - |
| ❌ PPARγ allosteric site | 2PRG | metaglidasen | 11% | 35% | - |
| ❌ c-ABL myristate pocket | 3CS9 | asciminib | 7% | 19% | - |
| ❌ p38α DFG-out pocket | 1P38 | BIRB 796 | 7% | 24% | - |
| ❌ ERK2 allosteric site | 2ERK | - | 6% | 19% | - |
| ❌ Caspase-1 dimer interface | 2HBQ | - | 5% | 25% | - |
| ❌ PKM2 subunit interface | 1ZJH | TEPP-46 | 4% | 17% | - |
| ❌ MMP-13 S1′ tunnel | 2OZR | non-zinc | 4% | 6% | - |
| ❌ TEM-1 allosteric site | 1JWP | CBT | 2% | 17% | - |
| ❌ IDH1 R132H dimer interface | 3MAP | ivosidenib | 2% | 7% | - |
| ❌ SHP-2 allosteric tunnel | 2SHP | SHP099 | 0% | 0% | - |

**The remaining gap is mostly sampling, not ranking.** Raising the cutoff from
top-5 to top-20 lifts the size-robust score only from **7/22 to 10/22**, just
3 pockets are detected-but-mis-ranked. The other 12 misses are not localized
at all even at top-20, so they are a sampling/localization ceiling (the NMA
ensemble never opens, or the detector never localizes the site tightly
enough) rather than a ranking failure. Under the older recall metric the
top-20 ceiling looked like 73%, which made the problem appear to be ranking;
it was largely the metric. The hard cases split into **oligomeric-interface
pockets** (Caspase-1, IDH1, PKM2) that form *between* subunits and are
invisible to single-chain analysis, and **large-rearrangement sites** (p38
DFG-out, c-ABL myristate) that need sampling beyond elastic-network modes.

Dimer-interface pockets are partly addressable with `--homodimer` (reads
BIOMT records and builds the biological assembly), though this benchmark's
single-chain-referenced scoring does not credit them. For large-rearrangement
sites the optional Boltz-2 backend samples more broadly, but its current
sequence-based integration is noisy, see the README's [Backends](../README.md#backends)
section.

```bash
python benchmarks/cryptic_benchmark.py --category cryptic                          # full run
python benchmarks/cryptic_benchmark.py --category cryptic --rank-by druggability    # ablation
python benchmarks/cryptic_benchmark.py --category cryptic --top-n 20                # detection ceiling
```

## Ranking strategies

`--rank-by` selects how pockets are ordered (cryptic benchmark pass rate,
NMA, N=20):

| Strategy | Description | Cryptic pass |
|----------|-------------|--------------|
| `crypticity` (default) | most cryptic sites first | **12 / 20** |
| `druggability` | peak open-state composite druggability | 10 / 20 |
| `balanced` | druggability with a mild persistence bonus | 8 / 20 |
| `persistence` | legacy persistence × druggability | 7 / 20 |

## Orthosteric / conformational controls

Crypticity ranking (the default) intentionally de-prioritizes always-open
sites, so for orthosteric / general pocket finding use `--rank-by
druggability`. Under the corrected contact-lining pipeline (NMA, `--rank-by
druggability`):

| Category | Result | Notes |
|----------|--------|-------|
| Orthosteric | 3 / 6 | hen lysozyme 100%, HIF-2α 96% (1.1 Å centroid), DHFR 50%; misses HIV protease, thrombin, trypsin (1S0Q numbering) |
| Conformational | 1 / 1 | adenylate kinase open→closed |

Orthosteric detection is a known relative weakness of the tight-contact
pipeline: the tool is tuned for transient cryptic sites, not always-open
active-site grooves.

## Speed (NMA backend, no GPU)

| Protein size | Time |
|-------------|------|
| ~130 residues (lysozyme) | 0.6s |
| ~170 residues (MDM2) | 1.1s |
| ~350 residues (K-Ras) | 0.9s |
| ~530 residues (HIV-1 RT chain A) | 8.4s |
