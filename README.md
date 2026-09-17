<p align="center">
  <img src="https://raw.githubusercontent.com/mooreneural/lacuna/main/docs/lacuna_logo.png"
       alt="Lacuna" width="720">
</p>

<p align="center"><strong>Cryptic binding pocket discovery via conformational ensemble analysis.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/lacuna-pockets/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lacuna-pockets"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.20533638"><img alt="DOI" src="https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20533638-blue"></a>
  <a href="https://www.biorxiv.org/content/10.64898/2026.08.14.744956v2"><img alt="bioRxiv" src="https://img.shields.io/badge/bioRxiv-10.64898%2F2026.08.14.744956-bc2635"></a>
  <a href="https://app.tamarind.bio/tools/lacuna"><img alt="Run on Tamarind Bio" src="https://img.shields.io/badge/Run_on-Tamarind_Bio-2ea44f"></a>
  <a href="https://neurosnap.ai/service/Lacuna%20Cryptic%20Pocket%20Discovery"><img alt="Run on Neurosnap" src="https://img.shields.io/badge/Run_on-Neurosnap-2ea44f"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mooreneural/lacuna/main/docs/cryptic_pocket_breathing.gif"
       alt="Adenosine kinase breathing along its lowest normal mode; as the interdomain cleft opens, a cryptic pocket (blue) is revealed and a ligand docks into it"
       width="62%">
</p>

<p align="center"><em>A protein breathes through Lacuna's conformational ensemble, and a cryptic
pocket appears. Adenosine kinase along its lowest normal mode (Lacuna's NMA backend): as the
interdomain cleft opens, the substrate pocket (blue) is exposed and 2-fluoroadenosine (orange,
the crystallographic ligand from holo 2PKK) docks into it. The motion is Lacuna's own mode-0
ensemble; the ligand is shown at its bound pose.</em></p>

Most protein structure predictors return one static conformation. But many
disease-relevant proteins are called undruggable not because they are
biologically intractable, but because no pocket is visible in their ground
state. K-Ras was considered undruggable for thirty years until a transient
cryptic pocket was found beneath its switch-II region. That pocket now backs
sotorasib and adagrasib.

Lacuna finds those pockets. It generates a conformational ensemble from any
input structure, detects pockets in every conformer, clusters them across the
ensemble to surface sites that appear only transiently, and ranks them with a
fitted model.

```bash
lacuna discover kras.pdb --conformers 20 --emit-boltz-constraints --emit-vina-boxes
```

## Install

```bash
pip install lacuna-pockets
```

Optional extras, for better conformational sampling or the sequence-assisted
ranker:

```bash
pip install "lacuna-pockets[openmm]"   # 100ps implicit-solvent MD
pip install "lacuna-pockets[plm]"      # PLM-assisted ranker (PyTorch, ESM-2)
pip install "lacuna-pockets[boltz]"    # Boltz-2 diffusion sampling (experimental, GPU)
pip install "lacuna-pockets[all]"      # everything
```

Requires Python 3.10+. The default backend needs no GPU, no force field and no
model weights.

<p align="center">
  <img src="https://raw.githubusercontent.com/mooreneural/lacuna/main/docs/bclxl_pocket_example.png"
       alt="A Lacuna-detected cryptic pocket on apo BCL-XL, in blue, overlapping the ABT-737 binding site"
       width="65%">
</p>

<p align="center"><em>Apo BCL-XL (1LXL): a Lacuna-detected pocket (blue) opening onto the
groove where ABT-737 (orange, from the holo structure 2YXJ) binds, a site never shown to
the detector. With <code>--detector surface-fusion</code> this site is recovered at rank 2,
Jaccard 0.36, centroid 5.6 Å. The default <code>alpha</code> detector places it third and
does not clear the size-robust bar, which is the specialisation this release is about.</em></p>

## Quick start

```bash
lacuna discover protein.pdb --conformers 20
```

Writes a ranked `pocket_report.json` plus, on request, Boltz YAML constraints
and AutoDock Vina boxes ready for docking. Full options in
**[docs/USAGE.md](docs/USAGE.md)**.

## How it works

1. **Ensemble generation.** N conformers from elastic-network normal mode
   analysis (default), OpenMM implicit-solvent MD, or Boltz-2 diffusion sampling.
2. **Pocket detection.** Grid-based alpha-point analysis per conformer: distance
   transform, local maxima in the 1.4-5.5 A interaction zone, clustered into
   candidates.
3. **Cross-ensemble clustering.** Greedy centroid merging matches corresponding
   pockets across every conformer, turning transient cavities into persistent
   sites with their own statistics.
4. **Druggability scoring.** Gaussian volume reward centred at 300 A³, plus
   enclosure, hydrophobicity and aromaticity (Halgren 2009), scored per conformer.
5. **Ranking.** A fitted linear model over 23 geometric and ensemble-derived
   features orders the sites. Each also carries a continuous crypticity score.

## Documentation

| | |
|---|---|
| **[Usage](docs/USAGE.md)** | CLI, Python API, backends, output formats, worked example |
| **[Ranking](docs/RANKING.md)** | Ranking strategies, the fitted model, crypticity |
| **[Benchmarks](docs/BENCHMARKS.md)** | Full results, head-to-head comparisons, negative results |
| **[Paper](paper/)** | Analysis, per-candidate data, and scripts that regenerate every figure |

## Results

On CryptoBench's designated test fold, Lacuna recovers **55.6%** of known
cryptic sites in its top five with the zero-dependency default and **66.1%**
with the optional PLM-assisted ranker, and the true site is present in the
candidate set for up to 92% of structures.

### Recovery by configuration

Cryptic-pocket recovery on CryptoBench's held-out test fold, under the
size-robust criterion (Jaccard >= 0.25, or centroid within 4 A of the site).

| Configuration | Top-5 recovery | Site found (coverage) |
|---|:--:|:--:|
| Default (CPU-only, zero-dependency) | 55.6% | 73.7% |
| + PLM reranker | 66.1% | 73.7% |
| `surface-fusion` detector | 73.9%¹ | 86.4%¹ |
| Union across detectors | n/a | **92.2%** |

¹ Held-out surface-fusion evaluation at 5 conformers (n=184); the other rows use
the 20-conformer cohort (n=180).

**[Full results, including head-to-head comparisons and where Lacuna loses →](docs/BENCHMARKS.md)**

> **The more interesting result is not Lacuna's score.** Across five
> candidate-generation methods evaluated in six configurations, coverage
> (whether a qualifying candidate is proposed at all) spans 14.5 points,
> while conversion (whether a method's own coverage reaches the top five)
> spans 36.7. fpocket has the highest coverage at 73.7% but the lowest
> top-5 recovery at 43.6%, while P2Rank converts 95.8% of the sites it
> covers. Two Lacuna rankers operating on the exact same candidate set
> differ by 10.6 points of top-5 recovery, isolating ranking directly.
>
> Union coverage saturates at 92.2%, rising to 98.6% for annotated sites
> containing at least eight residues. Candidate competition is also causal:
> adding synthetic competitors while holding the true site, real candidate
> set, and ranker fixed reduces top-5 recovery by 16.8 points on the
> training folds and 17.0 points on the held-out test fold. Detector
> consensus provides no measurable gain at a budget of five candidates,
> but gains 11.8 points at a budget of twenty.
>
> Moore CW. *Cryptic binding sites are detected but not ranked: coverage,
> conversion, and the limits of detector consensus.* bioRxiv 2026.
> [doi:10.64898/2026.08.11.743381](https://doi.org/10.64898/2026.08.11.743381)

## Citation

If you use Lacuna, please cite the software paper:

```bibtex
@article{moore2026lacuna,
  author  = {Moore, Clayton W.},
  title   = {Lacuna: Cryptic Binding Pocket Discovery
             via Conformational Ensemble Analysis},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.14.744956}
}
```

If you use the benchmark data or the coverage/conversion decomposition, please
also cite the accompanying analysis:

```bibtex
@article{moore2026coverage,
  author  = {Moore, Clayton W.},
  title   = {Cryptic binding sites are detected but not ranked:
             coverage, conversion, and the limits of detector consensus},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.11.743381}
}
```

To cite a specific software version, the archived releases carry their own DOIs
under the concept DOI [10.5281/zenodo.20533638](https://doi.org/10.5281/zenodo.20533638),
which always resolves to the newest.

## Acknowledgements

Lacuna is measured against, and builds on, work released openly by others:
**fpocket** (Le Guilloux et al. 2009), **P2Rank** (Krivák & Hoksza 2018),
**IF-SitePred** (Carbery et al. 2024) and **MDpocket** (Schmidtke et al. 2011).
Evaluation uses the **CryptoBench** (Škrhák et al. 2025) and **PocketMiner**
(Meller et al. 2023) datasets. Method credits: **ANM** (Atilgan et al. 2001),
**SiteMap druggability** (Halgren 2009), **enclosure scoring** (Schmidtke &
Barril 2010), **ESM-2** (Lin et al. 2023).

## License

**[MIT](LICENSE)**, free to use, study, modify, share, and embed in
closed-source or commercial work, with no copyleft obligation.

> Versions 0.2.0 through 0.3.1 were released under AGPL-3.0 and remain available
> under those terms. MIT applies from 1.0.0 onward. Lacuna moved back to a
> permissive license because its central recommendation is to combine several
> detectors, and copyleft makes that combination harder for exactly the people
> the work is aimed at.
