<img width="1066" height="760" alt="bclxl_pocket" src="https://github.com/user-attachments/assets/995fcb60-91ed-4bfc-83b3-602731cbe2f5" />

# Lacuna

**Cryptic binding pocket discovery via conformational ensemble analysis.**

Most protein structure predictors (AlphaFold, Boltz, Chai) give you one static structure. But ~70% of disease-relevant proteins are considered "undruggable" not because they're biologically intractable - it's because no pocket is visible in their ground state. K-Ras was "undruggable" for 30 years until a transient cryptic pocket was found in its switch-II region. That pocket now backs sotorasib and adagrasib.

Lacuna finds those pockets. It generates a conformational ensemble from any input structure, detects pockets per conformer, and clusters them across the ensemble to surface sites that only appear transiently - ranked by a continuous crypticity score.

```bash
lacuna discover kras.pdb --conformers 20 --emit-boltz-constraints --emit-vina-boxes
```

## Install

```bash
pip install lacuna-pockets
```

**Optional backends** (better conformational sampling):
```bash
pip install "lacuna-pockets[openmm]"   # 100ps implicit-solvent MD
pip install "lacuna-pockets[boltz]"    # Boltz-2 diffusion sampling (experimental, GPU)
pip install "lacuna-pockets[all]"      # everything
```

Requires Python 3.10+.

---

## Quick start

### CLI

```bash
# Discover pockets with defaults (NMA backend - physically grounded, no GPU needed)
lacuna discover protein.pdb --conformers 20

# Filter and limit output
lacuna discover protein.pdb --min-druggability 0.5 --min-persistence 0.3 --top 5

# Analyze a homodimer - detects pockets at the dimer interface (e.g. Caspase-1, IDH1)
# Reads BIOMT records from PDB; for best results use the biological assembly download from RCSB
lacuna discover protein.pdb --homodimer --conformers 20

# Optional Boltz-2 backend (experimental - see the Backends note)
lacuna discover protein.pdb --backend boltz --conformers 30

# Emit all docking file formats
lacuna discover protein.pdb --emit-boltz-constraints --emit-vina-boxes --emit-pocket-pdbs

# Generate docking files from a previous report
lacuna dock-prep kras_lacuna/pocket_report.json kras.pdb --format all
```

### Python API

```python
from lacuna import load_structure, detect_pockets, cluster_pockets
from lacuna.ensemble.nma_backend import NMABackend
from lacuna.io.structure import coords_array
from lacuna.io.writers import write_report, write_boltz_constraint

structure = load_structure("protein.pdb")
backend = NMABackend(seed=42)
coord_sets = backend.generate("protein.pdb", n_conformers=20)

all_coords = [coords_array(structure)] + coord_sets
pocket_lists = []
for ci, coords in enumerate(all_coords):
    pockets = detect_pockets(coords, structure)
    for p in pockets:
        p.conformer_idx = ci
    pocket_lists.append(pockets)

clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords))
for c in clusters[:5]:
    print(f"Rank {c.rank}  druggability={c.druggability:.3f}  "
          f"persistence={c.persistence:.0%}  cryptic={c.cryptic}")
    print(f"  Residues: {', '.join(c.lining_residues[:5])}")
```

---

## How it works

1. **Ensemble generation** - Generate N conformers via elastic network model normal mode analysis (built-in, default), OpenMM implicit-solvent MD, or experimental Boltz-2 diffusion sampling
2. **Pocket detection** - Grid-based alpha-point analysis per conformer: compute distance transform, find local maxima within the 1.4-5.5 Å interaction zone, cluster nearby alpha-points into pocket candidates
3. **Cross-ensemble clustering** - Greedy centroid merging clusters corresponding pockets across all conformers
4. **Druggability scoring** - Gaussian volume reward centered at 300 Å³ + enclosure + hydrophobicity + aromaticity (Halgren 2009), scored in each conformer
5. **Scoring & ranking** - Each site gets a continuous crypticity score (how much it opens relative to the apo state × druggability when open) and is flagged `cryptic: true` if present in <90% of conformers. Sites are then ordered by a **learned ranker** (default), a linear model over 23 geometric and ensemble features fitted to identify the true binding site; `--rank-by` selects an analytic rule instead

---

## Outputs

| File | Description |
|------|-------------|
| `pocket_report.json` | Ranked pocket metadata: centroid, volume + apo→open range, druggability, crypticity, persistence, lining residues |
| `pocket_N_site.pdb` | Pseudoatom PDB for PyMOL/ChimeraX visualization |
| `pocket_N_constraint.yaml` | Boltz YAML - add a SMILES and run `boltz predict` to dock into this site |
| `pocket_N_vina.conf` | AutoDock Vina / Gnina / QuickVina box config |

---

## Backends

| Backend | Install | Quality | Speed | Notes |
|---------|---------|---------|-------|-------|
| `nma` | built-in | good | ~0.1s/conf | Elastic-network normal modes (default) |
| `openmm` | `lacuna[openmm]` | good | ~2s/conf | Implicit-solvent MD, 100ps |
| `boltz` | `lacuna[boltz]` | experimental | ~100s/protein (GPU) | Diffusion sampling, noisy (see note below) |
| `random` | built-in | baseline | ~0.04s/conf | Gaussian backbone perturbation |

**Auto-selection order:** `boltz` → `openmm` → `nma` → `random`. On a plain `pip install lacuna`, the NMA backend runs automatically.

The `nma` backend samples physically meaningful collective motions - the same hinge-bending and breathing modes that open cryptic pockets in nature - without requiring a GPU or force field. It is the zero-dependency default.

> **Boltz backend status (honest note).** The `boltz` backend runs Boltz-2 diffusion sampling on a GPU, but it currently predicts each conformer *de novo from sequence* (not partial diffusion from the input structure), which yields structurally divergent, noisy ensembles (150-300+ pocket clusters vs NMA's ~35). In GPU benchmarking it did **not** reliably improve cryptic detection over NMA. A proper apo-templated integration with sequence-based residue mapping is planned; until then, NMA is the recommended backend.

---

## Benchmarks

**66.1% of known cryptic sites recovered in the top 5** on CryptoBench's held-out test fold with the sequence ranker (119/180, 95% CI 58.9-72.8), under a size-robust criterion; **55.6%** with the zero-dependency default. On the same structures that is level with P2Rank's 63.3% and more than twice fpocket's 28.3%.

Every number in this section comes from one code state, re-measured end to end after the last change to the ranker weights.

**Size-robust success criterion (top-5 pockets):** a pocket whose lining residues reach a **Jaccard overlap ≥ 0.25** with the known ligand-contact site (Jaccard = |found ∩ known| / |found ∪ known|), **or** whose center is within 4 Å of the site centroid. Lining residues use a true atomic-contact definition (any residue with an atom within 5 Å of the detected cavity). Recall is *not* used as the headline: a large pocket can engulf a small known site and score high recall while sitting nowhere near it. Both numbers print side by side in every benchmark script.

### Head-to-head against other detectors

All five paired on the same 180 test-fold structures under the identical criterion. MDpocket is given the **same NMA ensemble Lacuna uses**, so that row compares analysis pipelines rather than samplers.

| Detector | Kind | Size-robust top-5 | Paired vs `learned-plm` |
|----------|------|:-----------------:|------------------|
| **Lacuna** (`learned-plm`) | ensemble + PLM | **66.1%** (119/180) | - |
| P2Rank | single-structure, learned | 63.3% (114/180) | +2.8% [-4.4, +9.4] |
| **Lacuna** (`learned`, default) | ensemble | **55.6%** (100/180) | +10.6% [+6.1, +15.0] |
| MDpocket | ensemble (same input ensemble) | 43.9% (79/180) | +22.2% [+14.4, +30.0] |
| fpocket | single-structure, geometric | 28.3% (51/180) | +37.8% [+29.4, +46.1] |

With the sequence ranker Lacuna is **level with P2Rank**: nominally ahead by 2.8 points, but the interval spans zero, so parity is the honest word rather than a win. It beats every other detector here by margins whose intervals exclude zero.

**The zero-dependency default does not reach parity.** At 55.6% it trails P2Rank by 7.8 points (CI -15.0 to -0.6, excluding zero). An earlier version of this file called the default indistinguishable from P2Rank; that was measured before the conformer-invariant refit and is no longer accurate. The default still beats MDpocket by +11.7% (CI +3.9 to +19.4) and fpocket by a wide margin.

Union with P2Rank reaches 76.1% and Lacuna alone catches 23 structures P2Rank misses, so the tools remain complementary rather than redundant.

The sequence ranker is an **optional extra** (`pip install "lacuna-pockets[plm]"`) because it needs PyTorch and downloads an ESM-2 checkpoint. It is a separate strategy rather than the default so that identical commands give identical rankings on every machine, whether or not the extra is installed.

MDpocket is the closest relative of this work and the fair ensemble baseline. Its best configuration out of ten (isovalue and ranking rule both swept in its favour) is reported; at its default isovalue it scores 40.2%.

### Independent validation

Default `learned` strategy, with the optional sequence ranker in the last column.

| Benchmark | N | `learned` (default) | Legacy recall | `learned-plm` | Notes |
|-----------|--:|:-----------:|:-------------:|:-------------:|-------|
| CryptoBench (held-out test fold) | 180 | **55.6%** | 77% | **66.1%** | largest & most diverse; the headline |
| PocketMiner | 45 | **73%** (33/45) | 84% | **80%** (36/45) | per-residue cryptic labels |
| Curated apo/holo set (this repo) | 22 | **45%** (10/22) | 68% | 41% (9/22) | hand-picked literature cryptic pairs |
| COACH420 | 144 | **87%** (125/144) | 93% | not measured | *general* holo sites, not cryptic; see below |

On the curated 22 the default edges out the sequence ranker, 10/22 against 9/22. At that sample size the difference is one structure and means nothing on its own, but it is a reminder that the sequence ranker's advantage is established on CryptoBench and does not automatically transfer.

**On general binding sites, a general-purpose tool is better.** COACH420 holds
holo structures whose pocket is already open, which is an easier task and not the
one Lacuna is built for. Paired on the same 144 structures P2Rank scores 93.8%
against Lacuna's 86.8% (-6.9%, CI -12.5 to -1.4, excludes zero), having tied it
on cryptic sites. Each tool wins or ties where it was designed to. The higher
absolute number here reflects the easier task, not better performance:
cross-dataset comparisons of the headline are not meaningful.
**[Detail →](docs/BENCHMARKS.md#coach420-general-binding-sites-and-where-lacunas-specialisation-shows)**

Datasets: PocketMiner (Meller et al. 2023, *Nat. Commun.*); CryptoBench (Vavra et al. 2024, *Bioinformatics*). The CryptoBench split follows the dataset's own homology-separated folds, and the ranker was fitted only on train folds, so the test fold is genuinely unseen.

The curated 22-target set is the hardest of the three despite being the smallest: it was assembled from published cryptic-pocket case studies and is deliberately enriched for the large-motion sites this pipeline handles worst.

```bash
python benchmarks/cryptic_benchmark.py --category cryptic     # curated set (~4 min)
python benchmarks/pocketminer_benchmark.py                    # PocketMiner (auto-downloads)
python benchmarks/cryptobench_benchmark.py                    # CryptoBench test fold (~10 min)
python benchmarks/compare_detectors_cryptobench.py --analyze  # the head-to-head table
python benchmarks/compare_mdpocket.py --folds test            # vs MDpocket (needs mdpocket on PATH)
```

### Where the remaining gap is

Across the candidate set Lacuna generates, **some** cluster clears the criterion for 73.7% of structures, against the 66.1% `learned-plm` surfaces in the top 5. Ranking is therefore close to exhausted: it converts 89% of what detection makes available, and even perfect ordering would add only about 8 points. The remaining 26% is a detection gap, where no candidate clears the bar at any rank.

Several attempts to close it returned nothing measurable: spatial non-maximum suppression, merging adjacent sub-pockets, hard-negative mining, gradient boosting in place of the linear model, and importing P2Rank's own per-pocket confidence as a feature. That last one matters most: if per-point scoring were the missing ranking ingredient, handing the ranker P2Rank's opinion directly would have helped, and it did not (-1.1% on held-out data). P2Rank's advantage lies in proposing different candidates, not in ordering ours better.

The other limit is sampling. Stratifying the test fold by how far the pocket moves between apo and holo, Lacuna recovers 47% of the most-mobile quartile against P2Rank's 62%. The default NMA backend is harmonic and cannot generate large hinge or interface openings, and enhanced-temperature MD, metadynamics along an apo-derived collective variable, and SWISH scaled-water MD were each null against baseline at single-workstation sampling (see `benchmarks/experiments/`). Reliably catching those rare events needs tens to hundreds of nanoseconds across dozens of replicas per target, which is cluster scale, not a missing algorithm.

### Ranking

`--rank-by` selects how sites are ordered. The default `learned` is a linear model over 23 features (pocket geometry, druggability, and ensemble-derived terms such as how far a site's centroid wanders between conformers, which single-structure detectors cannot compute). It is trained on within-structure pairs, so it optimises ordering directly rather than classifying pockets in isolation.

On CryptoBench's test fold it recovers 55.6% against 17.8% for the previous `crypticity` default, rising to 66.1% with `learned-plm`. **On the curated 22-target set the ordering reverses**: `persistence` and `balanced` reach 13/22 where `learned` reaches 10/22, though at n=22 the intervals overlap heavily. If your targets resemble the classic literature case studies more than CryptoBench, the analytic strategies are worth trying. **[Full ablation →](docs/BENCHMARKS.md#ranking-strategies)**

Every pocket also carries a continuous **crypticity score** in [0, 1], the conformational-selection signature of a cryptic site:

```
opening    = (max_volume − apo_volume) / max_volume        # 1.0 if absent in the apo state
crypticity = opening × peak_open_state_druggability
```

A constitutive pocket already formed in the input scores ≈ 0; one that is absent in the apo structure and opens into a druggable cavity scores near 1. The JSON report also includes per-pocket volume dynamics (`apo_volume_A3`, `volume_range_A3`) and `max_druggability`.

NMA runtime is sub-second to ~7s per protein on a laptop CPU, no GPU required. **[Per-size timing →](docs/BENCHMARKS.md#speed-nma-backend-no-gpu)**

---

## Example: K-Ras switch-II

```bash
# Download K-Ras apo (from RCSB); NMA backend (default) recovers switch-II at rank 3
lacuna discover 4OBE.pdb \
    --conformers 20 \
    --emit-boltz-constraints \
    --output kras_pockets/

# pocket_0_constraint.yaml is ready - add your SMILES:
#   - ligand:
#       id: L
#       smiles: YOUR_SMILES_HERE
boltz predict kras_pockets/pocket_0_constraint.yaml
```

See [`examples/kras_cryptic.py`](examples/kras_cryptic.py) for a full annotated Python workflow.

---

## Input formats

Accepts PDB or mmCIF from any predictor or database:
- AlphaFold 2 / AlphaFold 3
- Boltz-1 / Boltz-2
- Chai-1
- RCSB PDB
- ESMFold, RoseTTAFold, OpenFold, etc.

---

## Citation

If you use Lacuna in published research, please cite:

> Moore, C. (2026). *Lacuna: Cryptic Binding Pocket Discovery via Conformational Ensemble Analysis.* https://github.com/mooreneural/lacuna

**BibTeX:**
```bibtex
@software{moore2026lacuna,
  author  = {Moore, Clayton W.},
  title   = {Lacuna: Cryptic Binding Pocket Discovery
             via Conformational Ensemble Analysis},
  year    = {2026},
  url     = {https://github.com/mooreneural/lacuna},
  version = {0.3.1}
}
```

**Methodology papers Lacuna builds on:**

- Atilgan et al. (2001) *Biophys. J.* 80(1):505-515 - Anisotropic Network Model (NMA backend)
- Halgren (2009) *J. Chem. Inf. Model.* 49(2):377-389 - SiteMap druggability scoring
- Le Guilloux et al. (2009) *BMC Bioinformatics* 10:168 - fpocket alpha-sphere approach
- Schmidtke & Barril (2010) *J. Med. Chem.* 53(15):5858-5867 - enclosure scoring

---

## License

**[GNU AGPL-3.0-or-later](LICENSE)** - free to use, study, modify, and share.
The AGPL's copyleft requires that if you distribute a modified version, **or run
a modified version as a network/hosted service**, you make the complete
corresponding source available under the same license.

A separate **[commercial license](LICENSE_COMMERCIAL)** removes the AGPL
copyleft obligation (for embedding Lacuna in closed-source products or hosted
services without releasing your own source) and adds warranty, indemnification,
support SLAs, and custom development.

> Versions ≤ 0.1.2 were released under the MIT License and remain available
> under those terms. AGPL-3.0 applies from version 0.2.0 onward.
