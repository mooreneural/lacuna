<img width="550" height="541" alt="bclxl_pocket" src="https://github.com/user-attachments/assets/6169cd25-c400-4579-8836-29fd3cccccd3" />

# Introduction

**Cryptic binding pocket discovery via conformational ensemble analysis.**

Most protein structure predictors (AlphaFold, Boltz, Chai) give you one static structure. But ~70% of disease-relevant proteins are considered "undruggable" not because they're biologically intractable - it's because no pocket is visible in their ground state. K-Ras was "undruggable" for 30 years until a transient cryptic pocket was found in its switch-II region. That pocket now backs sotorasib and adagrasib.

Lacuna finds those pockets. It generates a conformational ensemble from any input structure, detects pockets per conformer, and clusters them across the ensemble to surface sites that only appear transiently - ranked by a continuous crypticity score.

```
lacuna discover kras.pdb --conformers 20 --emit-boltz-constraints --emit-vina-boxes
```

---

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
5. **Crypticity scoring & ranking** - Each site gets a continuous crypticity score (how much it opens relative to the apo state × druggability when open) and is flagged `cryptic: true` if present in <90% of conformers. Pockets are ranked by crypticity by default; `--rank-by druggability` is available for always-open / orthosteric sites

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
| `nma` | built-in | good | ~0.1s/conf | Anisotropic Network Model normal mode analysis - hinge bending, breathing, twist motions |
| `openmm` | `lacuna[openmm]` | good | ~2s/conf | 100ps Langevin MD, GBn2 implicit solvent |
| `boltz` | `lacuna[boltz]` | experimental | ~100s/protein (GPU) | Boltz-2 diffusion sampling from sequence; high diversity but noisy (see note) |
| `random` | built-in | baseline | ~0.04s/conf | Correlated Gaussian backbone perturbation |

**Auto-selection order:** `boltz` → `openmm` → `nma` → `random`. On a plain `pip install lacuna`, the NMA backend runs automatically.

The `nma` backend samples physically meaningful collective motions - the same hinge-bending and breathing modes that open cryptic pockets in nature - without requiring a GPU or force field. It is the zero-dependency default.

> **Boltz backend status (honest note).** The `boltz` backend runs Boltz-2 diffusion sampling on a GPU, but it currently predicts each conformer *de novo from sequence* (not partial diffusion from the input structure), which yields structurally divergent, noisy ensembles (150-300+ pocket clusters vs NMA's ~35). In GPU benchmarking it did **not** reliably improve cryptic detection over NMA. A proper apo-templated integration with sequence-based residue mapping is planned; until then, NMA is the recommended backend.

---

## Benchmarks

**7 / 22 cryptic pockets localized (32%, size-robust criterion; NMA backend, crypticity ranking, 20 conformers).**

This curated result is cross-validated on two further independent datasets - **PocketMiner 31%** and **CryptoBench 18%** (the largest and hardest) - see [Independent validation](#independent-validation--three-benchmarks) below.

**Size-robust success criterion (top-5 pockets):** a pocket whose lining residues reach a **Jaccard overlap ≥ 0.25** with the known ligand-contact site (Jaccard = |found ∩ known| / |found ∪ known|), **or** whose center is within 4 Å of the site centroid. Lining residues use a true atomic-contact definition (any residue with an atom within 5 Å of the detected cavity). Reproduce with `python benchmarks/cryptic_benchmark.py --category cryptic`.

> **Why the number is lower than you may have seen before - please read.** Earlier releases reported this benchmark using plain **recall** (|found ∩ known| / |known| ≥ 30%), which gave **13/22 (59%)**. That metric is *size-gameable*: a large pocket engulfs most of a small known site and scores high recall while sitting nowhere near it. We verified this directly - a learned re-ranker reached 84% on the recall metric purely by ranking pockets on raw volume. We therefore switched the headline to a **size-robust** criterion (Jaccard, which penalizes oversized pockets, OR a ≤4 Å centroid hit). Under it the honest numbers roughly halve. Both criteria are printed side by side by every benchmark script; we lead with the robust one because it is the number we can defend on held-out data.
>
> Of the 22 cryptic targets, **2 pass the strict ≤4 Å centroid test** (IL-2, PTP1B) and 5 more clear Jaccard ≥ 0.25. Precise pocket-center localization is genuinely hard for elongated, partially-open cryptic grooves, which is why the centroid-only pass rate is low. `cryptic_benchmark.py` prints the full per-metric breakdown (centroid, Jaccard at 0.20/0.25/0.30, and legacy recall).

### Cryptic pockets - 7 / 22 (32%)

Sorted by Jaccard (size-robust overlap). ✅ = passes the size-robust criterion (Jaccard ≥ 0.25 **or** centroid ≤ 4 Å); recall is the legacy size-gameable metric, shown for contrast. "Rank" is the position of the best-matching top-5 pocket.

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

**The remaining gap is mostly sampling, not ranking.** Raising the cutoff from top-5 to top-20 lifts the size-robust score only from **7/22 to 10/22** - just 3 pockets are detected-but-mis-ranked. The other 12 misses are not localized at all even at top-20, so they are a sampling/localization ceiling (the NMA ensemble never opens or the detector never localizes the site tightly enough) rather than a ranking failure. This is the honest picture: under the older recall metric the top-20 ceiling looked like 73%, which made the problem appear to be ranking - it was largely the metric. The hard cases split into **oligomeric-interface pockets** (Caspase-1, IDH1, PKM2) that form *between* subunits and are invisible to single-chain analysis, and **large-rearrangement sites** (p38 DFG-out, c-ABL myristate) that need sampling beyond elastic-network modes.

Dimer-interface pockets are partly addressable with `--homodimer` (reads BIOMT records and builds the biological assembly), though this benchmark's single-chain-referenced scoring does not credit them. For large-rearrangement sites the optional Boltz-2 backend samples more broadly, but its current sequence-based integration is noisy - see [Backends](#backends).

### Independent validation - three benchmarks

Measured on three independent datasets (NMA + crypticity, top-5). Both criteria are reported: the **size-robust** headline (Jaccard ≥ 0.25 **or** ≤ 4 Å centroid) and the **legacy recall** number (≥ 30% recall **or** ≤ 4 Å centroid) that earlier releases led with.

| Benchmark | N | Size-robust | Legacy recall | Notes |
|-----------|--:|:-----------:|:-------------:|-------|
| Curated apo/holo set (this repo) | 22 | **32%** | 59% | literature cryptic pairs |
| PocketMiner (Meller 2023, *Nat. Commun.*) | 45 | **31%** | 60% | per-residue cryptic labels |
| CryptoBench test fold (Vavra 2024, *Bioinformatics*) | 180 | **18%** | 49% | largest & most diverse; harder |
| CryptoBench train folds (generalization check) | 749 | **13%** | 50% | brand-new pockets, held out from all tuning |

The two curated/field-standard sets converge at ~31-32% under the size-robust metric; **CryptoBench** - the field's largest cryptic set (1107 structures; 180 of its 222-structure held-out test fold evaluated here) - is harder at **18%**. The legacy recall column roughly doubles every number: that gap is the size-gaming headroom the recall metric leaves open (a large pocket covers a small known site without being localized on it), which is exactly why the size-robust number is the one we lead with.

**Generalization.** To check that these numbers are not an artifact of the specific test fold, we scored all 749 CryptoBench *train*-fold structures, which were never used in any tuning: **13%** size-robust (95% CI 10-16%) and 50% legacy. Both are statistically consistent with the test fold (overlapping confidence intervals), so the honest headline holds up on genuinely unseen pockets. Reproduce (each script prints both criteria):

```bash
python benchmarks/pocketminer_benchmark.py    # PocketMiner (auto-downloads)
python benchmarks/cryptobench_benchmark.py    # CryptoBench test fold (auto-downloads, ~10 min)
```

### Limitations and scaling (this ceiling is a compute problem)

The honest ceiling above (about 32% on the curated set, 13 to 18% on CryptoBench under the size-robust metric) is set by **conformational sampling**, not by ranking or pocket detection. At a top-20 cutoff the numbers rise only slightly, which means the pocket is usually not found-but-mis-ranked; it is simply never sampled in an open state.

The remaining misses concentrate in the **large-collective-motion classes**, hinge and oligomeric-interface openings. The default NMA backend is harmonic and cannot generate those motions. Molecular dynamics can in principle, but a cryptic opening is a **rare event**: in our tests, short trajectories (0.5 to 3 ns) essentially never caught one, and enhanced-temperature MD, metadynamics along an apo-derived collective variable, and SWISH scaled-water MD were all null at the sampling a single workstation affords (see `benchmarks/experiments/`).

Raising this ceiling is a **compute problem, not a missing algorithm**. Reliably observing rare openings needs orders of magnitude more MD sampling: tens to hundreds of nanoseconds per trajectory across dozens of independent replicas, aggregating microseconds per target, the scale used by the successful literature (for example PocketMiner's ~940,000 simulation windows and Folding@home-style datasets). As an anchor, the development GPU runs a small protein at roughly 300 ns/day; sampling rare openings across the 22 to 885 benchmark targets, with the frontier proteins several times slower, is tens to hundreds of GPU-days. **That is cluster or cloud GPU scale.** With that budget, the same pipeline could be driven by long multi-replica MD (or cosolvent MD) to attack the hinge and interface classes that are out of reach on a single machine.

### Orthosteric / conformational controls

Crypticity ranking (the default) intentionally de-prioritizes always-open sites, so for orthosteric / general pocket finding use `--rank-by druggability`. Under the corrected contact-lining pipeline (NMA, `--rank-by druggability`):

| Category | Result | Notes |
|----------|--------|-------|
| Orthosteric | 3 / 6 | hen lysozyme 100%, HIF-2α 96% (1.1 Å centroid), DHFR 50%; misses HIV protease, thrombin, trypsin (1S0Q numbering) |
| Conformational | 1 / 1 | adenylate kinase open→closed |

Orthosteric detection is a known relative weakness of the tight-contact pipeline - the tool is tuned for transient cryptic sites, not always-open active-site grooves.

### Crypticity score

Every reported pocket carries a continuous **crypticity score** in [0, 1] - the conformational-selection signature of a cryptic site, defined as how much the pocket opens relative to the apo/input structure × how druggable it is once open:

```
opening    = (max_volume − apo_volume) / max_volume        # 1.0 if absent in the apo state
crypticity = opening × peak_open_state_druggability
```

A constitutive pocket already formed in the input structure scores ≈ 0; a pocket absent in the apo structure that opens into a druggable cavity scores near 1. Ranking by crypticity is the **default** and recovers the most cryptic targets. The JSON report also includes per-pocket volume dynamics (`apo_volume_A3`, `volume_range_A3`) and `max_druggability`.

### Ranking strategies

`--rank-by` selects how pockets are ordered (cryptic benchmark pass rate, NMA, N=20):

| Strategy | Description | Cryptic pass |
|----------|-------------|--------------|
| `crypticity` (default) | most cryptic sites first | **12 / 20** |
| `druggability` | peak open-state composite druggability | 10 / 20 |
| `balanced` | druggability with a mild persistence bonus | 8 / 20 |
| `persistence` | legacy persistence × druggability | 7 / 20 |

### Speed (NMA backend, no GPU)

| Protein size | Time |
|-------------|------|
| ~130 residues (lysozyme) | 0.6s |
| ~170 residues (MDM2) | 1.1s |
| ~350 residues (K-Ras) | 0.9s |
| ~530 residues (HIV-1 RT chain A) | 8.4s |

### Head-to-head: Lacuna vs fpocket

fpocket detects pockets on a single static structure. Lacuna generates a conformational ensemble to expose sites that only become visible once the protein moves. Run side by side on the same structures under the same size-robust criterion (top-5, Jaccard ≥ 0.25 or centroid ≤ 4 Å), the two tools catch largely different pockets:

| Set | fpocket | Lacuna | **Combined (either)** |
|-----|:-------:|:------:|:----------------------:|
| CryptoBench test fold (n=180) | 28% (51/180) | 16% (29/180) | **38% (68/180)** |
| Curated cryptic set (n=22) | 18% (4/22) | 18% (4/22) | **36% (8/22)** |

On CryptoBench, Lacuna independently recovers **17 pockets that fpocket misses entirely**: sites invisible to single-structure geometric detection that only open once the ensemble samples them. On the curated set, the hit lists don't overlap at all: fpocket catches T4 lysozyme's buried cavity and PTP1B's allosteric site, while Lacuna catches the BCL-2/BCL-XL BH3 grooves, MDM2's p53-binding cleft, and IL-2's helix pocket, sites that open through conformational change rather than being present in one fixed geometry. Running both and taking the union beats either tool alone on both benchmarks.

> **Reproduce:**
> ```bash
> python benchmarks/compare_fpocket.py                            # 22 curated cryptic targets vs fpocket
> python benchmarks/compare_fpocket_cryptobench.py --folds test   # CryptoBench test fold vs fpocket (~7 min)
> python benchmarks/cryptic_benchmark.py --category cryptic       # 22 cryptic targets, NMA (~4 min)
> python benchmarks/cryptic_benchmark.py --quick                  # 10 conformers, faster
> python benchmarks/cryptic_benchmark.py --category cryptic --rank-by druggability  # ablation
> python benchmarks/cryptic_benchmark.py --category cryptic --top-n 20              # detection ceiling
> ```

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
  version = {0.2.0}
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
support SLAs, and custom development. Contact claytonwaynemoore@gmail.com.

> Versions ≤ 0.1.2 were released under the MIT License and remain available
> under those terms. AGPL-3.0 applies from version 0.2.0 onward.
