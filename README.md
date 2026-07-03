<p align="center">
  <img src="docs/lacuna_banner.png" alt="Lacuna - cryptic binding pocket discovery via conformational ensemble analysis" width="100%">
</p>

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

**13 / 22 cryptic pockets detected (59%, NMA backend, crypticity ranking, 20 conformers).**

This curated result is cross-validated on two further independent datasets - **PocketMiner 62%** and **CryptoBench 49%** (the largest and hardest) - see [Independent validation](#independent-validation--three-benchmarks) below.

Success criterion (top-5 pockets): a pocket whose lining residues overlap ≥30% with the known ligand-contact site, **or** whose center is within 4 Å of the site centroid. Lining residues use a true atomic-contact definition (any residue with an atom within 5 Å of the detected cavity). Reproduce with `python benchmarks/cryptic_benchmark.py --category cryptic`.

> **Transparency - please read.** These are OR-criterion pass counts. Of the 22 cryptic targets, **13 pass on residue overlap** and **2 also satisfy the strict ≤4 Å centroid test** (PTP1B, IL-2). Precise pocket-center localization is hard for elongated, partially-open cryptic grooves, so residue overlap (the criterion used by CryptoSite and PocketMiner) is the primary metric, reported alongside the strict centroid test. `cryptic_benchmark.py` prints the full per-metric breakdown.
>
> Earlier releases used a looser lining definition (residues within a ~13 Å sphere of the pocket center); the current atomic-contact criterion (≤5 Å from the detected cavity) is stricter and more conservative, which lowers the reported overlap. The figures here reflect that stricter criterion, cross-validated across three independent datasets.

### Cryptic pockets - 13 / 22 (59%)

| Protein | Apo PDB | Drug target | Overlap | Rank |
|---------|---------|-------------|--------:|:----:|
| ✅ T4 Lysozyme L99A cavity | 1L90 | - | 100% | 1 |
| ✅ Glucokinase allosteric site | 1V4S | activators | 100% | 2 |
| ✅ PTP1B allosteric helix site | 1A5Y | benzofurans | 94% | 5 |
| ✅ IL-2 helix-α1 site | 1M47 | - | 93% | 1 |
| ✅ K-Ras switch-II pocket | 4OBE | sotorasib/adagrasib | 79% | 3 |
| ✅ BCL-XL BH3 groove | 1LXL | navitoclax | 68% | 1 |
| ✅ HIV-1 RT NNRTI pocket | 1HMV | nevirapine | 62% | 4 |
| ✅ BCL-2 BH3 groove | 1G5M | venetoclax | 59% | 2 |
| ✅ Ricin A pterin pocket | 1RTC | - | 50% | 4 |
| ✅ MDM2 p53-binding cleft | 1Z1M | nutlin-3 | 47% | 1 |
| ✅ HCV NS5B thumb-site I | 1NB4 | VXR class | 47% | 4 |
| ✅ Src myristate pocket | 2SRC | - | 36% | 4 |
| ✅ PPARγ allosteric site | 2PRG | metaglidasen | 35% | 2 |
| ❌ Caspase-1 dimer interface | 2HBQ | - | 25% | - |
| ❌ p38α DFG-out pocket | 1P38 | BIRB 796 | 24% | - |
| ❌ ERK2 allosteric site | 2ERK | - | 19% | - |
| ❌ c-ABL myristate pocket | 3CS9 | asciminib | 19% | - |
| ❌ PKM2 subunit interface | 1ZJH | TEPP-46 | 17% | - |
| ❌ IDH1 R132H dimer interface | 3MAP | ivosidenib | 7% | - |
| ❌ MMP-13 S1′ tunnel | 2OZR | non-zinc | 6% | - |
| ❌ SHP-2 allosteric tunnel | 2SHP | SHP099 | 0% | - |
| ❌ TEM-1 allosteric site | 1JWP | CBT | 0% | - |

**The gap is ranking, not detection.** At a top-20 cutoff the ensemble already contains **16/22 (73%)** of the true pockets - several misses are detected but ranked below 5. The remaining hard cases split into two classes: **oligomeric-interface pockets** (Caspase-1, IDH1, PKM2) that form *between* subunits and are invisible to single-chain analysis, and **large-rearrangement sites** (p38 DFG-out, c-ABL myristate) that need sampling beyond elastic-network modes.

Dimer-interface pockets are partly addressable with `--homodimer` (reads BIOMT records and builds the biological assembly), though this benchmark's single-chain-referenced scoring does not credit them. For large-rearrangement sites the optional Boltz-2 backend samples more broadly, but its current sequence-based integration is noisy - see [Backends](#backends).

### Independent validation - three benchmarks

Cryptic-pocket recall measured on three independent datasets (NMA + crypticity, top-5, ≥30% residue overlap **or** ≤4 Å centroid):

| Benchmark | N | Recall | Notes |
|-----------|--:|:------:|-------|
| Curated apo/holo set (this repo) | 22 | **59%** | literature cryptic pairs |
| PocketMiner (Meller 2023, *Nat. Commun.*) | 45 | **62%** | per-residue cryptic labels |
| CryptoBench test fold (Vavra 2024, *Bioinformatics*) | 180 | **49%** | largest & most diverse; harder |

The curated and PocketMiner sets agree at ~60%; **CryptoBench** - the field's largest cryptic set (1107 structures; 180 of its 222-structure held-out test fold evaluated here) - is harder at **49%**, with a further ~18% of structures landing in the 20-29% overlap band just below the pass line (median overlap 29%). Two curated/field-standard sets converging at ~60% and the hardest comprehensive set at ~49% bound the honest recall. Reproduce:

```bash
python benchmarks/pocketminer_benchmark.py    # PocketMiner (auto-downloads)
python benchmarks/cryptobench_benchmark.py    # CryptoBench test fold (auto-downloads, ~10 min)
```

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

fpocket detects pockets on a single static structure. Lacuna generates a conformational ensemble - the critical difference for cryptic sites that are absent in the apo crystal.

| Target | fpocket 4.2 | Lacuna (NMA backend) |
|--------|------------|----------------------|
| 1HEL hen lysozyme (orthosteric) | ✅ rank 1 | ✅ 100% |
| 1L90 T4L L99A **(cryptic)** | ❌ not in top 5 | ✅ 100%, rank 1 |
| 4OBE K-Ras switch-II **(cryptic)** | ❌ not in top 5 | ✅ 79%, rank 3 |
| 1HPV HIV-1 protease (orthosteric) | ✅ rank 1 | ✅ rank 1 |
| **Score** | **2 / 4** | **4 / 4** |

T4L L99A and K-Ras switch-II are the canonical single-structure benchmark failures: the T4L cavity is physically absent in the apo crystal (<100 Å³), and the K-Ras switch-II pocket only opens during nucleotide exchange.

> **Reproduce:**
> ```bash
> python benchmarks/cryptic_benchmark.py --category cryptic   # 22 cryptic targets, NMA (~4 min)
> python benchmarks/cryptic_benchmark.py --quick              # 10 conformers, faster
> python benchmarks/cryptic_benchmark.py --category cryptic --rank-by druggability  # ablation
> python benchmarks/cryptic_benchmark.py --category cryptic --top-n 20              # detection ceiling
> python benchmarks/compare_fpocket.py                        # fpocket head-to-head
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
