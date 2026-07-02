<p align="center">
  <img src="docs/lacuna_banner.png" alt="Lacuna — cryptic binding pocket discovery via conformational ensemble analysis" width="100%">
</p>

# Lacuna

**Cryptic binding pocket discovery via conformational ensemble analysis.**

Most protein structure predictors (AlphaFold, Boltz, Chai) give you one static structure. But ~70% of disease-relevant proteins are considered "undruggable" not because they're biologically intractable - it's because no pocket is visible in their ground state. K-Ras was "undruggable" for 30 years until a transient cryptic pocket was found in its switch-II region. That pocket now backs sotorasib and adagrasib.

Lacuna finds those pockets. It generates a conformational ensemble from any input structure, detects pockets per conformer, and clusters them across the ensemble to surface sites that only appear transiently - ranked by druggability and persistence.

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
pip install "lacuna-pockets[boltz]"    # Boltz-2 partial diffusion (best quality, GPU recommended)
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

# Use Boltz-2 partial diffusion for highest-quality sampling
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

1. **Ensemble generation** - Generate N conformers via elastic network model normal mode analysis (built-in), OpenMM implicit-solvent MD, or Boltz-2 partial diffusion at varying noise levels
2. **Pocket detection** - Grid-based alpha-point analysis per conformer: compute distance transform, find local maxima within the 1.4–5.5 Å interaction zone, cluster nearby alpha-points into pocket candidates
3. **Cross-ensemble clustering** - Greedy centroid merging clusters corresponding pockets across all conformers
4. **Druggability scoring** - Gaussian volume reward centered at 300 Å³ + enclosure + hydrophobicity + aromaticity (Halgren 2009), scored in each conformer
5. **Crypticity scoring & ranking** - Each site gets a continuous crypticity score (how much it opens relative to the apo state × druggability when open) and is flagged `cryptic: true` if present in <90% of conformers. Pockets are ranked by peak open-state druggability by default; `--rank-by crypticity` surfaces the most cryptic sites first

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
| `boltz` | `lacuna[boltz]` | best | ~30s/conf (GPU) | Boltz-2 partial diffusion at varying noise fractions |
| `random` | built-in | baseline | ~0.04s/conf | Correlated Gaussian backbone perturbation |

**Auto-selection order:** `boltz` → `openmm` → `nma` → `random`. On a plain `pip install lacuna`, the NMA backend runs automatically.

The `nma` backend samples physically meaningful collective motions — the same hinge-bending and breathing modes that open cryptic pockets in nature — without requiring a GPU or force field. It replaces `random` as the zero-dependency default. For large-scale loop rearrangements or very deep cryptic sites, `boltz` remains the best option.

---

## Benchmarks

**17 / 20 cryptic pockets detected (85%, NMA backend, 20 conformers)** - exceeding the CryptoSite published benchmark rate on a statistically defensible N=20 set.

Success criterion (field standard, top-5 pockets): pocket centroid within 4 Å of the known binding-site centroid **or** ≥30% residue overlap. The numbers below are reproduced by the default configuration (`--backend nma --rank-by druggability`).

> **Transparency:** these are the OR-criterion pass counts. Reported per-metric: of the 20 cryptic targets, **17 pass on residue overlap** and **2 also satisfy the stricter centroid-distance test**. The centroid-of-binding-residues is an intentionally strict and somewhat ill-posed reference for elongated grooves, so the residue-overlap criterion (used by CryptoSite and PocketMiner) is the primary metric. `cryptic_benchmark.py` now prints the full per-metric breakdown.

### Cryptic pockets - 17 / 20 (85%)

| Protein | Apo PDB | Drug target | Overlap | Rank | Time |
|---------|---------|-------------|---------|------|------|
| ✅ T4L L99A hydrophobic cavity | 1L90 | - | 100% | 1 | 0.9s |
| ✅ K-Ras switch-II pocket | 4OBE | sotorasib / adagrasib | 93% | 3 | 0.9s |
| ✅ IL-2 helix-α1 site | 1M47 | - | 100% | 4 | 0.7s |
| ✅ MDM2 p53-binding cleft | 1Z1M | nutlin-3 | 47% | 4 | 1.1s |
| ✅ BCL-XL BH3 groove | 1LXL | navitoclax | 91% | 2 | 2.9s |
| ✅ BCL-2 BH3 groove | 1G5M | venetoclax | 96% | 5 | 1.2s |
| ✅ c-ABL myristate pocket | 3CS9 | asciminib | 44% | 1 | 1.7s |
| ✅ PTP1B allosteric helix site | 1A5Y | benzofuran inhibitors | 59% | 3 | 2.2s |
| ✅ p38α DFG-out pocket | 1P38 | BIRB 796 | 38% | 1 | 3.0s |
| ✅ HIV-1 RT NNRTI pocket | 1HMV | nevirapine | 94% | 2 | 8.4s |
| ✅ HCV NS5B thumb-site I | 1NB4 | VXR class | 60% | 5 | 4.7s |
| ✅ PPARγ allosteric AF-2 site | 2PRG | metaglidasen | 65% | 4 | 1.7s |
| ✅ Glucokinase allosteric site | 1V4S | B84 activator | 100% | 3 | 3.8s |
| ✅ MMP-13 S1′ allosteric tunnel | 2OZR | non-zinc inhibitors | 56% | 3 | 1.1s |
| ✅ Src myristate pocket | 2SRC | - | 48% | 5 | 4.0s |
| ✅ SHP-2 allosteric tunnel | 2SHP | SHP099 class | 47% | 1 | 6.1s |
| ✅ ERK2 allosteric site | 2ERK | - | 31% | 1 | 2.9s |
| ❌ Caspase-1 dimer interface | 2HBQ | - | 8% | - | 1.7s |
| ❌ IDH1 R132H dimer interface | 3MAP | ivosidenib | 21% | - | 3.7s |
| ❌ PKM2 subunit-interface activator | 1ZJH | TEPP-46 class | 25% | - | 4.3s |

Switching from the zero-dependency `random` backend to the physics-based `nma` backend recovers the former near-misses (IL-2 21%→100%, Src 28%→48%, SHP-2 and ERK2 now pass), and ranking by peak open-state druggability surfaces the right pocket into the top 5.

**All three remaining misses are oligomeric-interface pockets** - they form between subunits and cannot be seen in a single-chain analysis. They are addressable with the `--homodimer` flag, which reads BIOMT symmetry records and constructs the full biological assembly before analysis:

```bash
lacuna discover 2HBQ.pdb --homodimer --conformers 20   # Caspase-1 dimer interface
lacuna discover 3MAP.pdb --homodimer --conformers 20   # IDH1 R132H dimer interface
```

For the very hardest sites requiring large loop rearrangement, the optional Boltz-2 backend (`--backend boltz`) samples states unreachable by NMA.

### Conformational and orthosteric controls

| Category | Result | Notable entries |
|----------|--------|-----------------|
| Conformational | 1 / 1 (100%) | Adenylate kinase open→closed (rank 1) |
| Orthosteric | 5 / 6 (83%) | HIF-2α 100% (1.1 Å centroid), lysozyme 100%, thrombin, DHFR 72% |
| Orthosteric miss | - | Trypsin (1S0Q non-standard residue numbering - documented limitation) |

**Overall across all 27 proteins: 23 / 27 (85%).**

### Crypticity score

Every reported pocket now carries a continuous **crypticity score** in [0, 1] - the conformational-selection signature of a cryptic site, defined as how much the pocket opens relative to the apo/input structure × how druggable it is once open:

```
opening    = (max_volume − apo_volume) / max_volume        # 1.0 if absent in the apo state
crypticity = opening × peak_open_state_druggability
```

A constitutive pocket already formed in the input structure scores ≈ 0; a pocket absent in the apo structure that opens into a druggable cavity scores near 1. As an independent validation, ranking the benchmark **purely by crypticity** (`--rank-by crypticity`) still recovers 15/20 known cryptic pockets - the score discriminates true cryptic sites with no druggability tie-breaking. The JSON report also includes per-pocket volume dynamics (`apo_volume_A3`, `volume_range_A3`) and `max_druggability`.

### Ranking strategies

`--rank-by` selects how pockets are ordered (cryptic benchmark pass rate, NMA, N=20):

| Strategy | Description | Cryptic pass |
|----------|-------------|--------------|
| `druggability` (default) | peak open-state composite druggability | **17 / 20** |
| `persistence` | legacy persistence × druggability | 16 / 20 |
| `balanced` | druggability with a mild persistence bonus | 15 / 20 |
| `crypticity` | most cryptic sites first | 15 / 20 |

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
| 4OBE K-Ras switch-II **(cryptic)** | ❌ not in top 5 | ✅ 93%, rank 3 |
| 1HPV HIV-1 protease (orthosteric) | ✅ rank 1 | ✅ rank 1 |
| **Score** | **2 / 4** | **4 / 4** |

T4L L99A and K-Ras switch-II are the canonical single-structure benchmark failures: the T4L cavity is physically absent in the apo crystal (<100 Å³), and the K-Ras switch-II pocket only opens during nucleotide exchange.

> **Reproduce:**
> ```bash
> python benchmarks/cryptic_benchmark.py          # full 27-protein run, NMA backend (~5 min)
> python benchmarks/cryptic_benchmark.py --quick  # 10 conformers (~2 min)
> python benchmarks/cryptic_benchmark.py --category cryptic            # cryptic only
> python benchmarks/cryptic_benchmark.py --backend random --rank-by persistence  # ablations
> python benchmarks/compare_fpocket.py            # fpocket head-to-head
> ```

---

## Example: K-Ras switch-II

```bash
# Download K-Ras apo (from RCSB)
# Run with Boltz backend for highest-quality switch-II sampling
lacuna discover 4OBE.pdb \
    --backend boltz \
    --conformers 30 \
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

- Atilgan et al. (2001) *Biophys. J.* 80(1):505–515 - Anisotropic Network Model (NMA backend)
- Halgren (2009) *J. Chem. Inf. Model.* 49(2):377–389 - SiteMap druggability scoring
- Le Guilloux et al. (2009) *BMC Bioinformatics* 10:168 - fpocket alpha-sphere approach
- Schmidtke & Barril (2010) *J. Med. Chem.* 53(15):5858–5867 - enclosure scoring

---

## License

**[GNU AGPL-3.0-or-later](LICENSE)** — free to use, study, modify, and share.
The AGPL's copyleft requires that if you distribute a modified version, **or run
a modified version as a network/hosted service**, you make the complete
corresponding source available under the same license.

A separate **[commercial license](LICENSE_COMMERCIAL)** removes the AGPL
copyleft obligation (for embedding Lacuna in closed-source products or hosted
services without releasing your own source) and adds warranty, indemnification,
support SLAs, and custom development. Contact claytonwaynemoore@gmail.com.

> Versions ≤ 0.1.2 were released under the MIT License and remain available
> under those terms. AGPL-3.0 applies from version 0.2.0 onward.
