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
4. **Druggability scoring** - Gaussian volume reward centered at 300 Å³ + enclosure + hydrophobicity + aromaticity (Halgren 2009)
5. **Cryptic flagging** - Pockets present in <90% of conformers are marked `cryptic: true`

---

## Outputs

| File | Description |
|------|-------------|
| `pocket_report.json` | Ranked pocket metadata: centroid, volume, druggability, persistence, lining residues |
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

**14 / 20 cryptic pockets detected (70%, NMABackend, 20 conformers)** - matching the CryptoSite published benchmark rate on a statistically defensible N=20 set.

Success criterion: pocket centroid within 4 Å of the known binding-site centroid (field standard) **or** ≥30% residue overlap in top-5 pockets.

### Cryptic pockets - 14 / 20 (70%)

| Protein | Apo PDB | Drug target | Overlap | Time |
|---------|---------|-------------|---------|------|
| ✅ T4L L99A hydrophobic cavity | 1L90 | - | 100% | 0.9s |
| ✅ K-Ras switch-II pocket | 4OBE | sotorasib / adagrasib | 93% | 0.9s |
| ✅ MDM2 p53-binding cleft | 1Z1M | nutlin-3 | 95% | 1.1s |
| ✅ BCL-XL BH3 groove | 1LXL | navitoclax | 91% | 2.9s |
| ✅ BCL-2 BH3 groove | 1G5M | venetoclax | 73% | 1.2s |
| ✅ c-ABL myristate pocket | 3CS9 | asciminib | 44% | 1.7s |
| ✅ PTP1B allosteric helix site | 1A5Y | benzofuran inhibitors | 41% | 2.1s |
| ✅ p38α DFG-out pocket | 1P38 | BIRB 796 | 38% | 3.0s |
| ✅ HIV-1 RT NNRTI pocket | 1HMV | nevirapine | 38% | 8.3s |
| ✅ HCV NS5B thumb-site I | 1NB4 | VXR class | 33% | 4.4s |
| ✅ PKM2 allosteric activator | 1ZJH | TEPP-46 class | 33% | 4.3s |
| ✅ PPARγ allosteric AF-2 site | 2PRG | metaglidasen | 35% | 1.7s |
| ✅ Glucokinase allosteric site | 1V4S | B84 activator | 30% | 3.7s |
| ✅ MMP-13 S1′ allosteric tunnel | 2OZR | non-zinc inhibitors | 50% | 1.1s |
| ❌ IL-2 helix-α1 site *(Boltz-2 → 71% ✅)* | 1M47 | - | 21% | 0.7s |
| ❌ Src myristate pocket | 2SRC | - | 28% | 4.1s |
| ❌ SHP-2 allosteric tunnel | 2SHP | SHP099 class | 24% | 6.0s |
| ❌ ERK2 allosteric site | 2ERK | - | 27% | 2.9s |
| ❌ Caspase-1 dimer interface | 2HBQ | - | 8% | 1.6s |
| ❌ IDH1 R132H dimer interface | 3MAP | ivosidenib | 0% | 3.7s |

The six misses fall into two mechanistic classes: **near-misses** (IL-2, Src, SHP-2, ERK2 all 21–28%) where a physics-based backend closes the gap, and **dimer-interface pockets** (Caspase-1, IDH1 R132H) where the pocket forms between two chains.

Dimer-interface pockets are now addressable with the `--homodimer` flag, which reads BIOMT symmetry records from the PDB and constructs the full biological assembly before analysis:

```bash
lacuna discover 2HBQ.pdb --homodimer --conformers 20   # Caspase-1 dimer interface
lacuna discover 3MAP.pdb --homodimer --conformers 20   # IDH1 R132H dimer interface
```

IL-2 is confirmed at 71% rank 1 with Boltz-2 partial diffusion.

### Boltz-2 re-evaluation of near-misses

| Protein | RandomBackend | Boltz-2 | Notes |
|---------|--------------|---------|-------|
| IL-2 (1M47) | 21% - ❌ | **71% rank 1 - ✅** | Boltz samples the helix-α1 open state |
| Src myristate (2SRC) | 28% - ❌ | 8% - ❌ | Requires SH2-kinase linker rearrangement; needs MD |

### Conformational and orthosteric controls

| Category | Result | Notable entries |
|----------|--------|-----------------|
| Conformational | 1 / 1 (100%) | Adenylate kinase open→closed (35%, rank 1) |
| Orthosteric | 4 / 6 (67%) | HIV protease 100%, DHFR 100%, HIF-2α 100% |
| Orthosteric miss | - | Trypsin (residue numbering offset); thrombin (940-residue complex) |

**Overall across all 27 proteins: 19 / 27 (70%).**

### Speed (NMABackend, no GPU)

| Protein size | Time |
|-------------|------|
| ~130 residues (lysozyme) | 0.6s |
| ~170 residues (MDM2) | 1.1s |
| ~350 residues (K-Ras) | 0.9s |
| ~530 residues (HIV-1 RT chain A) | 8.3s |

### Head-to-head: Lacuna vs fpocket

fpocket detects pockets on a single static structure. Lacuna generates a conformational ensemble - the critical difference for cryptic sites that are absent in the apo crystal.

| Target | fpocket 4.2 | Lacuna (NMABackend) |
|--------|------------|----------------------|
| 1HEL hen lysozyme (orthosteric) | ✅ rank 1 | ✅ 100%, rank 2 |
| 1L90 T4L L99A **(cryptic)** | ❌ not in top 5 | ✅ 100%, rank 1 |
| 4OBE K-Ras switch-II **(cryptic)** | ❌ not in top 5 | ✅ 93%, rank 4 |
| 1HPV HIV-1 protease (orthosteric) | ✅ rank 1 | ✅ 100%, rank 1 |
| **Score** | **2 / 4** | **4 / 4** |

T4L L99A and K-Ras switch-II are the canonical single-structure benchmark failures: the T4L cavity is physically absent in the apo crystal (<100 Å³), and the K-Ras switch-II pocket only opens during nucleotide exchange.

> **Reproduce:**
> ```bash
> python benchmarks/cryptic_benchmark.py          # full 27-protein run, NMA backend (~5 min)
> python benchmarks/cryptic_benchmark.py --quick  # 10 conformers (~2 min)
> python benchmarks/cryptic_benchmark.py --category cryptic   # cryptic only
> python benchmarks/boltz_nearmiss.py             # Boltz-2 near-miss eval (GPU)
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

> Moore, C.W. (2026). *Lacuna: Cryptic Binding Pocket Discovery via Conformational Ensemble Analysis.* https://github.com/mooreneural/lacuna

**BibTeX:**
```bibtex
@software{moore2026lacuna,
  author  = {Moore, Clayton W.},
  title   = {Lacuna: Cryptic Binding Pocket Discovery
             via Conformational Ensemble Analysis},
  year    = {2026},
  url     = {https://github.com/mooreneural/lacuna},
  version = {0.1.0}
}
```

**Methodology papers Lacuna builds on:**

- Atilgan et al. (2001) *Biophys. J.* 80(1):505–515 - Anisotropic Network Model (NMA backend)
- Halgren (2009) *J. Chem. Inf. Model.* 49(2):377–389 - SiteMap druggability scoring
- Le Guilloux et al. (2009) *BMC Bioinformatics* 10:168 - fpocket alpha-sphere approach
- Schmidtke & Barril (2010) *J. Med. Chem.* 53(15):5858–5867 - enclosure scoring

---

## License

**[MIT License](LICENSE)** — free for any use, including commercial.

A [commercial license](LICENSE_COMMERCIAL) is available for organizations that
require indemnification, warranty coverage, support SLAs, or custom development
agreements. Contact claytonwaynemoore@gmail.com.
