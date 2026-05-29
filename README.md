# Lacuna

**Cryptic binding pocket discovery via conformational ensemble analysis.**

Most protein structure predictors (AlphaFold, Boltz, Chai) give you one static structure. But ~70% of disease-relevant proteins are considered "undruggable" not because they're biologically intractable — it's because no pocket is visible in their ground state. K-Ras was "undruggable" for 30 years until a transient cryptic pocket was found in its switch-II region. That pocket now backs sotorasib and adagrasib.

Lacuna finds those pockets. It generates a conformational ensemble from any input structure, detects pockets per conformer, and clusters them across the ensemble to surface sites that only appear transiently — ranked by druggability and persistence.

```
lacuna discover kras.pdb --conformers 20 --emit-boltz-constraints --emit-vina-boxes
```

---

## Install

```bash
pip install lacuna
```

**Optional backends** (better conformational sampling):
```bash
pip install "lacuna[openmm]"   # 100ps implicit-solvent MD
pip install "lacuna[boltz]"    # Boltz-2 partial diffusion (best quality, GPU recommended)
pip install "lacuna[all]"      # everything
```

Requires Python 3.10+.

---

## Quick start

### CLI

```bash
# Discover pockets with defaults (random backbone perturbation backend)
lacuna discover protein.pdb --conformers 20

# Filter and limit output
lacuna discover protein.pdb --min-druggability 0.5 --min-persistence 0.3 --top 5

# Use a physics-based backend for cryptic pockets
lacuna discover protein.pdb --backend boltz --conformers 30

# Emit all docking file formats
lacuna discover protein.pdb --emit-boltz-constraints --emit-vina-boxes --emit-pocket-pdbs

# Generate docking files from a previous report
lacuna dock-prep kras_lacuna/pocket_report.json kras.pdb --format all
```

### Python API

```python
from lacuna import load_structure, detect_pockets, cluster_pockets
from lacuna.ensemble.random_backend import RandomBackend
from lacuna.io.structure import coords_array
from lacuna.io.writers import write_report, write_boltz_constraint

structure = load_structure("protein.pdb")
backend = RandomBackend(seed=42)
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

1. **Ensemble generation** — Generate N conformers via backbone perturbation (built-in), OpenMM implicit-solvent MD, or Boltz-2 partial diffusion at varying noise levels
2. **Pocket detection** — Grid-based alpha-point analysis per conformer: compute distance transform, find local maxima within the 1.4–5.5 Å interaction zone, cluster nearby alpha-points into pocket candidates
3. **Cross-ensemble clustering** — Greedy centroid merging clusters corresponding pockets across all conformers
4. **Druggability scoring** — Gaussian volume reward centered at 300 Å³ + enclosure + hydrophobicity + aromaticity (Halgren 2009)
5. **Cryptic flagging** — Pockets present in <90% of conformers are marked `cryptic: true`

---

## Outputs

| File | Description |
|------|-------------|
| `pocket_report.json` | Ranked pocket metadata: centroid, volume, druggability, persistence, lining residues |
| `pocket_N_site.pdb` | Pseudoatom PDB for PyMOL/ChimeraX visualization |
| `pocket_N_constraint.yaml` | Boltz YAML — add a SMILES and run `boltz predict` to dock into this site |
| `pocket_N_vina.conf` | AutoDock Vina / Gnina / QuickVina box config |

---

## Backends

| Backend | Install | Quality | Speed | Notes |
|---------|---------|---------|-------|-------|
| `random` | built-in | baseline | ~0.04s/conf | Correlated Gaussian backbone perturbation |
| `openmm` | `lacuna[openmm]` | good | ~2s/conf | 100ps Langevin MD, GBn2 implicit solvent |
| `boltz` | `lacuna[boltz]` | best | ~30s/conf (GPU) | Boltz-2 partial diffusion at varying noise fractions |

**For truly cryptic pockets** use `boltz` or `openmm`. The `random` backend perturbs coordinates without a force field — it reliably finds surface pockets and shallow cryptic sites, but cannot sample large-scale loop rearrangements.

---

## Benchmarks

Tested against ground-truth binding sites on apo PDB structures using RandomBackend, 20 conformers, ≥30% residue overlap in top-5 pockets.

| Target | Pocket type | Result | Overlap | Rank | Time |
|--------|-------------|--------|---------|------|------|
| 1HEL hen lysozyme | Orthosteric (always open) | ✅ PASS | 75% | 4 | 0.8s |
| 1L90 T4L L99A | Cryptic hydrophobic cavity | ✅ PASS | 100% | 1 | 1.2s |
| 4OBE K-Ras WT apo | Cryptic switch-II pocket | ✅ PASS | 79% | 4 | 3.9s |
| 1HPV HIV protease apo | Active site (flap region) | ✅ PASS | 100% | 1 | 1.5s |

**4/4** known binding sites recovered (RandomBackend only).

Performance sweep on 1HEL (129 residues):

| Conformers | Total time | Per-conformer |
|-----------|-----------|---------------|
| 1 | 0.10s | 0.049s |
| 5 | 0.23s | 0.039s |
| 20 | 0.84s | 0.040s |
| 50 | 1.95s | 0.038s |

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

# pocket_0_constraint.yaml is ready — add your SMILES:
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

If you use Lacuna in published research, the methodology builds on:

- Halgren (2009) *J. Chem. Inf. Model.* 49(2):377–389 — SiteMap druggability scoring
- Le Guilloux et al. (2009) *BMC Bioinformatics* 10:168 — fpocket alpha-sphere approach  
- Schmidtke & Barril (2010) *J. Med. Chem.* 53(15):5858–5867 — enclosure scoring

---

## License

MIT
