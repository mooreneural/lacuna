# Usage

Command line, Python API, backends, and output formats.

## CLI

```bash
# Discover pockets with defaults (NMA backend, physically grounded, no GPU needed)
lacuna discover protein.pdb --conformers 20

# Filter and limit output
lacuna discover protein.pdb --min-druggability 0.5 --min-persistence 0.3 --top 5

# Analyze a homodimer, detecting pockets at the dimer interface (e.g. Caspase-1, IDH1).
# Reads BIOMT records from PDB; for best results use the biological assembly from RCSB.
lacuna discover protein.pdb --homodimer --conformers 20

# Optional Boltz-2 backend (experimental, see the note under Backends)
lacuna discover protein.pdb --backend boltz --conformers 30

# Emit all docking file formats
lacuna discover protein.pdb --emit-boltz-constraints --emit-vina-boxes --emit-pocket-pdbs

# Generate docking files from a previous report
lacuna dock-prep kras_lacuna/pocket_report.json kras.pdb --format all
```

`--rank-by` selects the ranking strategy; see [RANKING.md](RANKING.md).

## Python API

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

## Backends

| Backend | Install | Quality | Speed | Notes |
|---------|---------|---------|-------|-------|
| `nma` | built-in | good | ~0.1s/conf | Elastic-network normal modes (default) |
| `openmm` | `lacuna[openmm]` | good | ~2s/conf | Implicit-solvent MD, 100ps |
| `boltz` | `lacuna[boltz]` | experimental | ~100s/protein (GPU) | Diffusion sampling, noisy (see note) |
| `random` | built-in | baseline | ~0.04s/conf | Gaussian backbone perturbation |

**Auto-selection order:** `boltz` → `openmm` → `nma` → `random`. On a plain
`pip install lacuna-pockets`, the NMA backend runs automatically.

The `nma` backend samples physically meaningful collective motions, the same
hinge-bending and breathing modes that open cryptic pockets in nature, without
requiring a GPU or a force field. It is the zero-dependency default.

NMA runtime is sub-second to about 7s per protein on a laptop CPU.
**[Per-size timing →](BENCHMARKS.md#speed-nma-backend-no-gpu)**

> **Boltz backend status, honestly.** The `boltz` backend runs Boltz-2 diffusion
> sampling on a GPU, but it currently predicts each conformer *de novo from
> sequence* rather than by partial diffusion from the input structure, which
> yields structurally divergent, noisy ensembles (150-300+ pocket clusters
> against NMA's ~35). In GPU benchmarking it did **not** reliably improve cryptic
> detection over NMA. A proper apo-templated integration with sequence-based
> residue mapping is planned; until then NMA is the recommended backend.

## Outputs

| File | Description |
|------|-------------|
| `pocket_report.json` | Ranked pocket metadata: centroid, volume plus apo-to-open range, druggability, crypticity, persistence, lining residues |
| `pocket_N_site.pdb` | Pseudoatom PDB for PyMOL or ChimeraX visualisation |
| `pocket_N_constraint.yaml` | Boltz YAML; add a SMILES and run `boltz predict` to dock into this site |
| `pocket_N_vina.conf` | AutoDock Vina / Gnina / QuickVina box config |

The JSON report also includes per-pocket volume dynamics (`apo_volume_A3`,
`volume_range_A3`) and `max_druggability`.

### Per-pocket chain composition

`chain_shares` gives the fraction of a pocket's lining contributed by each chain,
and `interface` is true when two or more chains each contribute at least 10% of
the lining and at least 3 residues.

Read the flag in the direction it was measured. On the CryptoBench test fold
63.6% of proposed pockets are interfacial, rising to 73.6% of the top five, while
only 14.8% of annotated sites are. It is a reason to look harder at a candidate
before trusting it, not a reason to prefer one.

### `residue_propensity`

A per-residue cryptic-pocket score over the whole structure, aggregated from the
ranked pockets:

```python
import json
report = json.load(open("protein_lacuna/pocket_report.json"))
scores = report["residue_propensity"]      # {"LYS155:A": 1.85, ...}
scores.get("ALA42:A", 0.0)                 # residues lining no pocket are absent
```

Each residue's score is the inverse-rank-weighted sum, over pockets, of the
fraction of that pocket's conformers in which the residue lines it. A residue
lining the top-ranked pocket in every conformer scores 1.0 from that pocket
alone; residues absent from the map line no proposed pocket and should be read
as 0.0.

This is the quantity most cryptic-site predictors are evaluated on, as
per-residue ROC-AUC and AUPRC, so it is what to use when comparing Lacuna against
CryptoBench, PocketMiner, UniPocket or similar. On the CryptoBench held-out test
fold (n=231) it reaches **AUROC 0.774 and AUPRC 0.345**, against 0.249 AUPRC for
the same scorer run on the input structure alone and 0.115 for a burial
neighbour count. See `docs/BENCHMARKS.md`.

Note the units differ from `crypticity`: this is an unbounded sum over pockets,
not a score in [0, 1], so compare residues within a structure rather than across
structures.

## Input formats

Accepts PDB or mmCIF from any predictor or database:

- AlphaFold 2 / AlphaFold 3
- Boltz-1 / Boltz-2
- Chai-1
- RCSB PDB
- ESMFold, RoseTTAFold, OpenFold, and others

## Worked example: K-Ras switch-II

```bash
# Download K-Ras apo from RCSB; the NMA default recovers switch-II at rank 3
lacuna discover 4OBE.pdb \
    --conformers 20 \
    --emit-boltz-constraints \
    --output kras_pockets/

# pocket_0_constraint.yaml is ready; add your SMILES:
#   - ligand:
#       id: L
#       smiles: YOUR_SMILES_HERE
boltz predict kras_pockets/pocket_0_constraint.yaml
```

See [`examples/kras_cryptic.py`](../examples/kras_cryptic.py) for a fully
annotated Python workflow.
