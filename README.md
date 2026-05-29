# Lacuna

Cryptic binding pocket discovery via conformational ensemble analysis.

Lacuna takes any protein structure (from AlphaFold, Boltz, Chai, or the PDB), generates a conformational ensemble, and systematically discovers transient binding pockets that are invisible in static structures.

## Quick start

```bash
pip install lacuna
lacuna discover protein.pdb
```

## Backends

| Backend | Quality | Requirement |
|---|---|---|
| `random` | Lightweight (default) | None |
| `openmm` | Physics-based MD | `pip install lacuna[openmm]` |
| `boltz` | AI partial diffusion | `pip install lacuna[boltz]` + GPU |

## Output

- `pocket_report.json` — ranked pocket list with druggability scores
- `pocket_N_constraint.yaml` — Boltz docking constraint files
- `pocket_N_vina.conf` — AutoDock Vina box definitions
- `pocket_N_site.pdb` — pocket pseudoatoms for PyMOL/ChimeraX
