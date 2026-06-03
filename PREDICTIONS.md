# Lacuna Prospective Predictions

Pockets predicted computationally from apo structures, with no prior knowledge
of small-molecule binding at these sites. Each entry is a testable hypothesis:
if the pocket is real, a fragment screen or thermal shift assay targeting the
listed residues should show a hit rate above background.

---

## Prediction 1: Beta-catenin ARM7 cryptic sub-pocket

**Date:** 2026-06-02  
**Target:** Beta-catenin / CTNNB1 (UniProt P35222)  
**Structure used:** PDB 2Z6H (apo armadillo repeat domain, residues 149–691)  
**Backend:** Boltz-2 partial diffusion, 30 conformers  
**Status:** Prospective — no small molecule known to bind this surface

### Why this matters

Beta-catenin is the central effector of the WNT signalling pathway and is
constitutively active in ~70% of colorectal cancers, ~30% of hepatocellular
carcinomas, and many other solid tumours. Despite 40 years of effort, no
approved small molecule directly inhibits beta-catenin. The primary
therapeutic interface — the TCF/LEF binding groove — spans ~3000 Å² and
has been considered too large and flat for drug-like molecules.

### What Lacuna found

Running Boltz-2 on the apo structure surfaces a cryptic sub-pocket that is:

| Property | Value |
|----------|-------|
| Rank | 12 of 109 clusters |
| Druggability score | **0.855** (high; optimal pocket volume ~300 Å³) |
| Persistence | 16% (opens in ~5 of 31 conformers — genuinely cryptic) |
| Volume | 241 Å³ (drug-like range: 200–500 Å³) |
| Centroid | (−3.2, −25.5, 10.7) Å |

**Lining residues (ARM7–8 interface):**

| Residue | ARM repeat | Chemical character |
|---------|------------|--------------------|
| ARG515 | ARM7 | Positive (also contacts TCF/LEF at surface) |
| ARG542 | ARM7–8 loop | Positive |
| ALA543 | ARM8 | Hydrophobic |
| ARG549 | ARM8 | Positive |
| ARG550 | ARM8 | Positive |
| ARG565 | ARM8 | Positive |
| CYS573 | ARM8 | Nucleophilic — covalent warhead opportunity |
| ALA576 | ARM8 | Hydrophobic |

**Key observation:** CYS573 is partially buried in the cryptic state and
becomes solvent-exposed in ~16% of Boltz-2 conformers. This makes it a
candidate for covalent fragment screening (electrophilic warheads:
acrylamide, chloroacetamide).

### Structural context

The pocket sits on the concave face of ARM repeats 7–8, adjacent to but
distinct from the canonical TCF/LEF groove (which centres on ARM1–4 around
K312, R469, K508). The nearest pharmacological interface is the APC/axin
binding region (ARM6, ~residues 382–395); the predicted pocket shares no
lining residues with it.

This region corresponds to the "third face" of the ARM domain described in
structural biology literature as a potential protein–protein interaction
surface. No crystal structure shows a small molecule bound here.

### Validation strategy

**Tier 1 — biophysical (2–4 weeks, ~$10K):**
1. Express and purify CTNNB1 ARM domain (residues 149–691, well-established protocol)
2. Run a commercial DNA-encoded chemical library (DEL) screen against the ARM domain
3. Thermal shift assay (TSA/DSF) with 500–1000 fragment library, look for
   ΔTm ≥ 1°C hits that do NOT compete with a TCF peptide (ruling out
   canonical groove binders)
4. SPR or ITC to confirm direct binding of confirmed hits

**Tier 2 — structural (4–8 weeks if hits found):**
- Co-crystallise or cryo-EM of hit compounds with CTNNB1 ARM domain
- Confirm binding to ARM7–8 residues predicted by Lacuna

**Tier 3 — cellular (parallel with Tier 2):**
- STF luciferase reporter assay (WNT pathway readout) in HCT116 or SW480
  cells (both have CTNNB1-activating mutations)
- Look for inhibition of TOPFlash signal at non-cytotoxic concentrations

### Why CYS573 is actionable

Covalent fragment libraries targeting cysteines are now commercially available
(Enamine, Chembridge) and have yielded clinical candidates in KRAS (G12C),
BTK, and other targets. CYS573 is a non-catalytic cysteine with no known
function — an ideal covalent handle if the pocket is accessible.

### Docking files

Lacuna output files for this prediction:

```
ctnnb1_boltz_lacuna/
  pocket_report.json          — full ranked pocket list
  pocket_11_site.pdb          — pseudoatom PDB for PyMOL/ChimeraX (rank 12 = index 11)
  pocket_11_constraint.yaml   — Boltz YAML: add a SMILES and run boltz predict
  pocket_11_vina.conf         — AutoDock Vina / Gnina box centred on prediction
```

To dock a fragment into this pocket:
```yaml
# pocket_11_constraint.yaml — add your fragment SMILES:
version: 1
sequences:
  - protein:
      id: A
      sequence: "..."   # from output file
  - ligand:
      id: L
      smiles: "YOUR_FRAGMENT_SMILES"
constraints:
  pocket:
    - [A, 515]
    - [A, 542]
    - [A, 543]
    - [A, 549]
    - [A, 573]
```

---

## Adding predictions

To add a new prospective prediction, run:

```bash
lacuna discover YOUR_TARGET.pdb \
    --backend boltz \
    --conformers 30 \
    --emit-boltz-constraints \
    --emit-vina-boxes \
    --output predictions/YOUR_TARGET/
```

Then document the novel pockets (not overlapping known interfaces) in this
file with the residue list, druggability score, persistence, and proposed
validation assay.
