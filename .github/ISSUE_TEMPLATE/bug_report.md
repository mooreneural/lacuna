---
name: Bug report
about: Something crashed or produced an incorrect result
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
Command run (include flags/backend, e.g. `lacuna run 1abc.pdb --backend nma`)
and, if possible, the input structure (a PDB ID is enough if it's from the
PDB).

**Expected behavior**
What you expected to happen instead.

**Actual behavior**
What actually happened. Include the full error traceback if there is one.

**Environment**
- Lacuna version: (`python -c "import lacuna; print(lacuna.__version__)"`)
- Python version:
- OS:
- Backend used: NMA / OpenMM / Boltz-2 / random

**Additional context**
Anything else relevant (e.g. this only happens on multi-chain structures,
only at high conformer counts, etc.).
