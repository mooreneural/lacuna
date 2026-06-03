"""PDB/mmCIF structure reader using Biopython."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import PPBuilder

from lacuna.models import Atom, Residue, Structure

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "TYR", "CYS"}
_AROMATIC = {"PHE", "TRP", "TYR", "HIS"}

# Standard van der Waals radii in Å
VDW_RADII: dict[str, float] = {
    "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "H": 1.2,
}


def load_structure(path: str | Path, chain: str | None = None) -> Structure:
    """Parse a PDB or mmCIF file and return a Structure object.

    Args:
        path: Path to PDB or mmCIF file.
        chain: If given, load only this chain ID (e.g. "A"). Default loads all chains.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        if suffix in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        bio_struct = parser.get_structure(path.stem, str(path))

    atoms: list[Atom] = []
    residues: list[Residue] = []
    residue_map: dict[tuple[str, int], int] = {}  # (chain_id, res_seq) -> residue index

    serial = 0
    for model in bio_struct.get_models():
        for chain_obj in model.get_chains():
            if chain is not None and chain_obj.get_id() != chain:
                continue
            for res in chain_obj.get_residues():
                if res.get_id()[0] != " ":
                    continue  # skip HETATM / water
                res_name = res.get_resname().strip()
                chain_id = chain_obj.get_id()
                res_seq = res.get_id()[1]

                res_idx = len(residues)
                residues.append(Residue(
                    chain_id=chain_id,
                    seq_num=res_seq,
                    name=res_name,
                    atom_indices=[],
                ))
                residue_map[(chain_id, res_seq)] = res_idx

                for atom in res.get_atoms():
                    element = (atom.element or atom.get_name()[0]).upper().strip()
                    x, y, z = atom.get_coord()
                    atoms.append(Atom(
                        serial=serial,
                        name=atom.get_name().strip(),
                        res_name=res_name,
                        chain_id=chain_id,
                        res_seq=res_seq,
                        coords=(float(x), float(y), float(z)),
                        element=element,
                    ))
                    residues[-1].atom_indices.append(serial)
                    serial += 1
        break  # first model only

    # Build one-letter sequences per chain
    ppb = PPBuilder()
    sequence: dict[str, str] = {}
    for model in bio_struct.get_models():
        for chain_obj in model.get_chains():
            if chain is not None and chain_obj.get_id() != chain:
                continue
            seq = ""
            for pp in ppb.build_peptides(chain_obj):
                seq += str(pp.get_sequence())
            if seq:
                sequence[chain_obj.get_id()] = seq
        break

    return Structure(path=str(path), atoms=atoms, residues=residues, sequence=sequence)


def coords_array(structure: Structure) -> np.ndarray:
    """Return (N_atoms, 3) float32 coordinate array."""
    return np.array([a.coords for a in structure.atoms], dtype=np.float32)


def is_hydrophobic(res_name: str) -> bool:
    return res_name.upper() in _HYDROPHOBIC


def is_aromatic(res_name: str) -> bool:
    return res_name.upper() in _AROMATIC


def get_vdw_radius(element: str) -> float:
    return VDW_RADII.get(element.upper(), 1.7)


# ---------------------------------------------------------------------------
# Biological assembly helpers
# ---------------------------------------------------------------------------

def make_biological_assembly(path: Path | str, structure: Structure) -> Structure:
    """Apply BIOMT/struct_oper symmetry operations to create the biological assembly.

    Reads REMARK 350 BIOMT records (PDB) or _pdbx_struct_oper_list (mmCIF) and
    applies each non-identity operation to generate the full biological assembly.
    Returns the input structure unchanged if the file has no symmetry records or
    if the structure already has multiple chains (assumed to already be an assembly).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".cif", ".mmcif"):
        transforms = _parse_biomt_cif(path)
    else:
        transforms = _parse_biomt_pdb(path)

    non_identity = [(R, t) for R, t in transforms if not _is_identity(R, t)]
    if not non_identity:
        return structure

    # Map atom serial → residue index for rebuilding atom_indices in mates
    atom_to_res: dict[int, int] = {}
    for ri, res in enumerate(structure.residues):
        for serial in res.atom_indices:
            atom_to_res[serial] = ri

    existing_chains = sorted(set(a.chain_id for a in structure.atoms))
    chain_ids_per_mate = _allocate_chain_ids(existing_chains, len(non_identity))

    extra_atoms: list[Atom] = []
    extra_residues: list[Residue] = []
    extra_sequence: dict[str, str] = {}
    serial = len(structure.atoms)

    orig_coords = np.array([a.coords for a in structure.atoms])  # (N, 3)

    for mate_idx, (R, t) in enumerate(non_identity):
        chain_map = dict(zip(existing_chains, chain_ids_per_mate[mate_idx]))

        # Pre-create residue shells for this mate
        mate_residues: list[Residue] = [
            Residue(
                chain_id=chain_map.get(res.chain_id, res.chain_id),
                seq_num=res.seq_num,
                name=res.name,
                atom_indices=[],
            )
            for res in structure.residues
        ]

        new_coords = (R @ orig_coords.T).T + t  # (N, 3)

        for ai, (atom, pos) in enumerate(zip(structure.atoms, new_coords)):
            new_chain = chain_map.get(atom.chain_id, atom.chain_id)
            new_atom = Atom(
                serial=serial,
                name=atom.name,
                res_name=atom.res_name,
                chain_id=new_chain,
                res_seq=atom.res_seq,
                coords=(float(pos[0]), float(pos[1]), float(pos[2])),
                element=atom.element,
            )
            extra_atoms.append(new_atom)

            orig_ri = atom_to_res.get(atom.serial)
            if orig_ri is not None:
                mate_residues[orig_ri].atom_indices.append(serial)

            serial += 1

        extra_residues.extend(mate_residues)

        for old_chain, new_chain in chain_map.items():
            if old_chain in structure.sequence:
                extra_sequence[new_chain] = structure.sequence[old_chain]

    return Structure(
        path=structure.path,
        atoms=structure.atoms + extra_atoms,
        residues=structure.residues + extra_residues,
        sequence={**structure.sequence, **extra_sequence},
    )


def _parse_biomt_pdb(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (R, t) matrices from REMARK 350 BIOMT records in a PDB file."""
    rows: dict[int, dict[int, list[float]]] = {}

    with open(path) as f:
        for line in f:
            rec = line[:6].rstrip()
            if rec in ("ATOM", "HETATM"):
                break
            if not line.startswith("REMARK 350") or "BIOMT" not in line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                row_num = int(parts[2][-1]) - 1   # 0/1/2 from "BIOMT1/2/3"
                biomt_id = int(parts[3])
                vals = [float(x) for x in parts[4:8]]
                rows.setdefault(biomt_id, {})[row_num] = vals
            except (ValueError, IndexError):
                continue

    result = []
    for biomt_id in sorted(rows.keys()):
        mat = rows[biomt_id]
        if len(mat) == 3:
            R = np.array([[mat[i][j] for j in range(3)] for i in range(3)])
            t = np.array([mat[i][3] for i in range(3)])
            result.append((R, t))
    return result


def _parse_biomt_cif(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (R, t) matrices from _pdbx_struct_oper_list in an mmCIF file."""
    try:
        d = MMCIF2Dict(str(path))
    except Exception:
        return []

    key = "_pdbx_struct_oper_list.matrix[1][1]"
    if key not in d:
        return []

    n = len(d[key])
    result = []
    for i in range(n):
        try:
            R = np.array([
                [float(d[f"_pdbx_struct_oper_list.matrix[{r}][{c}]"][i]) for c in range(1, 4)]
                for r in range(1, 4)
            ])
            t = np.array([
                float(d[f"_pdbx_struct_oper_list.vector[{j}]"][i]) for j in range(1, 4)
            ])
            result.append((R, t))
        except (KeyError, ValueError, IndexError):
            continue
    return result


def _is_identity(R: np.ndarray, t: np.ndarray) -> bool:
    return bool(np.allclose(R, np.eye(3), atol=1e-3) and np.allclose(t, 0.0, atol=0.1))


def _allocate_chain_ids(existing: list[str], n_mates: int) -> list[list[str]]:
    """Return n_mates lists of new chain IDs, one new ID per chain in existing."""
    used = set(existing)
    candidates = (
        [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    )
    available = [c for c in candidates if c not in used]

    result: list[list[str]] = []
    for _ in range(n_mates):
        mate_ids: list[str] = []
        for chain in existing:
            if available:
                new_id = available.pop(0)
                used.add(new_id)
                mate_ids.append(new_id)
            else:
                mate_ids.append(chain + "+")  # last-resort fallback
        result.append(mate_ids)
    return result
