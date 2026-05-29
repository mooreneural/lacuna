"""PDB/mmCIF structure reader using Biopython."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBParser
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


def load_structure(path: str | Path) -> Structure:
    """Parse a PDB or mmCIF file and return a Structure object."""
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
        for chain in model.get_chains():
            for res in chain.get_residues():
                if res.get_id()[0] != " ":
                    continue  # skip HETATM / water
                res_name = res.get_resname().strip()
                chain_id = chain.get_id()
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
        for chain in model.get_chains():
            seq = ""
            for pp in ppb.build_peptides(chain):
                seq += str(pp.get_sequence())
            if seq:
                sequence[chain.get_id()] = seq
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
