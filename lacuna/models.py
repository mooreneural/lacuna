"""Core data models for Lacuna pocket discovery."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Atom:
    serial: int
    name: str
    res_name: str
    chain_id: str
    res_seq: int
    coords: tuple[float, float, float]
    element: str


@dataclass
class Residue:
    chain_id: str
    seq_num: int
    name: str
    atom_indices: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.name}{self.seq_num}:{self.chain_id}"


@dataclass
class Structure:
    """Parsed protein structure with atom and residue lists."""
    path: str
    atoms: list[Atom]
    residues: list[Residue]
    sequence: dict[str, str] = field(default_factory=dict)  # chain_id -> one-letter sequence


@dataclass
class Pocket:
    """A single binding pocket detected in one conformer."""
    centroid: tuple[float, float, float]
    volume_a3: float
    enclosure: float           # 0–1, fraction of grid points that are buried
    hydrophobic_fraction: float
    aromatic_count: int
    lining_residues: list[str]  # ["G12:A", "V29:A", ...]
    conformer_idx: int


@dataclass
class DrugabilityScore:
    volume_score: float
    enclosure_score: float
    hydrophobic_score: float
    aromatic_score: float
    composite: float  # 0–1


@dataclass
class PocketCluster:
    """Pocket cluster aggregated across the conformational ensemble."""
    rank: int
    centroid: tuple[float, float, float]
    volume_a3: float
    druggability: float
    persistence: float          # fraction of conformers where pocket is open
    cryptic: bool               # True if persistence < 0.9
    lining_residues: list[str]
    appears_in_conformers: list[int]
    member_pockets: list[Pocket] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "centroid": list(self.centroid),
            "volume_A3": round(self.volume_a3, 1),
            "druggability": round(self.druggability, 3),
            "persistence": round(self.persistence, 3),
            "cryptic": self.cryptic,
            "lining_residues": self.lining_residues,
            "appears_in_conformers": self.appears_in_conformers,
        }
