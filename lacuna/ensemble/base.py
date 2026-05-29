"""Abstract base class for ensemble generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EnsembleBackend(ABC):
    """Generates a conformational ensemble from a starting structure.

    Each conformer is returned as a (N_atoms, 3) float32 coordinate array
    in the same atom order as the input structure.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        **kwargs,
    ) -> list[np.ndarray]:
        """Return list of coordinate arrays, one per conformer."""
        ...
