"""Druggability scoring for detected pockets.

Based on the empirical model from:
  Halgren (2009) J. Chem. Inf. Model. 49(2):377-389
  Schmidtke & Barril (2010) J. Med. Chem. 53(15):5858-5867

Score components:
  - Volume: Gaussian reward centered at 300 Å³ (too small = can't bind drug,
             too large = not selective)
  - Enclosure: linear reward for buried pockets (open = solvent competition)
  - Hydrophobicity: linear reward (hydrophobic pockets drive binding via entropy)
  - Aromaticity: bonus for π-stacking contacts (common in drug-protein binding)

All components are [0,1] and combined as a weighted average.
"""

from __future__ import annotations

import math

from lacuna.models import DrugabilityScore, Pocket

# Volume scoring parameters
_OPTIMAL_VOLUME = 300.0   # Å³ center of Gaussian reward
_VOLUME_SIGMA = 200.0     # Å³ width

# Component weights
_W_VOLUME = 0.35
_W_ENCLOSURE = 0.30
_W_HYDROPHOBIC = 0.25
_W_AROMATIC = 0.10

# Aromatic saturation (above this count, diminishing returns)
_MAX_AROMATIC = 5


def score_pocket(pocket: Pocket) -> DrugabilityScore:
    """Compute druggability score for a single pocket."""
    vol_score = math.exp(
        -((pocket.volume_a3 - _OPTIMAL_VOLUME) ** 2) / (2 * _VOLUME_SIGMA ** 2)
    )
    enc_score = float(np.clip(pocket.enclosure, 0.0, 1.0))
    hyd_score = float(pocket.hydrophobic_fraction)
    arom_score = min(pocket.aromatic_count / _MAX_AROMATIC, 1.0)

    composite = (
        _W_VOLUME * vol_score
        + _W_ENCLOSURE * enc_score
        + _W_HYDROPHOBIC * hyd_score
        + _W_AROMATIC * arom_score
    )

    return DrugabilityScore(
        volume_score=round(vol_score, 4),
        enclosure_score=round(enc_score, 4),
        hydrophobic_score=round(hyd_score, 4),
        aromatic_score=round(arom_score, 4),
        composite=round(composite, 4),
    )


# avoid circular import — numpy used only internally
import numpy as np  # noqa: E402
