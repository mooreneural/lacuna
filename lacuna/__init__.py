# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Lacuna - cryptic binding pocket discovery via conformational ensemble analysis."""

#: Single-sourced from the installed distribution metadata, so this and
#: pyproject.toml cannot disagree. They did through the whole 1.0.0 release:
#: pyproject said 1.0.0, this said 0.3.1, and PyPI therefore served a package
#: that reported the wrong version to anyone who asked it. The fallback covers
#: running from a source checkout that was never installed.
try:                                    # pragma: no cover - trivial branch
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    __version__ = _pkg_version("lacuna-pockets")
except Exception:                       # pragma: no cover
    __version__ = "1.0.2"

from lacuna.models import Pocket, PocketCluster, DrugabilityScore, Structure
from lacuna.io.structure import load_structure
from lacuna.pockets.detector import detect_pockets
from lacuna.pockets.clusterer import cluster_pockets

__all__ = [
    "load_structure",
    "detect_pockets",
    "cluster_pockets",
    "Pocket",
    "PocketCluster",
    "DrugabilityScore",
    "Structure",
]
