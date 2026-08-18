# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""The package version must match pyproject.toml.

They disagreed for the whole of 1.0.0: pyproject said 1.0.0 and
lacuna/__init__.py said 0.3.1, so PyPI served a distribution labelled 1.0.0 whose
own __version__ reported 0.3.1. Nothing caught it because nothing compared them.
"""
from __future__ import annotations

import re
from pathlib import Path

import lacuna

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared_version() -> str:
    for line in PYPROJECT.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    raise AssertionError("no version line in pyproject.toml")


def test_package_version_matches_pyproject():
    assert lacuna.__version__ == _declared_version(), (
        "lacuna.__version__ is %r but pyproject.toml declares %r"
        % (lacuna.__version__, _declared_version()))


def test_version_is_pep440_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _declared_version()), (
        "expected a plain X.Y.Z release version, got %r" % _declared_version())
