"""Backend auto-selection must reflect what is actually installed.

The regression this guards: backend modules import cleanly without their heavy
dependency, because the dependency is imported inside generate(). Selecting on
`except ImportError` around the module import therefore always chose boltz, and
a base install crashed at generation time instead of falling back to nma.
"""
from __future__ import annotations

import pytest

from lacuna import cli


@pytest.fixture
def installed(monkeypatch):
    """Pretend exactly the named third-party modules are importable."""
    def apply(*present: str):
        monkeypatch.setattr(cli, "_installed", lambda m: m in present)
    return apply


def test_falls_back_to_nma_when_nothing_optional_is_installed(installed):
    installed()
    assert cli._auto_backend().name == "nma"


def test_prefers_boltz_when_boltz_is_installed(installed):
    installed("boltz")
    assert cli._auto_backend().name == "boltz"


def test_prefers_openmm_over_nma_when_boltz_is_absent(installed):
    installed("openmm", "pdbfixer")
    assert cli._auto_backend().name == "openmm"


def test_openmm_needs_pdbfixer_too(installed):
    installed("openmm")
    assert cli._auto_backend().name == "nma"


def test_installed_is_false_for_a_missing_module():
    assert cli._installed("a_module_that_does_not_exist_9f3a") is False


def test_installed_is_true_for_a_stdlib_module():
    assert cli._installed("json") is True
