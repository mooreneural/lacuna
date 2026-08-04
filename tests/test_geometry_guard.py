"""The learned ranker must notice when detection geometry no longer matches its fit.

The weights read pocket size, burial and depth, so changing how pockets are carved
changes what those numbers mean. When CLUSTER_RADIUS_A went 4.0 to 2.0 the previous
weights fell to the random null on the new pockets, and nothing said so. These
tests cover the guard that makes that visible.
"""

from __future__ import annotations

import warnings

import pytest

from lacuna.pockets import clusterer, detector


@pytest.fixture(autouse=True)
def reset_warning_state():
    """The guard warns once per process; each test needs a clean slate."""
    clusterer._geometry_warned = False
    yield
    clusterer._geometry_warned = False


def test_shipped_constants_match_the_fitted_geometry():
    """Guards against changing a detection constant without refitting."""
    for name, expected in clusterer._FITTED_GEOMETRY.items():
        assert getattr(detector, name) == expected, (
            f"detector.{name} is {getattr(detector, name)} but the shipped ranker "
            f"was fitted at {expected}; refit via benchmarks/train_ranker.py --fit "
            f"and update _FITTED_GEOMETRY"
        )


def test_no_warning_when_geometry_is_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        clusterer._check_fitted_geometry()


def test_warns_when_a_detection_constant_drifts(monkeypatch):
    monkeypatch.setattr(detector, "CLUSTER_RADIUS_A", 4.0, raising=True)
    with pytest.warns(RuntimeWarning, match="different detection geometry"):
        clusterer._check_fitted_geometry()


def test_warning_names_the_offending_constant_and_both_values(monkeypatch):
    monkeypatch.setattr(detector, "MIN_VOLUME_A3", 30.0, raising=True)
    with pytest.warns(RuntimeWarning) as record:
        clusterer._check_fitted_geometry()
    msg = str(record[0].message)
    assert "MIN_VOLUME_A3=30.0" in msg
    assert "fitted at 80.0" in msg


def test_warns_only_once_per_process(monkeypatch):
    """A per-pocket warning would flood a run over hundreds of clusters."""
    monkeypatch.setattr(detector, "CLUSTER_RADIUS_A", 3.0, raising=True)
    with pytest.warns(RuntimeWarning):
        clusterer._check_fitted_geometry()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clusterer._check_fitted_geometry()  # second call stays silent
