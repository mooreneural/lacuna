"""Tests for the optional sequence-augmented ranker.

ESM-2 is an optional extra and downloads a checkpoint on first use, so nothing
here loads a model. These cover the parts that must hold on a plain install:
the pure aggregation function, the weight/feature alignment that a silent
``zip()`` truncation would otherwise hide, and the failure modes when the extra
is absent or the caller forgets to supply residue probabilities.
"""

from __future__ import annotations

import pytest

from lacuna.pockets import plm
from lacuna.pockets.clusterer import (
    _PLM_RANKER_FEATURES,
    _PLM_RANKER_WEIGHTS,
    _RANKER_FEATURES,
    RANK_STRATEGIES,
    cluster_pockets,
    learned_plm_score,
)


class TestPocketFeatures:
    def test_empty_lining_scores_zero(self):
        """A pocket whose residues have no probabilities contributes nothing
        rather than raising: absent signal is not an error."""
        got = plm.pocket_features(set(), {})
        assert got == dict.fromkeys(plm.FEATURES, 0.0)

    def test_residues_without_probabilities_are_skipped(self):
        got = plm.pocket_features({1, 2, 999}, {1: 0.8, 2: 0.4})
        assert got["plm_mean"] == pytest.approx(0.6)
        assert got["plm_max"] == pytest.approx(0.8)

    def test_statistics(self):
        probs = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.1}
        got = plm.pocket_features({1, 2, 3, 4}, probs)
        assert got["plm_mean"] == pytest.approx(0.625)
        assert got["plm_max"] == pytest.approx(0.9)
        assert got["plm_top3"] == pytest.approx(0.8)      # (0.9+0.8+0.7)/3
        assert got["plm_frac"] == pytest.approx(0.75)     # three of four >= 0.5

    def test_top3_handles_small_pockets(self):
        got = plm.pocket_features({1}, {1: 0.6})
        assert got["plm_top3"] == pytest.approx(0.6)

    def test_all_features_present(self):
        got = plm.pocket_features({1}, {1: 0.5})
        assert set(got) == set(plm.FEATURES)


class TestWeightAlignment:
    def test_weights_align_with_features(self):
        """zip() would silently truncate to the shorter sequence, so a feature
        added without a weight would be dropped from scoring unnoticed."""
        assert len(_PLM_RANKER_FEATURES) == len(_PLM_RANKER_WEIGHTS)

    def test_plm_features_end_with_the_sequence_block(self):
        """The sequence features are appended, so they are the tail of the list.

        This deliberately does NOT assert that the leading entries match
        ``_RANKER_FEATURES``. They did once, and that coupling is what let a
        rename of the geometry set silently repoint the PLM weights at different
        quantities while every test still passed. Each list must name the
        features its own weights were fitted on; see tests/test_ranker_integrity.py.
        """
        assert tuple(_PLM_RANKER_FEATURES[-len(plm.FEATURES):]) == plm.FEATURES

    def test_strategy_is_registered(self):
        assert "learned-plm" in RANK_STRATEGIES


class TestAvailability:
    def test_available_returns_bool(self):
        assert isinstance(plm.available(), bool)

    def test_head_ships_with_the_package(self):
        """The fitted head is a package artifact; without it the extra installs
        but cannot rank."""
        assert plm._HEAD_PATH.exists(), f"missing head at {plm._HEAD_PATH}"

    def test_requesting_plm_without_probabilities_is_a_clear_error(self):
        with pytest.raises(ValueError, match="plm_residue_probs"):
            cluster_pockets([[]], n_conformers=1, rank_by="learned-plm")


class TestScoring:
    def test_missing_features_score_as_zero(self, monkeypatch):
        """A cluster with no sequence features still scores, using only its
        geometric terms, instead of raising a KeyError mid-ranking."""
        class FakeCluster:
            pass

        feats = {name: 0.0 for name in _RANKER_FEATURES}
        monkeypatch.setattr("lacuna.pockets.clusterer.ranker_features",
                            lambda c: feats)
        assert learned_plm_score(FakeCluster(), {}) == pytest.approx(0.0)

    def test_higher_sequence_signal_scores_higher(self, monkeypatch):
        class FakeCluster:
            pass

        feats = {name: 0.0 for name in _RANKER_FEATURES}
        monkeypatch.setattr("lacuna.pockets.clusterer.ranker_features",
                            lambda c: feats)
        low = learned_plm_score(FakeCluster(), plm.pocket_features({1}, {1: 0.05}))
        high = learned_plm_score(FakeCluster(), plm.pocket_features({1}, {1: 0.95}))
        assert high > low
