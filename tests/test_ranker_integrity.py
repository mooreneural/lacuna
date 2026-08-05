"""Each shipped ranker must score the features its weights were fitted on.

A weight vector is only meaningful paired with the exact feature list used to fit
it. The existing length check does not catch a rename or reorder: swapping the
geometry set to conformer-invariant features kept both lists 23 long, so a derived
PLM list silently repointed 23 weights at different quantities and every test still
passed. These assertions pin the identity of each list, not just its size.
"""

from __future__ import annotations

import numpy as np
import pytest

from lacuna.pockets import clusterer as cl

#: Pinned so a rename has to be made deliberately, in two places, with this
#: test as the reminder that the weights need refitting to match.
EXPECTED_GEOMETRY = (
    "vol", "vol_p90", "vol_p10", "apo_vol", "drug", "max_drug", "cryp",
    "pers", "n_lin", "vol_per_lin", "enc", "hyd", "aro", "mem_per_conf",
    "bur_raw", "depth", "depth_p90", "mouth", "elong", "flat", "dcen",
    "centroid_std", "vol_cv",
)
EXPECTED_PLM_EXTRA = ("plm_mean", "plm_max", "plm_top3", "plm_frac")

#: The sequence ranker's own list, pinned in full. It currently agrees with
#: EXPECTED_GEOMETRY on the geometric block because both were fitted on the
#: conformer-invariant features, and that is fine: what must not happen is one
#: list being *computed* from the other, so that renaming a geometry feature
#: repoints these weights at different quantities without anything failing.
#: Pinning the literal is what forces a rename to be made deliberately here too.
EXPECTED_PLM = EXPECTED_GEOMETRY + EXPECTED_PLM_EXTRA


class TestFeatureIdentity:
    def test_geometry_features_are_exactly_what_the_weights_expect(self):
        assert cl._RANKER_FEATURES == EXPECTED_GEOMETRY

    def test_plm_features_end_with_the_sequence_block(self):
        assert cl._PLM_RANKER_FEATURES[-4:] == EXPECTED_PLM_EXTRA

    def test_plm_features_are_exactly_what_its_weights_expect(self):
        """Pinned in full, so a geometry rename cannot quietly follow through.

        An earlier version of this test asserted the PLM list differed from the
        geometry list, on the theory that equality implied it was derived. That
        conflated a source-code property with a value one: both lists are now
        legitimately fitted on the same invariant geometry block, and the
        inequality assertion blocked a correct refit. Comparing against a literal
        gives the protection that was actually wanted, because a rename changes
        the module without changing this expectation.
        """
        assert cl._PLM_RANKER_FEATURES == EXPECTED_PLM

    @pytest.mark.parametrize("names,weights", [
        ("_RANKER_FEATURES", "_RANKER_WEIGHTS"),
        ("_PLM_RANKER_FEATURES", "_PLM_RANKER_WEIGHTS"),
    ])
    def test_weights_align_with_features(self, names, weights):
        assert len(getattr(cl, names)) == len(getattr(cl, weights))


class TestFeaturesAreProvided:
    def test_ranker_features_supplies_every_scored_name(self, monkeypatch):
        """Both strategies must be scoreable from one ranker_features() call."""
        from lacuna.models import Pocket, PocketCluster
        import dataclasses

        fields = {f.name for f in dataclasses.fields(Pocket)}
        kw = {"centroid": (0.0, 0.0, 0.0), "volume_a3": 120.0,
              "lining_residues": ["ALA1:A"], "conformer_idx": 0}
        for name in fields - set(kw) - {"source", "score"}:
            kw[name] = 0.5
        mem = [Pocket(**kw)]
        c = PocketCluster(
            rank=1, centroid=(0.0, 0.0, 0.0), volume_a3=120.0, volume_min_a3=110.0,
            volume_max_a3=130.0, druggability=0.5, max_druggability=0.6,
            apo_volume_a3=100.0, crypticity=0.2, persistence=0.5, cryptic=True,
            lining_residues=["ALA1:A"], appears_in_conformers=[0], member_pockets=mem,
        )
        available = set(cl.ranker_features(c))
        for name in cl._RANKER_FEATURES:
            assert name in available, f"geometry ranker scores missing feature {name}"
        for name in cl._PLM_RANKER_FEATURES:
            if name.startswith("plm_"):
                continue  # supplied separately from residue probabilities
            assert name in available, f"PLM ranker scores missing feature {name}"


class TestScoringIsFinite:
    def test_geometry_score_is_a_real_number(self):
        from lacuna.models import Pocket, PocketCluster
        import dataclasses

        fields = {f.name for f in dataclasses.fields(Pocket)}
        kw = {"centroid": (0.0, 0.0, 0.0), "volume_a3": 250.0,
              "lining_residues": ["ALA1:A", "PHE2:A"], "conformer_idx": 0}
        for name in fields - set(kw) - {"source", "score"}:
            kw[name] = 0.4
        c = PocketCluster(
            rank=1, centroid=(0.0, 0.0, 0.0), volume_a3=250.0, volume_min_a3=200.0,
            volume_max_a3=300.0, druggability=0.5, max_druggability=0.7,
            apo_volume_a3=180.0, crypticity=0.4, persistence=0.6, cryptic=True,
            lining_residues=["ALA1:A", "PHE2:A"], appears_in_conformers=[0],
            member_pockets=[Pocket(**kw)],
        )
        assert np.isfinite(cl.learned_score(c))
