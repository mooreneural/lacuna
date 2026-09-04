"""The learned surface detector, and the NumPy evaluation of its ensemble.

The tree walk is the part worth testing hardest. It replaces LightGBM at
inference so the package keeps its numpy/scipy-only install, which means a
silent disagreement between the two would ship a detector that scored one way
in training and another way in production.
"""
from __future__ import annotations

import numpy as np
import pytest

from lacuna.io.structure import coords_array, load_structure
from lacuna.pockets import surface_detector as sd
from lacuna.pockets.detector import GRID_SPACING, _build_grid_context

pytestmark = pytest.mark.skipif(not sd.available(),
                                reason="surface_head.npz not built in this checkout")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A small two-helix structure is enough to exercise every code path."""
    lines = []
    n = 1
    for i in range(1, 41):
        z = i * 1.5
        for name, el, dx in (("N", "N", 0.0), ("CA", "C", 1.2),
                             ("C", "C", 2.4), ("O", "O", 3.0)):
            x = dx + (6.0 if i > 20 else 0.0)
            lines.append("ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                         % (n, name, i, x, np.sin(i) * 4.0, z % 30, el))
            n += 1
    path = tmp_path_factory.mktemp("s") / "mini.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    st = load_structure(path)
    coords = coords_array(st)
    return st, coords, _build_grid_context(coords, st, GRID_SPACING)


def test_numpy_walk_matches_a_hand_built_tree():
    # Two trees, one split each, so the expected sum is arithmetic rather than
    # a second implementation of the thing under test.
    m = {"feature": np.array([0, 0, 0, 1, 0, 0], np.int32),
         "threshold": np.array([0.5, 0, 0, 2.0, 0, 0], np.float64),
         "left": np.array([1, 0, 0, 4, 0, 0], np.int32),
         "right": np.array([2, 0, 0, 5, 0, 0], np.int32),
         "is_leaf": np.array([False, True, True, False, True, True]),
         "value": np.array([0.0, -1.0, 1.0, 0.0, 10.0, 20.0], np.float64),
         "roots": np.array([0, 3], np.int32)}
    X = np.array([[0.0, 0.0], [0.0, 5.0], [1.0, 0.0], [1.0, 5.0]])
    assert sd._predict_raw(X, m).tolist() == [9.0, 19.0, 11.0, 21.0]


def test_shell_voxels_lie_in_the_probe_shell(prepared):
    _st, _c, ctx = prepared
    vox = sd.shell_voxels(ctx)
    assert len(vox) > 0
    d = ctx.dist[vox[:, 0], vox[:, 1], vox[:, 2]]
    assert (d >= sd.SHELL_LO).all() and (d <= sd.SHELL_HI).all()


def test_shell_thinning_is_deterministic(prepared):
    _st, _c, ctx = prepared
    a = sd.shell_voxels(ctx, max_points=50)
    b = sd.shell_voxels(ctx, max_points=50)
    assert len(a) <= 50 and np.array_equal(a, b)


def test_feature_width_depends_on_sequence_probabilities(prepared):
    st, _c, ctx = prepared
    vox = sd.shell_voxels(ctx, max_points=200)
    geom = sd.surface_point_features(st, ctx, vox)
    assert geom.shape == (len(vox), len(sd.GEOM_FEATURES))
    probs = {a.res_seq: 0.9 for a in st.atoms}
    withplm = sd.surface_point_features(st, ctx, vox, probs)
    assert withplm.shape[1] == len(sd.GEOM_FEATURES) + len(sd.PLM_FEATURES)
    # The geometric columns must be untouched by adding sequence signal.
    assert np.allclose(geom, withplm[:, :geom.shape[1]])


def test_features_are_finite(prepared):
    st, _c, ctx = prepared
    vox = sd.shell_voxels(ctx, max_points=200)
    probs = {a.res_seq: 0.5 for a in st.atoms}
    assert np.isfinite(sd.surface_point_features(st, ctx, vox, probs)).all()


def test_scores_are_probabilities(prepared):
    st, _c, ctx = prepared
    vox = sd.shell_voxels(ctx, max_points=300)
    s = sd.score_points(sd.surface_point_features(st, ctx, vox), with_plm=False)
    assert s.shape == (len(vox),)
    assert (s >= 0).all() and (s <= 1).all()


def test_detect_returns_tagged_pockets(prepared):
    st, coords, _ctx = prepared
    pockets = sd.detect_pockets_surface(coords, st, max_pockets=5)
    assert len(pockets) <= 5
    for p in pockets:
        assert p.source == "surface"
        assert p.lining_residues
        assert np.isfinite(p.centroid).all()


def test_detect_runs_with_sequence_probabilities(prepared):
    st, coords, _ctx = prepared
    probs = {a.res_seq: 0.7 for a in st.atoms}
    pockets = sd.detect_pockets_surface(coords, st, plm_residue_probs=probs,
                                        max_pockets=5)
    assert all(p.source == "surface" for p in pockets)


def test_fuses_with_the_alpha_detector(prepared):
    """Both detectors' pockets must go through one clusterer without special-casing."""
    from lacuna.pockets.clusterer import cluster_pockets
    from lacuna.pockets.detector import detect_pockets

    st, coords, _ctx = prepared
    pooled = detect_pockets(coords, st) + sd.detect_pockets_surface(coords, st)
    if not pooled:
        pytest.skip("no pockets on this synthetic structure")
    clusters = cluster_pockets([pooled], n_conformers=1, rank_by="druggability")
    assert clusters
    assert clusters[0].rank == 1
