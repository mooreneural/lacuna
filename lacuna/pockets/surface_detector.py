# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Learned surface detector - proposals where geometry alone finds nothing.

Why a second detector at all
----------------------------
The alpha-sphere detector only proposes at concavities, so a site that is
shallow, flat, or not yet open is discarded before ranking ever sees it. On
CryptoBench's training folds that costs coverage rather than rank: nothing
qualifying is proposed at all for roughly a quarter of targets, while union
coverage across published detectors reaches 92.2%. No improvement to ranking
can reach those, because there is nothing in the pool to rank.

This detector has no concavity filter. It scores points on the probe-accessible
surface shell and proposes wherever the model says ligandable. On the targets
where the alpha detector proposes nothing qualifying, it puts a qualifying
cluster at the annotated site for about a third of them.

Why it is not a reimplementation of P2Rank
------------------------------------------
Two differences, both consequences of where it sits. It is trained on cryptic
sites specifically rather than general holo binding sites, and it can consume
per-residue sequence probabilities from ``lacuna.pockets.plm``, which lifted
held-out per-point AUC from 0.760 to 0.848 and recovery from 23.4% to 34.1%.
A structure-only detector cannot use that signal at all.

Features are fields, not neighbour lists
----------------------------------------
Every feature is read out of a smoothed 3D field rather than by walking a
neighbour list per point. That is not a micro-optimisation: the ensemble calls
this once per conformer, and a per-point KD-tree aggregation costs tens of
seconds per structure where a box filter over the occupancy grid costs
milliseconds. ``_GridContext`` already computes ``local_density`` exactly this
way, so this follows the design that was there.

The same function computes features for training and for inference. Train/serve
skew in a per-point model is silent and produces a detector that scores well
offline and proposes nonsense in the pipeline, so there is deliberately only
one implementation.

No new dependencies
-------------------
The fitted gradient-boosted ensemble is stored as flat arrays in
``surface_head.npz`` and evaluated by a vectorised tree walk in NumPy. Lacuna
stays installable with numpy, scipy and biopython alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from lacuna.models import Pocket, Structure
from lacuna.pockets.detector import (
    GRID_SPACING, _build_grid_context, characterize_pockets,
)

#: Voxels this far from the nearest heavy atom form the probe-accessible shell.
#: Below 1.4 A is inside the van der Waals surface; beyond 4.0 A is bulk solvent.
SHELL_LO, SHELL_HI = 1.4, 4.0
#: Radii, in Angstrom, of the two smoothing scales. The coarse one describes the
#: neighbourhood a ligand would occupy; the fine one resolves the local surface,
#: and their ratio distinguishes a groove from an open face.
COARSE_R, FINE_R = 8.0, 4.0
#: Points scoring above this quantile are clustered into proposals.
TOP_QUANTILE = 0.98
#: Angstrom; proposals closer than this are one site.
CLUSTER_EPS = 5.0

GEOM_FEATURES = (
    "dist", "density", "bulk_depth", "radial",
    "atom_density_coarse", "atom_density_fine", "density_ratio",
    "f_hydrophobic", "f_aromatic", "f_polar", "f_carbon",
)
PLM_FEATURES = ("plm_field_mean", "plm_field_max", "plm_field_fine")

_HEAD_PATH = Path(__file__).with_name("surface_head.npz")


def available() -> bool:
    """True when the fitted model ships with this installation."""
    return _HEAD_PATH.exists()


# ─────────────────────────────── feature fields ──────────────────────────────

def _box(radius: float) -> int:
    """uniform_filter window covering a sphere of this radius, in voxels."""
    return max(3, int(round(2 * radius / GRID_SPACING)) | 1)


def _property_fields(structure: Structure, ctx, plm_residue_probs=None) -> dict:
    """Smoothed atom-property fields, sampled later at the surface points.

    Counts are accumulated into the occupancy grid with ``np.add.at`` so that
    several atoms landing in one voxel all count, then box-filtered at two
    scales. Fractions are ratios of smoothed counts, which is what makes them
    stable where the neighbourhood is nearly empty.
    """
    from lacuna.io.structure import is_aromatic, is_hydrophobic

    shape = tuple(ctx.shape)
    idx = np.clip(np.round((ctx.coords - ctx.lo) / GRID_SPACING).astype(int),
                  0, np.asarray(shape) - 1)
    g = (idx[:, 0], idx[:, 1], idx[:, 2])

    atoms = structure.atoms
    props = {
        "all": np.ones(len(atoms), np.float32),
        "hyd": np.array([is_hydrophobic(a.res_name) for a in atoms], np.float32),
        "aro": np.array([is_aromatic(a.res_name) for a in atoms], np.float32),
        "pol": np.array([a.element.upper() in ("N", "O") for a in atoms], np.float32),
        "car": np.array([a.element.upper() == "C" for a in atoms], np.float32),
    }
    if plm_residue_probs:
        props["plm"] = np.array(
            [float(plm_residue_probs.get(a.res_seq, 0.0)) for a in atoms], np.float32)

    coarse, fine = _box(COARSE_R), _box(FINE_R)
    out = {}
    for name, w in props.items():
        grid = np.zeros(shape, np.float32)
        np.add.at(grid, g, w)
        out[name + "_coarse"] = uniform_filter(grid, size=coarse)
        out[name + "_fine"] = uniform_filter(grid, size=fine)
        if name == "plm":
            # Max rather than mean: one strongly predicted residue beside a point
            # is the signal, and averaging it against its neighbours hides it.
            out["plm_max"] = maximum_filter(grid, size=fine)
    return out


def surface_point_features(structure: Structure, ctx, pts_vox: np.ndarray,
                           plm_residue_probs=None) -> np.ndarray:
    """Per-point features at the given voxel indices.

    Used for both training and inference. See the module docstring on why there
    is only one implementation.
    """
    f = _property_fields(structure, ctx, plm_residue_probs)
    g = (pts_vox[:, 0], pts_vox[:, 1], pts_vox[:, 2])
    xyz = ctx.lo + pts_vox * GRID_SPACING

    eps = 1e-6
    a_c, a_f = f["all_coarse"][g], f["all_fine"][g]
    cols = [
        ctx.dist[g],
        ctx.local_density[g],
        ctx.bulk_depth[g],
        np.linalg.norm(xyz - ctx.protein_centroid, axis=1) / max(ctx.radius_gyration, eps),
        a_c,
        a_f,
        a_f / (a_c + eps),
        f["hyd_coarse"][g] / (a_c + eps),
        f["aro_coarse"][g] / (a_c + eps),
        f["pol_coarse"][g] / (a_c + eps),
        f["car_coarse"][g] / (a_c + eps),
    ]
    if plm_residue_probs:
        cols += [
            f["plm_coarse"][g] / (a_c + eps),
            f["plm_max"][g],
            f["plm_fine"][g] / (a_f + eps),
        ]
    return np.column_stack(cols).astype(np.float32)


def shell_voxels(ctx, max_points: int = 0) -> np.ndarray:
    """Probe-accessible shell voxels, optionally thinned by a fixed stride.

    Thinning is a stride rather than a random sample so a conformer scores the
    same way every run.
    """
    vox = np.argwhere((ctx.dist >= SHELL_LO) & (ctx.dist <= SHELL_HI))
    if max_points and len(vox) > max_points:
        vox = vox[:: int(np.ceil(len(vox) / max_points))]
    return vox


# ───────────────────────────── numpy tree ensemble ───────────────────────────

def _load_head(with_plm: bool):
    z = np.load(_HEAD_PATH)
    p = "plm_" if with_plm else "geom_"
    if p + "feature" not in z:
        raise KeyError(
            "surface_head.npz has no %smodel. Retrain with "
            "analysis/moores_pocket_law/train_surface_detector.py" % p)
    return {k: z[p + k] for k in
            ("feature", "threshold", "left", "right", "is_leaf", "value", "roots")}


def _predict_raw(X: np.ndarray, m: dict) -> np.ndarray:
    """Sum of leaf values over the ensemble, all samples walking together.

    Every sample descends one level per iteration, so the loop runs for the
    depth of the deepest tree rather than once per sample.
    """
    feature, thr = m["feature"], m["threshold"]
    left, right, is_leaf, value = m["left"], m["right"], m["is_leaf"], m["value"]
    out = np.zeros(len(X), np.float64)
    for root in m["roots"]:
        cur = np.full(len(X), int(root), np.int64)
        while True:
            act = ~is_leaf[cur]
            if not act.any():
                break
            n = cur[act]
            go_left = X[act, feature[n]] <= thr[n]
            cur[act] = np.where(go_left, left[n], right[n])
        out += value[cur]
    return out


def score_points(X: np.ndarray, with_plm: bool, head: dict | None = None) -> np.ndarray:
    """P(surface point lies at a ligandable site).

    ``head`` overrides the shipped model. Cross-validation needs to score a
    structure with a model that did not train on its fold, and there is no
    honest way to do that with a single baked-in ensemble.
    """
    return 1.0 / (1.0 + np.exp(-_predict_raw(X, head or _load_head(with_plm))))


# ──────────────────────────────── the detector ───────────────────────────────

def detect_pockets_surface(
    coords: np.ndarray,
    structure: Structure,
    plm_residue_probs: dict[int, float] | None = None,
    max_pockets: int = 10,
    max_points: int = 20000,
    head: dict | None = None,
) -> list[Pocket]:
    """Detect pockets in one conformer with the learned surface scorer.

    Mirrors ``detector.detect_pockets``'s signature so it is a drop-in detector
    in the ensemble loop. Proposals are characterised by ``characterize_pockets``
    so their features share one scale with every other detector and the
    crypticity and druggability maths stays coherent when they are fused.

    Passing ``plm_residue_probs`` (from ``lacuna.pockets.plm``) selects the
    sequence-aware model, which is substantially better; without it the
    geometry-only model is used.
    """
    if head is None and not available():
        raise FileNotFoundError(
            "surface_head.npz is not installed, so the learned surface detector "
            "cannot run. It ships with the package; a source checkout needs "
            "analysis/moores_pocket_law/train_surface_detector.py to build it.")

    ctx = _build_grid_context(coords, structure, GRID_SPACING)
    vox = shell_voxels(ctx, max_points)
    if len(vox) < 10:
        return []

    with_plm = bool(plm_residue_probs)
    X = surface_point_features(structure, ctx, vox, plm_residue_probs)
    s = score_points(X, with_plm, head)

    top = np.flatnonzero(s >= np.quantile(s, TOP_QUANTILE))
    if len(top) < 3:
        return []

    from lacuna.pockets.clusterer import _greedy_cluster
    xyz = (ctx.lo + vox * GRID_SPACING)[top]
    labels = _greedy_cluster(xyz.astype(float), eps=CLUSTER_EPS)

    # Rank proposals by total evidence, not by peak: a broad well-scored region
    # is a likelier site than one sharp voxel.
    groups = [(float(s[top][labels == c].sum()), xyz[labels == c].mean(axis=0))
              for c in range(labels.max() + 1)]
    groups.sort(key=lambda kv: -kv[0])
    centres = [c for _v, c in groups[:max_pockets]]

    pockets = []
    for strength, pocket in zip([v for v, _c in groups[:max_pockets]],
                                characterize_pockets(coords, structure, centres)):
        # characterize_pockets returns None where no open void surrounds the
        # centre. That is the honest outcome for a point the model liked on a
        # flat face, and such proposals are dropped rather than invented.
        if pocket is None:
            continue
        pocket.source = "surface"
        pocket.score = float(strength)
        pockets.append(pocket)
    return pockets
