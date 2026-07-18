# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Grid-based binding pocket detection using alpha-sphere-inspired segmentation.

Algorithm:
  1. Build protein occupancy grid (VDW spheres).
  2. Compute Euclidean distance transform: dist[i,j,k] = Å to nearest protein atom.
  3. Find LOCAL MAXIMA of the distance field inside the interaction zone.
     Each local max is an "alpha point" - a sphere that fits snugly into a
     protein concavity and touches multiple atoms on all sides.  This is the
     core idea behind fpocket / CASTp.
  4. Cluster nearby alpha points into pockets (DBSCAN-style dilation + labelling).
  5. Grow each pocket cluster outward into its surrounding interaction zone
     to compute volume, lining residues, and druggability features.

This correctly segments the protein into DISTINCT POCKETS rather than treating
the entire surface as one connected region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    distance_transform_edt,
    maximum_filter,
    binary_dilation,
    uniform_filter,
)
from scipy.spatial import cKDTree

from lacuna.models import Atom, Pocket, Residue, Structure
from lacuna.io.structure import is_aromatic, is_hydrophobic


GRID_SPACING = 1.0       # Å per voxel
PADDING = 5.0            # Å padding around bounding box
MIN_VOLUME_A3 = 80.0     # minimum pocket volume to report
# Real druggable pockets are ~200-1000 Å³; cap well below the over-merged-blob
# regime (2500-5000 Å³). Blobs above this both hurt real output and trivially
# game residue-overlap metrics (a 140-residue "pocket" overlaps any small site).
MAX_VOLUME_A3 = 1500.0

# Interaction zone
ALPHA_DIST_MIN = 1.6     # Å  - too small: inside VDW sphere
ALPHA_DIST_MAX = 6.0     # Å  - too large: bulk solvent alpha sphere

# Cluster radius: alpha points within this distance are merged into one pocket
CLUSTER_RADIUS_A = 4.0   # Å

# Lining residues: a residue lines the pocket if any of its atoms is within this
# distance of the detected cavity (the alpha-cluster void voxels). This is a true
# atomic-contact definition - replaces the earlier centroid+9 Å sphere, which
# swept in a large shell of non-lining residues and inflated residue overlap.
LINING_CONTACT_A = 5.0   # Å from any cavity voxel


@dataclass
class _GridContext:
    """Precomputed grid/geometry shared by pocket detection and characterization.

    Building this once per conformer lets ``detect_pockets`` (which scans every
    concavity) and ``characterize_pocket`` (which describes one given location)
    compute pocket features - volume, buriedness-weighted centroid, contact-based
    lining residues, enclosure, hydrophobicity, aromaticity - through the exact
    same code path, so pockets from different detectors are on one identical scale
    and can be fused in the ensemble clusterer.
    """
    coords: np.ndarray
    residues: list[Residue]
    atom_res_idx: dict[int, int]
    lo: np.ndarray
    shape: np.ndarray
    grid_spacing: float
    voxel_vol: float
    protein_mask: np.ndarray
    dist: np.ndarray
    local_density: np.ndarray
    atom_tree: cKDTree


def _build_grid_context(
    coords: np.ndarray,
    structure: Structure,
    grid_spacing: float,
) -> _GridContext:
    """Build the protein occupancy grid, distance field, buriedness field, and
    atom KD-tree for one conformer. See ``_GridContext``."""
    atoms = structure.atoms
    residues = structure.residues
    atom_res_idx = _build_atom_residue_map(atoms, residues)

    lo = coords.min(axis=0) - PADDING
    shape = np.ceil((coords.max(axis=0) + PADDING - lo) / grid_spacing).astype(int) + 1

    protein_mask = _build_protein_mask(coords, atoms, lo, shape, grid_spacing)
    # Distance transform (Å to nearest protein atom, for non-protein voxels)
    dist = distance_transform_edt(~protein_mask).astype(np.float32) * grid_spacing
    # Buriedness field: local fraction of occupied space (used for enclosure and the
    # hotspot-weighted centroid).
    local_density = uniform_filter(protein_mask.astype(np.float32), size=9)
    atom_tree = cKDTree(coords)

    return _GridContext(
        coords=coords,
        residues=residues,
        atom_res_idx=atom_res_idx,
        lo=lo,
        shape=shape,
        grid_spacing=grid_spacing,
        voxel_vol=grid_spacing ** 3,
        protein_mask=protein_mask,
        dist=dist,
        local_density=local_density,
        atom_tree=atom_tree,
    )


def _pocket_from_cavity(
    void_mask: np.ndarray,
    cluster_mask: np.ndarray,
    ctx: _GridContext,
) -> Pocket:
    """Describe one cavity (its void voxels + enclosing cluster mask) as a Pocket.

    ``void_mask`` are the empty voxels that make up the cavity (volume + centroid +
    lining come from these); ``cluster_mask`` is the enclosing region used for the
    buriedness/enclosure estimate. Shared by ``detect_pockets`` and
    ``characterize_pocket``.
    """
    grid_spacing = ctx.grid_spacing
    lo = ctx.lo

    # Hotspot-centered localization: weight cavity voxels by buriedness (local
    # protein density) so the reported center sits at the most enclosed, ligandable
    # sub-pocket rather than the geometric mean. For elongated or partially-open
    # cryptic pockets the geometric centroid drifts toward the open mouth; the
    # buriedness-weighted center tracks where a ligand's core actually binds,
    # tightening docking-box placement and localization.
    vox_indices = np.argwhere(void_mask) if void_mask.any() else np.argwhere(cluster_mask)
    vox_weights = ctx.local_density[vox_indices[:, 0], vox_indices[:, 1], vox_indices[:, 2]]
    if float(vox_weights.sum()) > 1e-6:
        centroid_vox = np.average(vox_indices, axis=0, weights=vox_weights)
    else:
        centroid_vox = vox_indices.mean(axis=0)
    centroid = tuple((centroid_vox * grid_spacing + lo).tolist())

    # Lining residues: any residue with an atom within LINING_CONTACT_A of the
    # cavity voxels. True atomic contact - not a centroid sphere - so the set
    # reflects the residues that actually wall the pocket and feed accurate
    # druggability + docking outputs.
    cavity_world = vox_indices * grid_spacing + lo  # (K, 3) cavity voxel centers
    neighbor_lists = ctx.atom_tree.query_ball_point(cavity_world, r=LINING_CONTACT_A)
    nearby: set[int] = set()
    for nl in neighbor_lists:
        nearby.update(nl)

    lining_res_set: set[int] = set()
    for ai in nearby:
        ri = ctx.atom_res_idx.get(int(ai))
        if ri is not None:
            lining_res_set.add(ri)

    lining_residues = [ctx.residues[ri].label for ri in sorted(lining_res_set)]

    enclosure_raw = float(ctx.local_density[cluster_mask].mean())
    enclosure = min(enclosure_raw / 0.4, 1.0)

    lining_res_objs = [ctx.residues[ri] for ri in sorted(lining_res_set)]
    hyd_frac = (
        sum(1 for r in lining_res_objs if is_hydrophobic(r.name))
        / max(len(lining_res_objs), 1)
    )
    arom_count = sum(1 for r in lining_res_objs if is_aromatic(r.name))

    volume = float(void_mask.sum()) * ctx.voxel_vol

    return Pocket(
        centroid=centroid,
        volume_a3=volume,
        enclosure=enclosure,
        hydrophobic_fraction=hyd_frac,
        aromatic_count=arom_count,
        lining_residues=lining_residues,
        conformer_idx=-1,
    )


def detect_pockets(
    coords: np.ndarray,
    structure: Structure,
    grid_spacing: float = GRID_SPACING,
    min_volume_a3: float = MIN_VOLUME_A3,
) -> list[Pocket]:
    """Detect binding pockets in a single conformer.

    Args:
        coords: (N_atoms, 3) float32 coordinate array for this conformer.
        structure: Structure object with atom metadata.
        grid_spacing: Voxel size in Å.
        min_volume_a3: Minimum pocket volume to report.

    Returns:
        List of Pocket objects, unsorted.
    """
    ctx = _build_grid_context(coords, structure, grid_spacing)

    # Alpha points: local maxima of the distance field in the interaction zone.
    # Local max size 3 → a point is max if no neighbour within 1 voxel is larger.
    local_max_mask = (ctx.dist == maximum_filter(ctx.dist, size=3))
    alpha_points = local_max_mask & (ctx.dist >= ALPHA_DIST_MIN) & (ctx.dist <= ALPHA_DIST_MAX)

    if not alpha_points.any():
        return []

    # Cluster alpha points: dilate so nearby ones merge, then label.
    cluster_radius_vox = max(1, int(np.ceil(CLUSTER_RADIUS_A / grid_spacing)))
    struct_el = ndimage.generate_binary_structure(3, 1)
    dilated_alpha = binary_dilation(alpha_points, structure=struct_el,
                                    iterations=cluster_radius_vox)
    labeled, n_labels = ndimage.label(dilated_alpha)

    if n_labels == 0:
        return []

    pockets: list[Pocket] = []
    for label_id in range(1, n_labels + 1):
        alpha_cluster = labeled == label_id

        # Volume from the alpha-cluster core (not the grown region) - avoids
        # over-inflation in large inter-chain spaces.
        alpha_void = alpha_cluster & (~ctx.protein_mask)
        volume = float(alpha_void.sum()) * ctx.voxel_vol
        if volume < min_volume_a3 or volume > MAX_VOLUME_A3:
            continue

        pockets.append(_pocket_from_cavity(alpha_void, alpha_cluster, ctx))

    return pockets


def _characterize_at(
    center: np.ndarray,
    ctx: _GridContext,
    radius: float,
) -> Pocket | None:
    """Describe the pocket at ``center`` against an already-built grid context.

    Returns ``None`` if there is no open (non-protein) void within ``radius`` of
    the center. See ``characterize_pocket`` for the rationale.
    """
    grid_spacing = ctx.grid_spacing
    c_vox = (center - ctx.lo) / grid_spacing
    rad_vox = radius / grid_spacing
    c_idx = np.round(c_vox).astype(int)
    rad_int = int(np.ceil(rad_vox))
    lo_b = np.maximum(c_idx - rad_int, 0)
    hi_b = np.minimum(c_idx + rad_int + 1, ctx.shape)
    if np.any(lo_b >= hi_b):
        return None  # center is off the grid

    zc = np.arange(lo_b[0], hi_b[0])[:, None, None]
    yc = np.arange(lo_b[1], hi_b[1])[None, :, None]
    xc = np.arange(lo_b[2], hi_b[2])[None, None, :]
    d2 = (zc - c_vox[0]) ** 2 + (yc - c_vox[1]) ** 2 + (xc - c_vox[2]) ** 2
    ball = np.zeros(tuple(ctx.shape), dtype=bool)
    ball[lo_b[0]:hi_b[0], lo_b[1]:hi_b[1], lo_b[2]:hi_b[2]] = d2 <= rad_vox ** 2

    void_mask = ball & (~ctx.protein_mask)
    if not void_mask.any():
        return None

    return _pocket_from_cavity(void_mask, ball, ctx)


def characterize_pockets(
    coords: np.ndarray,
    structure: Structure,
    centers: list[tuple[float, float, float]] | np.ndarray,
    grid_spacing: float = GRID_SPACING,
    radius: float = CLUSTER_RADIUS_A,
) -> list[Pocket | None]:
    """Characterize several locations against one shared grid (built once).

    Same semantics as ``characterize_pocket`` per center, but the expensive grid
    build is amortized across all centers - the right entry point when folding in
    a whole list of external-detector proposals for one conformer.
    """
    ctx = _build_grid_context(coords, structure, grid_spacing)
    return [_characterize_at(np.asarray(c, dtype=float), ctx, radius) for c in centers]


def characterize_pocket(
    coords: np.ndarray,
    structure: Structure,
    center: tuple[float, float, float] | np.ndarray,
    grid_spacing: float = GRID_SPACING,
    radius: float = CLUSTER_RADIUS_A,
) -> Pocket | None:
    """Describe the pocket at a given location using Lacuna's own geometry.

    Used to fold in candidate locations from an external detector (e.g. P2Rank):
    the external tool supplies where to look, and this returns a Pocket whose
    volume, centroid, lining residues, and druggability features are computed by
    the same code as ``detect_pockets`` - so external and built-in pockets share
    one scale and can be fused in the ensemble clusterer.

    The alpha-sphere volume filter is intentionally NOT applied here: the external
    detector's own model is the reason to trust this location, so a cavity outside
    the detector's usual size band is still described rather than discarded.

    Returns ``None`` if there is no open (non-protein) void within ``radius`` of the
    center - i.e. the point is fully buried in atoms or fully solvent-exposed with
    no concavity in this conformer. For many centers use ``characterize_pockets``.
    """
    ctx = _build_grid_context(coords, structure, grid_spacing)
    return _characterize_at(np.asarray(center, dtype=float), ctx, radius)


def _build_protein_mask(
    coords: np.ndarray,
    atoms: list[Atom],
    lo: np.ndarray,
    shape: np.ndarray,
    grid_spacing: float,
) -> np.ndarray:
    # Mark atom centers in the grid (vectorized - O(N_atoms))
    atom_grid = np.zeros(shape, dtype=bool)
    indices = np.clip(
        np.round((coords - lo) / grid_spacing).astype(int),
        0,
        np.array(shape) - 1,
    )
    atom_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # EDT gives distance from each voxel to its nearest atom center (in voxels).
    # Threshold at 1.7 Å (carbon VDW radius) to mark voxels inside any atom.
    # This approximates per-element radii (range 1.2–1.8 Å) with negligible error
    # at 1 Å grid spacing; avoids a per-atom Python loop.
    dist_vox = distance_transform_edt(~atom_grid)
    return dist_vox <= (1.7 / grid_spacing)


def _build_atom_residue_map(atoms: list[Atom], residues: list[Residue]) -> dict[int, int]:
    m: dict[int, int] = {}
    for ri, res in enumerate(residues):
        for ai in res.atom_indices:
            m[ai] = ri
    return m
