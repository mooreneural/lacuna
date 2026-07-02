# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Grid-based binding pocket detection using alpha-sphere-inspired segmentation.

Algorithm:
  1. Build protein occupancy grid (VDW spheres).
  2. Compute Euclidean distance transform: dist[i,j,k] = Å to nearest protein atom.
  3. Find LOCAL MAXIMA of the distance field inside the interaction zone.
     Each local max is an "alpha point" — a sphere that fits snugly into a
     protein concavity and touches multiple atoms on all sides.  This is the
     core idea behind fpocket / CASTp.
  4. Cluster nearby alpha points into pockets (DBSCAN-style dilation + labelling).
  5. Grow each pocket cluster outward into its surrounding interaction zone
     to compute volume, lining residues, and druggability features.

This correctly segments the protein into DISTINCT POCKETS rather than treating
the entire surface as one connected region.
"""

from __future__ import annotations

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
MAX_VOLUME_A3 = 5000.0

# Interaction zone
ALPHA_DIST_MIN = 1.6     # Å  — too small: inside VDW sphere
ALPHA_DIST_MAX = 6.0     # Å  — too large: bulk solvent alpha sphere

# Cluster radius: alpha points within this distance are merged into one pocket
CLUSTER_RADIUS_A = 4.0   # Å

# Lining residues: a residue lines the pocket if any of its atoms is within this
# distance of the detected cavity (the alpha-cluster void voxels). This is a true
# atomic-contact definition — replaces the earlier centroid+9 Å sphere, which
# swept in a large shell of non-lining residues and inflated residue overlap.
LINING_CONTACT_A = 5.0   # Å from any cavity voxel


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
    atoms = structure.atoms
    residues = structure.residues
    atom_res_idx = _build_atom_residue_map(atoms, residues)

    lo = coords.min(axis=0) - PADDING
    shape = np.ceil((coords.max(axis=0) + PADDING - lo) / grid_spacing).astype(int) + 1

    # 1. Protein occupancy grid
    protein_mask = _build_protein_mask(coords, atoms, lo, shape, grid_spacing)

    # 2. Distance transform (Å to nearest protein atom, for non-protein voxels)
    dist = distance_transform_edt(~protein_mask).astype(np.float32) * grid_spacing

    # 3. Alpha points: local maxima of dist in the interaction zone
    #    Local max size 3 → a point is max if no neighbour within 1 voxel is larger
    local_max_mask = (dist == maximum_filter(dist, size=3))
    alpha_points = local_max_mask & (dist >= ALPHA_DIST_MIN) & (dist <= ALPHA_DIST_MAX)

    if not alpha_points.any():
        return []

    # 4. Cluster alpha points: dilate so nearby ones merge, then label
    cluster_radius_vox = max(1, int(np.ceil(CLUSTER_RADIUS_A / grid_spacing)))
    struct_el = ndimage.generate_binary_structure(3, 1)
    dilated_alpha = binary_dilation(alpha_points, structure=struct_el,
                                    iterations=cluster_radius_vox)
    labeled, n_labels = ndimage.label(dilated_alpha)

    if n_labels == 0:
        return []

    # 5. For each cluster, compute volume, lining residues, and druggability props.
    # Pre-compute once — used per-pocket for enclosure scoring
    local_density = uniform_filter(protein_mask.astype(np.float32), size=9)
    # Atom KDTree for fast contact-based lining-residue lookup.
    atom_tree = cKDTree(coords)

    voxel_vol = grid_spacing ** 3
    pockets: list[Pocket] = []

    for label_id in range(1, n_labels + 1):
        alpha_cluster = labeled == label_id

        # Volume from the alpha-cluster core (not the grown region) — avoids
        # over-inflation in large inter-chain spaces.
        alpha_void = alpha_cluster & (~protein_mask)
        volume = float(alpha_void.sum()) * voxel_vol
        if volume < min_volume_a3 or volume > MAX_VOLUME_A3:
            continue

        # Hotspot-centered localization: weight cavity voxels by buriedness (local
        # protein density) so the reported center sits at the most enclosed,
        # ligandable sub-pocket rather than the geometric mean. For elongated or
        # partially-open cryptic pockets the geometric centroid drifts toward the
        # open mouth; the buriedness-weighted center tracks where a ligand's core
        # actually binds, tightening docking-box placement and localization.
        vox_indices = np.argwhere(alpha_void) if alpha_void.any() else np.argwhere(alpha_cluster)
        vox_weights = local_density[vox_indices[:, 0], vox_indices[:, 1], vox_indices[:, 2]]
        if float(vox_weights.sum()) > 1e-6:
            centroid_vox = np.average(vox_indices, axis=0, weights=vox_weights)
        else:
            centroid_vox = vox_indices.mean(axis=0)
        centroid = tuple((centroid_vox * grid_spacing + lo).tolist())

        # Lining residues: any residue with an atom within LINING_CONTACT_A of the
        # detected cavity (the alpha-cluster void voxels). True atomic contact —
        # not a centroid sphere — so the set reflects the residues that actually
        # wall the pocket and feed accurate druggability + docking outputs.
        cavity_world = vox_indices * grid_spacing + lo  # (K, 3) cavity voxel centers
        neighbor_lists = atom_tree.query_ball_point(cavity_world, r=LINING_CONTACT_A)
        nearby: set[int] = set()
        for nl in neighbor_lists:
            nearby.update(nl)

        lining_res_set: set[int] = set()
        for ai in nearby:
            ri = atom_res_idx.get(int(ai))
            if ri is not None:
                lining_res_set.add(ri)

        lining_residues = [residues[ri].label for ri in sorted(lining_res_set)]

        enclosure_raw = float(local_density[alpha_cluster].mean())
        enclosure = min(enclosure_raw / 0.4, 1.0)

        lining_res_objs = [residues[ri] for ri in sorted(lining_res_set)]
        hyd_frac = (
            sum(1 for r in lining_res_objs if is_hydrophobic(r.name))
            / max(len(lining_res_objs), 1)
        )
        arom_count = sum(1 for r in lining_res_objs if is_aromatic(r.name))

        pockets.append(Pocket(
            centroid=centroid,
            volume_a3=volume,
            enclosure=enclosure,
            hydrophobic_fraction=hyd_frac,
            aromatic_count=arom_count,
            lining_residues=lining_residues,
            conformer_idx=-1,
        ))

    return pockets


def _build_protein_mask(
    coords: np.ndarray,
    atoms: list[Atom],
    lo: np.ndarray,
    shape: np.ndarray,
    grid_spacing: float,
) -> np.ndarray:
    # Mark atom centers in the grid (vectorized — O(N_atoms))
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
