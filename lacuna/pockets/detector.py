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

from lacuna.models import Atom, Pocket, Residue, Structure
from lacuna.io.structure import get_vdw_radius, is_hydrophobic, is_aromatic


GRID_SPACING = 1.0       # Å per voxel
PADDING = 5.0            # Å padding around bounding box
MIN_VOLUME_A3 = 80.0     # minimum pocket volume to report
MAX_VOLUME_A3 = 5000.0

# Interaction zone
ALPHA_DIST_MIN = 1.6     # Å  — too small: inside VDW sphere
ALPHA_DIST_MAX = 6.0     # Å  — too large: bulk solvent alpha sphere

# Cluster radius: alpha points within this distance are merged into one pocket
CLUSTER_RADIUS_A = 4.0   # Å

# Pocket volume growth: expand each alpha-point cluster by this much
GROW_RADIUS_A = 3.0      # Å from any alpha point


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

    # 5. For each cluster, grow into the nearby interaction zone and compute props
    grow_vox = max(1, int(np.ceil(GROW_RADIUS_A / grid_spacing)))
    interaction_zone = (~protein_mask) & (dist >= 0.5) & (dist <= ALPHA_DIST_MAX + 1.0)

    voxel_vol = grid_spacing ** 3
    pockets: list[Pocket] = []

    for label_id in range(1, n_labels + 1):
        alpha_cluster = labeled == label_id
        # Expand the cluster into the surrounding interaction zone
        grown = binary_dilation(alpha_cluster, structure=struct_el, iterations=grow_vox)
        pocket_region = grown & interaction_zone

        volume = float(pocket_region.sum()) * voxel_vol
        if volume < min_volume_a3 or volume > MAX_VOLUME_A3:
            continue

        vox_indices = np.argwhere(pocket_region)
        centroid_vox = vox_indices.mean(axis=0)
        centroid = tuple((centroid_vox * grid_spacing + lo).tolist())

        # Lining residues: atoms within (pocket_radius + 4 Å) of centroid
        pocket_radius = (volume * 3 / (4 * 3.14159)) ** (1 / 3)
        search_dist = pocket_radius + 4.0
        centroid_arr = np.array(centroid)
        dists_to_center = np.linalg.norm(coords - centroid_arr, axis=1)
        nearby = np.where(dists_to_center < search_dist)[0]

        lining_res_set: set[int] = set()
        for ai in nearby:
            ri = atom_res_idx.get(ai)
            if ri is not None:
                lining_res_set.add(ri)

        lining_residues = [residues[ri].label for ri in sorted(lining_res_set)]

        # Enclosure: local protein density at the alpha point cluster centroid
        local_density = uniform_filter(protein_mask.astype(np.float32), size=9)
        enclosure_raw = float(local_density[alpha_cluster].mean())
        enclosure = min(enclosure_raw / 0.4, 1.0)  # normalise to [0,1]

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
    mask = np.zeros(shape, dtype=bool)
    for i, atom in enumerate(atoms):
        r = get_vdw_radius(atom.element)
        center_vox = ((coords[i] - lo) / grid_spacing).astype(int)
        r_vox = int(np.ceil(r / grid_spacing)) + 1

        ix0 = max(0, center_vox[0] - r_vox); ix1 = min(shape[0], center_vox[0] + r_vox + 1)
        iy0 = max(0, center_vox[1] - r_vox); iy1 = min(shape[1], center_vox[1] + r_vox + 1)
        iz0 = max(0, center_vox[2] - r_vox); iz1 = min(shape[2], center_vox[2] + r_vox + 1)

        xi = np.arange(ix0, ix1); yi = np.arange(iy0, iy1); zi = np.arange(iz0, iz1)
        xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
        dist2 = (
            ((xx - center_vox[0]) * grid_spacing) ** 2
            + ((yy - center_vox[1]) * grid_spacing) ** 2
            + ((zz - center_vox[2]) * grid_spacing) ** 2
        )
        mask[ix0:ix1, iy0:iy1, iz0:iz1] |= dist2 <= r ** 2
    return mask


def _build_atom_residue_map(atoms: list[Atom], residues: list[Residue]) -> dict[int, int]:
    m: dict[int, int] = {}
    for ri, res in enumerate(residues):
        for ai in res.atom_indices:
            m[ai] = ri
    return m
