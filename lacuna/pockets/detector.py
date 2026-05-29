"""Grid-based binding pocket detection.

Algorithm (no external dependencies):
  1. Build a 3D occupancy grid over the protein bounding box.
  2. Mark voxels as PROTEIN if any atom's VDW sphere overlaps them.
  3. Mark voxels as SOLVENT_EXPOSED by flood-filling from the grid boundary.
  4. Remaining voxels (inside protein surface but not bulk solvent) are cavity.
  5. Label connected cavity regions → candidate pockets.
  6. Filter by minimum volume, compute pocket centroid and lining residues.

This is a simplified implementation of the algorithm used by fpocket/CASTp.
Probe radius 1.4 Å (water). Grid spacing 1.0 Å balances speed and resolution.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage

from lacuna.models import Atom, Pocket, Residue, Structure
from lacuna.io.structure import get_vdw_radius, is_hydrophobic, is_aromatic


GRID_SPACING = 1.0      # Å per voxel
PROBE_RADIUS = 1.4      # Å, water probe
PADDING = 4.0           # Å padding around bounding box
MIN_VOLUME_A3 = 80.0    # Minimum pocket volume to report
MAX_VOLUME_A3 = 3000.0  # Ignore unrealistically large cavities


def detect_pockets(
    coords: np.ndarray,
    structure: Structure,
    grid_spacing: float = GRID_SPACING,
    min_volume_a3: float = MIN_VOLUME_A3,
) -> list[Pocket]:
    """Detect binding pockets in a single conformer.

    Args:
        coords: (N_atoms, 3) float32 coordinate array for this conformer.
        structure: Structure object with atom metadata (residue assignments, elements).
        grid_spacing: Voxel size in Å.
        min_volume_a3: Minimum cavity volume to report.

    Returns:
        List of Pocket objects, unsorted.
    """
    atoms = structure.atoms
    residues = structure.residues

    # Build spatial lookup: residue index for each atom
    atom_res_idx = _build_atom_residue_map(atoms, residues)

    # 1. Build occupancy grid
    lo = coords.min(axis=0) - PADDING
    hi = coords.max(axis=0) + PADDING
    shape = np.ceil((hi - lo) / grid_spacing).astype(int) + 1

    grid = np.zeros(shape, dtype=np.int8)  # 0=empty, 1=protein, 2=solvent

    # Mark protein voxels (VDW + probe sphere)
    for i, atom in enumerate(atoms):
        r = get_vdw_radius(atom.element) + PROBE_RADIUS
        center_vox = ((coords[i] - lo) / grid_spacing).astype(int)
        r_vox = int(np.ceil(r / grid_spacing)) + 1

        ix0, ix1 = max(0, center_vox[0] - r_vox), min(shape[0], center_vox[0] + r_vox + 1)
        iy0, iy1 = max(0, center_vox[1] - r_vox), min(shape[1], center_vox[1] + r_vox + 1)
        iz0, iz1 = max(0, center_vox[2] - r_vox), min(shape[2], center_vox[2] + r_vox + 1)

        xi = np.arange(ix0, ix1)
        yi = np.arange(iy0, iy1)
        zi = np.arange(iz0, iz1)
        xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")

        dist2 = (
            ((xx - center_vox[0]) * grid_spacing) ** 2
            + ((yy - center_vox[1]) * grid_spacing) ** 2
            + ((zz - center_vox[2]) * grid_spacing) ** 2
        )
        mask = dist2 <= r ** 2
        grid[ix0:ix1, iy0:iy1, iz0:iz1][mask] = 1

    # 2. Flood-fill from boundary to mark bulk solvent
    _flood_fill_solvent(grid)

    # 3. Cavity = voxels still marked 0 (not protein, not bulk solvent)
    cavity_mask = grid == 0

    # 4. Label connected cavity regions
    labeled, n_labels = ndimage.label(cavity_mask)

    # 5. Build Pocket objects
    voxel_vol = grid_spacing ** 3
    pockets: list[Pocket] = []

    for label_id in range(1, n_labels + 1):
        region = labeled == label_id
        volume = float(region.sum()) * voxel_vol

        if volume < min_volume_a3 or volume > MAX_VOLUME_A3:
            continue

        # Centroid in real space
        vox_indices = np.argwhere(region)
        centroid_vox = vox_indices.mean(axis=0)
        centroid = tuple((centroid_vox * grid_spacing + lo).tolist())

        # Find lining residues: atoms within (pocket_radius + 4 Å) of centroid.
        # Using a fixed radius misses lining atoms for large pockets, so we scale
        # with the estimated pocket sphere radius from its volume.
        pocket_radius = (volume * 3 / (4 * 3.14159)) ** (1 / 3)
        search_dist = pocket_radius + 4.0

        centroid_arr = np.array(centroid)
        dists = np.linalg.norm(coords - centroid_arr, axis=1)
        nearby_atom_indices = np.where(dists < search_dist)[0]

        lining_res_set: set[int] = set()
        for ai in nearby_atom_indices:
            ri = atom_res_idx.get(ai)
            if ri is not None:
                lining_res_set.add(ri)

        lining_residues = [residues[ri].label for ri in sorted(lining_res_set)]

        # Enclosure: fraction of voxels adjacent to protein vs. total boundary
        enclosure = _compute_enclosure(region, grid)

        # Hydrophobic fraction and aromatic count
        lining_res_objs = [residues[ri] for ri in sorted(lining_res_set)]
        hyd_frac = (
            sum(1 for r in lining_res_objs if is_hydrophobic(r.name)) / max(len(lining_res_objs), 1)
        )
        arom_count = sum(1 for r in lining_res_objs if is_aromatic(r.name))

        pockets.append(Pocket(
            centroid=centroid,
            volume_a3=volume,
            enclosure=enclosure,
            hydrophobic_fraction=hyd_frac,
            aromatic_count=arom_count,
            lining_residues=lining_residues,
            conformer_idx=-1,  # set by caller
        ))

    return pockets


def _build_atom_residue_map(atoms: list[Atom], residues: list[Residue]) -> dict[int, int]:
    """Map atom serial → residue index."""
    m: dict[int, int] = {}
    for ri, res in enumerate(residues):
        for ai in res.atom_indices:
            m[ai] = ri
    return m


def _flood_fill_solvent(grid: np.ndarray) -> None:
    """BFS from all boundary voxels to mark bulk solvent (value 2)."""
    shape = grid.shape
    q: deque[tuple[int, int, int]] = deque()

    # Seed from all boundary faces
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in [0, shape[2] - 1]:
                if grid[i, j, k] == 0:
                    grid[i, j, k] = 2
                    q.append((i, j, k))
        for k in range(shape[2]):
            for j_edge in [0, shape[1] - 1]:
                if grid[i, j_edge, k] == 0:
                    grid[i, j_edge, k] = 2
                    q.append((i, j_edge, k))
    for j in range(shape[1]):
        for k in range(shape[2]):
            for i_edge in [0, shape[0] - 1]:
                if grid[i_edge, j, k] == 0:
                    grid[i_edge, j, k] = 2
                    q.append((i_edge, j, k))

    neighbors = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    while q:
        x, y, z = q.popleft()
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if grid[nx, ny, nz] == 0:
                    grid[nx, ny, nz] = 2
                    q.append((nx, ny, nz))


def _compute_enclosure(region: np.ndarray, grid: np.ndarray) -> float:
    """Fraction of pocket boundary voxels that are adjacent to protein (vs. solvent)."""
    # Dilate pocket region by 1 voxel and intersect with protein
    dilated = ndimage.binary_dilation(region)
    shell = dilated & ~region
    if not shell.any():
        return 0.0
    protein_adjacent = (grid[shell] == 1).sum()
    total_shell = shell.sum()
    return float(protein_adjacent) / float(total_shell)
