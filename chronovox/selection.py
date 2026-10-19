"""Selecting voxels in a matrix, and moving, flipping, rotating, copying and pasting what is selected.
Pure numpy: no GL, UI or file imports.

A selection is a boolean mask the same shape as the matrix's voxels, (z, y, x), where True marks a selected
voxel. Only solid voxels are ever selected: dragging a box over empty space selects nothing there. Cells are
(x, y, z). Every function returns new arrays and leaves its inputs alone, so the caller can keep the old ones for
undo. Moving, flipping and rotating lift the selected voxels out and put them back down: they replace whatever
is at the new place, and the places they leave become empty.

While the same selection is being moved around it "floats": the voxels it covers are not lost but kept underneath
(``under``, the grid as it is with the selection lifted off), so moving it away uncovers them again. The covered
voxels are only given up when the selection changes or is cleared. Functions that take ``under`` put the selection
down on that instead of on the grid they were given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .editing import flood_region_3d

Cell = tuple[int, int, int]


# --- looking at a selection --------------------------------------------------------------------

def solid(mask: np.ndarray, voxels: np.ndarray) -> np.ndarray:
    """The selected cells that are really solid (an earlier edit may have emptied some of the selected ones)."""
    return mask & (voxels != 0)


def bounds(mask: np.ndarray) -> tuple[Cell, Cell] | None:
    """The smallest box holding every selected cell, as (low, high) inclusive, or None for an empty selection."""
    if not mask.any():
        return None
    xs = np.flatnonzero(mask.any(axis=(0, 1)))
    ys = np.flatnonzero(mask.any(axis=(0, 2)))
    zs = np.flatnonzero(mask.any(axis=(1, 2)))
    return (int(xs[0]), int(ys[0]), int(zs[0])), (int(xs[-1]), int(ys[-1]), int(zs[-1]))


def indices_of(mask: np.ndarray) -> np.ndarray:
    """A mask as the flat indices of its True cells: small to keep when only a few cells are selected."""
    return np.flatnonzero(mask).astype(np.uint32 if mask.size < 2 ** 32 else np.uint64)


def mask_from(indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask.flat[indices] = True
    return mask


# --- making a selection ------------------------------------------------------------------------

def box_mask(voxels: np.ndarray, a: Cell, b: Cell) -> np.ndarray:
    """The solid voxels inside the box spanned by two cells (any order, clipped to the matrix)."""
    depth, height, width = voxels.shape
    lo = [max(min(a[i], b[i]), 0) for i in range(3)]
    hi = [min(max(a[i], b[i]), (width, height, depth)[i] - 1) for i in range(3)]
    mask = np.zeros(voxels.shape, bool)
    if any(l > h for l, h in zip(lo, hi)):
        return mask
    region = (slice(lo[2], hi[2] + 1), slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
    mask[region] = voxels[region] != 0
    return mask


def wand_mask(voxels: np.ndarray, seed: Cell) -> np.ndarray:
    """Every voxel of the seed voxel's color that is connected to it through shared faces. Empty if the seed is empty."""
    x, y, z = seed
    depth, height, width = voxels.shape
    if not (0 <= x < width and 0 <= y < height and 0 <= z < depth) or voxels[z, y, x] == 0:
        return np.zeros(voxels.shape, bool)
    return flood_region_3d(voxels == voxels[z, y, x], seed)


def all_mask(voxels: np.ndarray) -> np.ndarray:
    return voxels != 0


# --- moving ------------------------------------------------------------------------------------

GRAZING = 0.12      # a face seen this edge-on (the ray's part along its normal) is too steep to drag a selection across


def grab_plane(cell: Cell, normal: Cell, direction) -> tuple[int, float]:
    """The plane a selection is dragged in when it is grabbed by a face, as (axis, where that plane cuts the axis).

    The plane is the face that was clicked: grabbing a top face (normal along y) drags in the plane of constant y, so
    the selection slides flat. ``cell`` and ``normal`` are the voxel hit and the face of it that was entered. If
    the face is seen too nearly edge-on to drag across, or the ray began inside the voxel and so entered no face, the plane
    that faces the viewer most directly is used instead, and it passes through the middle of the voxel.
    """
    ray = np.asarray(direction, float)
    length = float(np.linalg.norm(ray)) or 1.0
    axis = next((a for a in range(3) if normal[a]), None)
    if axis is not None and abs(ray[axis]) / length >= GRAZING:
        return axis, float(cell[axis] + (1 if normal[axis] > 0 else 0))
    facing = int(np.argmax(np.abs(ray)))
    return facing, cell[facing] + 0.5


def clamp_offset(mask: np.ndarray, offset: Cell) -> Cell:
    """The offset limited so that the selection stays inside the matrix: voxels are never pushed out of it."""
    box = bounds(mask)
    if box is None:
        return (0, 0, 0)
    lo, hi = box
    size = mask.shape[::-1]
    return tuple(int(min(max(offset[i], -lo[i]), size[i] - 1 - hi[i])) for i in range(3))


def _lift(voxels: np.ndarray, mask: np.ndarray):
    """(voxels without the selected ones, their cells as (z, y, x) arrays, their palette indices)."""
    chosen = solid(mask, voxels)
    zs, ys, xs = np.nonzero(chosen)
    values = voxels[zs, ys, xs]
    rest = voxels.copy()
    rest[chosen] = 0
    return rest, (zs, ys, xs), values


def move(voxels: np.ndarray, mask: np.ndarray, offset: Cell,
         under: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, Cell]:
    """Move the selected voxels by ``offset``. Returns (new voxels, new mask, the offset really used).

    The offset is clamped so the selection stays inside the matrix. Moved voxels replace what they land on, unless
    ``under`` is given: then they float over it, and what they covered before is back where they are no longer.
    """
    dx, dy, dz = clamp_offset(solid(mask, voxels), offset)
    rest, (zs, ys, xs), values = _lift(voxels, mask)
    if under is not None:
        rest = under.copy()
    zs, ys, xs = zs + dz, ys + dy, xs + dx
    rest[zs, ys, xs] = values
    moved = np.zeros(voxels.shape, bool)
    moved[zs, ys, xs] = True
    return rest, moved, (dx, dy, dz)


# --- flipping and rotating ---------------------------------------------------------------------

def array_axis(axis: int) -> int:
    """The numpy axis of a cell axis (0 = x, 1 = y, 2 = z) in a (z, y, x) array."""
    return 2 - axis


def flip(voxels: np.ndarray, mask: np.ndarray, axis: int,
         under: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the selected voxels across the middle of their bounding box, along x, y or z (0, 1, 2)."""
    chosen = solid(mask, voxels)
    box = bounds(chosen)
    if box is None:
        return voxels.copy(), np.zeros(voxels.shape, bool)
    (x0, y0, z0), (x1, y1, z1) = box
    region = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
    inside = chosen[region]
    values = np.where(inside, voxels[region], 0).astype(voxels.dtype)
    flipped_inside = np.flip(inside, array_axis(axis))
    flipped_values = np.flip(values, array_axis(axis))
    new = voxels.copy() if under is None else under.copy()
    view = new[region]
    if under is None:
        view[inside] = 0
    view[flipped_inside] = flipped_values[flipped_inside]
    new_mask = np.zeros(voxels.shape, bool)
    new_mask[region] = flipped_inside
    return new, new_mask


# The two cell axes a turn about each axis moves between: a quarter turn takes +u to +v (a right-handed turn).
TURN_PLANE = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


def rotate(voxels: np.ndarray, mask: np.ndarray, axis: int, quarter_turns: int = 1, under: np.ndarray | None = None):
    """Turn the selected voxels by quarter turns about an axis through the middle of their bounding box.

    ``axis`` is 0, 1 or 2 for x, y or z. One quarter turn about x takes +y toward +z, about y takes +z toward +x
    and about z takes +x toward +y; a negative count turns the other way. When the box has an odd size on one
    of the two axes that swap and an even size on the other, the turned block cannot be centered exactly and ends
    up half a cell to the low side. The block is moved just enough to stay inside the matrix.

    Returns (new voxels, new mask), or None if the turned block would not fit in the matrix at all.
    """
    chosen = solid(mask, voxels)
    box = bounds(chosen)
    if box is None:
        return voxels.copy(), np.zeros(voxels.shape, bool)
    (x0, y0, z0), (x1, y1, z1) = box
    region = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
    inside = chosen[region]
    values = np.where(inside, voxels[region], 0).astype(voxels.dtype)
    u, v = TURN_PLANE[axis]
    axes = (array_axis(u), array_axis(v))
    turned_inside = np.rot90(inside, quarter_turns, axes)
    turned_values = np.rot90(values, quarter_turns, axes)

    old_dims = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
    new_dims = turned_inside.shape[::-1]
    size = voxels.shape[::-1]
    if any(n > s for n, s in zip(new_dims, size)):
        return None
    low = [min(max(lo + (od - nd) // 2, 0), s - nd)
           for lo, od, nd, s in zip((x0, y0, z0), old_dims, new_dims, size)]
    target = (slice(low[2], low[2] + new_dims[2]), slice(low[1], low[1] + new_dims[1]), slice(low[0], low[0] + new_dims[0]))

    new = voxels.copy()
    new[chosen] = 0                                          # lift the selection out...
    if under is not None:
        new = under.copy()
    view = new[target]                                       # ...and put the turned copy down
    view[turned_inside] = turned_values[turned_inside]
    new_mask = np.zeros(voxels.shape, bool)
    new_mask[target] = turned_inside
    return new, new_mask


def delete(voxels: np.ndarray, mask: np.ndarray, under: np.ndarray | None = None) -> np.ndarray:
    """The voxels with the selected ones removed (and what they were covering showing again, given ``under``)."""
    if under is not None:
        return under.copy()
    new = voxels.copy()
    new[solid(mask, voxels)] = 0
    return new


# --- copying and pasting -----------------------------------------------------------------------

@dataclass
class Clip:
    """A copied block of voxels: a color for every cell of its bounding box, with alpha 0 for the empty ones.

    Colors are kept as RGBA and not as palette numbers, so a block can be pasted into any model.
    """
    rgba: np.ndarray                  # (z, y, x, 4) uint8

    @property
    def size(self) -> Cell:
        depth, height, width = self.rgba.shape[:3]
        return (width, height, depth)

    @property
    def solid(self) -> np.ndarray:
        return self.rgba[..., 3] != 0

    @property
    def count(self) -> int:
        return int(self.solid.sum())


def extract(voxels: np.ndarray, mask: np.ndarray, palette: np.ndarray) -> Clip | None:
    """Copy the selected voxels (with their colors) out of a matrix. None if nothing is selected."""
    chosen = solid(mask, voxels)
    box = bounds(chosen)
    if box is None:
        return None
    (x0, y0, z0), (x1, y1, z1) = box
    region = (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))
    inside = chosen[region]
    indices = voxels[region]
    rgba = np.zeros(inside.shape + (4,), np.uint8)
    rgba[inside] = palette[indices[inside].astype(np.intp) - 1]
    rgba[inside, 3] = 255
    return Clip(rgba)


def fit_origin(origin: Cell, block: Cell, size: Cell) -> Cell:
    """Where to put a block's low corner so it lies inside a matrix: moved in if it fits, else cut off at the low edge."""
    return tuple(int(min(max(o, 0), s - b)) if b <= s else 0 for o, b, s in zip(origin, block, size))


def origin_against_surface(cell: Cell, axis: int, side: int, block: Cell) -> Cell:
    """The low corner for a block set down on a surface, centered on ``cell`` across the surface.

    ``cell`` is the empty cell next to the surface, ``axis`` the axis the surface faces along and ``side`` which face
    of that cell touches it (-1 the low face, so the block rises away from the surface along +axis, and +1 the high
    face, so it extends along -axis). Along the other two axes the block is centered on the cell.
    """
    origin = [cell[i] - block[i] // 2 for i in range(3)]
    origin[axis] = cell[axis] if side < 0 else cell[axis] - (block[axis] - 1)
    return tuple(origin)


def paste(voxels: np.ndarray, clip: Clip, origin: Cell, index_for_color) -> tuple[np.ndarray, np.ndarray, int]:
    """Put a block into a matrix with its low corner at ``origin``.

    ``index_for_color(rgba)`` gives the palette index for a color (adding it if needed). Voxels of the block that
    fall outside the matrix are dropped. Returns (new voxels, mask of the pasted voxels, how many were dropped).
    """
    zs, ys, xs = np.nonzero(clip.solid)
    colors = clip.rgba[zs, ys, xs]
    unique, inverse = np.unique(colors, axis=0, return_inverse=True)
    palette_indices = np.array([index_for_color(tuple(int(c) for c in color)) for color in unique], voxels.dtype)
    values = palette_indices[inverse.reshape(-1)]
    dx, dy, dz = origin
    zs, ys, xs = zs + dz, ys + dy, xs + dx
    depth, height, width = voxels.shape
    inside = (zs >= 0) & (zs < depth) & (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    new = voxels.copy()
    new[zs[inside], ys[inside], xs[inside]] = values[inside]
    pasted = np.zeros(voxels.shape, bool)
    pasted[zs[inside], ys[inside], xs[inside]] = True
    return new, pasted, int((~inside).sum())


# --- the selection of the active matrix, shared by the viewer and the undo entries ------------------

class VoxelSelection:
    """Which voxels of which matrix are selected. One per open model; undo entries hold on to it so that
    undoing a move also puts the selection back."""

    def __init__(self) -> None:
        self.matrix = None
        self.mask: np.ndarray | None = None
        self.version = 0                 # goes up on every change, so a drawing of the selection knows when to refresh
        self.under: np.ndarray | None = None      # what is beneath a selection that is floating (see the module notes)
        self.under_matrix = None

    def set(self, matrix, mask: np.ndarray | None, keep_under: bool = False) -> None:
        """Select these voxels. A new selection ends any floating; ``keep_under`` is for the same selection having moved."""
        if mask is None or not mask.any():
            self.clear()
            return
        self.matrix, self.mask = matrix, mask
        if not keep_under:
            self.under = self.under_matrix = None
        self.version += 1

    def clear(self) -> None:
        self.under = self.under_matrix = None
        if self.matrix is not None:
            self.matrix, self.mask = None, None
            self.version += 1

    def start_floating_over(self, matrix, under: np.ndarray) -> None:
        """Make ``under`` what the current selection floats over (a paste floats over what it was pasted onto)."""
        self.under, self.under_matrix = under, matrix

    def under_for(self, matrix, voxels: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """What the selection floats over, beginning to float it if it is not floating yet.

        A floating selection is only trusted while the grid is still exactly what it was left as: the saved voxels
        underneath with the selected ones on top. If anything else has changed the grid since (another tool drew on
        it, say) the selection starts afresh and what it covered is given up.
        """
        chosen = solid(mask, voxels)
        if self.under is not None and self.under_matrix is matrix and self.under.shape == voxels.shape:
            expected = self.under.copy()
            expected[chosen] = voxels[chosen]
            if np.array_equal(expected, voxels):
                return self.under
        rest = voxels.copy()
        rest[chosen] = 0
        self.under, self.under_matrix = rest, matrix
        return rest

    def mask_for(self, matrix) -> np.ndarray | None:
        """The selection if it belongs to this matrix and still fits it (a resize makes it stale), else None."""
        if self.matrix is matrix and self.mask is not None and self.mask.shape == matrix.voxels.shape:
            return self.mask
        return None
