"""Moving, rotating and resizing matrices, each returning an undo entry. Pure numpy: no GL, UI or file imports."""
from __future__ import annotations

import numpy as np

from . import transform
from .selection import TURN_PLANE, array_axis
from .history import GroupEdit, MergeEdit, ReorientEdit, ResizeEdit, TransformEdit
from .model import (INT32_MAX, INT32_MIN, MAX_MATRIX_SIZE, MAX_MATRIX_VOXELS, MAX_RESOLUTION, POSITION_SCALE, Matrix,
                    Model)

Triple = tuple[int, int, int]


def resized_origin(matrix: Matrix, low: Triple) -> tuple[Triple, Triple]:
    """Position and pivot after adding ``low`` cells on the minimum side of each axis.

    The matrix formula is ``position + R f (p - pivot) + f pivot`` (f being the cell size).
    Adding n cells on the low side renumbers every cell p -> p + n, so keeping each voxel where it
    was needs pivot + n (the pivot stays on the same physical spot) and position - f n.
    """
    position = tuple(p - lo * matrix.cell_size_128 for p, lo in zip(matrix.position, low))
    pivot = tuple(p + lo * POSITION_SCALE for p, lo in zip(matrix.pivot, low))
    return position, pivot


def plan_resize(matrix: Matrix, low: Triple, high: Triple) -> tuple[np.ndarray, Triple, Triple]:
    """Plan growing or cropping a matrix, without changing it.

    ``low`` and ``high`` are cell counts (x, y, z) added to the minimum and maximum side of
    each axis; negative values crop. Existing voxels stay where they are in the model:
    the position and pivot are adjusted for cells added or removed on the low side, which
    moves the origin corner. Returns (new voxels, new position, new pivot).
    Raises ValueError if the result would be empty, too big, or out of range.
    """
    old_x, old_y, old_z = matrix.size
    new_size = tuple(s + lo + hi for s, lo, hi in zip((old_x, old_y, old_z), low, high))
    if any(n < 1 for n in new_size):
        raise ValueError("a matrix needs at least one cell on every axis")
    if any(n > MAX_MATRIX_SIZE for n in new_size):
        raise ValueError(f"a matrix can be at most {MAX_MATRIX_SIZE} cells on an axis")
    if new_size[0] * new_size[1] * new_size[2] > MAX_MATRIX_VOXELS:
        raise ValueError(f"a matrix can hold at most {MAX_MATRIX_VOXELS:,} voxels")

    position, pivot = resized_origin(matrix, low)
    if any(not INT32_MIN <= v <= INT32_MAX for v in position + pivot):
        raise ValueError("position out of range")

    # Axis order of the voxel array is (z, y, x); `old` and `new` are the overlapping ranges.
    old_slices, new_slices = [], []
    for axis in (2, 1, 0):  # z, y, x as indices into low/high/size
        lo, size, new = low[axis], matrix.size[axis], new_size[axis]
        start = max(0, -lo)                    # first old cell that survives
        stop = min(size, new - lo)             # one past the last old cell that survives
        old_slices.append(slice(start, max(start, stop)))
        new_slices.append(slice(start + lo, max(start, stop) + lo))
    voxels = np.zeros((new_size[2], new_size[1], new_size[0]), matrix.voxels.dtype)
    voxels[tuple(new_slices)] = matrix.voxels[tuple(old_slices)]
    return voxels, position, pivot


def resize_matrix(matrix: Matrix, low: Triple, high: Triple) -> ResizeEdit:
    """Resize a matrix in place and return the undo entry. Raises ValueError, changing nothing, if it cannot."""
    voxels, position, pivot = plan_resize(matrix, low, high)
    edit = ResizeEdit(matrix, matrix.voxels, voxels, (matrix.position, matrix.pivot), (position, pivot))
    edit.redo()
    return edit


def set_transform(matrix: Matrix, position: Triple | None = None, pivot: Triple | None = None,
                  rotation=None, resolution: int | None = None) -> TransformEdit | None:
    """Set any of position (1/128 units), pivot (1/128 cell), rotation (quaternion) and resolution;
    returns the undo entry, or None if nothing changed. Raises ValueError for values out of range."""
    new_position = tuple(int(v) for v in position) if position is not None else matrix.position
    new_pivot = tuple(int(v) for v in pivot) if pivot is not None else matrix.pivot
    new_rotation = transform.normalize(tuple(float(c) for c in rotation)) if rotation is not None else matrix.rotation
    new_scale = matrix.resolution if resolution is None else int(resolution)
    if not 1 <= new_scale <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be 1 to {MAX_RESOLUTION}")
    if any(not INT32_MIN <= v <= INT32_MAX for v in new_position + new_pivot):
        raise ValueError("position out of range")
    old = (matrix.position, matrix.pivot, matrix.rotation, matrix.resolution)
    new = (new_position, new_pivot, new_rotation, new_scale)
    if new == old:
        return None
    edit = TransformEdit(matrix, old, new)
    edit.redo()
    return edit


def move_pivot_in_place(matrix: Matrix, new_pivot: Triple) -> TransformEdit | None:
    """Change the pivot without visibly moving the matrix.

    The model position of local point p is ``position + R (p - pivot) + pivot``, so with a
    rotation a different pivot would shift the whole matrix. The position is compensated:
    ``position' = position + f (I - R)(pivot - pivot')``, rounded to 1/128, so an unrotated matrix
    is exact and a rotated one is within half a step of where it was.
    """
    r = transform.to_matrix(matrix.rotation)
    delta = matrix.cell_size * (np.eye(3) - r) @ (np.asarray(matrix.pivot, float) - np.asarray(new_pivot, float))
    position = tuple(int(round(p + d)) for p, d in zip(matrix.position, delta))
    return set_transform(matrix, position=position, pivot=tuple(int(v) for v in new_pivot))


# --- merging ---------------------------------------------------------------------------------------

ALIGNMENT_TOLERANCE = 0.004      # voxels: less than a 1/128 step (0.0078), so a fractional offset is caught


def _same_rotation(a: Matrix, b: Matrix) -> bool:
    dot = sum(x * y for x, y in zip(transform.normalize(a.rotation), transform.normalize(b.rotation)))
    return 1.0 - abs(dot) < 1e-8             # q and -q are the same rotation


def _local_origin(matrix: Matrix) -> np.ndarray:
    """Where the matrix's local origin sits in the model (position + f pivot - R f pivot, in units)."""
    s = 1.0 / POSITION_SCALE
    pivot = np.asarray(matrix.pivot, float) * s * matrix.cell_size
    return np.asarray(matrix.position, float) * s + pivot - transform.rotate(matrix.rotation, pivot)


def merge_offset(source: Matrix, target: Matrix) -> np.ndarray:
    """Where the source's cell (0, 0, 0) falls in the target's grid, in cells. Needs equal rotations and scales."""
    r = transform.to_matrix(target.rotation)
    return r.T @ (_local_origin(source) - _local_origin(target)) / target.cell_size


def merge_problem(source: Matrix, target: Matrix) -> str | None:
    """Why ``source`` cannot be merged into ``target``, or None if it can."""
    if source is target:
        return "a matrix cannot be merged into itself"
    if source.resolution != target.resolution:
        return (f"{source.name} and {target.name} have different resolutions "
                f"({source.resolution} and {target.resolution}). Give them the same resolution first.")
    if not _same_rotation(source, target):
        return f"{source.name} and {target.name} are rotated differently. Give them the same rotation first."
    offset = merge_offset(source, target)
    if np.abs(offset - np.round(offset)).max() > ALIGNMENT_TOLERANCE:
        x, y, z = (f"{v:.3f}".rstrip("0").rstrip(".") for v in offset)
        return (f"{source.name} is not on {target.name}'s voxel grid (offset {x}, {y}, {z} voxels). "
                "Snap its position to whole voxels first.")
    return None


def plan_merge(source: Matrix, target: Matrix) -> tuple[np.ndarray, Triple, Triple]:
    """Plan merging ``source`` into ``target`` without changing either.

    The target grows just enough to hold the source's occupied cells (its empty margins are ignored),
    keeping its own voxels where they are in the model. Where both have a voxel the source's wins;
    the source's empty cells leave the target alone. Returns (new voxels, new position, new pivot).
    Raises ValueError if they cannot be merged or the result would be too big.
    """
    problem = merge_problem(source, target)
    if problem:
        raise ValueError(problem)
    occupied = np.argwhere(source.voxels != 0)               # rows of (z, y, x)
    if not len(occupied):
        return target.voxels, target.position, target.pivot  # nothing to add
    first, last = occupied.min(axis=0)[::-1], occupied.max(axis=0)[::-1]    # as (x, y, z)
    shift = np.rint(merge_offset(source, target)).astype(int)
    lo, hi = first + shift, last + shift                      # the occupied cells, in the target's cell numbers
    size = np.array(target.size)
    low = tuple(int(v) for v in np.maximum(0, -lo))
    high = tuple(int(v) for v in np.maximum(0, hi - (size - 1)))
    voxels, position, pivot = plan_resize(target, low, high)
    block = source.voxels[first[2]:last[2] + 1, first[1]:last[1] + 1, first[0]:last[0] + 1]
    start = lo + np.array(low)                                # where the block lands in the grown grid
    window = voxels[start[2]:start[2] + block.shape[0], start[1]:start[1] + block.shape[1],
                    start[0]:start[0] + block.shape[2]]
    filled = block != 0
    window[filled] = block[filled]
    return voxels, position, pivot


def merge_matrix(model: Model, source: Matrix, target: Matrix) -> MergeEdit:
    """Merge ``source`` into ``target`` in place, removing the source, and return the undo entry.

    Raises ValueError, changing nothing, if they cannot be merged.
    """
    voxels, position, pivot = plan_merge(source, target)
    edit = MergeEdit(model, target, source, next(i for i, m in enumerate(model.matrices) if m is source),
                     target.voxels, voxels, (target.position, target.pivot), (position, pivot))
    edit.redo()
    return edit


def set_resolution(matrix: Matrix, resolution: int) -> TransformEdit | None:
    """Change a matrix's resolution, scaling it about the world origin.

    Making the cells smaller (or bigger) scales the whole matrix about (0, 0, 0): the position is multiplied by
    the same factor as the cells, so a model made of several matrices that all change resolution keeps its
    shape. The result is rounded to 1/128 unit. Returns the undo entry, or None if the resolution is unchanged.
    """
    if not 1 <= resolution <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be 1 to {MAX_RESOLUTION}")
    factor = (0.5 ** (resolution - 1)) / matrix.cell_size
    position = tuple(int(round(p * factor)) for p in matrix.position)
    return set_transform(matrix, position=position, resolution=resolution)


def set_all_resolutions(model: Model, resolution: int, matrices=None) -> GroupEdit | None:
    """Give every matrix of the model (or just ``matrices``) this resolution, each scaled about the world origin,
    as one undo step.

    Matrices that already have it are left alone. Returns None if there was nothing to change.
    Raises ValueError, changing nothing, for a resolution out of range.
    """
    if not 1 <= resolution <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be 1 to {MAX_RESOLUTION}")
    chosen = model.matrices if matrices is None else matrices
    edits = [e for m in chosen if (e := set_resolution(m, resolution)) is not None]
    if not edits:
        return None
    count = len(edits)
    return GroupEdit(f"resolution {resolution} for {count} matri{'x' if count == 1 else 'ces'}", edits)


# --- scale double and scale half ------------------------------------------------------------------------------

def _scaled(matrix: Matrix, voxels: np.ndarray, pivot: Triple, factor: float) -> ResizeEdit:
    """Swap in new voxels and pivot and return the undo entry.

    The matrix scales about the world origin, like a change of resolution does: the pivot (in cells) scales
    with the cells and the position scales by ``factor`` (rounded to 1/128), so the matrix and everything
    about it moves toward or away from the origin as a whole.
    """
    position = tuple(int(np.floor(p * factor + 0.5)) for p in matrix.position)
    if any(not INT32_MIN <= p <= INT32_MAX for p in position):
        raise ValueError("position out of range")
    edit = ResizeEdit(matrix, matrix.voxels, voxels, (matrix.position, matrix.pivot), (position, pivot))
    edit.redo()
    return edit


def scale_double(matrix: Matrix) -> ResizeEdit:
    """Double a matrix: every cell becomes a 2 x 2 x 2 block of the same color, so each side is twice as long.

    The matrix grows about the world origin, so its position doubles too; the pivot, measured in
    cells, doubles so that it stays at the same place on the matrix (the center stays the center). The
    resolution is not touched, so the matrix really is twice as big. Raises ValueError, changing
    nothing, if the result would be too big.
    """
    sx, sy, sz = matrix.size
    if max(sx, sy, sz) * 2 > MAX_MATRIX_SIZE:
        raise ValueError(f"a matrix can be at most {MAX_MATRIX_SIZE} cells on an axis")
    if sx * sy * sz * 8 > MAX_MATRIX_VOXELS:
        raise ValueError(f"a matrix can hold at most {MAX_MATRIX_VOXELS:,} voxels")
    pivot = tuple(2 * p for p in matrix.pivot)
    if any(not INT32_MIN <= p <= INT32_MAX for p in pivot):
        raise ValueError("pivot out of range")
    voxels = np.repeat(np.repeat(np.repeat(matrix.voxels, 2, axis=0), 2, axis=1), 2, axis=2)
    return _scaled(matrix, voxels, pivot, 2.0)


def scale_half(matrix: Matrix) -> ResizeEdit:
    """Halve a matrix: keep the corner voxel (the one with the lowest x, y and z) of every 2 x 2 x 2 block.

    A side of odd length keeps its last, lone cell, so a side of n cells becomes (n + 1) // 2. The matrix
    shrinks about the world origin, so its position halves, and the pivot halves. This throws information
    away: scaling back up does not restore what was lost (undo does). Raises ValueError if the matrix is already 1 x 1 x 1.
    """
    if matrix.size == (1, 1, 1):
        raise ValueError("the matrix is already as small as it can be")
    voxels = np.ascontiguousarray(matrix.voxels[::2, ::2, ::2])
    pivot = tuple(int(np.floor(p / 2 + 0.5)) for p in matrix.pivot)
    return _scaled(matrix, voxels, pivot, 0.5)


# --- finer or coarser detail at the same size -------------------------------------------------------------------

def refine(matrix: Matrix, steps: int) -> GroupEdit | None:
    """Give a matrix finer (steps > 0) or coarser (steps < 0) detail without changing how big it is.

    Each step toward finer detail doubles the cells on every axis (see ``scale_double``) and makes the resolution one
    higher, so the cells are half as big again: the matrix looks as it did, with eight times the cells. Each step toward
    coarser detail halves the cells (``scale_half``) and lowers the resolution by one. That is the pair of changes a
    resolution change normally needs a scale double or half to go with.

    Returns the undo entry, or None for 0 steps. Raises ValueError, changing nothing, if the resolution would leave 1 to
    ``MAX_RESOLUTION`` or the matrix would get too big on the way.
    """
    if steps == 0:
        return None
    if not 1 <= matrix.resolution + steps <= MAX_RESOLUTION:
        raise ValueError(f"resolution must stay between 1 and {MAX_RESOLUTION}")
    direction = 1 if steps > 0 else -1
    edits = []
    try:
        for _ in range(abs(steps)):
            edits.append((scale_double if direction > 0 else scale_half)(matrix))
            changed = set_resolution(matrix, matrix.resolution + direction)
            if changed is not None:
                edits.append(changed)
    except ValueError:
        for edit in reversed(edits):
            edit.undo()
        raise
    return GroupEdit(f"{'finer' if steps > 0 else 'coarser'} detail of {matrix.name}", edits)


# --- turning and flipping the voxels inside a matrix --------------------------------------------------------------

def _turn_matrix_3x3(axis: int, quarter_turns: int) -> np.ndarray:
    """How cell axes (x, y, z) map under quarter turns about ``axis``: +u goes to +v, with (u, v) from ``TURN_PLANE``."""
    u, v = TURN_PLANE[axis]
    one = np.eye(3)
    one[:, u], one[:, v] = np.eye(3)[:, v], -np.eye(3)[:, u]            # the image of +u is +v, and of +v is -u
    return np.linalg.matrix_power(one, quarter_turns % 4).round().astype(int)


def _reoriented(matrix: Matrix, voxels: np.ndarray, q: np.ndarray, label: str) -> ReorientEdit | None:
    """Replace a matrix's voxels by ``voxels``, which are the old ones moved by ``q`` about the matrix's center.

    ``q`` is a 3 x 3 matrix of signed axis swaps acting on cell coordinates (x, y, z). The matrix's rotation is not
    touched. The matrix stays put as a whole: its center keeps its place in the model, and its pivot stays at the same
    spot on the voxels (it turns or flips with them), so a pivot in the middle stays in the middle. Returns the undo
    entry, or None if nothing changed.
    """
    old_size = np.array(matrix.size, float)
    new_size = np.array(voxels.shape[::-1], float)
    centre, new_centre = old_size / 2, new_size / 2                              # in cells
    s = float(POSITION_SCALE)
    f = matrix.cell_size
    pivot = np.array(matrix.pivot, float) / s                                    # cells
    r = transform.to_matrix(matrix.rotation)
    position = np.array(matrix.position, float) / s                              # units
    world_centre = position + r @ (f * (centre - pivot)) + f * pivot             # where the middle of the matrix is
    new_pivot = new_centre + q @ (pivot - centre)
    new_position = world_centre - r @ (f * (new_centre - new_pivot)) - f * new_pivot
    state = (tuple(int(round(v * s)) for v in new_position), tuple(int(round(v * s)) for v in new_pivot))
    if any(not INT32_MIN <= v <= INT32_MAX for v in state[0] + state[1]):
        raise ValueError("position out of range")
    if np.array_equal(voxels, matrix.voxels) and state == (matrix.position, matrix.pivot):
        return None
    edit = ReorientEdit(matrix, matrix.voxels, np.ascontiguousarray(voxels), (matrix.position, matrix.pivot), state, label)
    edit.redo()
    return edit


def turn_matrix(matrix: Matrix, axis: int, quarter_turns: int = 1) -> ReorientEdit | None:
    """Turn the voxels of a matrix by quarter turns about the x, y or z axis (0, 1, 2) through its center.

    One quarter turn about x takes +y toward +z, about y takes +z toward +x and about z takes +x toward +y; a negative
    count goes the other way. The matrix's own rotation is not changed: its cells are turned, so a turn about x or y or
    z swaps the matrix's sizes on the two axes involved (a 4 x 2 x 6 matrix turned about y becomes 6 x 2 x 4).
    Returns the undo entry, or None for a whole number of full turns.
    """
    turns = quarter_turns % 4
    if turns == 0:
        return None
    u, v = TURN_PLANE[axis]
    voxels = np.rot90(matrix.voxels, turns, (array_axis(u), array_axis(v)))
    return _reoriented(matrix, voxels, _turn_matrix_3x3(axis, turns), f"turn about {'XYZ'[axis]}")


def flip_matrix(matrix: Matrix, axis: int) -> ReorientEdit | None:
    """Mirror the voxels of a matrix along the x, y or z axis (0, 1, 2). The size and the rotation do not change."""
    q = np.eye(3, dtype=int)
    q[axis, axis] = -1
    return _reoriented(matrix, np.flip(matrix.voxels, array_axis(axis)), q, f"flip {'XYZ'[axis]}")
