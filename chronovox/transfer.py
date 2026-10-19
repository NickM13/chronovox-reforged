"""Moving matrices and colors from one model to another (between tabs). No GL or UI imports."""
from __future__ import annotations

import numpy as np

from .matrix_tools import group_anchor
from .model import MAX_MATRICES, MAX_PALETTE, POSITION_SCALE, Matrix, Model
from .naming import variant_name


def _missing_colors(source: Model, target: Model, indices: np.ndarray) -> list[tuple]:
    """The colors of these 1-based source palette indices that the target does not have yet, each once."""
    wanted, seen = [], set()
    for index in indices:
        rgba = tuple(int(c) for c in source.palette[int(index) - 1])
        if rgba not in seen and not target.find_color(rgba):
            seen.add(rgba)
            wanted.append(rgba)
    return wanted


def copy_matrix(source: Model, matrix: Matrix, target: Model) -> tuple[Matrix, int]:
    """Add a copy of ``matrix`` (which belongs to ``source``) to the end of ``target``. Returns (copy, index).

    Every voxel keeps its color: colors the target palette lacks are appended to it, the rest are matched
    by value. Position, pivot, rotation and resolution are kept. The name is kept if it is free in the
    target, otherwise it gets a number (keeping the game suffix). Raises ValueError, changing nothing,
    if the target is full of matrices or colors.
    """
    if len(target.matrices) >= MAX_MATRICES:
        raise ValueError(f"a model can hold at most {MAX_MATRICES} matrices")
    used = np.unique(matrix.voxels)
    used = used[used > 0]
    if len(target.palette) + len(_missing_colors(source, target, used)) > MAX_PALETTE:
        raise ValueError(f"the palette would be over {MAX_PALETTE} colors")
    lookup = np.zeros(len(source.palette) + 1, np.uint16)       # source index -> target index; 0 stays empty
    for index in used:
        lookup[index] = target.index_for_color(tuple(int(c) for c in source.palette[int(index) - 1]))
    taken = {m.name for m in target.matrices}
    name = matrix.name if matrix.name not in taken else variant_name(matrix.name, taken)
    copy = Matrix(name, lookup[matrix.voxels], matrix.position, matrix.pivot, matrix.rotation, matrix.resolution)
    target.matrices.append(copy)
    return copy, len(target.matrices) - 1


def import_colors(source: Model, target: Model, used_only: bool = True) -> int:
    """Append the source's colors that the target lacks (matched by value) and return how many were added.

    With ``used_only`` the colors no voxel in the source uses are left out. Stops at a full palette
    and raises ValueError only if nothing at all could be added because of it.
    """
    mask = source.used_palette_mask() if used_only else np.ones(len(source.palette), bool)
    indices = np.flatnonzero(mask) + 1
    wanted = _missing_colors(source, target, indices)
    room = MAX_PALETTE - len(target.palette)
    if wanted and room <= 0:
        raise ValueError(f"palette is full ({MAX_PALETTE} colors)")
    for rgba in wanted[:room]:
        target.index_for_color(rgba)
    return min(len(wanted), room)


def group_shift(matrices, focus) -> tuple[int, int, int]:
    """How far to move all of these matrices, in 1/128 units, to put their common center on ``focus``.

    The center is the average of the matrices' centers (see ``group_anchor``) and ``focus`` is a point in model
    space, in units. The move is a whole number of units, as for a new matrix, so pasted matrices stay on the grid
    and keep their places relative to each other.
    """
    center = group_anchor(matrices)
    return tuple(int(round(f - c)) * POSITION_SCALE for f, c in zip(focus, center))


def centered_position(matrix: Matrix, focus) -> tuple[int, int, int]:
    """The position that puts the middle of one matrix on ``focus`` (see ``group_shift``)."""
    return tuple(p + d for p, d in zip(matrix.position, group_shift([matrix], focus)))
