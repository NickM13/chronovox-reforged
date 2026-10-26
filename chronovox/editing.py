"""Voxel editing tools: ray picking and edits. No GL, UI or file imports.

Cells are (x, y, z) integers; ``matrix.voxels`` is indexed [z, y, x].
Rays are given in the matrix's local space, where the grid spans [0, size].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .model import Matrix, Model

Cell = tuple[int, int, int]


class Tool(Enum):
    PLACE = "Place"
    ERASE = "Erase"
    PAINT = "Replace"
    PICK = "Pick color"
    SELECT = "Select"
    FILL_APPEND = "Fill: append"
    FILL_ERASE = "Fill: erase"
    FILL_REPLACE = "Fill: replace"


FILL_TOOLS = (Tool.FILL_APPEND, Tool.FILL_ERASE, Tool.FILL_REPLACE)


class Brush(Enum):
    """How an action is applied: to single voxels, to a box, or by flood fill."""
    VOXEL = "Voxel"
    BOX = "Box"
    FILL = "Fill"


# What you choose is an action (what happens) and a brush (how it is applied). Pick takes no brush, and for
# Select the Voxel and Box brushes both drag out a box, while Fill selects a connected region.
ACTIONS = (Tool.PLACE, Tool.ERASE, Tool.PAINT, Tool.PICK, Tool.SELECT)
BRUSHES = (Brush.VOXEL, Brush.BOX, Brush.FILL)

# The fill brush does a different job for each action: Place appends a layer against a face,
# Erase deletes a connected region, Paint recolors one.
FILL_FOR_ACTION = {Tool.PLACE: Tool.FILL_APPEND, Tool.ERASE: Tool.FILL_ERASE, Tool.PAINT: Tool.FILL_REPLACE}


def effective_tool(action: Tool, brush: Brush) -> Tool:
    """The tool that does the work for an action drawn with a brush.

    Voxel and Box brushes use the action's own tool (a box is built from the same edits), the Fill brush
    maps to the fill tool for that action, and Pick ignores the brush.
    """
    if brush is Brush.FILL and action in FILL_FOR_ACTION:
        return FILL_FOR_ACTION[action]
    return action


class VoxelTool(Enum):
    """What the toolbar offers in Voxel mode: one button each, whatever action and brush it is made of inside."""
    PLACE = "Place"
    ERASE = "Erase"
    PAINT = "Replace"
    PICK = "Pick color"
    SELECT_BOX = "Box select"
    SELECT_WAND = "Wand select"
    FILL = "Fill"


# The order of the toolbox.
VOXEL_TOOLS = tuple(VoxelTool)

# (action, brush) behind each tool. Holding Shift turns the single-voxel tools into box tools, so there are no
# box buttons.
TOOL_PARTS = {
    VoxelTool.PLACE: (Tool.PLACE, Brush.VOXEL),
    VoxelTool.ERASE: (Tool.ERASE, Brush.VOXEL),
    VoxelTool.PAINT: (Tool.PAINT, Brush.VOXEL),
    VoxelTool.PICK: (Tool.PICK, Brush.VOXEL),
    VoxelTool.SELECT_BOX: (Tool.SELECT, Brush.VOXEL),
    VoxelTool.SELECT_WAND: (Tool.SELECT, Brush.FILL),
    VoxelTool.FILL: (Tool.PLACE, Brush.FILL),       # append; replace is Tool.PAINT with the same brush
}


def voxel_tool_for(action: Tool, brush: Brush) -> VoxelTool:
    """The toolbar tool an action and brush amount to (a Box brush counts as its Voxel brush)."""
    if action is Tool.PICK:
        return VoxelTool.PICK
    fill = brush is Brush.FILL
    return {
        Tool.PLACE: VoxelTool.FILL if fill else VoxelTool.PLACE,
        Tool.ERASE: VoxelTool.ERASE,                       # erasing a region is done by selecting it and deleting
        Tool.PAINT: VoxelTool.FILL if fill else VoxelTool.PAINT,
        Tool.SELECT: VoxelTool.SELECT_WAND if fill else VoxelTool.SELECT_BOX,
    }[action]


@dataclass(frozen=True)
class Hit:
    cell: Cell     # the solid voxel that was hit
    normal: Cell   # axis-aligned normal of the face that was entered (0,0,0 if the ray began inside)


def _in_bounds(cell, size) -> bool:
    return all(0 <= c < s for c, s in zip(cell, size))


def _slab(size, o, d):
    """Where a ray crosses the grid box [0, size].

    Returns (t_enter, t_exit, enter_axis, enter_sign, exit_axis), or None if the ray
    misses the box or points away from it. ``enter_axis`` is None when the ray
    starts inside the box.
    """
    t_enter, t_exit = -math.inf, math.inf
    enter_axis, enter_sign, exit_axis = None, 0, None
    for a in range(3):
        if abs(d[a]) < 1e-12:
            if not 0 <= o[a] <= size[a]:
                return None
            continue
        t1, t2 = (0 - o[a]) / d[a], (size[a] - o[a]) / d[a]
        near, far = min(t1, t2), max(t1, t2)
        if near > t_enter:
            t_enter, enter_axis, enter_sign = near, a, (-1 if d[a] > 0 else 1)
        if far < t_exit:
            t_exit, exit_axis = far, a
    if t_enter > t_exit or t_exit < 0:
        return None
    if t_enter < 0:
        t_enter, enter_axis = 0.0, None
    return t_enter, t_exit, enter_axis, enter_sign, exit_axis


def raycast(voxels: np.ndarray, origin, direction) -> Hit | None:
    """First solid voxel along a ray (Amanatides-Woo grid traversal)."""
    size = voxels.shape[::-1]  # (x, y, z)
    o = [float(v) for v in origin]
    d = [float(v) for v in direction]

    crossing = _slab(size, o, d)
    if crossing is None:
        return None
    t_enter, _, enter_axis, enter_sign, _ = crossing

    cell = [min(max(int(math.floor(o[a] + d[a] * t_enter)), 0), size[a] - 1) for a in range(3)]
    normal = [0, 0, 0]
    if enter_axis is not None:
        normal[enter_axis] = enter_sign

    step = [1 if d[a] >= 0 else -1 for a in range(3)]
    t_max, t_delta = [], []
    for a in range(3):
        if abs(d[a]) < 1e-12:
            t_max.append(math.inf)
            t_delta.append(math.inf)
        else:
            boundary = cell[a] + (1 if step[a] > 0 else 0)
            t_max.append((boundary - o[a]) / d[a])
            t_delta.append(abs(1.0 / d[a]))

    while True:
        if voxels[cell[2], cell[1], cell[0]] != 0:
            return Hit(tuple(cell), tuple(normal))
        a = t_max.index(min(t_max))
        cell[a] += step[a]
        if not 0 <= cell[a] < size[a]:
            return None
        normal = [0, 0, 0]
        normal[a] = -step[a]
        t_max[a] += t_delta[a]


def ray_box_distance(shape_xyz, origin, direction) -> float | None:
    """Distance along the ray (in units of ``direction``) to where it enters the box [0, shape], or None."""
    crossing = _slab(shape_xyz, [float(v) for v in origin], [float(v) for v in direction])
    return None if crossing is None else crossing[0]


def exit_face(shape_xyz, origin, direction) -> tuple[Cell, int, int] | None:
    """Where a ray leaves the grid, as (cell, axis, inward sign), or None if it misses the grid.

    The cell is the one just inside the wall the ray goes out through. ``axis`` is that wall's
    axis (0 = x, 1 = y, 2 = z) and the sign is the direction pointing from the wall into the grid.
    """
    o = [float(v) for v in origin]
    d = [float(v) for v in direction]
    crossing = _slab(shape_xyz, o, d)
    if crossing is None or crossing[4] is None:
        return None
    t_exit, axis = crossing[1], crossing[4]
    cell = [min(max(int(math.floor(o[a] + d[a] * t_exit)), 0), shape_xyz[a] - 1) for a in range(3)]
    cell[axis] = shape_xyz[axis] - 1 if d[axis] > 0 else 0
    return tuple(cell), axis, (-1 if d[axis] > 0 else 1)


def exit_cell(shape_xyz, origin, direction) -> Cell | None:
    """The cell where a ray leaves the grid: a cell on one of the back faces.

    Back faces are the walls, floor or ceiling of the matrix outline that face away
    from the viewer. None if the ray misses the matrix.
    """
    face = exit_face(shape_xyz, origin, direction)
    return face[0] if face else None


def find_target(voxels: np.ndarray, tool: Tool, origin, direction,
                back_face: bool = False) -> Cell | None:
    """The cell a tool would act on for this ray, or None.

    Place targets the empty cell next to the hit face; the other tools target the
    voxel that was hit. When nothing is hit, Place falls back to the back face of
    the matrix, and so do the other tools if ``back_face`` is set (used for boxes,
    whose corners may be empty cells).
    """
    hit = raycast(voxels, origin, direction)
    size = voxels.shape[::-1]
    if hit is None:
        return exit_cell(size, origin, direction) if tool is Tool.PLACE or back_face else None
    if tool is Tool.PLACE:
        cell = tuple(c + n for c, n in zip(hit.cell, hit.normal))
        if cell == hit.cell or not _in_bounds(cell, size):
            return None
        return cell
    return hit.cell


def expansion_for(cell: Cell, size: Cell) -> tuple[Cell, Cell] | None:
    """How a matrix has to grow to take a cell beyond its edge, as (cells to add on the low side, on the high side).

    Each is one cell on the axis the cell lies beyond, and 0 on the others. None if the cell is inside already.
    """
    low = tuple(1 if c < 0 else 0 for c in cell)
    high = tuple(1 if c >= s else 0 for c, s in zip(cell, size))
    return (low, high) if any(low) or any(high) else None


def symmetric_expansion(cell: Cell, size: Cell, mirror: tuple[bool, bool, bool],
                        radial: "Radial | None") -> tuple[Cell, Cell] | None:
    """How a matrix has to grow to take a cell and all its mirrored and turned copies, as (low, high) like expansion_for.

    On the axes the symmetry works across (mirrored ones, and the two a radial turn moves things on) the growth is the
    same on both sides, so the center the copies are made around does not move. None if every copy is inside already.
    """
    cells = symmetric_cells(cell, size, mirror, radial)
    low = [max(0, -min(c[i] for c in cells)) for i in range(3)]
    high = [max(0, max(c[i] for c in cells) - size[i] + 1) for i in range(3)]
    across = [bool(m) for m in mirror]
    if radial is not None and radial[1] > 1:
        across = [a or i != radial[0] for i, a in enumerate(across)]
    for i in range(3):
        if across[i]:
            low[i] = high[i] = max(low[i], high[i])
    return (tuple(low), tuple(high)) if any(low) or any(high) else None


def cell_after_expansion(cell: Cell, low: Cell) -> Cell:
    """Where a cell is numbered once the matrix has grown by ``low`` cells on its low sides (the cells all move up)."""
    return tuple(c + l for c, l in zip(cell, low))


def place_face(voxels: np.ndarray, origin, direction, outside: bool = False) -> tuple[Cell, int, int] | None:
    """The cell Place would fill for this ray and the face the ray is pointing at, as (cell, axis, side).

    The cell is the one ``find_target`` gives for Place. The face is the side of that cell touching the
    voxel that was hit, or lying on the back wall of the matrix when nothing was hit. ``axis`` is 0, 1 or 2
    for x, y or z; ``side`` is -1 for the low face of the cell along that axis and +1 for the high one.
    With ``outside`` a face on the edge of the matrix gives the cell just beyond it (-1 or the size along that axis),
    where the matrix would have to grow; otherwise there is no target there.
    """
    hit = raycast(voxels, origin, direction)
    size = voxels.shape[::-1]
    if hit is None:
        wall = exit_face(size, origin, direction)
        if wall is None:
            return None
        cell, axis, inward = wall
        return cell, axis, -inward                      # the wall is on the side the ray leaves through
    cell = tuple(c + n for c, n in zip(hit.cell, hit.normal))
    if cell == hit.cell or (not outside and not _in_bounds(cell, size)):
        return None
    axis = next(a for a in range(3) if hit.normal[a])
    return cell, axis, -hit.normal[axis]


def mirror_cells(cell: Cell, size: Cell, axes: tuple[bool, bool, bool]) -> list[Cell]:
    """The cell and its reflections across the center planes of the enabled axes (x, y, z). The cell itself comes first.

    With several axes on, every combination is included (2, 4 or 8 cells); cells on a plane are not repeated.
    """
    cells = [tuple(cell)]
    for axis, on in enumerate(axes):
        if on:
            for c in list(cells):
                flipped = c[:axis] + (size[axis] - 1 - c[axis],) + c[axis + 1:]
                if flipped not in cells:
                    cells.append(flipped)
    return cells


def mirror_boxes(a: Cell, b: Cell, size: Cell, axes: tuple[bool, bool, bool]) -> list[tuple[Cell, Cell]]:
    """The box between two corners and its reflections, as corner pairs (the box itself first)."""
    boxes = [(tuple(a), tuple(b))]
    for axis, on in enumerate(axes):
        if on:
            for lo, hi in list(boxes):
                flip = lambda c: c[:axis] + (size[axis] - 1 - c[axis],) + c[axis + 1:]
                other = (flip(lo), flip(hi))
                if other not in boxes and (other[1], other[0]) not in boxes:
                    boxes.append(other)
    return boxes


def mirror_targets(target: "FillTarget", size: Cell, axes: tuple[bool, bool, bool]) -> list["FillTarget"]:
    """A fill target and its reflections. Reflecting across the face's own axis turns the append direction around."""
    targets = [target]
    for axis, on in enumerate(axes):
        if on:
            for t in list(targets):
                cell = t.cell[:axis] + (size[axis] - 1 - t.cell[axis],) + t.cell[axis + 1:]
                sign = -t.sign if t.axis == axis else t.sign
                flipped = FillTarget(cell, t.axis, sign)
                if flipped not in targets:
                    targets.append(flipped)
    return targets


Radial = tuple[int, int]   # (axis the copies turn around, how many copies make a full turn)

_PLANE = ((1, 2), (2, 0), (0, 1))   # the two axes that turn around axis 0, 1, 2


def _turn(angle: float) -> tuple[float, float]:
    """cos and sin of an angle, exact at the quarter turns so they round-trip cleanly."""
    c, s = math.cos(angle), math.sin(angle)
    return (0.0 if abs(c) < 1e-9 else c), (0.0 if abs(s) < 1e-9 else s)


def rotate_cells(cells: np.ndarray, size: Cell, radial: Radial, step: int) -> np.ndarray:
    """Turn (N, 3) cells around the center of the matrix by ``step`` copies' worth, to the nearest cells.

    Cells that land outside the grid are left out of range for the caller to drop.
    """
    axis, count = radial
    u, v = _PLANE[axis]
    cos, sin = _turn(2.0 * math.pi * step / count)
    du = cells[:, u] + 0.5 - size[u] / 2
    dv = cells[:, v] + 0.5 - size[v] / 2
    out = np.array(cells, dtype=np.int64)
    out[:, u] = np.floor(du * cos - dv * sin + size[u] / 2 + 1e-9).astype(np.int64)
    out[:, v] = np.floor(du * sin + dv * cos + size[v] / 2 + 1e-9).astype(np.int64)
    return out


def symmetric_cells(cell: Cell, size: Cell, mirror: tuple[bool, bool, bool], radial: Radial | None) -> list[Cell]:
    """The cell and every copy of it under the radial turns and the mirrors. The cell itself comes first."""
    turned = [tuple(cell)]
    if radial is not None and radial[1] > 1:
        for step in range(1, radial[1]):
            turned.append(tuple(int(c) for c in rotate_cells(np.array([cell]), size, radial, step)[0]))
    cells: list[Cell] = []
    for c in turned:
        for m in mirror_cells(c, size, mirror):
            if m not in cells:
                cells.append(m)
    return cells


def symmetric_targets(target: "FillTarget", size: Cell, mirror: tuple[bool, bool, bool],
                      radial: Radial | None) -> list["FillTarget"]:
    """A fill target and its copies. A turn that is not a quarter turn only keeps a face on the turning axis flat."""
    turned = [target]
    if radial is not None and radial[1] > 1:
        axis, count = radial
        u, v = _PLANE[axis]
        for step in range(1, count):
            cell = tuple(int(c) for c in rotate_cells(np.array([target.cell]), size, radial, step)[0])
            if target.axis == axis:
                turned.append(FillTarget(cell, target.axis, target.sign))
            elif (4 * step) % count == 0:                       # a quarter turn: the face moves to the other axis
                cos, sin = _turn(2.0 * math.pi * step / count)
                normal = [0, 0, 0]
                normal[target.axis] = target.sign
                nu, nv = normal[u], normal[v]
                ru, rv = round(nu * cos - nv * sin), round(nu * sin + nv * cos)
                new_axis = u if ru else v
                turned.append(FillTarget(cell, new_axis, ru or rv))
    targets: list[FillTarget] = []
    for t in turned:
        for m in mirror_targets(t, size, mirror):
            if m not in targets:
                targets.append(m)
    return targets


def symmetric_box_cells(a: Cell, b: Cell, size: Cell, mirror: tuple[bool, bool, bool], radial: Radial) -> np.ndarray:
    """Every in-range cell covered by the box between two corners and its radial turns and mirrors, as (N, 3)."""
    lo = [max(min(a[i], b[i]), 0) for i in range(3)]
    hi = [min(max(a[i], b[i]), size[i] - 1) for i in range(3)]
    if any(l > h for l, h in zip(lo, hi)):
        return np.zeros((0, 3), np.int64)
    grid = np.mgrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].reshape(3, -1).T
    parts = [grid if step == 0 else rotate_cells(grid, size, radial, step) for step in range(radial[1])]
    for axis, on in enumerate(mirror):
        if on:
            flipped = []
            for part in parts:
                f = part.copy()
                f[:, axis] = size[axis] - 1 - f[:, axis]
                flipped.append(f)
            parts += flipped
    cells = np.concatenate(parts)
    inside = np.all((cells >= 0) & (cells < np.array(size)), axis=1)
    return np.unique(cells[inside], axis=0)


def apply_cells(model: Model, matrix: Matrix, tool: Tool, cells: np.ndarray,
                color: tuple[int, int, int, int], only_color: int | None = None) -> bool:
    """Like apply_box, for an arbitrary set of (N, 3) cells. Returns True if the voxel grid changed."""
    if len(cells) == 0:
        return False
    z, y, x = cells[:, 2], cells[:, 1], cells[:, 0]
    current = matrix.voxels[z, y, x]
    if tool is Tool.ERASE:
        new = np.zeros_like(current)
    elif tool is Tool.PLACE:
        new = np.full_like(current, model.index_for_color(color))
    elif tool is Tool.PAINT:
        chosen = (current != 0) if only_color is None else (current == only_color)
        if not chosen.any() or (only_color is not None and model.find_color(color) == only_color):
            return False
        new = np.where(chosen, model.index_for_color(color), current).astype(current.dtype)
    else:
        return False
    changed = bool((new != current).any())
    matrix.voxels[z, y, x] = new
    return changed


def apply_tool(model: Model, matrix: Matrix, tool: Tool, cell: Cell,
               color: tuple[int, int, int, int]) -> bool:
    """Apply a tool at a cell. Returns True if the voxel grid changed."""
    x, y, z = cell
    if not _in_bounds(cell, matrix.size):
        return False
    current = int(matrix.voxels[z, y, x])
    if tool is Tool.ERASE:
        new = 0
    elif tool is Tool.PLACE:
        new = model.index_for_color(color)
    elif tool is Tool.PAINT:
        if current == 0:
            return False
        new = model.index_for_color(color)
    else:
        return False
    if new == current:
        return False
    matrix.voxels[z, y, x] = new
    return True


def apply_box(model: Model, matrix: Matrix, tool: Tool, a: Cell, b: Cell,
              color: tuple[int, int, int, int], only_color: int | None = None) -> bool:
    """Apply a tool to every cell of the box spanned by two corner cells (inclusive).

    The corners can be given in any order and the box is clipped to the grid.
    Place fills the whole box and erase clears it. Paint recolors the existing voxels: with
    ``only_color`` (a palette index) just the ones of that color, so a box started on a voxel replaces
    that voxel's color and leaves the rest alone; without it, every voxel in the box.
    Returns True if the voxel grid changed.
    """
    size = matrix.size
    lo = [max(min(a[i], b[i]), 0) for i in range(3)]
    hi = [min(max(a[i], b[i]), size[i] - 1) for i in range(3)]
    if any(l > h for l, h in zip(lo, hi)):
        return False
    region = matrix.voxels[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]  # a view
    if tool is Tool.ERASE:
        changed = bool(region.any())
        region[...] = 0
    elif tool is Tool.PLACE:
        index = model.index_for_color(color)
        changed = bool((region != index).any())
        region[...] = index
    elif tool is Tool.PAINT:
        mask = (region != 0) if only_color is None else (region == only_color)
        if not mask.any():
            return False
        if only_color is not None and model.find_color(color) == only_color:
            return False                                   # already that color: change nothing, add nothing
        index = model.index_for_color(color)
        changed = bool((region[mask] != index).any())
        region[mask] = index
    else:
        return False
    return changed


# --- flood fills -------------------------------------------------------------------

def flood_region(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """The 4-connected region of True cells of a 2D mask that contains ``seed`` (row, column).

    Works on runs of cells rather than single cells, so large flat regions are cheap.
    Returns an all-False array if the seed cell is False or outside the mask.
    """
    rows, cols = mask.shape
    region = np.zeros(mask.shape, bool)
    r0, c0 = seed
    if not (0 <= r0 < rows and 0 <= c0 < cols) or not mask[r0, c0]:
        return region

    padded = np.zeros((rows, cols + 2), np.int8)
    padded[:, 1:-1] = mask
    change = np.diff(padded, axis=1)                 # +1 where a run starts, -1 just past where it ends
    starts = [np.flatnonzero(change[r] == 1).tolist() for r in range(rows)]
    ends = [np.flatnonzero(change[r] == -1).tolist() for r in range(rows)]

    from bisect import bisect_right
    first = bisect_right(starts[r0], c0) - 1
    seen = {(r0, first)}
    stack = [(r0, first)]
    while stack:
        r, i = stack.pop()
        lo, hi = starts[r][i], ends[r][i]
        region[r, lo:hi] = True
        for nr in (r - 1, r + 1):
            if not 0 <= nr < rows:
                continue
            row_starts, row_ends = starts[nr], ends[nr]
            j = bisect_right(row_ends, lo)           # first run in that row that ends after `lo`
            while j < len(row_starts) and row_starts[j] < hi:
                if (nr, j) not in seen:
                    seen.add((nr, j))
                    stack.append((nr, j))
                j += 1
    return region


def flood_region_3d(mask: np.ndarray, seed: Cell) -> np.ndarray:
    """The face-connected (6-neighbour) region of True voxels of a (z, y, x) mask containing ``seed`` (x, y, z).

    Cells that only touch along an edge or a corner are not connected. Returns an all-False array if the
    seed is False or outside the mask.

    The mask is cut into runs along x, the runs are linked to the overlapping runs in the four
    neighbouring rows, and the region is grown over those links a whole frontier at a time, so a big
    solid region costs a few array operations per step and not a Python loop per voxel.
    """
    depth, height, width = mask.shape
    sx, sy, sz = seed
    region = np.zeros(mask.shape, bool)
    if not (0 <= sx < width and 0 <= sy < height and 0 <= sz < depth) or not mask[sz, sy, sx]:
        return region

    padded = np.zeros((depth, height, width + 2), np.int8)
    padded[:, :, 1:-1] = mask
    change = np.diff(padded, axis=2)                      # +1 where a run starts, -1 just past where it ends
    zs, ys, xs = np.nonzero(change == 1)                  # run starts, in (z, y, x) order
    xe = np.nonzero(change == -1)[2]                      # run ends (exclusive), in the same order
    count = len(xs)
    big = width + 2
    row = zs.astype(np.int64) * height + ys               # which (z, y) row each run is in
    by_start = row * big + xs                             # sorted: rows in order, runs in order within a row
    by_end = row * big + xe

    # Runs touching a run in the next row up in y, or in the next row up in z, by overlapping in x.
    links_a, links_b = [], []
    for step, valid in ((1, ys < height - 1), (height, zs < depth - 1)):
        runs = np.flatnonzero(valid)
        other = row[runs] + step
        first = np.searchsorted(by_end, other * big + xs[runs], side="right")     # first run ending after we start
        last = np.searchsorted(by_start, other * big + xe[runs], side="left")     # past the last starting before we end
        counts = np.maximum(last - first, 0)
        total = int(counts.sum())
        if total:
            links_a.append(np.repeat(runs, counts))
            links_b.append(np.repeat(first - (np.cumsum(counts) - counts), counts) + np.arange(total))
    if links_a:
        a, b = np.concatenate(links_a), np.concatenate(links_b)
        src, dst = np.concatenate((a, b)), np.concatenate((b, a))            # links work in both directions
        order = np.argsort(src, kind="stable")
        neighbours = dst[order]
        pointer = np.concatenate(([0], np.cumsum(np.bincount(src, minlength=count))))
    else:
        neighbours, pointer = np.zeros(0, np.int64), np.zeros(count + 1, np.int64)

    start = int(np.searchsorted(by_start, (sz * height + sy) * big + sx, side="right")) - 1   # the run holding the seed
    reached = np.zeros(count, bool)
    reached[start] = True
    frontier = np.array([start])
    while frontier.size:
        counts = pointer[frontier + 1] - pointer[frontier]
        total = int(counts.sum())
        if not total:
            break
        offsets = np.repeat(pointer[frontier] - (np.cumsum(counts) - counts), counts) + np.arange(total)
        found = neighbours[offsets]
        found = np.unique(found[~reached[found]])
        reached[found] = True
        frontier = found

    taken = np.flatnonzero(reached)
    paint = np.zeros((depth, height, width + 1), np.int8)
    paint[zs[taken], ys[taken], xs[taken]] = 1
    paint[zs[taken], ys[taken], xe[taken]] = -1
    return np.cumsum(paint, axis=2, dtype=np.int8)[:, :, :-1] > 0


@dataclass(frozen=True)
class FillTarget:
    cell: Cell   # append: the empty cell to start from; erase and replace: the voxel that was clicked
    axis: int    # axis of the clicked face (0 = x, 1 = y, 2 = z); append stays in the plane perpendicular to it
    sign: int    # append: direction from the solid backing toward the empty layer (+1 or -1)


def pick_fill(voxels: np.ndarray, tool: Tool, origin, direction, outside: bool = False) -> FillTarget | None:
    """What a fill tool would act on for this ray, or None.

    Append starts from the empty cell in front of the face that was hit (or, if nothing is
    hit, from the cell at the back wall of the matrix). Erase and replace need a voxel to be hit. With ``outside``,
    append can also start from a cell just beyond the matrix, where the matrix would have to grow.
    """
    hit = raycast(voxels, origin, direction)
    size = voxels.shape[::-1]
    if tool is Tool.FILL_APPEND:
        if hit is None:
            face = exit_face(size, origin, direction)
            return FillTarget(*face) if face else None
        if hit.normal == (0, 0, 0):
            return None
        cell = tuple(c + n for c, n in zip(hit.cell, hit.normal))
        if not outside and not _in_bounds(cell, size):
            return None
        axis = _normal_axis(hit.normal)
        return FillTarget(cell, axis, hit.normal[axis])
    if hit is None or hit.normal == (0, 0, 0):
        return None
    axis = _normal_axis(hit.normal)
    return FillTarget(hit.cell, axis, hit.normal[axis])


def _normal_axis(normal: Cell) -> int:
    return next(a for a, n in enumerate(normal) if n)


def _layer(voxels: np.ndarray, axis: int, layer: int) -> np.ndarray:
    """A writable 2D view of one layer: all cells whose coordinate on ``axis`` (0=x, 1=y, 2=z) is ``layer``."""
    return np.moveaxis(voxels, 2 - axis, 0)[layer]


def _in_layer(cell: Cell, axis: int) -> tuple[int, int]:
    """The (row, column) of a cell inside its layer, matching ``_layer``."""
    index = (cell[2], cell[1], cell[0])              # array order: z, y, x
    return tuple(index[i] for i in range(3) if i != 2 - axis)


MAX_TOLERANCE = 100       # percent of the distance between black and white


def similar_mask(voxels: np.ndarray, palette: np.ndarray, index: int, tolerance: float = 0.0) -> np.ndarray:
    """Which voxels have the palette color ``index`` (1-based) or one within ``tolerance`` percent of it.

    The distance is the straight distance between the RGB values, as a percentage of the distance from black to
    white. At 0 only the color itself matches. Empty cells never match.
    """
    if tolerance <= 0 or not len(palette):
        return voxels == index
    rgb = palette[:, :3].astype(float)
    distance = np.linalg.norm(rgb - rgb[index - 1], axis=1) / (255.0 * math.sqrt(3.0)) * 100.0
    allowed = np.concatenate(([False], distance <= tolerance + 1e-9))
    return allowed[np.minimum(voxels, len(allowed) - 1)]


def apply_fill(model: Model, matrix: Matrix, tool: Tool, target: FillTarget,
               color: tuple[int, int, int, int], contiguous: bool = True, tolerance: float = 0.0) -> int:
    """Flood fill a matrix. Returns how many voxels changed (0 if none).

    * Append: within one layer, the empty cells next to the clicked face that have a solid voxel (of
      any color) behind them, connected to the start cell. A back wall of the matrix counts as solid.
    * Erase: every voxel of the clicked color connected to the clicked voxel in 3D (sharing faces), removed.
    * Replace: the same 3D region, recolored. With ``tolerance`` colors close to the clicked one count too, and
      without ``contiguous`` it is every such voxel in the matrix, not only the ones connected to the clicked voxel.
    """
    voxels = matrix.voxels
    axis = target.axis
    size = matrix.size[axis]
    layer = target.cell[axis]
    if not _in_bounds(target.cell, matrix.size) or tool not in FILL_TOOLS:
        return 0
    plane = _layer(voxels, axis, layer)
    seed = _in_layer(target.cell, axis)

    if tool is Tool.FILL_APPEND:
        backing = layer - target.sign
        solid_behind = (_layer(voxels, axis, backing) != 0) if 0 <= backing < size else np.ones(plane.shape, bool)
        region = flood_region((plane == 0) & solid_behind, seed)
        if not region.any():
            return 0
        plane[region] = model.index_for_color(color)
        return int(region.sum())

    x, y, z = target.cell
    clicked = int(voxels[z, y, x])
    if clicked == 0:
        return 0
    if tool is Tool.FILL_REPLACE:
        if tolerance <= 0 and model.find_color(color) == clicked:
            return 0                                   # already that color: change nothing, add nothing
        matches = similar_mask(voxels, model.palette, clicked, tolerance)
        region = flood_region_3d(matches, target.cell) if contiguous else matches
        new = model.index_for_color(color)
        region = region & (voxels != new)
        voxels[region] = new
        return int(region.sum())
    region = flood_region_3d(voxels == clicked, target.cell)
    voxels[region] = 0
    return int(region.sum())
