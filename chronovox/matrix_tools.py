"""Move, resize and rotate handles for the active matrix. Pure numpy: no GL, UI or file imports.

The viewer feeds it the camera eye, the combined view-projection matrix and the mouse; it
decides which handle is under the cursor, turns drags into changes of the matrix, and reports
the line segments to draw. Move and rotate change the matrix live as you drag; resize shows
an outline and applies when you let go. Finishing a drag returns an undo entry.

With several matrices selected, Move moves all of them by the same amount and Rotate swings all of them
about the center of the selection (the average of their centers); Resize still works on the active one.

The Origin tool uses the same handles as Move, but they carry the pivot of the active matrix and leave the
matrix where it is (the position is adjusted to make up for it). It always acts on the active matrix alone.

Movement is fine by default (any 1/128 voxel) and snaps to whole voxels while ``snap`` is
held; rotation snaps to 15 degree steps. Move handles run along the world axes (and a small square between each pair of axes moves in that plane),
rotation rings
turn about the world axes through the pivot, and resize handles sit on the six faces of the
matrix and run along its own (possibly rotated) axes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import transform
from .gizmos import (AXES, AXIS_COLORS, HIGHLIGHT_COLOR, MatrixTool, closest_param_on_line, pick_handle,
                     ray_plane_point, ring_points)
from .editing import raycast, ray_box_distance
from .history import GroupEdit, ResizeEdit, TransformEdit
from .matrix_ops import resize_matrix, resized_origin, set_transform
from .model import POSITION_SCALE, Matrix

HANDLE_SCALE = 0.12        # handle length as a fraction of the distance to the camera
ROTATE_SNAP_DEGREES = 15.0
PIVOT_COLOR = (0.92, 0.92, 0.92, 0.9)

Key = tuple  # ("axis", a) | ("plane", a) | ("ring", a) | ("face", (a, sign)); a plane is named by its normal axis
PLANE_SIZE = 0.3            # side of a move plane's square, as a fraction of the handle length; one corner is the origin
PLANE_FILL_ALPHA = 0.12     # how strongly a plane square is tinted
PLANE_FILL_ALPHA_ACTIVE = 0.32   # ... while the mouse is over it or it is being dragged


def model_point(position, pivot, rotation, local, scale: float = 1.0) -> np.ndarray:
    """Model-space point of a local point: position + R f (p - pivot) + f pivot, f being the cell size."""
    s = 1.0 / POSITION_SCALE
    pivot_v = np.asarray(pivot, float) * s
    return (np.asarray(position, float) * s
            + transform.rotate(rotation, scale * (np.asarray(local, float) - pivot_v)) + scale * pivot_v)


def to_local(matrix: Matrix, point) -> np.ndarray:
    """Inverse of ``model_point`` for a model-space point."""
    s = 1.0 / POSITION_SCALE
    f = matrix.cell_size
    pivot_v = np.asarray(matrix.pivot, float) * s
    r = transform.to_matrix(matrix.rotation)
    return r.T @ (np.asarray(point, float) - np.asarray(matrix.position, float) * s - f * pivot_v) / f + pivot_v


def pivot_world(matrix: Matrix) -> np.ndarray:
    return (np.asarray(matrix.position, float) + matrix.cell_size * np.asarray(matrix.pivot, float)) / POSITION_SCALE


def matrix_center(matrix: Matrix) -> np.ndarray:
    """Model-space center of a matrix."""
    return model_point(matrix.position, matrix.pivot, matrix.rotation, np.array(matrix.size, float) / 2, matrix.cell_size)


def group_anchor(matrices) -> np.ndarray:
    """The center of several matrices: the average of their centers.

    Swinging every matrix about this point leaves it where it is, so the handles do not jump when a rotation ends.
    """
    return np.mean([matrix_center(m) for m in matrices], axis=0)


def rotate_about(matrix: Matrix, state: tuple, anchor, axis, angle: float) -> tuple[tuple, tuple]:
    """Where a matrix ends up, as (position, rotation), when it is turned ``angle`` radians about a world axis
    through ``anchor``. ``state`` is the matrix's (position, pivot, rotation) before the turn.

    The pivot point of the matrix swings about the anchor and the matrix turns by the same amount, so every
    part of it moves as if the matrices were one rigid piece. About its own pivot a matrix only turns.
    """
    position, pivot, rotation = state
    s = POSITION_SCALE
    pivot_offset = matrix.cell_size * np.asarray(pivot, float) / s
    pivot_point = np.asarray(position, float) / s + pivot_offset
    turn = transform.from_axis_angle(axis, angle)
    moved = np.asarray(anchor, float) + transform.rotate(turn, pivot_point - np.asarray(anchor, float))
    new_position = tuple(int(round(v)) for v in (moved - pivot_offset) * s)
    return new_position, transform.normalize(transform.multiply(turn, rotation))


def plane_square(anchor, normal: int, length: float, eye=None) -> np.ndarray:
    """The square of a move plane as a polyline: its outline (the first five points, closed) and then both
    diagonals, which are only there so the whole square can be picked and not just its edges.

    One corner is at ``anchor`` and the square lies along the two axes other than ``normal``, the axis the
    plane is perpendicular to. With a camera position ``eye`` the square is in the quadrant the camera is
    looking into: along each of its two axes it points toward the side of the anchor the camera is on.
    """
    u, v = [a for a in range(3) if a != normal]
    su, sv = (1.0, 1.0) if eye is None else tuple(1.0 if eye[a] >= anchor[a] else -1.0 for a in (u, v))
    corners = []
    for cu, cv in ((0.0, 0.0), (PLANE_SIZE, 0.0), (PLANE_SIZE, PLANE_SIZE), (0.0, PLANE_SIZE)):
        point = np.array(anchor, float)
        point[u] += su * cu * length
        point[v] += sv * cv * length
        corners.append(point)
    c0, c1, c2, c3 = corners
    return np.array([c0, c1, c2, c3, c0, c2, c1, c3])


def axis_world(matrix: Matrix, axis: int) -> np.ndarray:
    return transform.rotate(matrix.rotation, AXES[axis])


@dataclass
class GizmoDrag:
    tool: MatrixTool
    key: Key
    matrix: Matrix
    start_state: tuple          # (position, pivot, rotation)
    anchor: np.ndarray          # world point the handle is attached to at the start
    direction: np.ndarray       # world direction of the handle (axis, ring axis or face axis)
    t0: float = 0.0             # where along the axis the mouse first grabbed it
    v0: np.ndarray | None = None  # rotate: vector from the pivot to the first mouse point
    sign: int = 1               # resize: +1 for a maximum-side face, -1 for a minimum-side face
    cells: int = 0              # resize: cells added (negative: cropped) at that face
    changed: bool = False
    others: list = field(default_factory=list)          # the other selected matrices, moved and rotated along
    other_states: list = field(default_factory=list)    # their (position, pivot, rotation) at the start

    def matrices(self) -> list[Matrix]:
        return [self.matrix, *self.others]


class MatrixTools:
    def __init__(self) -> None:
        self.tool = MatrixTool.MOVE
        self.hover: Key | None = None
        self.drag: GizmoDrag | None = None

    # -- handle geometry ------------------------------------------------------------
    @staticmethod
    def handle_length(anchor, eye) -> float:
        return max(float(np.linalg.norm(np.asarray(anchor, float) - np.asarray(eye, float))) * HANDLE_SCALE, 0.5)

    def _anchor(self, matrix: Matrix, anchor):
        """Where the handles sit: the pivot of the active matrix, or the given point (never for the Origin tool)."""
        if anchor is None or self.tool is MatrixTool.PIVOT:
            return pivot_world(matrix)
        return np.asarray(anchor, float)

    def candidates(self, matrix: Matrix, eye, anchor=None) -> list[tuple[Key, np.ndarray]]:
        """Pickable handles of the current tool as (key, polyline) pairs.

        ``anchor`` is where the move and rotate handles sit; by default the active matrix's pivot.
        """
        anchor = self._anchor(matrix, anchor)
        length = self.handle_length(anchor, eye)
        out = []
        if self.tool in (MatrixTool.MOVE, MatrixTool.PIVOT):
            for a in range(3):
                out.append((("axis", a), np.array([anchor, anchor + np.array(AXES[a]) * length])))
            for a in range(3):                                  # after the axes, which win where they overlap
                out.append((("plane", a), plane_square(anchor, a, length, eye)))
        elif self.tool is MatrixTool.ROTATE:
            for a in range(3):
                out.append((("ring", a), ring_points(anchor, a, length * 0.9)))
        else:
            size = np.array(matrix.size, float)
            for a in range(3):
                for sign in (-1, 1):
                    center = size / 2
                    center[a] = size[a] if sign > 0 else 0.0
                    face = model_point(matrix.position, matrix.pivot, matrix.rotation, center, matrix.cell_size)
                    out.append((("face", (a, sign)), np.array([face, face + axis_world(matrix, a) * sign * length * 0.4])))
        return out

    def update_hover(self, matrix: Matrix, eye, view_proj, cursor, width: float, height: float,
                     anchor=None) -> None:
        if self.drag is None:
            self.hover = pick_handle(self.candidates(matrix, eye, anchor), view_proj, cursor, width, height)

    # -- dragging -------------------------------------------------------------------
    def begin_drag(self, matrix: Matrix, ray, others=(), anchor=None) -> bool:
        """Start dragging the hovered handle. False if there is none or the mouse ray cannot grab it.

        ``others`` are further selected matrices that move and rotate along with ``matrix``; ``anchor`` is where
        the move and rotate handles sit (see ``candidates``).
        """
        key = self.hover
        if key is None:
            return False
        origin, direction = ray
        state = (matrix.position, matrix.pivot, matrix.rotation)
        kind = key[0]
        anchor = self._anchor(matrix, anchor)
        if kind == "axis":
            axis = np.array(AXES[key[1]])
            t0 = closest_param_on_line(origin, direction, anchor, axis)
            if t0 is None:
                return False
            self.drag = GizmoDrag(MatrixTool.MOVE, key, matrix, state, anchor, axis, t0=t0)
        elif kind == "plane":
            axis = np.array(AXES[key[1]])                       # the plane's normal
            point = ray_plane_point(origin, direction, anchor, axis)
            if point is None:
                return False
            self.drag = GizmoDrag(MatrixTool.MOVE, key, matrix, state, anchor, axis, v0=point)
        elif kind == "ring":
            axis = np.array(AXES[key[1]])
            point = ray_plane_point(origin, direction, anchor, axis)
            if point is None or np.linalg.norm(point - anchor) < 1e-6:
                return False
            self.drag = GizmoDrag(MatrixTool.ROTATE, key, matrix, state, anchor, axis, v0=point - anchor)
        else:
            a, sign = key[1]
            size = np.array(matrix.size, float)
            center = size / 2
            center[a] = size[a] if sign > 0 else 0.0
            face = model_point(matrix.position, matrix.pivot, matrix.rotation, center, matrix.cell_size)
            axis = axis_world(matrix, a)
            t0 = closest_param_on_line(origin, direction, face, axis)
            if t0 is None:
                return False
            self.drag = GizmoDrag(MatrixTool.RESIZE, key, matrix, state, face, axis, t0=t0, sign=sign)
        if self.tool is MatrixTool.PIVOT:
            self.drag.tool = MatrixTool.PIVOT
        elif self.drag.tool is not MatrixTool.RESIZE:
            self.drag.others = [o for o in others if o is not matrix]
            self.drag.other_states = [(o.position, o.pivot, o.rotation) for o in self.drag.others]
        return True

    def update_drag(self, ray, snap: bool) -> bool:
        """Follow the mouse. Returns True if the matrix (or the resize proposal) changed."""
        drag = self.drag
        if drag is None:
            return False
        origin, direction = ray
        m = drag.matrix
        if drag.tool in (MatrixTool.MOVE, MatrixTool.PIVOT):
            if drag.key[0] == "plane":
                point = ray_plane_point(origin, direction, drag.anchor, drag.direction)
                if point is None:
                    return False
                offsets = {a: point[a] - drag.v0[a] for a in range(3) if a != drag.key[1]}   # both axes of the plane
            else:
                t = closest_param_on_line(origin, direction, drag.anchor, drag.direction)
                if t is None:
                    return False
                offsets = {drag.key[1]: t - drag.t0}
            if drag.tool is MatrixTool.PIVOT:
                changed = self._drag_pivot(m, drag, offsets, snap)
                drag.changed = drag.changed or changed
                return changed
            position = list(drag.start_state[0])
            grid = POSITION_SCALE * m.cell_size             # snapping goes by one voxel of this matrix, not a whole unit
            for axis, offset in offsets.items():
                value = drag.start_state[0][axis] + offset * POSITION_SCALE
                position[axis] = int(round(round(value / grid) * grid) if snap else round(value))
            changed = tuple(position) != m.position
            m.position = tuple(position)
            moved = [p - p0 for p, p0 in zip(m.position, drag.start_state[0])]  # the others move by the same amount
            for other, state in zip(drag.others, drag.other_states):
                other.position = tuple(p + d for p, d in zip(state[0], moved))
        elif drag.tool is MatrixTool.ROTATE:
            point = ray_plane_point(origin, direction, drag.anchor, drag.direction)
            if point is None:
                return False
            v1 = point - drag.anchor
            angle = math.atan2(float(drag.direction @ np.cross(drag.v0, v1)), float(drag.v0 @ v1))
            if snap:
                step = math.radians(ROTATE_SNAP_DEGREES)
                angle = round(angle / step) * step
            position, rotation = rotate_about(m, drag.start_state, drag.anchor, drag.direction, angle)
            changed = rotation != m.rotation or position != m.position
            m.position, m.rotation = position, rotation
            for other, state in zip(drag.others, drag.other_states):
                other.position, other.rotation = rotate_about(other, state, drag.anchor, drag.direction, angle)
        else:
            t = closest_param_on_line(origin, direction, drag.anchor, drag.direction)
            if t is None:
                return False
            axis = drag.key[1][0]
            cells = int(round((t - drag.t0) * drag.sign / m.cell_size))     # a cell is f units wide
            cells = max(cells, 1 - m.size[axis])          # never crop the matrix away entirely
            changed = cells != drag.cells
            drag.cells = cells
        drag.changed = drag.changed or changed
        return changed

    @staticmethod
    def _drag_pivot(m: Matrix, drag: GizmoDrag, offsets: dict, snap: bool) -> bool:
        """Move the pivot by the world offsets (units, per axis) from where the drag began, keeping the matrix in place.

        A pivot step of d cells along the matrix's own axes moves the pivot point by f R d in the world, so the wanted
        world offset becomes R^T offset / f cells. With ``snap`` the pivot lands on a whole cell.
        """
        position0, pivot0, _ = drag.start_state
        world = np.zeros(3)
        for axis, offset in offsets.items():
            world[axis] = offset
        rotation = transform.to_matrix(m.rotation)
        pivot = np.asarray(pivot0, float) / POSITION_SCALE + rotation.T @ world / m.cell_size   # in cells
        if snap:
            pivot = np.round(pivot)
        new_pivot = tuple(int(round(v * POSITION_SCALE)) for v in pivot)
        delta = POSITION_SCALE * m.cell_size * (np.eye(3) - rotation) @ (
            (np.asarray(pivot0, float) - np.asarray(new_pivot, float)) / POSITION_SCALE)
        new_position = tuple(int(round(p + d)) for p, d in zip(position0, delta))
        changed = new_pivot != m.pivot or new_position != m.position
        m.pivot, m.position = new_pivot, new_position
        return changed

    def end_drag(self, commit: bool) -> tuple[TransformEdit | ResizeEdit | GroupEdit | None, str | None]:
        """Finish a drag. Returns (undo entry or None, error message or None).

        Cancelling puts a moved or rotated matrix back. A committed resize is applied here.
        """
        drag, self.drag = self.drag, None
        if drag is None:
            return None, None
        m = drag.matrix
        if drag.tool is MatrixTool.RESIZE:
            if not commit or drag.cells == 0:
                return None, None
            axis = drag.key[1][0]
            low, high = [0, 0, 0], [0, 0, 0]
            (high if drag.sign > 0 else low)[axis] = drag.cells
            try:
                return resize_matrix(m, tuple(low), tuple(high)), None
            except ValueError as exc:
                return None, str(exc)
        members = list(zip(drag.matrices(), [drag.start_state, *drag.other_states]))
        new_states = [(x.position, x.pivot, x.rotation) for x, _ in members]
        for x, state in members:                       # put everything back, then apply the change as undoable edits
            x.position, x.pivot, x.rotation = state
        if not commit:
            return None, None
        edits = [set_transform(x, *new) for (x, state), new in zip(members, new_states) if new != state]
        if not edits:
            return None, None
        if len(edits) == 1:
            return edits[0], None
        what = "move" if drag.tool is MatrixTool.MOVE else "rotation"
        return GroupEdit(f"{what} of {len(edits)} matrices", edits), None

    def drag_summary(self) -> str:
        drag = self.drag
        if drag is None:
            return ""
        m = drag.matrix
        if drag.tool is MatrixTool.PIVOT:
            return "Origin " + ", ".join(f"{(p - n * POSITION_SCALE / 2) / POSITION_SCALE:.3f}"
                                         for p, n in zip(m.pivot, m.size)) + " from center"
        if drag.tool is MatrixTool.MOVE:
            return "Position " + ", ".join(f"{p / POSITION_SCALE:.4f}" for p in m.position)
        if drag.tool is MatrixTool.ROTATE:
            return "Rotation " + ", ".join(f"{d:.1f}" for d in transform.to_euler(m.rotation)) + " deg"
        axis = "XYZ"[drag.key[1][0]]
        side = "+" if drag.sign > 0 else "-"
        return f"Resize {side}{axis}: {drag.cells:+d} cell(s)"

    # -- drawing --------------------------------------------------------------------
    def fills(self, matrix: Matrix, eye, anchor=None) -> tuple[np.ndarray, np.ndarray]:
        """Translucent triangles (N, 3, 3) and RGBA colors (N, 4) for the move planes; none for the other tools.

        A square is faintly tinted, and a little more while the mouse is over it or it is being dragged.
        """
        if self.tool not in (MatrixTool.MOVE, MatrixTool.RESIZE, MatrixTool.PIVOT):
            return np.zeros((0, 3, 3), np.float32), np.zeros((0, 4), np.float32)
        active = self.drag.key if self.drag else self.hover
        anchor = self._anchor(matrix, anchor)
        length = self.handle_length(anchor, eye)
        triangles, colors = [], []
        if self.tool is MatrixTool.RESIZE:             # the resize handles end in small solid boxes
            for key, points in self.candidates(matrix, eye, anchor):
                if key[0] != "face":
                    continue
                color = (*(HIGHLIGHT_COLOR if key == active else AXIS_COLORS[key[1][0]]), 0.9)
                for quad in _box_faces(_handle_box(points, length)):
                    for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                        triangles.append(tri)
                        colors.append(color)
            return np.array(triangles, np.float32).reshape(-1, 3, 3), np.array(colors, np.float32).reshape(-1, 4)
        for a in range(3):
            c0, c1, c2, c3 = plane_square(anchor, a, length, eye)[:4]
            alpha = PLANE_FILL_ALPHA_ACTIVE if active == ("plane", a) else PLANE_FILL_ALPHA
            for tri in ((c0, c1, c2), (c0, c2, c3)):
                triangles.append(tri)
                colors.append((*AXIS_COLORS[a], alpha))
        return np.array(triangles, np.float32).reshape(-1, 3, 3), np.array(colors, np.float32).reshape(-1, 4)

    def overlay(self, matrix: Matrix, eye, anchor=None) -> tuple[np.ndarray, np.ndarray]:
        """Line segments (N, 2, 3) and RGBA colors (N, 4) for the handles, the pivot and any resize preview."""
        segments: list[tuple] = []
        colors: list[tuple] = []

        def add(a, b, color):
            segments.append((a, b))
            colors.append(color)

        active = self.drag.key if self.drag else self.hover
        anchor = self._anchor(matrix, anchor)
        length = self.handle_length(anchor, eye)

        for key, points in self.candidates(matrix, eye, anchor):
            if key[0] == "plane":
                points = points[:5]                    # the outline only: the diagonals are for picking
            axis = key[1] if key[0] != "face" else key[1][0]
            color = (*(HIGHLIGHT_COLOR if key == active else AXIS_COLORS[axis]), 1.0)
            for a, b in zip(points, points[1:]):
                add(a, b, color)
            if key[0] == "face":                       # a resize handle ends in a small box
                corners = _handle_box(points, length)
                for c0 in range(8):
                    for c1 in range(c0 + 1, 8):
                        if bin(c0 ^ c1).count("1") == 1:
                            add(corners[c0], corners[c1], color)
            elif key[0] not in ("ring", "plane"):     # a small chevron on the tip of a move arrow
                tip, back = points[1], points[0] - points[1]
                direction = back / max(np.linalg.norm(back), 1e-9)
                helper = np.array(AXES[(axis + 1) % 3]) if abs(direction @ np.array(AXES[(axis + 1) % 3])) < 0.9 \
                    else np.array(AXES[(axis + 2) % 3])
                side = np.cross(direction, helper)
                side /= max(np.linalg.norm(side), 1e-9)
                up = np.cross(direction, side)
                for offset in (side, -side, up, -up):
                    add(tip, tip + direction * length * 0.1 + offset * length * 0.05, color)

        for a in range(3):                             # a small cross marking the pivot
            d = np.array(AXES[a]) * length * 0.07
            add(anchor - d, anchor + d, PIVOT_COLOR)

        drag = self.drag
        if drag is not None and drag.tool is MatrixTool.RESIZE and drag.cells != 0:
            axis = drag.key[1][0]
            low, high = [0, 0, 0], [0, 0, 0]
            (high if drag.sign > 0 else low)[axis] = drag.cells
            position, pivot = resized_origin(matrix, tuple(low))
            size = [s + lo + hi for s, lo, hi in zip(matrix.size, low, high)]
            corners = {(x, y, z): model_point(position, pivot, matrix.rotation, (x * size[0], y * size[1], z * size[2]),
                                              matrix.cell_size)
                       for x in (0, 1) for y in (0, 1) for z in (0, 1)}
            for c0 in corners:
                for c1 in corners:
                    if c0 < c1 and sum(a != b for a, b in zip(c0, c1)) == 1:
                        add(corners[c0], corners[c1], (*HIGHLIGHT_COLOR, 1.0))

        return np.array(segments, np.float32).reshape(-1, 2, 3), np.array(colors, np.float32).reshape(-1, 4)


def _handle_box(points, length: float) -> np.ndarray:
    """The 8 corners of the small box on the end of a face handle (corner i has bit 0, 1, 2 for the +side of its axis)."""
    tip, back = points[1], points[0] - points[1]
    direction = back / max(np.linalg.norm(back), 1e-9)
    helper = np.array(AXES[0]) if abs(direction @ np.array(AXES[0])) < 0.9 else np.array(AXES[1])
    side = np.cross(direction, helper)
    side /= max(np.linalg.norm(side), 1e-9)
    up = np.cross(direction, side)
    half = length * 0.045
    return np.array([tip + (sx * direction + sy * side + sz * up) * half
                     for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])


def _box_faces(corners: np.ndarray) -> list:
    """The six faces of ``_handle_box`` corners as quads (corner index = x + 2 y + 4 z)."""
    c = corners
    return [(c[0], c[1], c[3], c[2]), (c[4], c[5], c[7], c[6]), (c[0], c[1], c[5], c[4]),
            (c[2], c[3], c[7], c[6]), (c[0], c[2], c[6], c[4]), (c[1], c[3], c[7], c[5])]


def pick_matrix(matrices, ray) -> int | None:
    """Which matrix a mouse ray points at, as an index into ``matrices``, or None.

    ``matrices`` is a list of Matrix (use None for ones that must be skipped, such as hidden ones).
    A matrix whose voxels are hit wins over one that is only hit inside its empty bounds. The
    nearest voxel hit wins; among empty bounds the smallest matrix wins, so a small empty matrix
    sitting inside a big one can still be clicked.
    """
    origin, direction = np.asarray(ray[0], float), np.asarray(ray[1], float)
    voxel_hits, box_hits = [], []
    for index, m in enumerate(matrices):
        if m is None:
            continue
        o = to_local(m, origin)
        d = transform.to_matrix(m.rotation).T @ direction / m.cell_size    # same ray parameter as in the model
        hit = raycast(m.voxels, o, d)
        if hit is not None:
            voxel_hits.append((ray_box_distance((1, 1, 1), o - np.array(hit.cell), d), index))
        else:
            t = ray_box_distance(m.size, o, d)
            if t is not None:
                box_hits.append((m.size[0] * m.size[1] * m.size[2], t, index))
    if voxel_hits:
        return min(voxel_hits)[1]
    return min(box_hits)[2] if box_hits else None
