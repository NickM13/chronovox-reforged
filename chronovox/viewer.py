"""Viewer window: orbit camera, face-culled rendering, imgui panels.

Window and input via glfw, drawing via moderngl, UI via Dear ImGui (imgui-bundle).
"""
from __future__ import annotations

import math
import shutil
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import glfw
from pyglm import glm
import moderngl
import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from . import csm, cvox, icon, shading, thumbnails
from .camera import Camera
from .gizmos import MatrixTool, Mode
from . import grid as ground_grid
from .editing import (FILL_FOR_ACTION, FILL_TOOLS, TOOL_PARTS, VOXEL_TOOLS, Brush, FillTarget, Tool, VoxelTool, apply_box,
                      apply_fill, apply_tool, cell_after_expansion, effective_tool, symmetric_expansion, find_target,
                      MAX_TOLERANCE, apply_cells, mirror_boxes, pick_fill, symmetric_box_cells, symmetric_cells, symmetric_targets, place_face, raycast, voxel_tool_for)
from .toolbar import ToolButton, icon_button, push_selected_colors, set_images, tool_strip
from .history import (GroupEdit, History, MatrixListEdit, MatrixMoveEdit, MergeEdit, PaletteEdit, ReorientEdit, SelectionEdit,
                      VoxelEdit)
from .mesh import Mesh, build_mesh
from . import transform as xf
from .matrix_ops import (flip_matrix, merge_matrix, move_pivot_in_place, refine, resize_matrix, scale_double, scale_half,
                         set_all_resolutions, set_resolution, set_transform, turn_matrix)
from .matrix_tools import MatrixTools, group_anchor, pick_matrix, to_local
from .model import MAX_RESOLUTION, POSITION_SCALE, Matrix, Model
from .info_bar import fit_text, matrix_summary
from . import explorer, filemanager, trash
from .icons import IconSet
from .recent import add_recent, remove_recent
from .tabs import Tab, tab_label
from . import selection as voxsel
from .workspace_ui import WorkspacePanel
from .transfer import copy_matrix, group_shift, import_colors
from . import __version__
from .preferences import DEFAULT_EXPLORER_WIDTH, Preferences
from .ui_fields import NumberFields, ctrl_down, shift_down

VERTEX_SHADER = """
#version 330
uniform mat4 u_view_proj;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
in vec4 in_color;
in float in_ao;
out vec3 v_normal;
out vec4 v_color;
out float v_ao;
void main() {
    gl_Position = u_view_proj * u_model * vec4(in_pos, 1.0);
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    v_ao = in_ao;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform float u_shade_base;      // brightness by face direction: base + weights . normal, see chronovox.shading
uniform vec3 u_shade_weights;
uniform float u_ao_strength;
in vec3 v_normal;
in vec4 v_color;
in float v_ao;
out vec4 f_color;
void main() {
    float shade = u_shade_base + dot(normalize(v_normal), u_shade_weights);
    float occlusion = mix(1.0, mix(0.35, 1.0, v_ao), u_ao_strength);
    f_color = vec4(v_color.rgb * shade * occlusion, 1.0);
}
"""

FLAT_VERTEX_SHADER = """
#version 330
uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform vec3 u_eye;
uniform float u_pull;      // world-space shift toward the camera, so lines lying on a voxel's surface win the depth test
in vec3 in_pos;
void main() {
    vec3 world = (u_model * vec4(in_pos, 1.0)).xyz;
    vec3 toward = u_eye - world;
    float length_to_eye = length(toward);
    if (length_to_eye > 1e-6) world += toward / length_to_eye * u_pull;
    gl_Position = u_view_proj * vec4(world, 1.0);
}
"""

FLAT_FRAGMENT_SHADER = """
#version 330
uniform vec3 u_color;
uniform float u_alpha;
out vec4 f_color;
void main() { f_color = vec4(u_color, u_alpha); }
"""

GRID_VERTEX_SHADER = """
#version 330
uniform mat4 u_view_proj;
uniform float u_lift;      // world-space shift along y, toward the viewer's side of the grid plane
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
out vec3 v_world;
void main() {
    vec3 world = in_pos + vec3(0.0, u_lift, 0.0);
    gl_Position = u_view_proj * vec4(world, 1.0);
    v_color = in_color;
    v_world = world;
}
"""

GRID_FRAGMENT_SHADER = """
#version 330
uniform vec3 u_center;
uniform float u_radius;
uniform float u_fade;      // 1 = fade out toward the edge of the grid, 0 = no fade
in vec4 v_color;
in vec3 v_world;
out vec4 f_color;
void main() {
    float fade = 1.0;
    if (u_fade > 0.5) fade = 1.0 - smoothstep(0.5 * u_radius, u_radius, distance(v_world.xz, u_center.xz));
    f_color = vec4(v_color.rgb, v_color.a * fade);
}
"""

PREVIEW_DISPLAY_SIZE = 128   # pixels across a model's picture on the home page
HOVER_FACE_COLOR = (0.35, 0.62, 1.0)   # blue: "selected". Green would vanish against green voxels
HOVER_FACE_ALPHA = 0.2        # a faint tint on the face a Place click would build against
HOVER_FACE_EDGE_ALPHA = 0.55

BACKGROUND = (0.16, 0.17, 0.20)
BOUNDS_COLOR = (0.45, 0.50, 0.58)
SELECTED_BOUNDS_COLOR = (0.35, 0.62, 1.0)   # outline of the matrices selected along with the active one
PRIMARY_BOUNDS_COLOR = (1.0, 1.0, 1.0)      # outline of the active (primary) matrix while several are selected
INACTIVE_BOUNDS_ALPHA = 0.35  # outlines of the matrices that are not selected
SELECTION_OUTLINE = (0.55, 0.78, 1.0)
TOOL_COLORS = {Tool.SELECT: SELECTION_OUTLINE, Tool.PLACE: (0.35, 1.0, 0.45), Tool.ERASE: (1.0, 0.35, 0.30),
               Tool.PAINT: (1.0, 0.85, 0.25), Tool.PICK: (0.35, 0.85, 1.0),
               Tool.FILL_APPEND: (0.35, 1.0, 0.45), Tool.FILL_ERASE: (1.0, 0.35, 0.30),
               Tool.FILL_REPLACE: (1.0, 0.85, 0.25)}
NEW_MATRIX_SIZE = 32
MAX_UI_SIZE = 1024  # the size fields stop here, well past what the editor handles comfortably
SWATCH_COLUMNS = 12
CAMERA_CONTROLS = (
    ("Alt+LMB", "Orbit"),
    ("RMB", "Rotate around focus"),
    ("MMB", "Pan"),
    ("Ctrl+Alt+LMB", "Pan"),
    ("Wheel", "Zoom"),
    ("RMB + W A S D", "Fly: move the focus"),
    ("RMB + Q / E", "Fly: move down / up"),
    ("RMB + Shift", "Fly faster"),
)
KEYBIND_SECTIONS = (
    ("Camera", CAMERA_CONTROLS),
    ("General", (
        ("Tab", "Switch between Voxel and Matrix mode"),
        ("T / E / R / K", "Voxel tools: place, erase, replace (recolor), pick color"),
        ("S", "Select: box select; press again for wand select"),
        ("F", "Fill; press again to switch between append and replace"),
        ("W / E / R / G", "Matrix tools: move, rotate, resize, origin (pivot)"),
        ("1 - 7 / 1 - 4", "The same tools by their place in the toolbox (Voxel mode / Matrix mode)"),
        ("Shift (held)", "Place, paint and erase drag out a box"),
        ("P", "Preview mode"),
        ("Home", "Frame the whole model"),
        ("Space", "Focus on the voxel under the cursor, else the selected matrix"),
        ("Ctrl+B", "Show or fold the Explorer sidebar"),
        ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
        ("Ctrl+N / O / S", "New / open / save (Shift+S: save as)"),
        ("Ctrl+W", "Close the tab"),
        ("Ctrl+Tab", "Next tab (Shift: previous)"),
    )),
    ("Matrices", (
        ("F2", "Rename the selected matrix"),
        ("Delete", "Delete the matrix"),
        ("Ctrl+A", "Select all matrices (with a select tool in Voxel mode: all voxels)"),
        ("Ctrl+D", "Deselect all matrices (with a select tool in Voxel mode: the voxels)"),
        ("Ctrl+M", "Merge the selected matrices"),
        ("Ctrl+Up / Down", "Move the matrix up or down the list"),
        ("[  /  ]", "Coarser / finer resolution of the selected matrices"),
        ("Shift+[  /  Shift+]", "The same, keeping the size"),
        ("Ctrl+click", "Add a matrix to the selection"),
        ("Shift+click", "Select a range in the list"),
    )),
    ("Voxel selection", (
        ("Ctrl+C / X / V", "Copy / cut / paste"),
        ("Ctrl+Shift+V", "Paste matrices at the camera focus instead of where they were copied from"),
        ("X / Y / Z", "Flip the selection (Shift: turn it)"),
        ("Arrows, PgUp / PgDn", "Nudge the selection"),
        ("Delete", "Delete the selected voxels"),
        ("Esc", "Deselect"),
    )),
)
RESOLUTION_BOX_WIDTH = 34.0   # the small resolution box on every row of the matrix list
MATRIX_DRAG = "MATRIX"   # ImGui drag-and-drop payload type for reordering the matrix list
DOCK_FLAGS = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse   # still scrolls
              | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
              | imgui.WindowFlags_.no_nav_inputs | imgui.WindowFlags_.no_bring_to_front_on_focus)
DEFAULT_RIGHT_WIDTH, DEFAULT_RIGHT_SPLIT = 340.0, 0.4
PREFERENCE_CHECK_SECONDS = 3.0
# Tab belongs to the mode switch, so ImGui must not use it (or the arrow keys) to hop between fields.
PANEL_FLAGS = imgui.WindowFlags_.no_nav_inputs


def cells_per_unit(matrix: Matrix) -> str:
    """How many cells fit in one unit along an axis at the matrix's resolution, as words: "4 cells"."""
    count = 2 ** (matrix.resolution - 1)
    return f"{count} cell" + ("" if count == 1 else "s")


def cube_edges() -> np.ndarray:
    """Line list for the 12 edges of the unit cube."""
    corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], np.float32)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    return np.array([corners[i] for e in edges for i in e], np.float32)


# --- scene --------------------------------------------------------------------

def matrix_transform(matrix: Matrix) -> glm.mat4:
    """model = position + R f (p - pivot) + f pivot, with f the cell size; local cells in, units out."""
    s = 1.0 / POSITION_SCALE
    f = float(matrix.cell_size)
    x, y, z, w = matrix.rotation
    rotation = glm.mat4_cast(glm.quat(w, x, y, z))
    pivot = glm.vec3(*matrix.pivot) * s
    position = glm.vec3(*matrix.position) * s
    return glm.translate(position + f * pivot) * rotation * glm.scale(glm.vec3(f)) * glm.translate(-pivot)


@dataclass
class Stroke:
    """A button-held edit.

    ``base`` is a copy of the matrix taken when the stroke began. Every pick during
    the stroke is made against it, so the edits being drawn never change what the
    cursor is aiming at: no stacking toward the camera, no digging deeper.
    """
    tool: Tool
    matrix: Matrix
    base: np.ndarray
    last_cursor: tuple[float, float]
    prefix: object = None        # the edit that grew the matrix to take the first voxel, to undo together with the stroke


@dataclass
class BoxDrag:
    """Drag out a box between the cell clicked and the cell released on, and apply the action to it.

    ``base`` and ``palette`` are copies taken when the drag began. Picking always uses
    ``base``, and with a live preview the matrix is reset to it before each re-fill,
    so the preview can follow the cursor and be cancelled cleanly.
    """
    tool: Tool
    start: tuple[int, int, int]
    end: tuple[int, int, int]
    base: np.ndarray
    palette: np.ndarray
    live: bool
    only_color: int | None = None   # paint: the palette index of the voxel the box started on; only that color changes

    def corners(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        lo = tuple(min(a, b) for a, b in zip(self.start, self.end))
        hi = tuple(max(a, b) for a, b in zip(self.start, self.end))
        return lo, hi


@dataclass
class SelectionDrag:
    """A mouse drag with the Select action: either dragging out a box to select, or moving what is selected.

    ``base`` is a copy of the voxels from when a move began, so the preview can follow the cursor and be cancelled.
    """
    kind: str                                  # "box" or "move"
    old_mask: np.ndarray | None = None         # the selection before the drag
    under: np.ndarray | None = None            # move: what the selection floats over
    add: bool = False                          # box: Shift was held, so add to the selection
    start: tuple[int, int, int] = (0, 0, 0)    # box: the cell the drag began on
    end: tuple[int, int, int] = (0, 0, 0)      # box: the cell it is on now
    base: np.ndarray | None = None             # move: the voxels before the move
    axis: int = 0                              # move: the axis the plane is perpendicular to (that of the face grabbed)
    plane: float = 0.0                         # move: where that plane cuts the axis
    point: np.ndarray | None = None            # move: where the mouse was on the plane when it began
    offset: tuple[int, int, int] = (0, 0, 0)   # move: how far the selection has moved


@dataclass
class DrawItem:
    matrix: Matrix
    transform: glm.mat4
    mesh: Mesh
    vao: moderngl.VertexArray | None
    buffers: list
    visible: bool = True
    highlight_key: object = None     # which selection (if any) is colored into the mesh, to know when to rebuild it

    def release(self) -> None:
        if self.vao is not None:
            self.vao.release()
        for buf in self.buffers:
            buf.release()
        self.vao, self.buffers = None, []


def _leaf_edits(edit):
    """The single edits inside a group of edits, which may hold groups themselves."""
    for inner in getattr(edit, "edits", ()):
        yield from _leaf_edits(inner)
    if not hasattr(edit, "edits"):
        yield edit


def model_bounds(model: Model) -> tuple[glm.vec3, float]:
    """Center and bounding radius of all matrices (rotation included)."""
    points = []
    for m in model.matrices:
        sx, sy, sz = m.size
        t = matrix_transform(m)
        for corner in np.array(np.meshgrid([0, sx], [0, sy], [0, sz])).reshape(3, -1).T:
            points.append(np.array(t * glm.vec4(*corner, 1.0))[:3])
    if not points:
        return glm.vec3(0), 10.0
    pts = np.array(points)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return glm.vec3(*((lo + hi) / 2)), float(np.linalg.norm(hi - lo) / 2) or 1.0


def demo_model() -> Model:
    """Small built-in model so the viewer shows something without a file."""
    palette = np.array([[220, 70, 60, 255], [80, 200, 90, 255], [70, 120, 230, 255],
                        [240, 220, 90, 255]], np.uint8)
    n = 24
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((xx - n / 2 + .5) ** 2 + (yy - n / 2 + .5) ** 2 + (zz - n / 2 + .5) ** 2)
    ball = np.where(r < n / 2, 1 + (xx // 4 + yy // 4 + zz // 4) % 3, 0).astype(np.uint16)
    slab = np.full((2, 6, 14), 4, np.uint16)
    s = POSITION_SCALE
    half = math.radians(40) / 2  # 40 degrees about Z
    return Model(palette, [
        Matrix("ball", ball),
        Matrix("slab", slab, position=(30 * s, 8 * s, 5 * s),
               rotation=(0.0, 0.0, math.sin(half), math.cos(half))),
    ])


# --- app ----------------------------------------------------------------------

class Viewer:
    def __init__(self, visible: bool = True, preferences_path: Path | None = None) -> None:
        """``preferences_path``: where to load and save preferences; None (the default) uses and writes none."""
        self.preferences_path = preferences_path
        prefs = Preferences.load(preferences_path) if preferences_path else Preferences()
        icon.set_app_id()           # before the window exists, so the taskbar shows our icon
        if not glfw.init():
            raise RuntimeError("could not initialise glfw")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.VISIBLE, False)   # shown below, once its saved size and position are applied
        self.window = glfw.create_window(prefs.window_width, prefs.window_height, "Chronovox Reforged", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create window")
        self._restore_window(prefs)
        self._set_window_icon()
        if visible:
            glfw.show_window(self.window)
        if prefs.maximized:  # after showing it, so Windows remembers the position and size to restore to
            glfw.maximize_window(self.window)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.ctx = moderngl.create_context()
        self.icons = IconSet(self.ctx)                # the picture icons from assets/icons
        # The picture icons are in the project but off for now: the tools and buttons use the icons drawn in code.
        # Turn this on to use the pictures (a name with no file still falls back to its drawn icon).
        self.picture_icons = False
        if self.picture_icons:
            set_images(self.icons)
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.prog["u_shade_base"].value = shading.BASE
        self.prog["u_shade_weights"].value = shading.WEIGHTS

        imgui.create_context()
        imgui.get_io().set_ini_filename("")  # no imgui.ini written next to the working dir
        self.imgui_impl = GlfwRenderer(self.window)
        self._imgui_scroll = glfw.set_scroll_callback(self.window, self._on_scroll)
        # The click that brings the window back to the front only focuses it: it must not draw, select or drag.
        self._swallowed: set[int] = set()            # buttons held since such a click, until they are let go
        self._swallow_press = 0                      # frames left in which a press counts as the focusing click
        self._imgui_mouse_button = glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        self._imgui_focus = glfw.set_window_focus_callback(self.window, self._on_focus)
        self._dropped: list[str] = []      # files dropped on the window, opened at the start of the next frame
        glfw.set_drop_callback(self.window, self._on_drop)

        self.flat = self.ctx.program(vertex_shader=FLAT_VERTEX_SHADER,
                                     fragment_shader=FLAT_FRAGMENT_SHADER)
        self.cube_vbo = self.ctx.buffer(cube_edges().tobytes())
        self.cube_vao = self.ctx.vertex_array(self.flat, [(self.cube_vbo, "3f", "in_pos")])

        self.grid_prog = self.ctx.program(vertex_shader=GRID_VERTEX_SHADER,
                                          fragment_shader=GRID_FRAGMENT_SHADER)
        self.grid_vbo = self.ctx.buffer(reserve=ground_grid.VERTEX_COUNT * 7 * 4)  # rewritten every frame
        self.grid_vao = self.ctx.vertex_array(self.grid_prog,
                                              [(self.grid_vbo, "3f 4f", "in_pos", "in_color")])

        # One face of a unit cell, in the XY plane: two triangles for the tint and four lines for the edge.
        self.face_fill_vbo = self.ctx.buffer(np.array(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)], np.float32).tobytes())
        self.face_fill_vao = self.ctx.vertex_array(self.flat, [(self.face_fill_vbo, "3f", "in_pos")])
        self.face_edge_vbo = self.ctx.buffer(np.array(
            [(0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 0)],
            np.float32).tobytes())
        self.face_edge_vao = self.ctx.vertex_array(self.flat, [(self.face_edge_vbo, "3f", "in_pos")])

        self.overlay_vbo = self.ctx.buffer(reserve=4096 * 7 * 4)  # rewritten every frame
        self.overlay_vao = self.ctx.vertex_array(self.grid_prog,
                                                 [(self.overlay_vbo, "3f 4f", "in_pos", "in_color")])

        self.camera = Camera()
        self.model: Model | None = None
        self.items: list[DrawItem] = []
        self.path: Path | None = None
        self.suggested_name = "model.cvox"  # default file name offered by Save as
        self.status = ""
        self._last_cursor = glfw.get_cursor_pos(self.window)
        self._keys_down: set[int] = set()
        self._lmb_was_down = False
        self.mirror = [False, False, False]             # edits are repeated across the center plane of each enabled axis
        self.radial_on = False                          # ... and turned around the center of the matrix, around radial_axis,
        self.radial_axis = 1                            # radial_count copies to a full turn
        self.radial_count = 4
        self.fill_contiguous = True       # Replace fill and the wand take only voxels connected to the clicked one (True) or all
        self.fill_tolerance = 0           # ... and colors within this many percent of the clicked color count too
        self.fill_replace = False         # the Fill tool recolors a region (True) or fills against a face (False)
        self.action = Tool.PLACE          # what a click does: Place, Erase, Paint or Pick
        self.brush = Brush.VOXEL          # how it is applied: one voxel at a time, a box, or a flood fill
        self._last_action = Tool.PLACE    # what a brush key returns to after Pick
        self.color = tuple(prefs.brush_color)  # current brush color, RGB 0-255
        self.active = 0                  # index of the matrix tools act on
        self.hover: tuple[int, int, int] | None = None  # cell the active tool would act on
        self.hover_index: int | None = None             # the matrix that cell is in
        self._target_restore: tuple[int, set] | None = None   # (active, selection) to go back to after an edit
        self.hover_face: tuple[int, int] | None = None  # Place: (axis, side) of the hover cell's face being built against
        self.fill_target: FillTarget | None = None      # what a fill tool would fill from, when it is active
        self.stroke: Stroke | None = None
        self.box: BoxDrag | None = None
        self.sel_drag: SelectionDrag | None = None
        self.voxel_clip: voxsel.Clip | None = None      # voxels copied with Ctrl+C
        self.clip_kind = "matrices"                      # what Ctrl+V pastes: "voxels" or "matrices", whichever was copied last
        self._sel_info_key: tuple | None = None
        self._sel_info: tuple[int, tuple | None] = (0, None)   # how many voxels are selected and their bounding box
        self.ambient_occlusion = prefs.ambient_occlusion
        self.show_inactive_outlines = prefs.show_inactive_outlines
        self.show_grid = prefs.show_grid
        self.history = History()
        self.mode = Mode.VOXEL
        self.vsel = voxsel.VoxelSelection()              # the selected voxels of the active matrix
        self.tabs = [Tab(camera=self.camera, history=self.history, voxel_selection=self.vsel)]   # the open models; the current one lives in the fields above
        self.current = 0                 # index of the current tab
        self.clipboard: Model | None = None   # the copied matrices, with the palette they were made with
        self.preview_mode = False        # picture mode: only the model is drawn, no panels, outlines or grid
        self._preview_since = 0.0        # when picture mode began, for the hint that fades out
        self._shown_uids: list[int] = []         # the tabs the bar showed last frame, in order
        self._bar_selected: int | None = None   # uid of the tab the tab bar last showed as selected
        self.matrix_tools = MatrixTools()
        self.new_matrix_name = ""
        self.renaming: int | None = None  # index of the matrix being renamed in the list
        self.rename_text = ""
        self._rename_focus = False
        self.new_matrix_size = [NEW_MATRIX_SIZE] * 3
        self.new_matrix_resolution = 1
        self._reset_layout = False  # set by View > Reset window layout, applied next frame
        self.right_width = prefs.right_dock_width    # the docked Model and Tools panels on the right
        self.right_split = prefs.right_dock_split    # the share of their height the Model panel takes
        self.show_new_matrix = False                 # the New matrix window
        self.live_box_preview = prefs.live_box_preview  # show the filled result while dragging a box
        self.last_dir = prefs.last_dir               # folder the file dialogs open in
        self.recent_files = [p for p in prefs.recent_files if p.lower().endswith(".cvox")]  # newest first
        self._recent_exists: dict[str, bool] = {}    # which of them are still on disk, refreshed now and then
        self._recent_checked = -10.0
        self._recent_to_remove: str | None = None    # a recent model asked to be taken off the list this frame
        self._recent_to_open: str | None = None      # one asked to be opened from its right-click menu
        self._recent_drop_missing = False            # "remove the ones that are gone" was chosen
        self.recent_view = prefs.recent_view         # "list" or "grid"
        self.explorer_visible = prefs.explorer_visible   # the Explorer: a tree of the models under some folders
        self.explorer_roots = list(prefs.explorer_roots)
        self.explorer_expanded = set(prefs.explorer_expanded)
        self.explorer_filter = ""
        self._recolor_rgb = [0.0, 0.0, 0.0]            # the color being picked in the palette's Recolor popup
        self._clone_source: str | None = None         # a model the Explorer asked to clone, waiting for a name
        self._clone_name = ""
        self._delete_candidate: str | None = None     # a model the Explorer asked to delete, waiting for the answer
        self.sidebar_tab = "explorer"                # which tab of the sidebar shows: "explorer" or "workspace"
        self._sidebar_tab_request: str | None = None # a tab to switch to by program (the Workspace menu)
        self._explorer_seen: str | None = None       # the current tab's file the sidebar last showed the way to
        self._explorer_reveal: str | None = None     # a file to open folders down to and scroll to
        self._explorer_reveal_tries = 0
        self.show_keybinds = False                   # the Help > Keybinds window
        self.explorer_width = prefs.explorer_width
        self.explorer_csm = prefs.explorer_csm       # also list legacy .csm models
        self._explorer_trees: dict[str, explorer.Node] = {}
        self._explorer_scanned = -100.0              # when the trees were last read from disk
        self._explorer_dirty = True                  # read them again at the next chance
        # the shared workspace: its login and file links live next to the preferences (nowhere without them)
        self.workspace = WorkspacePanel(self, preferences_path.parent if preferences_path else None, prefs.workspace_url)
        self._previews: dict[str, tuple] = {}        # path -> (file key, GL texture or None): the home page pictures
        self._preview_dir = preferences_path.parent / "thumbnails" if preferences_path else None
        thumbnails.prune(self._preview_dir, self.recent_files)
        self.bulk_resolution = prefs.bulk_resolution # what the "set all" box applies
        self.expand_on_edge = prefs.expand_on_edge   # placing on a face at the matrix edge grows the matrix to take it
        self.fields = NumberFields()                  # number fields that step with the arrow keys
        self._normal_bounds = (*glfw.get_window_pos(self.window), *glfw.get_window_size(self.window))
        self._maximized = False
        self._saved_prefs: Preferences | None = prefs if preferences_path else None
        self._voxel_generation = 0       # goes up whenever voxels may have changed
        self._usage_key: tuple | None = None
        self.selection: set[int] = set()  # ids of the matrices Ctrl+clicked in the list; the active one is always part of it
        self._usage = np.zeros(0, bool)  # which palette colors any voxel uses, valid for _usage_key
        self._last_pref_check = time.perf_counter()
        self.dirty = False
        self._title = ""
        self._drag: str | None = None  # active camera drag: orbit, pan, zoom or look
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def _set_window_icon(self) -> None:
        """The icon in the title bar, the taskbar and Alt+Tab. A missing or unreadable picture never stops the program."""
        try:
            icons = icon.window_icons()
            glfw.set_window_icon(self.window, len(icons), icons)
        except Exception:            # noqa: BLE001 - an icon is not worth a crash, whatever went wrong
            pass

    # -- preferences --
    def _restore_window(self, prefs: Preferences) -> None:
        """Apply the saved position to the (still hidden) window, if it is somewhere reachable."""
        if prefs.window_x is not None and self._on_a_monitor(prefs.window_x, prefs.window_y, prefs.window_width):
            glfw.set_window_pos(self.window, prefs.window_x, prefs.window_y)

    @staticmethod
    def _on_a_monitor(x: int, y: int, width: int) -> bool:
        """True if the window's title bar would be reachable on one of the connected monitors."""
        probe = (x + min(width, 200) // 2, y + 12)
        for monitor in glfw.get_monitors():
            mx, my, mw, mh = glfw.get_monitor_workarea(monitor)
            if mx <= probe[0] < mx + mw and my <= probe[1] < my + mh:
                return True
        return False

    def _track_window(self) -> None:
        """Remember the window's un-maximized bounds, so they can be restored next time."""
        self._maximized = bool(glfw.get_window_attrib(self.window, glfw.MAXIMIZED))
        iconified = bool(glfw.get_window_attrib(self.window, glfw.ICONIFIED))
        if not self._maximized and not iconified:
            x, y = glfw.get_window_pos(self.window)
            w, h = glfw.get_window_size(self.window)
            if w > 0 and h > 0:
                self._normal_bounds = (x, y, w, h)

    def collect_preferences(self) -> Preferences:
        x, y, w, h = self._normal_bounds
        return Preferences(
            window_width=max(w, 480), window_height=max(h, 320), window_x=x, window_y=y,
            maximized=self._maximized,
            ambient_occlusion=self.ambient_occlusion, show_grid=self.show_grid,
            show_inactive_outlines=self.show_inactive_outlines, live_box_preview=self.live_box_preview,
            right_dock_width=float(self.right_width), right_dock_split=float(self.right_split),
            brush_color=list(self.color), last_dir=self.last_dir,
            bulk_resolution=self.bulk_resolution, expand_on_edge=self.expand_on_edge,
            recent_files=list(self.recent_files),
            recent_view=self.recent_view, workspace_url=self.workspace.url_text.strip(),
            explorer_width=float(self.explorer_width), explorer_visible=self.explorer_visible, explorer_csm=self.explorer_csm, explorer_roots=list(self.explorer_roots),
            explorer_expanded=sorted(self.explorer_expanded))

    def save_preferences(self) -> None:
        """Write the preferences if they changed since the last write. Does nothing without a path."""
        if self.preferences_path is None:
            return
        prefs = self.collect_preferences()
        if prefs != self._saved_prefs and prefs.save(self.preferences_path):
            self._saved_prefs = prefs

    def _on_drop(self, _window, paths) -> None:
        self._dropped.extend(paths)

    def _open_dropped(self) -> None:
        """Open the files dropped on the window, each in its own tab. Anything but a model file is skipped."""
        if not self._dropped or self._busy():
            return
        paths, self._dropped = self._dropped, []
        skipped = [Path(p).name for p in paths if Path(p).suffix.lower() not in (".cvox", ".csm")]
        for path in paths:
            if Path(path).suffix.lower() in (".cvox", ".csm"):
                self.open_file(path)
        if skipped:
            self.status = f"Not a model file: {', '.join(skipped)}"

    def _remember_dir(self, path: Path) -> None:
        self.last_dir = str(path.parent)

    def _render_preview(self, model: Model) -> tuple[np.ndarray, "moderngl.Texture"]:
        """Draw a model once from the default viewing angle into a small square picture.

        Returns the pixels (RGBA, bottom row first as OpenGL has them) and a texture holding them.
        """
        ctx, size = self.ctx, thumbnails.SIZE
        model = model.opaque()
        items = [self._build_item(m, palette=model.palette) for m in model.matrices]
        texture = ctx.texture((size, size), 4)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        depth = ctx.depth_renderbuffer((size, size))
        fbo = ctx.framebuffer([texture], depth)
        fbo.use()
        ctx.viewport = (0, 0, size, size)
        fbo.clear(*BACKGROUND, 1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.front_face = "cw"
        camera = Camera()
        camera.frame(*model_bounds(model))
        view_proj = camera.projection(1.0) * camera.view()
        self.prog["u_ao_strength"].value = 0.5
        for item in items:
            if item.vao is not None:
                self.prog["u_view_proj"].write(view_proj.to_bytes())
                self.prog["u_model"].write(item.transform.to_bytes())
                item.vao.render()
        pixels = np.frombuffer(fbo.read(components=4), np.uint8).reshape(size, size, 4).copy()
        for item in items:
            item.release()
        fbo.release()
        depth.release()
        ctx.screen.use()
        return pixels, texture

    def _preview(self, path: str, budget: list[int]):
        """The home page picture of a model file: a texture, or None if it has none (yet).

        Pictures are kept on disk, so only a model that is new or was saved again has to be loaded and drawn;
        ``budget`` counts how many of those may still be drawn this frame, so a long list does not stall it.
        """
        key = thumbnails.file_key(path)
        known = self._previews.get(path)
        if known is not None and known[0] == key:
            return known[1]
        if known is not None and known[1] is not None:
            known[1].release()
        texture = None
        if key is not None:
            pixels = thumbnails.load(self._preview_dir, path, key)
            if pixels is not None:
                texture = self.ctx.texture((thumbnails.SIZE, thumbnails.SIZE), 4, pixels.tobytes())
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            elif budget[0] <= 0:
                self._previews.pop(path, None)
                return None                      # later: another picture has been drawn this frame already
            else:
                budget[0] -= 1
                try:
                    pixels, texture = self._render_preview(cvox.load(path))
                    thumbnails.store(self._preview_dir, path, key, pixels)
                except (OSError, cvox.CvoxError):
                    texture = None
        self._previews[path] = (key, texture)
        return texture

    def _remember_recent(self, path: Path) -> None:
        if path.suffix.lower() != ".cvox":          # only models in the editor's own format are listed
            return
        self.recent_files = add_recent(self.recent_files, str(path.resolve()))
        self._recent_checked = -10.0

    # -- model handling --
    def _selection_for(self, matrix: Matrix) -> np.ndarray | None:
        """The selected voxels of a matrix that are to be drawn highlighted, or None (not in picture or matrix mode)."""
        if not self.selection_visible():
            return None
        return self.vsel.mask_for(matrix)

    def _highlight_key(self, matrix: Matrix):
        return self.vsel.version if self._selection_for(matrix) is not None else None

    def _sync_highlights(self) -> None:
        """Rebuild the mesh of any matrix whose highlighted voxels are not the ones now selected."""
        for index, item in enumerate(self.items):
            if item.highlight_key != self._highlight_key(item.matrix):
                self._rebuild_item(index)

    def _build_item(self, matrix: Matrix, visible: bool = True, palette: np.ndarray | None = None) -> DrawItem:
        mask = self._selection_for(matrix)
        mesh = build_mesh(matrix.voxels, self.model.palette if palette is None else palette,
                          voxsel.solid(mask, matrix.voxels) if mask is not None else None)
        vao, buffers = None, []
        if mesh.face_count:
            vbo_p = self.ctx.buffer(mesh.positions.tobytes())
            vbo_n = self.ctx.buffer(mesh.normals.tobytes())
            vbo_c = self.ctx.buffer(mesh.colors.tobytes())
            vbo_a = self.ctx.buffer(mesh.ao.tobytes())
            ibo = self.ctx.buffer(mesh.indices.tobytes())
            buffers = [vbo_p, vbo_n, vbo_c, vbo_a, ibo]
            vao = self.ctx.vertex_array(
                self.prog,
                [(vbo_p, "3f", "in_pos"), (vbo_n, "3f", "in_normal"), (vbo_c, "4f1", "in_color"),
                 (vbo_a, "1f1", "in_ao")],
                index_buffer=ibo)
        return DrawItem(matrix, matrix_transform(matrix), mesh, vao, buffers, visible, self._highlight_key(matrix))

    def palette_usage(self) -> np.ndarray:
        """Which palette colors are used by a voxel, as a bool array. Recomputed only after voxels change."""
        key = (self._voxel_generation, len(self.model.palette))
        if key != self._usage_key:
            self._usage, self._usage_key = self.model.used_palette_mask(), key
        return self._usage

    def _rebuild_item(self, index: int) -> None:
        self._voxel_generation += 1
        old = self.items[index]
        old.release()
        self.items[index] = self._build_item(old.matrix, old.visible)

    def set_model(self, model: Model, path: Path | None = None) -> None:
        self._voxel_generation += 1
        for item in self.items:
            item.release()
        self.items = []
        self.model, self.path = model, path
        self.suggested_name = "model.cvox"
        self.items = [self._build_item(m) for m in model.matrices]
        self.active = -1                              # nothing selected: the voxel tools work on whatever they hit
        self.selection = set()
        self.renaming = None
        self.hover = None
        self.stroke = None
        self.box = None
        self.sel_drag = None
        self.vsel.clear()
        self.matrix_tools.drag = None
        self.matrix_tools.hover = None
        self.history.clear()
        self.dirty = False
        self.camera.frame(*model_bounds(model))

    # -- tabs --
    def _store_tab(self) -> None:
        """Park the current model's state in its tab, so another tab can be loaded."""
        tab = self.tabs[self.current]
        tab.model, tab.items, tab.path, tab.suggested_name = self.model, self.items, self.path, self.suggested_name
        tab.active, tab.history, tab.dirty, tab.camera = self.active, self.history, self.dirty, self.camera
        tab.voxel_selection = self.vsel
        tab.selection = self.selection

    def _load_tab(self, index: int) -> None:
        self.current = index
        tab = self.tabs[index]
        self.model, self.items, self.path, self.suggested_name = tab.model, tab.items, tab.path, tab.suggested_name
        # The mode and the chosen tool belong to the editor, not to a tab: they stay as they are when tabs change.
        self.active, self.history, self.dirty, self.camera = tab.active, tab.history, tab.dirty, tab.camera
        self.vsel = tab.voxel_selection
        self.sel_drag = None
        self._voxel_generation += 1
        self._usage_key = None
        self.selection = tab.selection
        self.renaming = None
        self.hover = None
        self.fill_target = None
        self.stroke = None
        self.box = None
        self.matrix_tools.drag = None
        self.matrix_tools.hover = None

    def tab_name(self, index: int) -> str:
        """What the tab shows: the file name, or "untitled", with a * while it has unsaved changes."""
        if index == self.current:
            return "Home" if self.model is None else tab_label(self.path, self.dirty, self.suggested_name)
        tab = self.tabs[index]
        return "Home" if tab.model is None else tab_label(tab.path, tab.dirty, tab.suggested_name)

    def switch_tab(self, index: int) -> bool:
        if not 0 <= index < len(self.tabs) or index == self.current or self._busy():
            return False
        self._store_tab()
        self._load_tab(index)
        return True

    def _is_fresh(self) -> bool:
        """True for an untitled model nobody has changed since it was created."""
        return (self.model is not None and self.path is None and not self.dirty and not self.history.can_undo
                and not self.history.can_redo)

    def _open_tab(self) -> None:
        """Make a new tab current. Its model is empty (the home page) until a model is put in it. A tab that is
        still the home page is used as it is."""
        if self.model is None:
            return
        self._store_tab()
        self.tabs.append(Tab())
        self._load_tab(len(self.tabs) - 1)

    def close_tab(self, index: int | None = None, ask: bool = True) -> bool:
        """Close a tab (default: the current one), asking first if it has unsaved changes.

        Closing the last tab leaves the home page in its place.
        """
        index = self.current if index is None else index
        if self._busy() or not 0 <= index < len(self.tabs):
            return False
        self._store_tab()
        tab = self.tabs[index]
        if ask and tab.dirty and not self._ask(
                "Unsaved changes", f"Close {self.tab_name(index).lstrip('*')} and discard its unsaved changes?"):
            return False
        for item in tab.items:
            item.release()
        del self.tabs[index]
        if not self.tabs:
            self.tabs = [Tab()]
            self._load_tab(0)
            return True
        self._load_tab(min(self.current - 1 if index < self.current else self.current, len(self.tabs) - 1))
        return True

    def copy_matrix_to_tab(self, index: int, tab_index: int) -> bool:
        """Copy a matrix of this model into another tab's model (as one undo step there). This tab stays current."""
        if (self.model is None or self._busy() or tab_index == self.current or not 0 <= tab_index < len(self.tabs)
                or not 0 <= index < len(self.items)):
            return False
        source_model, matrix, origin = self.model, self.items[index].matrix, self.current
        self.switch_tab(tab_index)
        try:
            try:
                copy, at = copy_matrix(source_model, matrix, self.model)
            except ValueError as exc:
                message = f"Cannot copy {matrix.name}: {exc}"
            else:
                self.history.push(MatrixListEdit(self.model, copy, at, added=True))
                self._sync_items()
                self.active = at
                self.dirty = True
                message = f"Copied {matrix.name} to {self.tab_name(tab_index).lstrip('*')} as {copy.name}"
        finally:
            self.switch_tab(origin)
        self.status = message
        return True

    def copy_selected_matrix(self) -> bool:
        """Put a copy of every selected matrix on the clipboard.

        The copies stay there when the model changes or its tab closes, and keep their places relative to each other.
        """
        indices = self.selected_indices()
        if self.model is None or not indices:
            return False
        holder = Model(self.model.palette.copy())
        for i in indices:
            copy_matrix(self.model, self.items[i].matrix, holder)
        self.clipboard = holder
        self.clip_kind = "matrices"
        what = holder.matrices[0].name if len(indices) == 1 else f"{len(indices)} matrices"
        self.status = f"Copied {what}. Ctrl+V pastes in place, Ctrl+Shift+V at the camera focus."
        return True

    def paste_matrix(self, at_focus: bool = False) -> bool:
        """Add the clipboard's matrices to the current model, where they were copied from.

        With ``at_focus`` their common center goes on the camera focus instead. One undo step. The pasted matrices become the selection. If any of them cannot be added nothing is.
        """
        if self.model is None or self._busy() or self.clipboard is None:
            return False
        copies, edits = [], []
        try:
            for matrix in self.clipboard.matrices:
                copy, at = copy_matrix(self.clipboard, matrix, self.model)
                copies.append(copy)
                edits.append(MatrixListEdit(self.model, copy, at, added=True))
        except ValueError as exc:
            for copy in copies:                           # take back the ones already added
                self.model.remove_matrix(copy)
            self.status = f"Cannot paste: {exc}"
            return False
        if at_focus:
            shift = group_shift(copies, tuple(self.camera.target))
            for copy in copies:
                copy.position = tuple(p + d for p, d in zip(copy.position, shift))
        self.history.push(edits[0] if len(edits) == 1 else GroupEdit(f"paste of {len(edits)} matrices", edits))
        self._sync_items()
        self.active = self.model.matrices.index(copies[0])
        self.selection = {id(c) for c in copies}
        self.dirty = True
        self.status = f"Pasted {copies[0].name}" if len(copies) == 1 else f"Pasted {len(copies)} matrices"
        return True

    def import_palette_from_tab(self, tab_index: int, used_only: bool = True) -> bool:
        """Add the colors of another tab's model that this one lacks to this palette."""
        if self.model is None or tab_index == self.current or not 0 <= tab_index < len(self.tabs):
            return False
        self._store_tab()
        try:
            added = import_colors(self.tabs[tab_index].model, self.model, used_only)
        except ValueError as exc:
            self.status = str(exc)
            return False
        name = self.tab_name(tab_index).lstrip("*")
        self.status = (f"Added {added} color{'s' if added != 1 else ''} from {name}" if added
                       else f"{name} has no colors this palette lacks")
        return True

    def new_model(self) -> None:
        self._open_tab()
        n = NEW_MATRIX_SIZE
        model = Model()
        # Centered on the origin in X and Z, standing on the ground plane: corner at (-16, 0, -16).
        model.new_matrix((n, n, n), "Matrix", (-(n // 2) * POSITION_SCALE, 0, -(n // 2) * POSITION_SCALE))
        self.set_model(model)
        self.status = f"New model with one {n}x{n}x{n} matrix"

    def open_file(self, path: str | Path) -> None:
        path = Path(path)
        legacy = path.suffix.lower() == ".csm"
        if not legacy:
            for i in range(len(self.tabs)):          # already open: go to that tab
                known = self.path if i == self.current else self.tabs[i].path
                if known is not None and known.resolve() == path.resolve():
                    fresh = self.current if self._is_fresh() else None
                    self.switch_tab(i)
                    if fresh is not None:         # the untitled tab it was opened from has nothing in it: close it
                        self.close_tab(fresh, ask=False)
                    self.status = f"{path.name} is already open"
                    return
        try:
            model = csm.load(path) if legacy else cvox.load(path)
        except (OSError, cvox.CvoxError, csm.CsmError) as exc:
            self.status = f"Could not open {path.name}: {exc}"
            if not path.exists():
                self.recent_files = remove_recent(self.recent_files, str(path.resolve()))
            return
        self._remember_dir(path)
        self._remember_recent(path)
        self._explorer_dirty = True
        if not self._is_fresh():              # a fresh untitled model has nothing to lose: the file takes its place
            self._open_tab()
        if legacy:
            # A converted model has no .cvox file yet: keep it unsaved so Save asks where to put it.
            self.set_model(model, None)
            self.suggested_name = path.with_suffix(".cvox").name
            self.dirty = True
            self.status = (f"Imported {path.name}: {len(model.matrices)} matrices, "
                           f"{len(model.palette):,} colors. Save to keep it as .cvox.")
            return
        opaque = model.opaque()
        self.set_model(opaque, path)
        self.status = f"Opened {path.name}" + (" (palette alpha reset to 255)" if opaque is not model else "")

    def _tk_root(self):
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        return root

    def _ask(self, title: str, message: str) -> bool:
        from tkinter import messagebox
        root = self._tk_root()
        try:
            return messagebox.askyesno(title, message, parent=root)
        finally:
            root.destroy()

    def _confirm_discard(self) -> bool:
        """Before quitting: True if no tab has unsaved changes or the user agrees to lose them."""
        self._store_tab()
        unsaved = [self.tab_name(i).lstrip("*") for i, tab in enumerate(self.tabs) if tab.dirty]
        if not unsaved:
            return True
        return self._ask("Unsaved changes", f"Discard unsaved changes in {', '.join(unsaved)}?")

    def _choose_file(self) -> None:
        from tkinter import filedialog
        root = self._tk_root()
        try:
            chosen = filedialog.askopenfilename(
                title="Open", initialdir=self.last_dir or None, filetypes=[("Chronovox models", "*.cvox *.csm"), ("Chronovox model", "*.cvox"),
                           ("Legacy Chronovox Studio model", "*.csm"), ("All files", "*.*")])
        finally:
            root.destroy()
        if chosen:
            self.open_file(chosen)

    def save(self, save_as: bool = False) -> None:
        if self.model is None:
            return
        path = self.path
        if save_as or path is None:
            from tkinter import filedialog
            root = self._tk_root()
            try:
                chosen = filedialog.asksaveasfilename(
                    title="Save as", defaultextension=".cvox", initialdir=self.last_dir or None,
                    initialfile=path.name if path else self.suggested_name,
                    filetypes=[("Chronovox model", "*.cvox")])
            finally:
                root.destroy()
            if not chosen:
                return
            path = Path(chosen)
        # The file keeps only colors some voxel uses; unused ones stay in the palette here for editing.
        packed = self.model.compacted()
        dropped = len(self.model.palette) - len(packed.palette)
        try:
            cvox.save(packed, path, compress=True)
        except (OSError, cvox.CvoxError) as exc:
            self.status = f"Could not save {path.name}: {exc}"
            return
        self.path = path
        self._explorer_dirty = True
        self._remember_dir(path)
        self._remember_recent(path)
        self.dirty = False
        self.status = f"Saved {path.name}" + (f" (left out {dropped} unused color{'s' if dropped != 1 else ''})"
                                              if dropped else "")

    @property
    def tool(self) -> Tool:
        """The tool doing the work right now: the action, or its fill tool when the Fill brush is chosen."""
        return effective_tool(self.action, self.brush)

    @tool.setter
    def tool(self, value: Tool) -> None:
        """Select a tool by its own name. A fill tool means that action with the Fill brush."""
        for action, fill in FILL_FOR_ACTION.items():
            if value is fill:
                self.action, self.brush = action, Brush.FILL
                return
        self.set_action(value)
        self.brush = Brush.VOXEL

    @property
    def voxel_tool(self) -> VoxelTool:
        """The toolbar tool that is chosen, whatever action and brush it is made of."""
        return voxel_tool_for(self.action, self.brush)

    def set_voxel_tool(self, tool: VoxelTool) -> None:
        if self._busy():
            return
        action, brush = TOOL_PARTS[tool]
        if tool is VoxelTool.FILL and self.fill_replace:
            action = Tool.PAINT
        self.action, self.brush = action, brush
        if action is not Tool.PICK:
            self._last_action = action

    def set_fill_replace(self, replace: bool) -> None:
        """Choose what the Fill tool does: recolor a connected region (True) or fill against the clicked face (False)."""
        if self._busy():
            return
        self.fill_replace = replace
        if self.voxel_tool is VoxelTool.FILL:
            self.set_voxel_tool(VoxelTool.FILL)             # the action behind it follows
        self.status = "Fill: " + ("replace the color of a region" if replace else "append against a face")

    def key_voxel_tool(self, tool: VoxelTool) -> None:
        """A tool key was pressed. Pressing Fill again switches its mode, Select again swaps box and wand."""
        if tool is VoxelTool.FILL and self.voxel_tool is VoxelTool.FILL:
            self.set_fill_replace(not self.fill_replace)
        elif tool is VoxelTool.SELECT_BOX and self.voxel_tool is VoxelTool.SELECT_BOX:
            self.set_voxel_tool(VoxelTool.SELECT_WAND)
        elif tool is VoxelTool.SELECT_BOX and self.voxel_tool is VoxelTool.SELECT_WAND:
            self.set_voxel_tool(VoxelTool.SELECT_BOX)
        else:
            self.set_voxel_tool(tool)

    def use_select_tool(self) -> None:
        """Switch to a select tool (the one in use if it already is one), so a selection is shown and can be moved."""
        if self.action is not Tool.SELECT:
            self.set_voxel_tool(VoxelTool.SELECT_BOX)

    def set_action(self, action: Tool) -> None:
        if self._busy():
            return
        self.action = action
        if action is not Tool.PICK:
            self._last_action = action

    def set_brush(self, brush: Brush) -> None:
        """Choose a brush. After Pick, which takes none, the previous action comes back."""
        if self._busy():
            return
        if self.action is Tool.PICK:
            self.action = self._last_action
        self.brush = brush

    def _effective_brush(self, shift: bool) -> Brush:
        """The brush in use: Shift turns the Voxel brush into a box for as long as it is held."""
        if self.action is Tool.PICK:
            return Brush.VOXEL
        return Brush.BOX if shift and self.brush is Brush.VOXEL else self.brush

    def focus_active_matrix(self) -> None:
        """Make the center of the selected matrices (just the active one if only one is) the point the camera orbits."""
        chosen = [self.items[i].matrix for i in self.selected_indices()]
        if not chosen:
            return
        self.camera.focus_on(glm.vec3(*group_anchor(chosen)))
        self.status = f"Focus on {chosen[0].name}" if len(chosen) == 1 else f"Focus on {len(chosen)} selected matrices"

    def focus_hovered_voxel(self) -> bool:
        """In voxel mode, make the center of the voxel under the cursor the point the camera orbits. False if none is."""
        item = self._hover_item()
        if self.mode is not Mode.VOXEL or self.hover is None or item is None:
            return False
        center = item.transform * glm.vec4(*(c + 0.5 for c in self.hover), 1.0)
        self.camera.focus_on(glm.vec3(center))
        x, y, z = self.hover
        self.status = f"Focus on cell {x}, {y}, {z}"
        return True

    def add_matrix(self, size: tuple[int, int, int], name: str | None = None, resolution: int = 1) -> bool:
        """Add an empty matrix centered on the camera focus and make it active."""
        if self.model is None:
            return False
        cell = 0.5 ** (resolution - 1)                           # a cell's size in units
        half = glm.vec3(*size) * cell / 2
        corner = [round(c / cell) for c in self.camera.target - half]  # whole cells
        position = tuple(int(round(c * cell * POSITION_SCALE)) for c in corner)
        try:
            matrix = self.model.new_matrix(size, name, position, resolution)
        except ValueError as exc:
            self.status = str(exc)
            return False
        self.history.push(MatrixListEdit(self.model, matrix, len(self.model.matrices) - 1, added=True))
        self._sync_items()
        self.active = len(self.items) - 1
        self.dirty = True
        self.status = f"Added {matrix.name} [{size[0]}x{size[1]}x{size[2]}]" + (f", resolution {resolution}" if resolution > 1 else "")
        return True

    def _sync_items(self) -> None:
        """Make the drawn items match the model's matrices: keep existing ones, build new ones, drop the rest."""
        self._voxel_generation += 1
        existing = {id(item.matrix): item for item in self.items}
        items = []
        for matrix in self.model.matrices:
            item = existing.pop(id(matrix), None)
            items.append(item if item is not None else self._build_item(matrix))
        for stale in existing.values():
            stale.release()
        self.items = items
        self.active = max(-1, min(self.active, len(items) - 1))       # -1 (nothing selected) stays
        self.renaming = None
        self.hover = None
        self.fill_target = None
        self.matrix_tools.hover = None

    def reorder_matrix(self, index: int, new_index: int) -> bool:
        """Move a matrix to another place in the list as one undo step. The list order is the saved order."""
        if self.model is None or self._busy() or not 0 <= index < len(self.items):
            return False
        new_index = max(0, min(new_index, len(self.items) - 1))
        if new_index == index:
            return False
        matrix = self.items[index].matrix
        old = self.model.move_matrix(matrix, new_index)
        self.history.push(MatrixMoveEdit(self.model, matrix, old, new_index))
        self._sync_items()
        self.active = new_index
        self.dirty = True
        self.status = f"Moved {matrix.name} to position {new_index + 1}"
        return True

    def move_active_matrix(self, step: int) -> bool:
        """Move the active matrix up (-1) or down (+1) the list."""
        return self.reorder_matrix(self.active, self.active + step)

    def selected_indices(self) -> list[int]:
        """The rows highlighted in the matrix list, lowest first. Always contains the active matrix.

        The Ctrl+click selection only counts while the active matrix is part of it: when something else
        changes which matrix is active (a new matrix, a delete, an undo), it falls back to just that one.
        """
        item = self.active_item()
        if item is None:
            return []
        if id(item.matrix) not in self.selection:
            return [self.active]
        return [i for i, it in enumerate(self.items) if id(it.matrix) in self.selection]

    def toggle_selection(self, index: int) -> None:
        """Ctrl+click: add a matrix to the selection, or take it out again (one always stays)."""
        if not 0 <= index < len(self.items):
            return
        group = set(self.selected_indices())
        if index in group:
            if len(group) == 1:
                return
            group.discard(index)
            if index == self.active:
                self.active = min(group)
        else:
            group.add(index)
            self.active = index
        self.selection = {id(self.items[i].matrix) for i in group}

    def select_range(self, index: int, additive: bool = False) -> None:
        """Shift+click: select every matrix from the active one to this one, in list order.

        The active matrix stays the anchor, so clicking further along extends the same range. With ``additive``
        (Ctrl+Shift+click) the range is added to what is already selected instead of replacing it.
        """
        if not 0 <= index < len(self.items) or not 0 <= self.active < len(self.items):
            return
        low, high = sorted((self.active, index))
        group = set(range(low, high + 1))
        if additive:
            group |= set(self.selected_indices())
        self.selection = {id(self.items[i].matrix) for i in group}

    def merge_selected(self) -> bool:
        """Merge every selected matrix into the one highest in the list, as one undo step.

        The others go away and the top one grows to hold them. Where matrices overlap, the one lower in the
        list wins. If any of them cannot be merged nothing changes.
        """
        indices = self.selected_indices()
        if self.model is None or self._busy():
            return False
        if len(indices) < 2:
            self.status = "Ctrl+click two or more matrices to select them, then merge"
            return False
        target = self.items[indices[0]].matrix
        sources = [self.items[i].matrix for i in indices[1:]]
        edits = []
        try:
            for source in sources:
                edits.append(merge_matrix(self.model, source, target))
        except ValueError as exc:
            for edit in reversed(edits):
                edit.undo()
            self.status = f"Cannot merge: {exc}"
            return False
        self.history.push(GroupEdit(f"merge of {len(sources) + 1} matrices into {target.name}", edits))
        self._sync_items()
        self._rebuild_item(self.model.matrices.index(target))      # its voxels and position changed
        self.active = self.model.matrices.index(target)
        self.selection = set()
        self.dirty = True
        x, y, z = target.size
        self.status = f"Merged {len(sources)} into {target.name} ({x}x{y}x{z}). Ctrl+Z undoes it."
        return True

    def merge_into(self, source_index: int, target_index: int) -> bool:
        """Merge one matrix into another as one undo step. The source goes away; the target grows to hold it."""
        if self.model is None or self._busy() or source_index == target_index:
            return False
        if not (0 <= source_index < len(self.items) and 0 <= target_index < len(self.items)):
            return False
        source, target = self.items[source_index].matrix, self.items[target_index].matrix
        try:
            edit = merge_matrix(self.model, source, target)
        except ValueError as exc:
            self.status = f"Cannot merge: {exc}"
            return False
        self.history.push(edit)
        self._sync_items()
        self._rebuild_item(self.model.matrices.index(target))      # its voxels and position changed
        self.active = self.model.matrices.index(target)
        self.dirty = True
        x, y, z = target.size
        self.status = f"Merged {source.name} into {target.name} ({x}x{y}x{z}). Ctrl+Z undoes it."
        return True

    def _busy(self) -> bool:
        return (self.stroke is not None or self.box is not None or self.sel_drag is not None
                or self.matrix_tools.drag is not None)

    def delete_matrix(self, index: int | None = None) -> bool:
        """Delete a matrix (default: the active one) as one undo step."""
        index = self.active if index is None else index
        if self.model is None or self._busy() or not 0 <= index < len(self.items):
            return False
        matrix = self.items[index].matrix
        edit = MatrixListEdit(self.model, matrix, index, added=False)
        edit.redo()
        self.history.push(edit)
        self.active = index                  # the next matrix takes its place in the list
        self._sync_items()
        self.dirty = True
        self.status = f"Deleted {matrix.name}. Ctrl+Z brings it back."
        return True

    def clone_matrix(self, index: int | None = None) -> bool:
        """Duplicate a matrix (default: the active one) as one undo step. The copy is selected."""
        index = self.active if index is None else index
        if self.model is None or self._busy() or not 0 <= index < len(self.items):
            return False
        try:
            copy, at = self.model.duplicate_matrix(self.items[index].matrix)
        except ValueError as exc:
            self.status = str(exc)
            return False
        self.history.push(MatrixListEdit(self.model, copy, at, added=True))
        self._sync_items()
        self.active = at
        self.dirty = True
        self.status = f"Cloned {self.items[index].matrix.name} as {copy.name}"
        return True

    def start_rename(self, index: int | None = None) -> None:
        """Begin renaming a matrix (default: the active one) in the Model panel."""
        index = self.active if index is None else index
        if not 0 <= index < len(self.items):
            return
        self.active = index
        self.renaming = index
        self.rename_text = self.items[index].matrix.name
        self._rename_focus = True

    def _finish_rename(self, commit: bool) -> None:
        index, self.renaming = self.renaming, None
        if not commit or index is None or not 0 <= index < len(self.items):
            return
        matrix = self.items[index].matrix
        try:
            if self.model.rename_matrix(matrix, self.rename_text):
                self.dirty = True
                self.status = f"Renamed to {matrix.name}"
        except ValueError as exc:
            self.status = f"Could not rename: {exc}"

    # -- editing --
    def active_item(self) -> DrawItem | None:
        return self.items[self.active] if 0 <= self.active < len(self.items) else None

    @property
    def radial(self) -> tuple[int, int] | None:
        """(axis, copies) while radial symmetry is on and makes copies, else None."""
        return (self.radial_axis, self.radial_count) if self.radial_on and self.radial_count > 1 else None

    def brush_rgba(self) -> tuple[int, int, int, int]:
        return (*self.color, 255)

    def select_palette_color(self, index: int) -> None:
        """Make the palette color with this 1-based index the brush color."""
        self.color = tuple(int(c) for c in self.model.palette[index - 1][:3])

    def recolor_palette_color(self, index: int, rgb: tuple[int, int, int]) -> bool:
        """Change the palette color with this 1-based index to ``rgb``, which recolors every voxel that uses it."""
        if self.model is None or not 1 <= index <= len(self.model.palette):
            return False
        old = tuple(int(c) for c in self.model.palette[index - 1])
        new = (*rgb, old[3])
        if new == old:
            return False
        was_brush = self.model.find_color(self.brush_rgba()) == index
        edit = PaletteEdit(self.model, index - 1, old, new)
        edit.redo()
        self.history.push(edit)
        for i in range(len(self.items)):
            self._rebuild_item(i)
        if was_brush:
            self.color = tuple(rgb)
        self.dirty = True
        self.status = f"Recolored palette color {index}"
        return True

    def _apply(self, tool: Tool, cell: tuple[int, int, int]) -> bool:
        """Apply a tool to the active matrix without rebuilding. True if voxels changed."""
        item = self.active_item()
        if item is None:
            return False
        if tool is Tool.PICK:
            x, y, z = cell
            index = int(item.matrix.voxels[z, y, x])
            if index:
                self.select_palette_color(index)
            return False
        changed = False
        try:
            for target in symmetric_cells(cell, item.matrix.size, tuple(self.mirror), self.radial):
                changed |= apply_tool(self.model, item.matrix, tool, target, self.brush_rgba())
        except ValueError as exc:
            self.status = str(exc)
        return changed

    def _commit(self) -> None:
        self.dirty = True
        self._rebuild_item(self.active)

    def edit(self, tool: Tool, cell: tuple[int, int, int]) -> None:
        """Apply one edit to the active matrix. All voxel edits go through here or a stroke."""
        item = self.active_item()
        if item is None:
            return
        if (any(self.mirror) or self.radial) and tool is not Tool.PICK:
            before = item.matrix.voxels.copy()               # several voxels change: record the difference
            if self._apply(tool, cell):
                self._record(item.matrix, before)
                self._commit()
            return
        x, y, z = cell
        inside = all(0 <= c < n for c, n in zip(cell, item.matrix.size))
        old =int(item.matrix.voxels[z, y, x]) if inside else 0
        if self._apply(tool, cell):
            new = int(item.matrix.voxels[z, y, x])
            flat = np.ravel_multi_index((z, y, x), item.matrix.voxels.shape)
            self.history.push(VoxelEdit(item.matrix, np.array([flat], np.uint32),
                                        np.array([old], np.uint16), np.array([new], np.uint16)))
            self._commit()

    def fill(self, item: DrawItem) -> int:
        """Flood fill from the current fill target as one undo step. Returns how many voxels changed."""
        target = self.fill_target
        if target is None:
            return 0
        prefix = None
        if self._outside(target.cell, item.matrix):                 # appending past the edge: grow the matrix first
            prefix, cell = self._expand_for(target.cell)
            if cell is None:
                return 0
            target = FillTarget(cell, target.axis, target.sign)
            item = self.active_item()
        before = item.matrix.voxels.copy()
        try:
            changed = 0
            for each in symmetric_targets(target, item.matrix.size, tuple(self.mirror), self.radial):
                changed += apply_fill(self.model, item.matrix, self.tool, each, self.brush_rgba(),
                                      self.fill_contiguous, self.fill_tolerance)
        except ValueError as exc:   # palette full
            changed = 0
            self.status = str(exc)
        if changed:
            self._commit()
            self._record(item.matrix, before, prefix)
            self.status = f"{self.tool.value}: {changed:,} voxels"
        else:
            if prefix is not None:
                self._record(item.matrix, before, prefix)           # nothing filled: take the growing back
            if not self.status.startswith("palette"):
                self.status = f"{self.tool.value}: nothing to fill there"
        return changed

    # -- voxel selection --
    def selection_visible(self) -> bool:
        """Whether the selection is shown and acted on: only with the Select action in voxel mode, and not in picture mode.

        With another action or in matrix mode the selection is kept but hidden, and nothing treats it as selected: the
        other tools ignore it and the keys that work on it (arrows, X Y Z, Delete, Esc) do their usual jobs.
        """
        return self.mode is Mode.VOXEL and self.action is Tool.SELECT and not self.preview_mode

    def selection_mask(self) -> np.ndarray | None:
        """The selected voxels of the active matrix as a mask, or None if nothing is selected there."""
        item = self.active_item()
        return self.vsel.mask_for(item.matrix) if item is not None else None

    def selection_info(self) -> tuple[int, tuple | None]:
        """(how many voxels are selected, the box around them), or (0, None). Cached: it is asked for every frame."""
        item, mask = self.active_item(), self.selection_mask()
        if mask is None:
            return 0, None
        key = (self.vsel.version, self._voxel_generation, id(item.matrix))
        if self._sel_info_key != key:
            chosen = voxsel.solid(mask, item.matrix.voxels)
            self._sel_info_key, self._sel_info = key, (int(chosen.sum()), voxsel.bounds(chosen))
        return self._sel_info

    def selection_count(self) -> int:
        return self.selection_info()[0]

    def set_selection(self, mask: np.ndarray | None, keep_under: bool = False) -> None:
        item = self.active_item()
        self.vsel.set(item.matrix, mask, keep_under) if item is not None else self.vsel.clear()

    def select_all(self) -> None:
        item = self.active_item()
        if item is not None:
            self.use_select_tool()                            # a selection is only shown with a select tool
            self.set_selection(voxsel.all_mask(item.matrix.voxels))
            self.status = f"Selected {self.selection_count():,} voxels"

    def deselect(self) -> None:
        self.vsel.clear()

    def _selection_edit(self, new_voxels: np.ndarray, new_mask: np.ndarray | None, label: str,
                        keep_under: bool = False, float_over_old: bool = False) -> bool:
        """Change the active matrix's voxels and the selection together as one undo step.

        ``keep_under``: it is the same selection, moved, so it keeps floating over what it floated over.
        ``float_over_old``: the new selection floats over the voxels as they were before (a paste).
        """
        item = self.active_item()
        matrix = item.matrix
        old_mask = self.selection_mask()
        before = matrix.voxels.copy()
        matrix.voxels[...] = new_voxels
        voxel_edit = VoxelEdit.between(matrix, before)
        empty = np.zeros(0, np.uint32)
        old = voxsel.indices_of(old_mask) if old_mask is not None else empty
        new = voxsel.indices_of(new_mask) if new_mask is not None else empty
        if voxel_edit is None and np.array_equal(old, new):
            return False
        self.history.push(SelectionEdit(voxel_edit, self.vsel, matrix, old, new, f"{label} in {matrix.name}",
                                        floating=keep_under))
        self.set_selection(new_mask, keep_under)
        if float_over_old and new_mask is not None:
            self.vsel.start_floating_over(matrix, before)
        self._commit()
        return True

    def _selection_action(self, work, what: str) -> bool:
        """Run ``work(voxels, mask, under)``, which returns (new voxels, new mask) or None, and record it.

        The selection floats: ``under`` is what is beneath it, so the voxels it covers are not lost.
        """
        item, mask = self.active_item(), self.selection_mask()
        if item is None or mask is None or self._busy():
            return False
        under = self.vsel.under_for(item.matrix, item.matrix.voxels, mask)
        result = work(item.matrix.voxels, mask, under)
        if result is None:
            self.status = f"Cannot {what}: it would not fit in the matrix"
            return False
        done = self._selection_edit(result[0], result[1], f"{what} of {self.selection_count():,} voxels",
                                    keep_under=True)
        if done:
            self.status = f"{what[0].upper()}{what[1:]}: {self.selection_count():,} voxels. Ctrl+Z undoes it."
        return done

    def selection_nudge(self, offset: tuple[int, int, int]) -> bool:
        def work(voxels, mask, under):
            new, new_mask, used = voxsel.move(voxels, mask, offset, under)
            return (new, new_mask) if used != (0, 0, 0) else None
        return self._selection_action(work, "move")

    def selection_flip(self, axis: int) -> bool:
        return self._selection_action(lambda v, m, u: voxsel.flip(v, m, axis, u), f"flip {'XYZ'[axis]}")

    def selection_rotate(self, axis: int, turns: int = 1) -> bool:
        return self._selection_action(lambda v, m, u: voxsel.rotate(v, m, axis, turns, u), f"turn about {'XYZ'[axis]}")

    def selection_delete(self) -> bool:
        item, mask = self.active_item(), self.selection_mask()
        if item is None or mask is None or self._busy():
            return False
        count = self.selection_count()
        under = self.vsel.under_for(item.matrix, item.matrix.voxels, mask)
        done = self._selection_edit(voxsel.delete(item.matrix.voxels, mask, under), None, f"deletion of {count:,} voxels")
        if done:
            self.status = f"Deleted {count:,} voxels"
        return done

    def copy_selection(self) -> bool:
        item, mask = self.active_item(), self.selection_mask()
        if item is None or mask is None:
            return False
        clip = voxsel.extract(item.matrix.voxels, mask, self.model.palette)
        if clip is None:
            return False
        self.voxel_clip, self.clip_kind = clip, "voxels"
        self.status = f"Copied {clip.count:,} voxels. Ctrl+V pastes them where the cursor is."
        return True

    def cut_selection(self) -> bool:
        return self.copy_selection() and self.selection_delete()

    def paste_voxels(self) -> bool:
        """Put the copied voxels into the active matrix with their middle on the camera focus; they become the selection."""
        item, clip = self.active_item(), self.voxel_clip
        if item is None or clip is None or self._busy():
            return False
        origin, where = self._paste_origin(item, clip)
        try:
            new, mask, dropped = voxsel.paste(item.matrix.voxels, clip, origin, self.model.index_for_color)
        except ValueError as exc:                     # palette full
            self.status = str(exc)
            return False
        if not self._selection_edit(new, mask, f"paste of {clip.count:,} voxels", float_over_old=True):
            self.status = "Nothing to paste"
            return False
        self.use_select_tool()                                # so the pasted block is selected and can be moved
        self.status = (f"Pasted {clip.count - dropped:,} voxels {where}"
                       + (f" ({dropped:,} did not fit)" if dropped else ""))
        return True

    def _paste_origin(self, item: DrawItem, clip: voxsel.Clip) -> tuple[tuple[int, int, int], str]:
        """Where to put a pasted block, as (its low corner, a few words saying where that is).

        With the cursor over the matrix the block is set down on whatever is under it, against that surface and
        centered along it, as when placing a voxel. Otherwise (the cursor is on a panel or the menu, or off the matrix)
        its middle goes on the camera focus. Either way it is moved in just enough to lie inside the matrix.
        """
        ray = None if imgui.get_io().want_capture_mouse else self._mouse_ray_local(item, glfw.get_framebuffer_size(self.window))
        found = place_face(item.matrix.voxels, *ray) if ray is not None else None
        if found is not None:
            origin, where = voxsel.origin_against_surface(found[0], found[1], found[2], clip.size), "at the cursor"
        else:
            focus = to_local(item.matrix, np.array(self.camera.target, float))
            origin = tuple(int(round(focus[i] - clip.size[i] / 2)) for i in range(3))
            where = "at the camera focus"
        return voxsel.fit_origin(origin, clip.size, item.matrix.size), where

    def copy_selected(self) -> bool:
        """Ctrl+C: the selected voxels if there are any, otherwise the selected matrices."""
        if self.selection_visible() and self.selection_mask() is not None:
            return self.copy_selection()
        return self.copy_selected_matrix()

    def paste(self, at_focus: bool = False) -> bool:
        """Ctrl+V: whatever was copied last, voxels or matrices. Matrices go where they were copied from, or with
        ``at_focus`` (Ctrl+Shift+V) to the camera focus."""
        if self.clip_kind == "voxels" and self.voxel_clip is not None:
            return self.paste_voxels()
        return self.paste_matrix(at_focus)

    def can_paste(self) -> bool:
        if self.clip_kind == "voxels":
            return self.voxel_clip is not None and self.active_item() is not None
        return self.clipboard is not None

    def _camera_nudges(self) -> dict:
        """What each arrow key does to a selection: the matrix's own axis that points most nearly the way the key does
        on screen. Left and Right follow the screen; Up and Down go away from and toward the camera."""
        item = self.active_item()
        inverse = glm.inverse(item.transform)
        eye, target = np.array(self.camera.eye(), float), np.array(self.camera.target, float)
        forward = target - eye
        forward[1] = 0.0
        forward = forward / np.linalg.norm(forward) if np.linalg.norm(forward) > 1e-4 else np.array([0.0, 0.0, 1.0])
        right = np.cross((0.0, 1.0, 0.0), forward)

        def local(vector):
            moved = inverse * glm.vec4(*vector, 0.0)
            return (moved.x, moved.y, moved.z)

        def step(axis: int, direction: float) -> tuple[int, int, int]:
            out = [0, 0, 0]
            out[axis] = 1 if direction >= 0 else -1
            return tuple(out)

        right_local, forward_local = local(right), local(forward)
        # Left/Right take the horizontal axis that points most nearly across the screen; Up/Down always take the other
        # one, so at a diagonal view the four arrows still go four different ways.
        across, away = (0, 2) if abs(right_local[0]) >= abs(right_local[2]) else (2, 0)
        r, f = step(across, right_local[across]), step(away, forward_local[away])
        negate = lambda step: tuple(-n for n in step)
        return {imgui.Key.right_arrow: r, imgui.Key.left_arrow: negate(r),
                imgui.Key.up_arrow: f, imgui.Key.down_arrow: negate(f)}

    def _selection_keys(self, shift: bool, escape: bool, x_key: bool, y_key: bool, z_key: bool) -> None:
        """Keys that work on the selected voxels: Esc deselects, X Y Z flip (with Shift, turn) and the arrows move."""
        if self.sel_drag is not None or self.selection_mask() is None:
            return
        if escape:
            self.deselect()
            return
        for axis, pressed in enumerate((x_key, y_key, z_key)):
            if pressed:
                self.selection_rotate(axis) if shift else self.selection_flip(axis)
                return
        nudges = self._camera_nudges()
        for key, step in nudges.items():
            if imgui.is_key_pressed(key):                    # with the usual key repeat when held
                self.selection_nudge(step)
                return
        for key, step in ((imgui.Key.page_up, (0, 1, 0)), (imgui.Key.page_down, (0, -1, 0))):
            if imgui.is_key_pressed(key):
                self.selection_nudge(step)
                return

    @staticmethod
    def _plane_point(origin, direction, axis: int, value: float):
        """Where a ray crosses the plane whose coordinate on ``axis`` is ``value``, or None if it runs along it."""
        if abs(direction[axis]) < 1e-9:
            return None
        return origin + direction * ((value - origin[axis]) / direction[axis])

    def _begin_selection_drag(self, item: DrawItem, fb_size: tuple[int, int], shift: bool) -> None:
        """The mouse went down with the Select action: start moving the selection, drag out a box, or use the wand."""
        ray = self._mouse_ray_local(item, fb_size)
        voxels, mask = item.matrix.voxels, self.selection_mask()
        hit = raycast(voxels, *ray) if ray is not None else None
        if hit is not None and mask is not None and not shift and mask[hit.cell[2], hit.cell[1], hit.cell[0]]:
            # a selected face was grabbed: drag the selection, with either select tool
            origin, direction = np.array(ray[0], float), np.array(ray[1], float)
            axis, plane = voxsel.grab_plane(hit.cell, hit.normal, direction)   # the face that was grabbed
            point = self._plane_point(origin, direction, axis, plane)
            if point is not None:
                under = self.vsel.under_for(item.matrix, voxels, mask)
                self.sel_drag = SelectionDrag("move", old_mask=mask.copy(), base=voxels.copy(), axis=axis,
                                              plane=plane, point=point, under=under)
                return
        if self.brush is Brush.FILL:                               # the wand: every connected voxel of the same color
            if hit is None:
                if not shift:
                    self.deselect()
                return
            chosen = voxsel.wand_mask(voxels, hit.cell, self.model.palette, self.fill_contiguous, self.fill_tolerance)
            self.set_selection(chosen | mask if (shift and mask is not None) else chosen)
            self.status = f"Selected {self.selection_count():,} voxels"
            return
        cell = find_target(voxels, Tool.SELECT, *ray, back_face=True) if ray is not None else None
        if cell is None:
            if not shift:
                self.deselect()
            return
        self.sel_drag = SelectionDrag("box", old_mask=mask, add=shift, start=cell, end=cell)
        if not shift and mask is not None:
            self.deselect()                                      # the old selection goes at once, not when the box is let go

    def _update_selection_drag(self, item: DrawItem, fb_size: tuple[int, int]) -> None:
        drag = self.sel_drag
        ray = self._mouse_ray_local(item, fb_size)
        if ray is None:
            return
        if drag.kind == "box":
            cell = find_target(item.matrix.voxels, Tool.SELECT, *ray, back_face=True)
            if cell is not None:
                drag.end = cell
            return
        point = self._plane_point(np.array(ray[0], float), np.array(ray[1], float), drag.axis, drag.plane)
        if point is None:
            return
        offset = tuple(0 if i == drag.axis else int(round(point[i] - drag.point[i])) for i in range(3))
        new, new_mask, used = voxsel.move(drag.base, drag.old_mask, offset, drag.under)
        if used != drag.offset:                                  # a live preview, put back if the drag is cancelled
            drag.offset = used
            item.matrix.voxels[...] = new
            self.vsel.set(item.matrix, new_mask, keep_under=True)
            self._rebuild_item(self.active)

    def _finish_selection_drag(self, item: DrawItem) -> None:
        drag, self.sel_drag = self.sel_drag, None
        matrix = item.matrix
        if drag.kind == "box":
            lo = tuple(min(a, b) for a, b in zip(drag.start, drag.end))
            hi = tuple(max(a, b) for a, b in zip(drag.start, drag.end))
            chosen = voxsel.box_mask(matrix.voxels, lo, hi)
            if drag.add and drag.old_mask is not None:
                chosen = chosen | drag.old_mask
            self.set_selection(chosen if chosen.any() else None)
            if chosen.any():
                self.status = f"Selected {self.selection_count():,} voxels"
            return
        if drag.offset == (0, 0, 0):
            return
        edit = VoxelEdit.between(matrix, drag.base)
        count = int(voxsel.solid(self.vsel.mask, matrix.voxels).sum())
        self.history.push(SelectionEdit(edit, self.vsel, matrix, voxsel.indices_of(drag.old_mask),
                                        voxsel.indices_of(self.vsel.mask), f"move of {count:,} voxels in {matrix.name}",
                                        floating=True))
        self.dirty = True
        self.status = f"Moved {count:,} voxels by {drag.offset[0]}, {drag.offset[1]}, {drag.offset[2]}"

    def _cancel_selection_drag(self, item: DrawItem | None) -> None:
        drag, self.sel_drag = self.sel_drag, None
        if drag is not None and drag.kind == "box" and not drag.add and drag.old_mask is not None and item is not None:
            self.vsel.set(item.matrix, drag.old_mask)            # cancelled: the selection that was replaced comes back
        if drag is not None and drag.kind == "move" and item is not None and drag.offset != (0, 0, 0):
            item.matrix.voxels[...] = drag.base
            self.vsel.set(item.matrix, drag.old_mask, keep_under=True)
            self._rebuild_item(self.active)

    def _expand_for(self, cell: tuple[int, int, int]):
        """Grow the active matrix by one cell on each side a target cell, or a mirrored or turned copy of it, lies beyond.

        Returns (the edit that did it, the target's cell in the grown matrix), or (None, None) if it could not grow. The
        edit is not in the history yet: it is recorded together with the voxels placed, as one undo step.
        """
        matrix = self.active_item().matrix
        grow = symmetric_expansion(cell, matrix.size, tuple(self.mirror), self.radial)
        if grow is None:
            return None, cell
        try:
            edit = resize_matrix(matrix, *grow)
        except ValueError as exc:
            self.status = f"Cannot grow {matrix.name}: {exc}"
            return None, None
        self._rebuild_item(self.active)                  # the grid changed shape and the matrix moved
        return edit, cell_after_expansion(cell, grow[0])

    def _outside(self, cell, matrix: Matrix) -> bool:
        """Whether the cell, or a mirrored or turned copy of it, lies beyond the matrix: it has to grow to take it."""
        if cell is None:
            return False
        return any(not all(0 <= c < n for c, n in zip(copy, matrix.size))
                   for copy in symmetric_cells(cell, matrix.size, tuple(self.mirror), self.radial))

    def _record(self, matrix: Matrix, before: np.ndarray, prefix=None) -> None:
        """Add one undo step for everything that changed in a matrix since ``before`` was copied.

        ``prefix`` is an edit that grew the matrix just before: it is undone together with the change, and taken back
        if nothing was changed after all.
        """
        edit = VoxelEdit.between(matrix, before)
        if prefix is not None:
            if edit is None:
                prefix.undo()                              # nothing was placed: the matrix does not stay bigger
                self._refresh_matrix(matrix, rebuild=True)
                return
            self.history.push(GroupEdit(f"voxels placed past the edge of {matrix.name}", [prefix, edit]))
            self.dirty = True
            return
        if edit is not None:
            self.history.push(edit)
            self.dirty = True

    def _end_stroke(self) -> None:
        stroke, self.stroke = self.stroke, None
        if stroke is not None:
            self._record(stroke.matrix, stroke.base, stroke.prefix)

    def undo(self) -> None:
        self._step(self.history.undo, "Undid", "undo")

    def redo(self) -> None:
        self._step(self.history.redo, "Redid", "redo")

    def _step(self, action, verb: str, noun: str) -> None:
        if self._busy():
            return  # finish the drag first
        edit = action()
        if edit is None:
            self.status = f"Nothing to {noun}"
            return
        if isinstance(edit, GroupEdit) and any(isinstance(e, (MergeEdit, MatrixListEdit)) for e in edit.edits):
            self._sync_items()
            targets = [e.target for e in edit.edits if isinstance(e, MergeEdit)]
            for target in dict.fromkeys(id(t) for t in targets):          # each merge target once
                self._refresh_matrix(next(t for t in targets if id(t) == target), rebuild=True)
            present = [m for m in (targets + [e.matrix for e in edit.edits if isinstance(e, MatrixListEdit)])
                       if any(m is x for x in self.model.matrices)]
            if present:
                self.active = self.model.matrices.index(present[0])
            else:
                self.active = max(-1, min(self.active, len(self.items) - 1))
        elif isinstance(edit, GroupEdit):
            changed = {id(leaf.matrix) for leaf in _leaf_edits(edit) if hasattr(leaf, "indices") or hasattr(leaf, "new_voxels")}
            for index, item in enumerate(self.items):
                if id(item.matrix) in changed:
                    self._rebuild_item(index)                         # their voxels changed
                item.transform = matrix_transform(item.matrix)
        elif isinstance(edit, MergeEdit):
            self._sync_items()
            self._refresh_matrix(edit.target, rebuild=True)
            self.active = max(0, min(self.model.matrices.index(edit.target), len(self.items) - 1))
        elif isinstance(edit, (MatrixListEdit, MatrixMoveEdit)):
            self._sync_items()
            present = [i for i, m in enumerate(self.model.matrices) if m is edit.matrix]
            self.active = present[0] if present else max(0, min(edit.index, len(self.items) - 1))
        elif isinstance(edit, PaletteEdit):
            for i in range(len(self.items)):
                self._rebuild_item(i)                              # the meshes carry the colors
        else:
            for i, item in enumerate(self.items):
                if item.matrix is edit.matrix:
                    if self.active >= 0:                      # with nothing selected, undo does not select the matrix
                        self.active = i
                    self._rebuild_item(i)
                    break
        if isinstance(edit, ReorientEdit) or (isinstance(edit, GroupEdit) and any(
                isinstance(leaf, ReorientEdit) for leaf in _leaf_edits(edit))):
            self.vsel.clear()
        self.hover = None
        self.dirty = True
        self.status = f"{verb} {edit.describe()}"

    def _view_proj(self, fb_size: tuple[int, int]) -> glm.mat4:
        return self.camera.projection(fb_size[0] / max(fb_size[1], 1)) * self.camera.view()

    def _mouse_ray_world(self, fb_size: tuple[int, int], cursor=None):
        """Cursor ray in model space as (origin, direction) numpy vectors, or None if outside the window."""
        win_w, win_h = glfw.get_window_size(self.window)
        if win_w <= 0 or win_h <= 0:
            return None
        mx, my = cursor if cursor is not None else glfw.get_cursor_pos(self.window)
        if not (0 <= mx < win_w and 0 <= my < win_h):
            return None
        nx, ny = 2 * mx / win_w - 1, 1 - 2 * my / win_h
        inv = glm.inverse(self._view_proj(fb_size))
        near, far = inv * glm.vec4(nx, ny, -1, 1), inv * glm.vec4(nx, ny, 1, 1)
        near, far = near.xyz / near.w, far.xyz / far.w
        return np.array(near, float), np.array(far - near, float)

    def _mouse_ray_local(self, item: DrawItem, fb_size: tuple[int, int], cursor=None):
        """Cursor ray in the local space of an item, as (origin, direction), or None."""
        ray = self._mouse_ray_world(fb_size, cursor)
        if ray is None:
            return None
        to_local = glm.inverse(item.transform)
        origin = to_local * glm.vec4(*ray[0], 1)
        direction = to_local * glm.vec4(*ray[1], 0)
        return tuple(origin.xyz), tuple(direction.xyz)

    def _edit_pool(self) -> list[int]:
        """The matrices the voxel tools may change: the selected ones, or every visible one if none is selected."""
        if self.active_item() is None:
            return [i for i, item in enumerate(self.items) if item.visible]
        return [i for i in self.selected_indices() if self.items[i].visible]

    def _edit_target(self, fb_size: tuple[int, int]) -> int | None:
        """The matrix under the cursor that a voxel tool would act on: the first one the cursor ray hits in the pool."""
        pool = self._edit_pool()
        if len(pool) <= 1:
            return pool[0] if pool else None
        ray = self._mouse_ray_world(fb_size)
        if ray is None:
            return None
        allowed = set(pool)
        found = pick_matrix([item.matrix if i in allowed else None for i, item in enumerate(self.items)], ray)
        return found

    def _hover_item(self) -> DrawItem | None:
        if self.hover_index is not None and 0 <= self.hover_index < len(self.items):
            return self.items[self.hover_index]
        return self.active_item()

    def _take_target(self, index: int, keep: bool) -> None:
        """Make the matrix a voxel tool is about to change the active one, for the length of that edit.

        Edits all work on the active matrix. With nothing (or several matrices) selected the tool picks its matrix
        by what the cursor is over, so the selection is put back afterwards, unless ``keep`` (the Select action,
        where clicking a matrix's voxels does select it).
        """
        if index == self.active:
            return
        if not keep and self._target_restore is None:
            self._target_restore = (self.active, self.selection)
        self.active = index

    def _release_target(self) -> None:
        if self._target_restore is not None:
            self.active, self.selection = self._target_restore
            self._target_restore = None
            if self.active >= len(self.items):
                self.active = -1

    def _update_hover(self, fb_size: tuple[int, int]) -> None:
        self.hover_face = None
        if self.stroke is not None or self.box is not None or self.sel_drag is not None:
            return  # a stroke sets the hover itself; a box draws its own outline
        self.hover = None
        self.hover_index = None
        self.fill_target = None
        if self.mode is not Mode.VOXEL:
            return
        index = None if (self._drag is not None or imgui.get_io().want_capture_mouse) else self._edit_target(fb_size)
        item = self.items[index] if index is not None else None
        self.hover_index = index
        if (item is None or not item.visible or self._drag is not None
                or imgui.get_io().want_capture_mouse
                or not glfw.get_window_attrib(self.window, glfw.FOCUSED)
                or self._key(glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT)):
            return
        ray = self._mouse_ray_local(item, fb_size)
        if ray is not None and self.tool in FILL_TOOLS:
            self.fill_target = pick_fill(item.matrix.voxels, self.tool, *ray,
                                         outside=self.expand_on_edge and self.tool is Tool.FILL_APPEND)
            self.hover = self.fill_target.cell if self.fill_target else None
        elif ray is not None and self.tool is Tool.PLACE:
            single = self._effective_brush(self._key(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)) is Brush.VOXEL
            found = place_face(item.matrix.voxels, *ray, outside=self.expand_on_edge and single)   # beyond the edge too
            if found is not None:
                self.hover, self.hover_face = found[0], found[1:]
        elif ray is not None:
            shift = self._key(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
            # A box may start on an empty back-face cell for any action.
            boxing = self._effective_brush(shift) is Brush.BOX
            # Erase and Place may start a box on an empty back-face cell; Paint must start on a voxel.
            self.hover = find_target(item.matrix.voxels, self.tool, *ray,
                                     back_face=(boxing and self.action is not Tool.PAINT) or self.action is Tool.SELECT)

    # -- matrix mode --
    def set_mode(self, mode: Mode) -> None:
        busy = self._busy()
        if busy or mode is self.mode:
            return
        self.mode = mode
        self.hover = None
        self.matrix_tools.hover = None
        self.status = f"{mode.value} mode"

    def set_preview(self, on: bool) -> None:
        """Picture mode: hide every panel, outline, grid and handle and show just the model."""
        if on == self.preview_mode or self._busy() or (on and self.model is None):
            return
        self.preview_mode = on
        self._preview_since = time.perf_counter()
        self.hover = None
        self.fill_target = None
        self.matrix_tools.hover = None

    def toggle_mode(self) -> None:
        self.set_mode(Mode.MATRIX if self.mode is Mode.VOXEL else Mode.VOXEL)

    def _refresh_matrix(self, matrix: Matrix, rebuild: bool) -> None:
        """Bring the drawn item for a matrix in line with it after its transform or voxels changed."""
        for i, item in enumerate(self.items):
            if item.matrix is matrix:
                if rebuild:
                    self._rebuild_item(i)
                else:
                    item.transform = matrix_transform(matrix)

    def _push_matrix_edit(self, edit, rebuild: bool) -> None:
        self.history.push(edit)
        self.dirty = True
        self._refresh_matrix(edit.matrix, rebuild)

    def set_matrix_transform(self, item: DrawItem, **changes) -> None:
        """Set position / pivot / rotation of a matrix as one undo step (values as set_transform takes them)."""
        try:
            edit = set_transform(item.matrix, **changes)
        except ValueError as exc:
            self.status = str(exc)
            return
        if edit is not None:
            self._push_matrix_edit(edit, rebuild=False)

    def reorient_selected(self, kind: str, axis: int, reverse: bool = False) -> bool:
        """Turn ("turn") or mirror ("flip") the voxels inside every selected matrix, as one undo step.

        Each matrix is changed about its own center and keeps its rotation; see ``turn_matrix`` and ``flip_matrix``.
        ``reverse`` turns the other way.
        """
        if self._busy():
            return False
        matrices = [self.items[i].matrix for i in self.selected_indices()]
        if not matrices:
            return False
        edits = []
        try:
            for matrix in matrices:
                edit = flip_matrix(matrix, axis) if kind == "flip" else turn_matrix(matrix, axis, -1 if reverse else 1)
                if edit is not None:
                    edits.append(edit)
        except ValueError as exc:
            for edit in reversed(edits):
                edit.undo()
            self.status = f"Cannot change the matrix: {exc}"
            return False
        what = f"{'Flipped' if kind == 'flip' else 'Turned'} about {'XYZ'[axis]}"
        if not edits:
            self.status = "Nothing changed: the matrix looks the same that way"
            return False
        self.history.push(edits[0] if len(edits) == 1 else GroupEdit(f"{kind} of {len(edits)} matrices", edits))
        for index, item in enumerate(self.items):
            if any(item.matrix is e.matrix for e in edits):
                self._rebuild_item(index)
            item.transform = matrix_transform(item.matrix)
        self.vsel.clear()                                  # the grid changed under any selected voxels
        self.hover = None
        self.dirty = True
        self.status = f"{what}: {', '.join(e.matrix.name for e in edits[:3])}{'...' if len(edits) > 3 else ''}. Ctrl+Z undoes it."
        return True

    def _refine(self, steps_by_matrix: list[tuple[Matrix, int]], what: str) -> bool:
        """Change the detail of matrices (each by its own number of steps, see ``refine``) as one undo step.

        Either all of them change or none does. ``what`` starts the status message.
        """
        done, edits = [], []
        try:
            for matrix, steps in steps_by_matrix:
                edit = refine(matrix, steps)
                if edit is not None:
                    edits.append(edit)
                    done.append(matrix)
        except ValueError as exc:
            for edit in reversed(edits):
                edit.undo()
            self.status = f"Cannot change the detail: {exc}"
            return False
        if not edits:
            return False
        self.history.push(edits[0] if len(edits) == 1 else GroupEdit(f"change of detail for {len(edits)} matrices", edits))
        for item in self.items:
            if any(item.matrix is m for m in done):
                self._rebuild_item(self.items.index(item))       # the voxels changed
            item.transform = matrix_transform(item.matrix)
        self.hover = None
        self.dirty = True
        sizes = ", ".join(f"{m.name} {m.size[0]}x{m.size[1]}x{m.size[2]}" for m in done[:3]) + ("..." if len(done) > 3 else "")
        self.status = f"{what}: {sizes}, resolution {done[0].resolution}. The size is kept. Ctrl+Z undoes it."
        return True

    def set_matrix_resolution(self, item: DrawItem, level: int) -> None:
        """Set a matrix's resolution (1 to 8) as one undo step, keeping its pivot point where it is."""
        level = max(1, min(int(level), MAX_RESOLUTION))
        try:
            edit = set_resolution(item.matrix, level)
        except ValueError as exc:
            self.status = str(exc)
            return
        if edit is not None:
            self._push_matrix_edit(edit, rebuild=False)
            self.status = f"{item.matrix.name}: resolution {level} ({cells_per_unit(item.matrix)} per unit)"

    def scale_matrix(self, item: DrawItem, double: bool) -> bool:
        """Double (every cell becomes 2x2x2) or halve (keep each 2x2x2 block's corner) a matrix, as one undo step."""
        try:
            edit = (scale_double if double else scale_half)(item.matrix)
        except ValueError as exc:
            self.status = f"Cannot scale {item.matrix.name}: {exc}"
            return False
        self._push_matrix_edit(edit, rebuild=True)
        x, y, z = item.matrix.size
        self.status = f"{item.matrix.name} {'doubled' if double else 'halved'} to {x}x{y}x{z}"
        return True

    def _push_group(self, edits: list, label: str, rebuild: set) -> None:
        """Record edits to several matrices as one undo step and refresh them (``rebuild``: ids of those whose voxels changed)."""
        self.history.push(edits[0] if len(edits) == 1 else GroupEdit(label, edits))
        for index, item in enumerate(self.items):
            if id(item.matrix) in rebuild:
                self._rebuild_item(index)
            item.transform = matrix_transform(item.matrix)
        self.vsel.clear()
        self.hover = None
        self.dirty = True

    def scale_selected(self, double: bool) -> bool:
        """Scale double or half every selected matrix, all as one undo step. Either all change or none does."""
        matrices = [self.items[i].matrix for i in self.selected_indices()]
        if not matrices or self._busy():
            return False
        edits = []
        try:
            for matrix in matrices:
                edits.append((scale_double if double else scale_half)(matrix))
        except ValueError as exc:
            for edit in reversed(edits):
                edit.undo()
            self.status = f"Cannot scale: {exc}"
            return False
        what = "doubled" if double else "halved"
        self._push_group(edits, f"scale {'double' if double else 'half'} of {len(edits)} matrices", {id(m) for m in matrices})
        self.status = (f"{matrices[0].name} {what}" if len(matrices) == 1 else f"{len(matrices)} matrices {what}") + ". Ctrl+Z undoes it."
        return True

    def center_selected_pivots(self) -> bool:
        """Put the pivot of every selected matrix in its middle, without moving any of them, as one undo step."""
        matrices = [self.items[i].matrix for i in self.selected_indices()]
        if not matrices or self._busy():
            return False
        edits = []
        try:
            for matrix in matrices:
                edit = move_pivot_in_place(matrix, tuple(n * POSITION_SCALE // 2 for n in matrix.size))
                if edit is not None:
                    edits.append(edit)
        except ValueError as exc:
            for edit in reversed(edits):
                edit.undo()
            self.status = str(exc)
            return False
        if not edits:
            self.status = "The pivot is already in the middle"
            return False
        self._push_group(edits, f"centered pivot of {len(edits)} matrices", set())
        self.status = "Pivot centered. Ctrl+Z undoes it."
        return True

    def set_all_resolutions(self, level: int, selected_only: bool = False) -> bool:
        """Give every matrix (or every selected one) this resolution as one undo step. Each is scaled about the
        world origin."""
        if self.model is None or self._busy():
            return False
        level = max(1, min(int(level), MAX_RESOLUTION))
        matrices = [self.items[i].matrix for i in self.selected_indices()] if selected_only else None
        try:
            edit = set_all_resolutions(self.model, level, matrices)
        except ValueError as exc:
            self.status = str(exc)
            return False
        if edit is None:
            self.status = ("The selected matrices are" if selected_only else "Every matrix is") + \
                f" already at resolution {level}"
            return False
        self.history.push(edit)
        for item in self.items:
            item.transform = matrix_transform(item.matrix)
        self.hover = None
        self.dirty = True
        self.status = f"Set {len(edit.edits)} matrices to resolution {level}. Ctrl+Z undoes it."
        return True

    def step_resolution(self, delta: int, keep_size: bool = False) -> bool:
        """Make the cells of every selected matrix finer (+1, the ] key) or coarser (-1, the [ key) by one resolution.

        All of them change as one undo step, each scaled about the world origin like any resolution change. A matrix
        that is already at the end of the range stays as it is while the others move on.

        With ``keep_size`` (Shift with the key) the cells are also doubled or halved, so the matrix keeps its size and
        only its detail changes: the same as the resolution change followed by Scale double or half.
        """
        indices = self.selected_indices()
        if not indices or self._busy():
            return False
        matrices = [self.items[i].matrix for i in indices]
        movable = [m for m in matrices if 1 <= m.resolution + delta <= MAX_RESOLUTION]
        if not movable:
            only = matrices[0]
            self.status = (f"{only.name} is already at resolution {only.resolution}" if len(matrices) == 1 else
                           f"All {len(matrices)} selected matrices are already at resolution "
                           f"{'1' if delta < 0 else MAX_RESOLUTION}")
            return False
        if keep_size:
            return self._refine([(m, delta) for m in movable], "Finer detail" if delta > 0 else "Coarser detail")
        if len(matrices) == 1:
            self.set_matrix_resolution(self.items[indices[0]], matrices[0].resolution + delta)
            return True
        edits = []
        try:
            for matrix in movable:
                edit = set_resolution(matrix, matrix.resolution + delta)
                if edit is not None:
                    edits.append(edit)
        except ValueError as exc:                       # one could not be scaled: leave all of them as they were
            for edit in reversed(edits):
                edit.undo()
            self.status = str(exc)
            return False
        self.history.push(GroupEdit(f"resolution {'+1' if delta > 0 else '-1'} for {len(edits)} matrices", edits))
        for item in self.items:
            item.transform = matrix_transform(item.matrix)
        self.hover = None
        self.dirty = True
        skipped = len(matrices) - len(movable)
        self.status = (f"Resolution {'finer' if delta > 0 else 'coarser'} for {len(edits)} matri{'x' if len(edits) == 1 else 'ces'}"
                       + (f" ({skipped} already at the limit)" if skipped else "") + ". Ctrl+Z undoes it.")
        return True

    def set_matrix_pivot(self, item: DrawItem, pivot) -> None:
        try:
            edit = move_pivot_in_place(item.matrix, pivot)
        except ValueError as exc:
            self.status = str(exc)
            return
        if edit is not None:
            self._push_matrix_edit(edit, rebuild=False)

    def resize_matrix_by(self, item: DrawItem, low, high) -> None:
        """Add (or, if negative, crop) cells on the low and high side of each axis as one undo step."""
        try:
            edit = resize_matrix(item.matrix, tuple(low), tuple(high))
        except ValueError as exc:
            self.status = f"Could not resize: {exc}"
            return
        self._push_matrix_edit(edit, rebuild=True)
        x, y, z = item.matrix.size
        self.status = f"Resized {item.matrix.name} to {x}x{y}x{z}"

    def _gizmo_group(self) -> tuple[list[Matrix], np.ndarray | None]:
        """The other selected matrices and where the move and rotate handles sit.

        With one matrix selected the handles sit on its pivot (anchor None); with several they sit at their
        center and the tools act on all of them.
        """
        active = self.active_item()
        chosen = [self.items[i].matrix for i in self.selected_indices()]
        others = [m for m in chosen if active is None or m is not active.matrix]
        return others, (group_anchor(chosen) if others else None)

    def _finish_gizmo(self, commit: bool) -> None:
        tools = self.matrix_tools
        members = tools.drag.matrices()
        edit, error = tools.end_drag(commit)
        if error:
            self.status = f"Could not resize: {error}"
        if edit is not None:
            if isinstance(edit, GroupEdit):                # several matrices moved or turned together
                self.history.push(edit)
                self.dirty = True
                for member in members:
                    self._refresh_matrix(member, rebuild=False)
            else:
                self._push_matrix_edit(edit, rebuild=hasattr(edit, "new_voxels"))
            self.status = f"Changed {edit.describe()}"
        else:
            for member in members:
                self._refresh_matrix(member, rebuild=False)

    def _handle_matrix_input(self, fb_size: tuple[int, int]) -> None:
        tools = self.matrix_tools
        lmb = self._mouse_down(glfw.MOUSE_BUTTON_LEFT)
        item = self.active_item()
        alt = self._key(glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT)
        ctrl = self._key(glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL)
        camera_busy = self._drag is not None or alt

        if tools.drag is not None:
            if item is None or camera_busy or self._key(glfw.KEY_ESCAPE):
                self._finish_gizmo(commit=False)
            elif not lmb:
                self._finish_gizmo(commit=True)
            else:
                ray = self._mouse_ray_world(fb_size)
                if ray is not None and tools.update_drag(ray, snap=ctrl):
                    for member in tools.drag.matrices():
                        self._refresh_matrix(member, rebuild=False)
        else:
            tools.hover = None
            free = (not camera_busy and not imgui.get_io().want_capture_mouse
                    and glfw.get_window_attrib(self.window, glfw.FOCUSED))
            if free:
                width, height = glfw.get_window_size(self.window)
                if item is not None and item.visible:
                    eye = np.array(self.camera.eye(), float)
                    cursor = glfw.get_cursor_pos(self.window)
                    tools.update_hover(item.matrix, eye, np.array(self._view_proj(fb_size), float),
                                       cursor, width, height, self._gizmo_group()[1])
                if lmb and not self._lmb_was_down:
                    ray = self._mouse_ray_world(fb_size)
                    if ray is not None and not (tools.hover is not None and item is not None
                                                and tools.begin_drag(item.matrix, ray, *self._gizmo_group())):
                        picked = pick_matrix([i.matrix if i.visible else None for i in self.items], ray)
                        if picked is not None and ctrl:
                            self.toggle_selection(picked)         # Ctrl+click adds it to the selection or drops it
                        elif picked is not None:
                            self.active = picked
                            self.selection = set()
                        elif not ctrl:
                            self.deselect_matrices()              # a click on empty space drops the selection
        self._lmb_was_down = lmb

    def select_all_matrices(self) -> None:
        """Select every matrix."""
        if self._busy() or not self.items:
            return
        self.selection = {id(item.matrix) for item in self.items}
        if self.active < 0:
            self.active = 0
        self.vsel.clear()
        self.status = f"Selected {len(self.items)} matrices"

    def select_all_smart(self) -> None:
        """Ctrl+A: all the voxels of the matrix while a select tool is in use, otherwise all the matrices."""
        if self.mode is Mode.VOXEL and self.selection_visible() and self.active_item() is not None:
            self.select_all()
        else:
            self.select_all_matrices()

    def deselect_smart(self) -> None:
        """Ctrl+D: drop the voxel selection while a select tool is in use, otherwise select no matrix."""
        if self.mode is Mode.VOXEL and self.selection_visible():
            self.deselect()
        else:
            self.deselect_matrices()

    def deselect_matrices(self) -> None:
        """Select no matrix at all."""
        if self._busy():
            return
        self.active = -1
        self.selection = set()
        self.vsel.clear()
        self.hover = None

    def _ctrl_click_matrix(self, fb_size: tuple[int, int]) -> None:
        """Add the matrix under the cursor to the selection, or take it out. Clicking empty space does nothing."""
        ray = self._mouse_ray_world(fb_size)
        if ray is None:
            return
        picked = pick_matrix([i.matrix if i.visible else None for i in self.items], ray)
        if picked is not None:
            self.toggle_selection(picked)

    def _handle_editing_input(self, fb_size: tuple[int, int]) -> None:
        in_flight = self.stroke is not None or self.box is not None or self.sel_drag is not None
        pressing = self._mouse_down(glfw.MOUSE_BUTTON_LEFT) and not self._lmb_was_down
        if (pressing and not in_flight and self.hover is not None and self.hover_index is not None
                and not self._key(glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL)
                and not self._key(glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT) and self._drag is None):
            self._take_target(self.hover_index, keep=self.action is Tool.SELECT)
        self._edit_input(fb_size)
        if self.stroke is None and self.box is None and self.sel_drag is None:
            self._release_target()                    # the edit (if any) is over

    def _edit_input(self, fb_size: tuple[int, int]) -> None:
        lmb = self._mouse_down(glfw.MOUSE_BUTTON_LEFT)
        item = self.active_item()
        alt = self._key(glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT)
        shift = self._key(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        interrupted = item is None or self._drag is not None or alt
        if self.box is not None:
            # A right click (look, or zoom with Alt) leaves the box alone; other camera drags and Alt cancel it.
            right_drag = self._drag in ("look", "zoom") and not alt
            if (interrupted and not (right_drag and item is not None)) or self._key(glfw.KEY_ESCAPE):
                self._cancel_box(item)
            elif not lmb:
                self._finish_box(item)
            elif not imgui.get_io().want_capture_mouse:
                ray = self._mouse_ray_local(item, fb_size)
                cell = find_target(self.box.base, self.box.tool, *ray, back_face=True) if ray else None
                if cell is not None and cell != self.box.end:
                    self.box.end = cell
                    if self.box.live:
                        self._preview_box(item)
        elif self.stroke is not None:
            if not lmb or interrupted:
                self._end_stroke()
            elif not imgui.get_io().want_capture_mouse:
                self._continue_stroke(item, fb_size)
        elif self.sel_drag is not None:
            if interrupted or self._key(glfw.KEY_ESCAPE):
                self._cancel_selection_drag(item)
            elif not lmb:
                self._finish_selection_drag(item)
            elif not imgui.get_io().want_capture_mouse:
                self._update_selection_drag(item, fb_size)
        elif (lmb and not self._lmb_was_down and not alt and self._key(glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL)
              and self._drag is None and not imgui.get_io().want_capture_mouse):
            self._ctrl_click_matrix(fb_size)         # Ctrl+click selects matrices; it never edits voxels
        elif (lmb and not self._lmb_was_down and self.action is Tool.SELECT and item is not None and not alt
              and self._drag is None and not imgui.get_io().want_capture_mouse):
            self._begin_selection_drag(item, fb_size, shift)
        elif lmb and not self._lmb_was_down and self.hover is not None and item is not None:
            brush = self._effective_brush(shift)
            if self.action is Tool.PICK:
                self.edit(self.action, self.hover)
            elif brush is Brush.FILL:
                self.fill(item)
            elif brush is Brush.BOX:
                x, y, z = self.hover
                # A paint box replaces the color of the voxel it started on, and only that color.
                only = int(item.matrix.voxels[z, y, x]) if self.action is Tool.PAINT else None
                self.box = BoxDrag(self.action, self.hover, self.hover, item.matrix.voxels.copy(),
                                   self.model.palette.copy(), self.live_box_preview, only)
                if self.box.live:
                    self._preview_box(item)
            else:
                prefix = None
                if self.action is Tool.PLACE and self._outside(self.hover, item.matrix):
                    prefix, cell = self._expand_for(self.hover)          # a face on the edge: grow the matrix by one
                    if cell is None:
                        self._lmb_was_down = lmb
                        return
                    item = self.active_item()
                self.stroke = Stroke(self.action, item.matrix, item.matrix.voxels.copy(),
                                     glfw.get_cursor_pos(self.window), prefix)
                self._continue_stroke(item, fb_size, first=True)
        self._lmb_was_down = lmb

    def _preview_box(self, item: DrawItem) -> None:
        """Reset the matrix to its pre-drag state, then fill the current box into it."""
        box = self.box
        item.matrix.voxels[...] = box.base
        try:
            self._apply_box(item, box)
        except ValueError as exc:
            self.status = str(exc)
        self._rebuild_item(self.active)

    def _apply_box(self, item: DrawItem, box: BoxDrag) -> bool:
        """Apply a box drag, and its reflections across the mirror planes."""
        changed = False
        if self.radial:
            cells = symmetric_box_cells(*box.corners(), item.matrix.size, tuple(self.mirror), self.radial)
            return apply_cells(self.model, item.matrix, box.tool, cells, self.brush_rgba(), box.only_color)
        for a, b in mirror_boxes(*box.corners(), item.matrix.size, tuple(self.mirror)):
            changed |= apply_box(self.model, item.matrix, box.tool, a, b, self.brush_rgba(), box.only_color)
        return changed

    def _cancel_box(self, item: DrawItem | None) -> None:
        box, self.box = self.box, None
        if box is not None and box.live and item is not None:
            item.matrix.voxels[...] = box.base
            self.model.palette = box.palette
            self._rebuild_item(self.active)

    def _finish_box(self, item: DrawItem) -> None:
        box, self.box = self.box, None
        if box.live:
            # The preview already holds the filled box; keep it if it changed anything.
            if np.array_equal(item.matrix.voxels, box.base):
                self.model.palette = box.palette
            else:
                self._record(item.matrix, box.base)
            return
        try:
            changed = self._apply_box(item, box)
        except ValueError as exc:
            self.status = str(exc)
            return
        if changed:
            self._commit()
            self._record(item.matrix, box.base)

    def _continue_stroke(self, item: DrawItem, fb_size: tuple[int, int], first: bool = False) -> None:
        """Apply the tool along the cursor path since the last frame."""
        stroke = self.stroke
        x0, y0 = stroke.last_cursor
        x1, y1 = glfw.get_cursor_pos(self.window)
        steps = 1 if first else min(max(int(math.hypot(x1 - x0, y1 - y0) / 3), 1), 64)  # no gaps on fast moves
        changed = False
        for i in range(1, steps + 1):
            t = i / steps
            ray = self._mouse_ray_local(item, fb_size, (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            if ray is None:
                continue
            cell = find_target(stroke.base, stroke.tool, *ray)
            if cell is None:
                continue
            self.hover = cell
            changed |= self._apply(stroke.tool, cell)
        stroke.last_cursor = (x1, y1)
        if changed:
            self._commit()

    # -- input --
    def _on_focus(self, window, focused) -> None:
        if self._imgui_focus:
            self._imgui_focus(window, focused)
        if focused:
            self._swallow_press = 2

    def _on_mouse_button(self, window, button, action, mods) -> None:
        if action == glfw.PRESS and self._swallow_press:
            self._swallowed.add(button)              # neither the editor nor ImGui ever sees this press
            self._swallow_press = 0
            return
        if action == glfw.RELEASE and button in self._swallowed:
            self._swallowed.discard(button)
            return
        if self._imgui_mouse_button:
            self._imgui_mouse_button(window, button, action, mods)

    def _mouse_down(self, button: int) -> bool:
        """Whether a mouse button is held, not counting a click that only focused the window."""
        if glfw.get_mouse_button(self.window, button) != glfw.PRESS:
            self._swallowed.discard(button)
            return False
        return button not in self._swallowed

    def _on_scroll(self, window, dx, dy) -> None:
        if self._imgui_scroll:
            self._imgui_scroll(window, dx, dy)
        # While a camera drag is active the cursor is hidden and ImGui may think it is over
        # a panel, so the wheel must not depend on want_capture_mouse then.
        if self._drag is not None or not imgui.get_io().want_capture_mouse:
            self.camera.scroll(dy)

    def _key(self, *keys: int) -> bool:
        return any(glfw.get_key(self.window, k) == glfw.PRESS for k in keys)

    def _pressed_once(self, key: int) -> bool:
        """True on the frame a key goes down."""
        down = self._key(key)
        was = key in self._keys_down
        (self._keys_down.add if down else self._keys_down.discard)(key)
        return down and not was

    def _handle_shortcuts(self) -> None:
        typing = imgui.get_io().want_text_input
        ctrl = self._key(glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL)
        shift = self._key(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        p_key, escape = self._pressed_once(glfw.KEY_P), self._pressed_once(glfw.KEY_ESCAPE)
        if self.preview_mode:                      # only leaving picture mode (and framing the model) works here
            if p_key or escape:
                self.set_preview(False)
            elif self._pressed_once(glfw.KEY_SPACE):
                self.focus_active_matrix()
            return
        if p_key and not ctrl and not typing:
            self.set_preview(True)
            return
        o, s_, n = (self._pressed_once(k) for k in (glfw.KEY_O, glfw.KEY_S, glfw.KEY_N))
        digits = [self._pressed_once(k) for k in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5, glfw.KEY_6,
                                                  glfw.KEY_7)]
        e_key, r_key, k_key, f_key, t_key = (self._pressed_once(k) for k in (
            glfw.KEY_E, glfw.KEY_R, glfw.KEY_K, glfw.KEY_F, glfw.KEY_T))
        x_key, a_key = self._pressed_once(glfw.KEY_X), self._pressed_once(glfw.KEY_A)
        brush_keys = [self._pressed_once(k) for k in (glfw.KEY_V, glfw.KEY_B)]       # only used with Ctrl: paste, the sidebar
        space = self._pressed_once(glfw.KEY_SPACE)
        f2 = self._pressed_once(glfw.KEY_F2)
        tab = self._pressed_once(glfw.KEY_TAB)
        z, y = self._pressed_once(glfw.KEY_Z), self._pressed_once(glfw.KEY_Y)
        d, delete = self._pressed_once(glfw.KEY_D), self._pressed_once(glfw.KEY_DELETE)
        up, down = self._pressed_once(glfw.KEY_UP), self._pressed_once(glfw.KEY_DOWN)
        merge_key = self._pressed_once(glfw.KEY_M)
        c_key = self._pressed_once(glfw.KEY_C)
        v_key = brush_keys[0]          # V is also the Voxel brush key: a key press can only be read once per frame
        w = self._pressed_once(glfw.KEY_W)
        g_key = self._pressed_once(glfw.KEY_G)
        coarser, finer = self._pressed_once(glfw.KEY_LEFT_BRACKET), self._pressed_once(glfw.KEY_RIGHT_BRACKET)
        if typing:
            return
        if self.selection_visible() and not ctrl:
            self._selection_keys(shift, escape, x_key, y, z)
        if tab and ctrl:
            self.switch_tab((self.current + (-1 if shift else 1)) % len(self.tabs))
        elif tab:
            self.toggle_mode()
        if f2:
            self.start_rename()
        if space and not ctrl:
            if not self.focus_hovered_voxel():
                self.focus_active_matrix()
        if ctrl and o:
            self._choose_file()
        elif ctrl and s_:
            self.save(save_as=shift)
        elif ctrl and n:
            self.new_model()
        elif ctrl and w:
            self.close_tab()
        elif ctrl and c_key:
            self.copy_selected()
        elif ctrl and v_key:
            self.paste(at_focus=shift)
        elif ctrl and x_key and self.selection_visible():
            self.cut_selection()
        elif ctrl and a_key:
            self.select_all_smart()
        elif ctrl and z:
            self.redo() if shift else self.undo()
        elif ctrl and y:
            self.redo()
        elif ctrl and brush_keys[1]:
            self.explorer_visible = not self.explorer_visible
        elif ctrl and d:
            self.deselect_smart()
        elif ctrl and (up or down):
            self.move_active_matrix(-1 if up else 1)
        elif not ctrl and (coarser or finer):
            self.step_resolution(1 if finer else -1, keep_size=shift)         # Shift also scales, keeping the size
        elif ctrl and merge_key:
            self.merge_selected()
        elif delete and not ctrl:
            if self.selection_visible() and self.selection_mask() is not None:
                self.selection_delete()               # with voxels selected, Delete removes them and not the matrix
            else:
                self.delete_matrix()
        elif not ctrl and self._drag != "look":
            # Tool letters. W A S D Q E fly the camera, but only while the right mouse button is held, so they are free
            # here (this branch is skipped while flying). X Y Z flip a voxel selection.
            if self.mode is Mode.VOXEL:
                for tool, pressed in ((VoxelTool.PLACE, t_key), (VoxelTool.ERASE, e_key), (VoxelTool.PAINT, r_key),
                                      (VoxelTool.PICK, k_key), (VoxelTool.SELECT_BOX, s_), (VoxelTool.FILL, f_key)):
                    if pressed:
                        self.key_voxel_tool(tool)
                for tool, pressed in zip(VOXEL_TOOLS, digits):          # the number keys pick by position in the toolbox
                    if pressed:
                        self.set_voxel_tool(tool)
            elif self.matrix_tools.drag is None:
                for tool, pressed in ((MatrixTool.MOVE, w), (MatrixTool.RESIZE, r_key), (MatrixTool.ROTATE, e_key),
                                  (MatrixTool.PIVOT, g_key)):
                    if pressed:
                        self.matrix_tools.tool = tool
                for tool, pressed in zip(MatrixTool, digits):
                    if pressed:
                        self.matrix_tools.tool = tool

    def _pick_drag(self) -> str | None:
        """Unity scene-view mapping from held buttons and modifiers."""
        down = self._mouse_down
        alt = self._key(glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT)
        ctrl = self._key(glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL)
        if down(glfw.MOUSE_BUTTON_MIDDLE):
            return "pan"
        if down(glfw.MOUSE_BUTTON_LEFT) and alt:
            return "pan" if ctrl else "orbit"
        if down(glfw.MOUSE_BUTTON_RIGHT):
            return "zoom" if alt else "look"
        return None

    def _handle_camera_input(self, dt: float, height: int) -> None:
        x, y = glfw.get_cursor_pos(self.window)
        dx, dy = x - self._last_cursor[0], y - self._last_cursor[1]
        self._last_cursor = (x, y)

        wanted = self._pick_drag()
        if self._drag is None:
            # A drag may only start over the 3D view, never from an ImGui panel.
            if wanted and not imgui.get_io().want_capture_mouse:
                self._drag = wanted
                if wanted == "look":
                    glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                    self._last_cursor = glfw.get_cursor_pos(self.window)
            return
        if wanted != self._drag:
            if self._drag == "look":
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                self._last_cursor = glfw.get_cursor_pos(self.window)
            self._drag = None
            return

        cam = self.camera
        if self._drag in ("orbit", "look"):
            cam.orbit(dx, dy)
        elif self._drag == "pan":
            cam.pan(dx, dy, height)
        elif self._drag == "zoom":
            cam.zoom_drag(dx, dy)

    def _handle_movement(self, dt: float) -> None:
        """While the right mouse button is held to look around: WASD moves the camera, Q/E go down/up, Shift is faster."""
        if self._drag != "look" or imgui.get_io().want_text_input or self._key(glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
            return  # not flying, typing, or a Ctrl shortcut such as Ctrl+S
        axis = lambda pos, neg: float(self._key(pos)) - float(self._key(neg))
        self.camera.fly_move(axis(glfw.KEY_D, glfw.KEY_A), axis(glfw.KEY_E, glfw.KEY_Q),
                             axis(glfw.KEY_W, glfw.KEY_S), dt,
                             self._key(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT))

    # -- ui --
    def _draw_menu(self) -> None:
        if not imgui.begin_main_menu_bar():
            return
        if imgui.begin_menu("File"):
            if imgui.menu_item("New", "Ctrl+N", False)[0]:
                self.new_model()
            if imgui.menu_item("Open... (.cvox or legacy .csm)", "Ctrl+O", False)[0]:
                self._choose_file()
            if imgui.menu_item("Save", "Ctrl+S", False, self.model is not None)[0]:
                self.save()
            if imgui.menu_item("Save as...", "Ctrl+Shift+S", False, self.model is not None)[0]:
                self.save(save_as=True)
            imgui.separator()
            if imgui.menu_item("Demo model", "", False)[0]:
                self._open_tab()
                self.set_model(demo_model())
                self.status = "Showing demo model"
            if imgui.menu_item("Close tab", "Ctrl+W", False)[0]:
                self.close_tab()
            imgui.separator()
            if imgui.menu_item("Quit", "", False)[0]:
                glfw.set_window_should_close(self.window, True)
            imgui.end_menu()
        if imgui.begin_menu("Edit"):
            if imgui.menu_item("Undo", "Ctrl+Z", False, self.history.can_undo)[0]:
                self.undo()
            if imgui.menu_item("Redo", "Ctrl+Y / Ctrl+Shift+Z", False, self.history.can_redo)[0]:
                self.redo()
            imgui.separator()
            has_matrix = self.active_item() is not None
            has_voxels = self.selection_visible() and self.selection_mask() is not None
            if imgui.menu_item("Copy", "Ctrl+C", False, has_matrix)[0]:
                self.copy_selected()
            if imgui.menu_item("Cut", "Ctrl+X", False, has_voxels)[0]:
                self.cut_selection()
            if imgui.menu_item("Paste", "Ctrl+V", False, self.can_paste())[0]:
                self.paste()
            if imgui.menu_item("Paste at camera focus", "Ctrl+Shift+V", False, self.can_paste())[0]:
                self.paste(at_focus=True)
            if imgui.begin_menu("Selection", has_matrix and self.mode is Mode.VOXEL):
                if imgui.menu_item("Select all voxels", "", False)[0]:
                    self.select_all()
                if imgui.menu_item("Deselect", "Esc", False, has_voxels)[0]:
                    self.deselect()
                imgui.separator()
                for axis in range(3):
                    if imgui.menu_item(f"Flip {'XYZ'[axis]}", "XYZ"[axis], False, has_voxels)[0]:
                        self.selection_flip(axis)
                for axis in range(3):
                    if imgui.menu_item(f"Turn about {'XYZ'[axis]}", f"Shift+{'XYZ'[axis]}", False, has_voxels)[0]:
                        self.selection_rotate(axis)
                imgui.separator()
                if imgui.menu_item("Delete voxels", "Del", False, has_voxels)[0]:
                    self.selection_delete()
                imgui.end_menu()
            imgui.end_menu()
        if imgui.begin_menu("Matrix"):
            item = self.active_item()
            has_matrix = item is not None
            if imgui.menu_item("Clone", "", False, has_matrix)[0]:
                self.clone_matrix()
            if imgui.menu_item("Delete", "Del", False, has_matrix)[0]:
                self.delete_matrix()
            if imgui.menu_item("Merge selected", "Ctrl+M", False, len(self.selected_indices()) > 1)[0]:
                self.merge_selected()
            if imgui.menu_item("Select all", "Ctrl+A", False, bool(self.items))[0]:
                self.select_all_matrices()
            if imgui.menu_item("Deselect all", "Ctrl+D", False, has_matrix)[0]:
                self.deselect_matrices()
            imgui.separator()
            if imgui.menu_item("Scale double", "", False, has_matrix)[0]:
                self.scale_selected(True)
            if imgui.menu_item("Scale half", "", False, has_matrix)[0]:
                self.scale_selected(False)
            if imgui.menu_item("Finer resolution", "]", False, has_matrix)[0]:
                self.step_resolution(1)
            if imgui.menu_item("Coarser resolution", "[", False, has_matrix)[0]:
                self.step_resolution(-1)
            imgui.separator()
            if imgui.menu_item("Center pivot", "", False, has_matrix)[0]:
                self.center_selected_pivots()
            imgui.separator()
            for axis, name in enumerate("XYZ"):
                if imgui.menu_item(f"Turn {name}", "", False, has_matrix)[0]:
                    self.reorient_selected("turn", axis)
            for axis, name in enumerate("XYZ"):
                if imgui.menu_item(f"Flip {name}", "", False, has_matrix)[0]:
                    self.reorient_selected("flip", axis)
            imgui.end_menu()
        if imgui.begin_menu("View"):
            _, self.ambient_occlusion = imgui.menu_item("Ambient occlusion", "", self.ambient_occlusion)
            _, self.expand_on_edge = imgui.menu_item("Grow at edges", "", self.expand_on_edge)
            imgui.separator()
            if imgui.menu_item("Preview mode", "P", False, self.model is not None)[0]:
                self.set_preview(True)
            _, self.live_box_preview = imgui.menu_item("Preview boxes while dragging", "", self.live_box_preview)
            _, self.show_grid = imgui.menu_item("Ground grid", "", self.show_grid)
            _, self.show_inactive_outlines = imgui.menu_item(
                "Outline inactive matrices", "", self.show_inactive_outlines)
            if imgui.menu_item("Frame model", "Home", False)[0] and self.model:
                self.camera.frame(*model_bounds(self.model))
            if imgui.menu_item("Focus on selection", "Space", False)[0]:
                self.focus_active_matrix()
            imgui.separator()
            if imgui.menu_item("Explorer sidebar", "Ctrl+B", self.explorer_visible)[0]:
                self.explorer_visible = not self.explorer_visible
            if imgui.menu_item("Reset window layout", "", False)[0]:
                self._reset_layout = True
                self.explorer_width = DEFAULT_EXPLORER_WIDTH
                self.right_width, self.right_split = DEFAULT_RIGHT_WIDTH, DEFAULT_RIGHT_SPLIT
            imgui.end_menu()
        if imgui.begin_menu("Workspace"):
            if imgui.menu_item("Show Workspace", "", self.workspace.open)[0]:
                self.show_sidebar_tab("workspace")
            who = self.workspace.session
            if who is not None:
                imgui.separator()
                imgui.text_disabled(f"Logged in as {who.name or who.email}")
                if imgui.menu_item("Log out", "", False)[0]:
                    self.workspace.log_out()
            imgui.end_menu()
        if imgui.begin_menu("Help"):
            if imgui.menu_item("Keybinds", "", self.show_keybinds)[0]:
                self.show_keybinds = not self.show_keybinds
            imgui.end_menu()
        imgui.end_main_menu_bar()

    def _keep_window_on_screen(self) -> None:
        """Call right after ``imgui.begin``: push the current window back inside the screen."""
        pos, size = imgui.get_window_pos(), imgui.get_window_size()
        display = imgui.get_io().display_size
        top = imgui.get_frame_height() + self._tab_bar_height()  # stay below the menu and tab bars, above the info bar
        bottom = display.y - self._bar_height()
        left = self._sidebar_width()
        x = min(max(pos.x, left), max(display.x - self._right_width() - size.x, left))
        y = min(max(pos.y, top), max(bottom - size.y, top))
        if (x, y) != (pos.x, pos.y):
            imgui.set_window_pos(imgui.ImVec2(x, y))

    def _draw_rename_field(self) -> None:
        """Inline name editor: Enter or clicking away applies, Escape cancels."""
        if self._rename_focus:
            imgui.set_keyboard_focus_here()
            self._rename_focus = False
        imgui.set_next_item_width(-1)
        flags = imgui.InputTextFlags_.enter_returns_true | imgui.InputTextFlags_.auto_select_all
        entered, self.rename_text = imgui.input_text("##rename", self.rename_text, flags)
        if imgui.is_key_pressed(imgui.Key.escape):
            self._finish_rename(commit=False)
        elif entered or imgui.is_item_deactivated():
            self._finish_rename(commit=True)

    def _right_width(self) -> float:
        """How much of the right edge the docked Model and Tools panels take (none on the home page)."""
        if self.preview_mode or self.model is None:
            return 0.0
        return self.right_width

    def _begin_dock(self, name: str, x: float, y: float, width: float, height: float) -> None:
        """Start a window fixed in place; finish it with ``_end_dock``."""
        imgui.set_next_window_pos(imgui.ImVec2(x, y))
        imgui.set_next_window_size(imgui.ImVec2(width, height))
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_min_size, imgui.ImVec2(1.0, 1.0))
        imgui.begin(name, None, DOCK_FLAGS)

    @staticmethod
    def _end_dock() -> None:
        imgui.end()
        imgui.pop_style_var(2)

    def _drag_handle(self, name: str, x: float, y: float, width: float, height: float, cursor):
        """A thin strip that lights up like a window border and can be dragged. Returns the mouse position while dragged."""
        imgui.set_next_window_pos(imgui.ImVec2(x, y))
        imgui.set_next_window_size(imgui.ImVec2(width, height))
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_background | imgui.WindowFlags_.no_move
                 | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_nav_inputs
                 | imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_focus_on_appearing)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, imgui.ImVec2(1.0, 1.0))
        imgui.begin(name, None, flags)
        imgui.invisible_button("##grab", imgui.ImVec2(width, height))
        grabbed = imgui.is_item_active()
        if grabbed or imgui.is_item_hovered():
            imgui.set_mouse_cursor(cursor)
            color = imgui.get_color_u32(imgui.Col_.separator_active if grabbed else imgui.Col_.separator_hovered)
            thick = 3.0
            if width < height:
                a, b = imgui.ImVec2(x + (width - thick) / 2, y), imgui.ImVec2(x + (width + thick) / 2, y + height)
            else:
                a, b = imgui.ImVec2(x, y + (height - thick) / 2), imgui.ImVec2(x + width, y + (height + thick) / 2)
            imgui.get_window_draw_list().add_rect_filled(a, b, color)
        mouse = imgui.get_io().mouse_pos if grabbed else None
        imgui.end()
        imgui.pop_style_var(2)
        return mouse

    def _draw_right_dock(self) -> None:
        """The Model panel above the Tools panel, fixed to the right edge. Drag the left edge for the width and the
        line between them for how the height is shared."""
        display = imgui.get_io().display_size
        top = imgui.get_frame_height()
        total = max(display.y - self._bar_height() - top, 1.0)
        self.right_width = max(200.0, min(self.right_width, display.x * 0.6))
        self.right_split = max(0.1, min(self.right_split, 0.9))
        width = self.right_width
        x = display.x - width
        model_height = round(total * self.right_split)
        self._begin_dock("Model", x, top, width, model_height)
        self._draw_model_panel()
        self._end_dock()
        self._begin_dock("Tools", x, top + model_height, width, total - model_height)
        self._draw_tools_panel()
        self._end_dock()
        mouse = self._drag_handle("##dock_width", x - 2.0, top, 5.0, total, imgui.MouseCursor_.resize_ew)
        if mouse is not None:
            self.right_width = max(200.0, min(display.x - mouse.x, display.x * 0.6))
        mouse = self._drag_handle("##dock_split", x, top + model_height - 2.0, width, 5.0, imgui.MouseCursor_.resize_ns)
        if mouse is not None:
            self.right_split = max(0.1, min((mouse.y - top) / total, 0.9))

    def _draw_new_matrix_window(self) -> None:
        """The window the New button opens: a name and a size, then Create."""
        if not self.show_new_matrix:
            return
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(imgui.ImVec2((display.x - self._right_width()) / 2, display.y / 3),
                                  imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.0))
        opened, self.show_new_matrix = imgui.begin("New matrix", True, PANEL_FLAGS | imgui.WindowFlags_.always_auto_resize)
        if opened:
            imgui.set_next_item_width(220)
            _, self.new_matrix_name = imgui.input_text_with_hint("Name", "automatic", self.new_matrix_name)
            changed, size = self.fields.row("newsize", "Size", self.new_matrix_size, step=1)
            if changed:
                self.new_matrix_size = [min(max(v, 1), MAX_UI_SIZE) for v in size]
            changed, level = self.fields.number("newresolution", "Resolution", self.new_matrix_resolution, step=1,
                                                shift_step=1, width=RESOLUTION_BOX_WIDTH + 60, buttons=True)
            if changed:
                self.new_matrix_resolution = max(1, min(level, MAX_RESOLUTION))
            imgui.text_disabled(f"{2 ** (self.new_matrix_resolution - 1)} cell"
                                + ("" if self.new_matrix_resolution == 1 else "s") + " per unit")
            if imgui.button("Create"):
                if self.add_matrix(tuple(self.new_matrix_size), self.new_matrix_name, self.new_matrix_resolution):
                    self.new_matrix_name = ""
                    self.show_new_matrix = False
        imgui.end()

    def _draw_model_panel(self) -> None:
        """The contents of the Model panel: the matrix list, scrolling on its own, and the buttons that stay below it."""
        if self.model:
            drops = []
            resolution_changes = []
            row_actions = []
            picked_rows = set(self.selected_indices())
            fh, gap = imgui.get_frame_height(), imgui.get_style().item_spacing.y
            footer = (max(fh, 28.0) if self.picture_icons else fh) + fh + 3 * gap + 2   # the buttons and the resolution row
            imgui.begin_child("##matrixlist", imgui.ImVec2(0, max(imgui.get_content_region_avail().y - footer, 60.0)))
            # The checkboxes and resolution boxes keep their usual spacing, but the name is as tall as the whole row, spacing
            # included, as if it had inner padding: the names touch, and there is no strip between two of them that
            # clicks on nothing. ImGui adds the checkbox's frame padding to the row to line a name up with the checkbox
            # text; a negative spacing of that size takes it out again so each row is exactly one name tall.
            style = imgui.get_style()
            gap_x, gap_y = float(style.item_spacing.x), float(style.item_spacing.y)    # the usual spacing, before changing it
            row_pitch = imgui.get_frame_height() + gap_y
            text_height = imgui.get_text_line_height()
            lift = style.frame_padding.y + gap_y / 2     # the name starts above the checkbox's top, so it is centered on it
            imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(gap_x, -style.frame_padding.y))
            imgui.push_style_var(imgui.StyleVar_.selectable_text_align,          # the name's text centered in its bar
                                 imgui.ImVec2(0.0, (row_pitch - text_height) / 2 / (row_pitch - text_height)))
            for i, item in enumerate(self.items):
                sx, sy, sz = item.matrix.size
                imgui.push_id(i)
                row_top = imgui.get_cursor_pos_y()
                row_start, row_width = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail().x
                row_draw = imgui.get_window_draw_list()
                row_draw.channels_split(2)                 # the row's highlight is painted under its widgets afterwards
                row_draw.channels_set_current(1)
                _, item.visible = imgui.checkbox("##visible", item.visible)
                imgui.same_line()
                highlighted = False
                if self.renaming == i:
                    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(gap_x, gap_y))
                    self._draw_rename_field()                       # a row made of boxes only: its spacing as usual
                    imgui.pop_style_var()
                else:
                    flags = imgui.SelectableFlags_.allow_double_click | imgui.internal.SelectableFlagsPrivate_.no_pad_with_half_spacing
                    width = max(imgui.get_content_region_avail().x - RESOLUTION_BOX_WIDTH - 8, 40)
                    # ImGui sets a name 3 px lower than its row to line it up with the checkbox's text. The cursor goes up by
                    # that and by half the gap, which puts the bar half the gap above the checkbox's top and as far below
                    # its bottom: centered. The row's own height is put back after the resolution box.
                    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - lift)
                    menu_id = f"##rowmenu{i}"                    # the row stays lit while its right-click menu is up
                    for color in (imgui.Col_.header, imgui.Col_.header_hovered, imgui.Col_.header_active):
                        imgui.push_style_color(color, imgui.ImVec4(0, 0, 0, 0))   # the whole row is highlighted below instead
                    name_clicked = imgui.selectable(f"{item.matrix.name}  [{sx}x{sy}x{sz}]", False, flags,
                                                    imgui.ImVec2(width, row_pitch))[0]
                    imgui.pop_style_color(3)
                    bar_top, bar_bottom = imgui.get_item_rect_min().y, imgui.get_item_rect_max().y
                    if name_clicked:
                        if shift_down():
                            self.select_range(i, additive=ctrl_down())      # Ctrl+Shift adds to the selection
                        elif ctrl_down():
                            self.toggle_selection(i)
                        else:
                            self.active = i
                            self.selection = set()
                            if imgui.is_mouse_double_clicked(0):
                                self.start_rename(i)
                    if imgui.is_item_hovered() and imgui.is_mouse_released(imgui.MouseButton_.right):
                        imgui.open_popup(menu_id)
                    # The list's tight row spacing must not leak into the menu: it has the usual look of the others.
                    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(gap_x, gap_y))
                    imgui.push_style_var(imgui.StyleVar_.selectable_text_align, imgui.ImVec2(0.0, 0.0))
                    if imgui.begin_popup(menu_id, imgui.WindowFlags_.no_move):
                        if imgui.menu_item("Clone", "", False)[0]:
                            row_actions.append(("clone", i))
                        if imgui.menu_item("Delete", "Del", False)[0]:
                            row_actions.append(("delete", i))
                        imgui.end_popup()
                    imgui.pop_style_var(2)
                    if imgui.begin_drag_drop_source():
                        imgui.set_drag_drop_payload_py_id(MATRIX_DRAG, i)
                        imgui.text(item.matrix.name)
                        imgui.end_drag_drop_source()
                    if imgui.begin_drag_drop_target():
                        payload = imgui.accept_drag_drop_payload_py_id(MATRIX_DRAG)
                        if payload is not None:
                            # Ctrl held while dropping merges the dragged matrix into this one.
                            drops.append((payload.data_id, i, ctrl_down()))
                        imgui.end_drag_drop_target()
                    imgui.same_line()
                    typed, level = self.fields.number(f"row.{i}", "##resolution", item.matrix.resolution, step=1,
                                                      shift_step=1, flags=imgui.InputTextFlags_.enter_returns_true,
                                                      width=RESOLUTION_BOX_WIDTH)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Resolution (1 to 8): a higher number means finer cells. Press Enter to apply.")
                    if typed:
                        resolution_changes.append((item, level))
                    imgui.set_cursor_pos_y(row_top + row_pitch)               # the next row starts where it always did
                    row_rect = (imgui.ImVec2(row_start.x, bar_top), imgui.ImVec2(row_start.x + row_width, bar_bottom))
                    lit = i in picked_rows or imgui.is_popup_open(menu_id)
                    hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(*row_rect)
                    highlighted = lit or hovered
                    if lit or hovered:
                        row_draw.channels_set_current(0)
                        row_draw.add_rect_filled(*row_rect, imgui.get_color_u32(
                            imgui.Col_.header if lit else imgui.Col_.header_hovered))
                if i % 2 and not highlighted:                  # every other row a shade lighter, so it reads as a list
                    row_draw.channels_set_current(0)
                    stripe_top = row_start.y - lift
                    row_draw.add_rect_filled(imgui.ImVec2(row_start.x, stripe_top),
                                             imgui.ImVec2(row_start.x + row_width, stripe_top + row_pitch),
                                             imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.045)))
                row_draw.channels_merge()
                imgui.pop_id()
            imgui.pop_style_var(2)
            imgui.dummy(imgui.ImVec2(0.0, 0.0))             # ImGui wants an item after the cursor was moved by hand
            imgui.end_child()
            for source, target, merging in drops:   # applied after the loop, which must not see the list change
                self.merge_into(source, target) if merging else self.reorder_matrix(source, target)
            for item, level in resolution_changes:
                self.set_matrix_resolution(item, level)
            for action, index in row_actions:           # also after the loop: they change the list
                self.clone_matrix(index) if action == "clone" else self.delete_matrix(index)
            button_height = max(imgui.get_frame_height(), 28.0) if self.picture_icons else imgui.get_frame_height()
            if imgui.button("New", imgui.ImVec2(0, button_height)):
                self.show_new_matrix = True
            imgui.same_line()
            if icon_button("Delete", "delete", button_height):
                self.delete_matrix()
            imgui.same_line()
            picked = len(self.selected_indices())
            imgui.begin_disabled(picked < 2)
            if icon_button(f"Merge ({picked})" if picked > 1 else "Merge", "merge", button_height):
                self.merge_selected()
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("Ctrl+click matrices to select several, then merge them into the top one.")
            imgui.text("Resolution:")
            imgui.same_line()
            changed, level = self.fields.number("bulk", "##bulkresolution", self.bulk_resolution, step=1, shift_step=1,
                                                width=RESOLUTION_BOX_WIDTH + 30, buttons=True)
            if changed:
                self.bulk_resolution = max(1, min(level, MAX_RESOLUTION))
            imgui.same_line()
            if imgui.button("All"):
                self.set_all_resolutions(self.bulk_resolution)
            imgui.same_line()
            if imgui.button("Selected"):
                self.set_all_resolutions(self.bulk_resolution, selected_only=True)

    def _draw_keybinds_window(self) -> None:
        """Help > Keybinds: every shortcut, in sections of two columns (what to press, and what it does)."""
        if not self.show_keybinds:
            return
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(imgui.ImVec2(display.x / 2, display.y / 2), imgui.Cond_.first_use_ever, imgui.ImVec2(0.5, 0.5))
        imgui.set_next_window_size(imgui.ImVec2(640, min(560, display.y - 120)), imgui.Cond_.first_use_ever)
        opened, self.show_keybinds = imgui.begin("Keybinds", True, PANEL_FLAGS)
        if opened:
            for title, binds in KEYBIND_SECTIONS:
                imgui.separator_text(title)
                if imgui.begin_table(f"##keys_{title}", 2, imgui.TableFlags_.sizing_fixed_fit):
                    imgui.table_setup_column("keys", imgui.TableColumnFlags_.width_fixed, 170.0)
                    imgui.table_setup_column("action", imgui.TableColumnFlags_.width_stretch)
                    for keys, action in binds:
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.text(keys)
                        imgui.table_next_column()
                        imgui.text_disabled(action)
                    imgui.end_table()
        imgui.end()

    def _draw_tools_panel(self) -> None:
        """The contents of the Tools panel."""
        if self.mode is Mode.MATRIX:
            self._draw_matrix_tools()
            return
        if self.voxel_tool is VoxelTool.FILL:
            imgui.text_disabled("Fill")
            for label, replace in (("Append", False), ("Replace", True)):
                if imgui.radio_button(f"{label}##fillmode", self.fill_replace is replace):
                    self.set_fill_replace(replace)
                if not replace:
                    imgui.same_line()
        if self.action is not Tool.SELECT and self.action is not Tool.PICK:
            imgui.text_disabled("Mirror")
            for axis, name in enumerate("XYZ"):
                imgui.same_line()
                _, self.mirror[axis] = imgui.checkbox(f"{name}##mirror", self.mirror[axis])
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Repeat edits across the center of the matrix along {name}")
            _, self.radial_on = imgui.checkbox("Radial##radial", self.radial_on)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Repeat edits turned around the center of the matrix, like spokes of a wheel")
            if self.radial_on:
                for axis, name in enumerate("XYZ"):
                    imgui.same_line()
                    if imgui.radio_button(f"{name}##radial_axis", self.radial_axis == axis):
                        self.radial_axis = axis
                imgui.set_next_item_width(90.0)
                _, count = imgui.input_int("Copies##radial", self.radial_count)
                self.radial_count = max(2, min(count, 32))
        if self.brush is Brush.FILL and self.action in (Tool.PAINT, Tool.SELECT):
            self._draw_match_options()
        self._draw_selection_tools()
        if self.box is not None:
            lo, hi = self.box.corners()
            imgui.text("Box: " + " x ".join(str(h - l + 1) for l, h in zip(lo, hi)))
        imgui.separator()

        changed, rgb = imgui.color_edit3("Color", [c / 255 for c in self.color])
        if changed:
            self.color = tuple(int(round(c * 255)) for c in rgb)
        if self.model is not None:
            index = self.model.find_color(self.brush_rgba())
            usage = self.palette_usage()
            if not index:
                imgui.text_disabled("New color, added when used")
            else:
                note = " (unused)" if index - 1 < len(usage) and not usage[index - 1] else ""
                imgui.text_disabled(f"Palette index {index}{note}")
            imgui.separator()
            self._draw_palette()

    def _draw_mode_switch(self) -> None:
        """Two buttons side by side in the top left corner of the view that show the mode and switch it (Tab does the same)."""
        top = imgui.get_frame_height() + self._tab_bar_height() + 8.0
        imgui.set_next_window_pos(imgui.ImVec2(self._sidebar_width() + 8.0, top))
        flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
                 | imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_nav_inputs
                 | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_focus_on_appearing
                 | imgui.WindowFlags_.no_bring_to_front_on_focus)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(5.0, 5.0))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4.0, 4.0))
        if imgui.begin("##modeswitch", None, flags):
            self._mode_buttons(width=72.0)
        imgui.end()
        imgui.pop_style_var(2)

    def _mode_buttons(self, width: float) -> None:
        for mode in Mode:
            count = push_selected_colors(self.mode is mode)
            if imgui.button(f"{mode.value}##mode", imgui.ImVec2(width, imgui.get_frame_height() * 1.1)):
                self.set_mode(mode)
            imgui.pop_style_color(count)
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"{mode.value} mode (Tab switches)")
            if mode is not list(Mode)[-1]:
                imgui.same_line()

    VOXEL_TOOL_ICONS = {
        VoxelTool.PLACE: "place", VoxelTool.ERASE: "erase", VoxelTool.PAINT: "paint", VoxelTool.PICK: "pick",
        VoxelTool.SELECT_BOX: "select_box", VoxelTool.SELECT_WAND: "wand",
    }
    VOXEL_TOOL_KEYS = {
        VoxelTool.PLACE: "T", VoxelTool.ERASE: "E", VoxelTool.PAINT: "R", VoxelTool.PICK: "K",
        VoxelTool.SELECT_BOX: "S", VoxelTool.SELECT_WAND: "S again", VoxelTool.FILL: "F",
    }

    def _brush_float(self) -> tuple[float, float, float]:
        return tuple(c / 255 for c in self.color)

    def _draw_tool_strip(self) -> None:
        """The toolbox: a small vertical window of icon buttons beside the explorer, for the tools of the current mode."""
        if self.mode is Mode.VOXEL:
            buttons = []
            for number, tool in enumerate(VOXEL_TOOLS, start=1):
                if tool is VoxelTool.FILL:
                    label, icon = ("Fill: replace", "fill_replace") if self.fill_replace else ("Fill: append", "fill_face")
                    buttons.append(ToolButton(tool, label, icon, f"{self.VOXEL_TOOL_KEYS[tool]}, again switches; {number}", ""))
                else:
                    buttons.append(ToolButton(tool, tool.value, self.VOXEL_TOOL_ICONS[tool], f"{self.VOXEL_TOOL_KEYS[tool]}, {number}", ""))
            selected = self.voxel_tool
        else:
            icons = {MatrixTool.MOVE: "move", MatrixTool.RESIZE: "resize", MatrixTool.ROTATE: "rotate",
                     MatrixTool.PIVOT: "pivot"}
            letters = {MatrixTool.MOVE: "W", MatrixTool.RESIZE: "R", MatrixTool.ROTATE: "E", MatrixTool.PIVOT: "G"}
            buttons = [ToolButton(tool, tool.value, icons[tool], f"{letters[tool]}, {number}", "")
                       for number, tool in enumerate(MatrixTool, start=1)]
            selected = self.matrix_tools.tool
        view_top = imgui.get_frame_height() + self._tab_bar_height()
        view_bottom = imgui.get_io().display_size.y - self._bar_height()
        imgui.set_next_window_pos(imgui.ImVec2(self._sidebar_width() + 8.0, (view_top + view_bottom) / 2),
                                  imgui.Cond_.always, imgui.ImVec2(0.0, 0.5))          # vertically centered at the left
        flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
                 | imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_nav_inputs
                 | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_focus_on_appearing
                 | imgui.WindowFlags_.no_bring_to_front_on_focus)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(5.0, 5.0))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4.0, 4.0))
        if imgui.begin("##toolstrip", None, flags):
            picked = tool_strip(buttons, selected, self._brush_float())
            if picked is not None:
                if self.mode is Mode.VOXEL:
                    self.set_voxel_tool(picked)
                elif self.matrix_tools.drag is None:
                    self.matrix_tools.tool = picked
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_match_options(self) -> None:
        """What counts as the same for Replace fill and the wand: connected or anywhere, and how close in color."""
        _, self.fill_contiguous = imgui.checkbox("Contiguous", self.fill_contiguous)
        if imgui.is_item_hovered():
            imgui.set_tooltip("On: only voxels connected to the clicked one.\nOff: every matching voxel in the matrix.")
        imgui.set_next_item_width(imgui.get_content_region_avail().x * 0.6)
        _, self.fill_tolerance = imgui.slider_int("Tolerance", self.fill_tolerance, 0, MAX_TOLERANCE, "%d%%")
        if imgui.is_item_hovered():
            imgui.set_tooltip("How different a color may be from the clicked one and still match.\n0 means the same color only.")

    def _draw_selection_tools(self) -> None:
        """What can be done with the selected voxels, shown while the Select action is chosen or something is selected."""
        if not self.selection_visible():
            return
        count, box = self.selection_info()
        imgui.separator()
        if count:
            (x0, y0, z0), (x1, y1, z1) = box
            imgui.text(f"{count:,} voxels selected ({x1 - x0 + 1} x {y1 - y0 + 1} x {z1 - z0 + 1})")
        else:
            imgui.text_disabled("Nothing selected")
        if imgui.button("Select all"):
            self.select_all()
        imgui.same_line()
        imgui.begin_disabled(not count)
        if imgui.button("Deselect"):
            self.deselect()
        imgui.same_line()
        if imgui.button("Delete##voxels"):
            self.selection_delete()
        for axis in range(3):
            if axis:
                imgui.same_line()
            if imgui.button(f"Flip {'XYZ'[axis]}"):
                self.selection_flip(axis)
        for axis in range(3):
            if axis:
                imgui.same_line()
            if imgui.button(f"Turn {'XYZ'[axis]}"):
                self.selection_rotate(axis, -1 if shift_down() else 1)
        if imgui.button("Copy##voxels"):
            self.copy_selection()
        imgui.same_line()
        if imgui.button("Cut##voxels"):
            self.cut_selection()
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(self.voxel_clip is None)
        if imgui.button("Paste##voxels"):
            self.paste_voxels()
        imgui.end_disabled()

    def _draw_matrix_tools(self) -> None:
        tools = self.matrix_tools
        imgui.push_text_wrap_pos(0.0)
        imgui.text_disabled("Click a matrix in the view to select it. Esc cancels a drag.")
        imgui.pop_text_wrap_pos()
        if tools.drag is not None:
            imgui.text(tools.drag_summary())
        imgui.separator()

        item = self.active_item()
        if item is None:
            imgui.text_disabled("No matrix selected.")
            return
        m = item.matrix
        imgui.text(m.name)
        enter = imgui.InputTextFlags_.enter_returns_true
        numeric = enter | imgui.InputTextFlags_.parse_empty_ref_val   # an emptied number field means 0
        step = float(POSITION_SCALE)

        changed, v = self.fields.row("position", "Position", [p / step for p in m.position], step=1.0,
                                     shift_step=1 / POSITION_SCALE, is_float=True, flags=numeric)
        if changed:
            self.set_matrix_transform(item, position=tuple(int(round(c * step)) for c in v))

        changed, v = self.fields.row("size", "Size", list(m.size), step=1, flags=enter)
        if changed:
            target = [max(int(n), 1) for n in v]
            self.resize_matrix_by(item, (0, 0, 0), [t - c for t, c in zip(target, m.size)])

        # Shown from the middle of the matrix, so 0, 0, 0 is the center; it is stored from the origin corner.
        middle = [n * step / 2 for n in m.size]
        changed, v = self.fields.row("pivot", "Pivot", [(p - c) / step for p, c in zip(m.pivot, middle)], step=1.0,
                                     shift_step=1 / POSITION_SCALE, is_float=True, flags=numeric)
        if changed:
            self.set_matrix_pivot(item, tuple(int(round(c * step + mid)) for c, mid in zip(v, middle)))

        euler = xf.to_euler(m.rotation)
        changed, v = self.fields.row("rotation", "Rotation", [float(e) + 0.0 for e in euler], step=1.0,
                                     shift_step=15.0, is_float=True, fmt="%.2f", flags=numeric)
        if changed:
            self.set_matrix_transform(item, rotation=xf.from_euler(*v))

        changed, level = self.fields.number("resolution", "Resolution", m.resolution, step=1, shift_step=1,
                                            flags=enter, buttons=True)
        if changed:
            self.set_matrix_resolution(item, level)
        imgui.text_disabled(f"{cells_per_unit(m)} per unit")

    def _mark_unused_swatch(self) -> None:
        """Strike through the swatch just drawn: no voxel uses this color, so it is left out of saved files."""
        top_left, bottom_right = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw = imgui.get_window_draw_list()
        corner_a, corner_b = imgui.ImVec2(top_left.x + 2, bottom_right.y - 2), imgui.ImVec2(bottom_right.x - 2, top_left.y + 2)
        draw.add_line(corner_a, corner_b, imgui.get_color_u32(imgui.ImVec4(0, 0, 0, 0.9)), 4.0)      # dark edge,
        draw.add_line(corner_a, corner_b, imgui.get_color_u32(imgui.ImVec4(1, 1, 1, 0.95)), 2.0)     # white core: shows on any color
        if imgui.is_item_hovered():
            imgui.set_tooltip("Unused: no voxel has this color, so it is not saved")

    def _draw_palette(self) -> None:
        palette = self.model.palette
        usage = self.palette_usage()
        unused = int((~usage).sum())
        imgui.text(f"Palette ({len(palette)})" + (f", {unused} unused" if unused else ""))
        if len(self.tabs) > 1:
            imgui.same_line()
            if imgui.small_button("Import colors..."):
                imgui.open_popup("import_colors")
        if imgui.begin_popup("import_colors"):
            imgui.text_disabled("Add the colors this palette lacks from:")
            for i, tab in enumerate(self.tabs):
                if i != self.current and imgui.selectable(f"{self.tab_name(i)}##import{tab.uid}", False)[0]:
                    self.import_palette_from_tab(i)
            imgui.end_popup()
        if not len(palette):
            return
        current = self.model.find_color(self.brush_rgba())
        rows = (len(palette) + SWATCH_COLUMNS - 1) // SWATCH_COLUMNS
        imgui.begin_child("palette", imgui.ImVec2(0, 0), 0, PANEL_FLAGS)
        clipper = imgui.ListClipper()
        clipper.begin(rows, 22.0)
        while clipper.step():
            for row in range(clipper.display_start, clipper.display_end):
                for col in range(SWATCH_COLUMNS):
                    i = row * SWATCH_COLUMNS + col
                    if i >= len(palette):
                        break
                    r, g, b = (int(c) for c in palette[i][:3])
                    if col:
                        imgui.same_line()
                    imgui.push_id(i)
                    flags = imgui.ColorEditFlags_.no_alpha | imgui.ColorEditFlags_.no_tooltip
                    if imgui.color_button("##swatch", imgui.ImVec4(r / 255, g / 255, b / 255, 1), flags,
                                          imgui.ImVec2(18, 18)):
                        self.select_palette_color(i + 1)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Click to use, right click to recolor")
                        if imgui.is_mouse_released(imgui.MouseButton_.right):
                            self._recolor_rgb = [r / 255, g / 255, b / 255]
                            imgui.open_popup("##recolor")
                    if imgui.begin_popup("##recolor"):
                        imgui.text(f"Recolor palette color {i + 1}")
                        _, self._recolor_rgb = imgui.color_picker3("##recolor_picker", self._recolor_rgb)
                        if imgui.button("Recolor", imgui.ImVec2(90, 0)):
                            self.recolor_palette_color(i + 1, tuple(int(round(c * 255)) for c in self._recolor_rgb))
                            imgui.close_current_popup()
                        imgui.same_line()
                        if imgui.button("Cancel", imgui.ImVec2(90, 0)):
                            imgui.close_current_popup()
                        imgui.end_popup()
                    if i < len(usage) and not usage[i]:
                        self._mark_unused_swatch()
                    if i + 1 == current:
                        top_left, bottom_right = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                        imgui.get_window_draw_list().add_rect(
                            top_left, bottom_right, imgui.get_color_u32(imgui.ImVec4(1, 1, 1, 1)),
                            0.0, 2.0)
                    imgui.pop_id()
        imgui.end_child()

    def _draw_axis_gizmo(self) -> None:
        """Axis indicator in the bottom-right corner: +X/+Y/+Z solid, -X/-Y/-Z hollow."""
        width, height = imgui.get_io().display_size.x, imgui.get_io().display_size.y
        radius = 46.0
        center = imgui.ImVec2(width - self._right_width() - radius - 36, height - self._bar_height() - radius - 36)
        draw = imgui.get_background_draw_list()
        colors = {"X": (0.93, 0.33, 0.33), "Y": (0.45, 0.85, 0.40), "Z": (0.38, 0.58, 1.0)}
        rgba = lambda c, a=1.0: imgui.get_color_u32(imgui.ImVec4(c[0], c[1], c[2], a))

        draw.add_circle_filled(center, radius + 18, imgui.get_color_u32(imgui.ImVec4(0, 0, 0, 0.28)), 48)
        for name, sign, x, y, depth in self.camera.gizmo_axes():  # far to near
            tip = imgui.ImVec2(center.x + x * radius, center.y - y * radius)
            fade = 1.0 if depth <= 0.2 else 0.65  # axes pointing away from the viewer are drawn fainter
            color = colors[name]
            draw.add_line(center, tip, rgba(color, 0.9 * fade if sign > 0 else 0.45 * fade), 2.0 if sign > 0 else 1.0)
            if sign > 0:
                draw.add_circle_filled(tip, 10.0, rgba(color, fade), 20)
                size = imgui.calc_text_size(name)
                draw.add_text(imgui.ImVec2(tip.x - size.x / 2, tip.y - size.y / 2),
                              imgui.get_color_u32(imgui.ImVec4(0.05, 0.05, 0.07, 1)), name)
            else:
                draw.add_circle_filled(tip, 11.0, rgba(color, 0.18 * fade), 20)
                draw.add_circle(tip, 11.0, rgba(color, 0.8 * fade), 20, 1.5)
                label = "-" + name
                size = imgui.calc_text_size(label)
                draw.add_text(imgui.ImVec2(tip.x - size.x / 2, tip.y - size.y / 2),
                              rgba(color, fade), label)

    def _bar_height(self) -> float:
        return imgui.get_frame_height() + 6.0

    def _tab_bar_height(self) -> float:
        return imgui.get_frame_height() + 4.0

    def _draw_tab_bar(self) -> None:
        """One tab per open model, just under the menu. Drop a matrix from the Model list onto a tab to copy it there."""
        display = imgui.get_io().display_size
        left = self._sidebar_width()                      # the Explorer sidebar runs the full height beside the tabs
        imgui.set_next_window_pos(imgui.ImVec2(left, imgui.get_frame_height()))
        imgui.set_next_window_size(imgui.ImVec2(max(display.x - left - self._right_width(), 1.0), self._tab_bar_height()))
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
                 | imgui.WindowFlags_.no_nav_inputs | imgui.WindowFlags_.no_focus_on_appearing
                 | imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_scrollbar)
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(4.0, 2.0))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, imgui.ImVec2(1.0, 1.0))   # else a window is 32 px tall at least
        imgui.begin("##tabbar", None, flags)
        clicked: int | None = None
        closing: int | None = None
        new_tab = False
        drops: list[tuple[int, int]] = []
        shown_at: list[tuple[float, int]] = []         # where each tab is drawn, to follow drags that reorder them
        unchanged = self._shown_uids == [tab.uid for tab in self.tabs]   # no tab came or went since the last frame
        wanted = self.tabs[self.current].uid
        forced = self._bar_selected != wanted          # the current tab changed by itself: make the bar follow
        bar_flags = (imgui.TabBarFlags_.fitting_policy_scroll | imgui.TabBarFlags_.no_close_with_middle_mouse_button
                     | imgui.TabBarFlags_.reorderable)
        # Only the current tab is blue; the others are a plain dark grey, so it stands out.
        grey, lit, blue = imgui.ImVec4(0.17, 0.17, 0.18, 1.0), imgui.ImVec4(0.27, 0.28, 0.31, 1.0), imgui.ImVec4(0.18, 0.43, 0.78, 1.0)
        tab_colors = ((imgui.Col_.tab, grey), (imgui.Col_.tab_hovered, lit), (imgui.Col_.tab_selected, blue),
                      (imgui.Col_.tab_dimmed, grey), (imgui.Col_.tab_dimmed_selected, blue),
                      (imgui.Col_.tab_selected_overline, imgui.ImVec4(0.45, 0.72, 1.0, 1.0)))
        for color, value in tab_colors:
            imgui.push_style_color(color, value)
        if imgui.begin_tab_bar("##tabs", bar_flags):
            for i, tab in enumerate(self.tabs):
                label = f"{self.tab_name(i)}###tab{tab.uid}"
                item_flags = imgui.TabItemFlags_.set_selected if forced and tab.uid == wanted else 0
                selected, keep = imgui.begin_tab_item(label, True, item_flags)
                shown_at.append((imgui.get_item_rect_min().x, i))
                if imgui.is_item_hovered():
                    path = self.path if i == self.current else tab.path
                    has_model = self.model is not None if i == self.current else tab.model is not None
                    imgui.set_tooltip(str(path) if path else ("Not saved yet" if has_model else "Home"))
                    if imgui.is_mouse_clicked(imgui.MouseButton_.middle):
                        closing = i
                if imgui.begin_drag_drop_target():
                    payload = imgui.accept_drag_drop_payload_py_id(MATRIX_DRAG)
                    if payload is not None:
                        drops.append((payload.data_id, i))
                    imgui.end_drag_drop_target()
                if selected:
                    clicked = i
                    imgui.end_tab_item()
                if not keep:
                    closing = i
            new_tab = imgui.tab_item_button("+", imgui.TabItemFlags_.trailing | imgui.TabItemFlags_.no_tooltip)
            imgui.end_tab_bar()
        imgui.pop_style_color(len(tab_colors))
        imgui.end()
        imgui.pop_style_var(3)
        if new_tab:
            self._open_tab()
        elif closing is not None:
            # Clicking an X also selects that tab in the bar; it must not become the current one, and if the
            # close is cancelled the bar has to go back to the tab that was current.
            if not self.close_tab(closing):
                self._bar_selected = None
        elif forced:
            self._bar_selected = wanted
        elif clicked is not None and clicked != self.current:
            self.switch_tab(clicked)
            self._bar_selected = self.tabs[self.current].uid
        for source, target in drops:
            self.copy_matrix_to_tab(source, target)
        if unchanged and not new_tab and closing is None and len(shown_at) == len(self.tabs):
            # A tab that was just added has no place in the bar yet, so only a bar that was already laid out can
            # have been dragged: then the tab list takes the order the tabs are drawn in.
            order = [i for _, i in sorted(shown_at)]
            if order != list(range(len(self.tabs))):
                self.tabs = [self.tabs[i] for i in order]
                self.current = next(i for i, tab in enumerate(self.tabs) if tab.uid == wanted)
        self._shown_uids = [tab.uid for tab in self.tabs]

    def _draw_info_bar(self) -> None:
        """A strip along the bottom: the active matrix, the mode and cell under the cursor, the last message."""
        display = imgui.get_io().display_size
        height = self._bar_height()
        imgui.set_next_window_pos(imgui.ImVec2(0, display.y - height))
        imgui.set_next_window_size(imgui.ImVec2(display.x, height))
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
                 | imgui.WindowFlags_.no_nav_inputs | imgui.WindowFlags_.no_focus_on_appearing
                 | imgui.WindowFlags_.no_bring_to_front_on_focus)
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(10.0, 3.0))
        imgui.push_style_var(imgui.StyleVar_.window_min_size, imgui.ImVec2(1.0, 1.0))
        imgui.begin("##infobar", None, flags)
        measure = lambda text: imgui.calc_text_size(text).x
        gap = "   |   "

        right = f"{self._fps:.0f} fps{gap}Chronovox Reforged {__version__}"
        right_x = display.x - measure(right) - 10.0
        imgui.set_cursor_pos_x(right_x)
        imgui.text_disabled(right)
        imgui.same_line(0.0, 0.0)

        item = self.active_item()
        if self.model is None:
            parts = [self.status] if self.status else []
        else:
            parts = [f"{self.mode.value} mode" + (f": {self.tool.value}" if self.mode is Mode.VOXEL else "")]
            if self.mode is Mode.VOXEL and self.hover is not None and self._hover_item() is not None:
                x, y, z = self.hover
                parts.append(f"cell {x}, {y}, {z}")
            if self.selection_visible() and self.selection_count():
                parts.append(f"{self.selection_count():,} selected")
            if self.status:                # before the matrix, so a long line cuts the matrix and not the message
                parts.append(self.status)
            parts.append(matrix_summary(item.matrix) if item else "no matrix")
        left = fit_text(gap.join(parts), right_x - 30.0, measure)
        imgui.set_cursor_pos_x(10.0)
        imgui.text(left)
        imgui.end()
        imgui.pop_style_var(3)

    def _draw_home(self) -> None:
        """The page a tab shows before it has a model: start a new one, open one, or pick a recent one."""
        display = imgui.get_io().display_size
        width = min(640.0, display.x - 40.0)
        left = self._sidebar_width()
        width = min(width, display.x - left - 40.0)
        imgui.set_next_window_pos(imgui.ImVec2((display.x + left) / 2, (display.y + imgui.get_frame_height()) / 2),
                                  imgui.Cond_.always, imgui.ImVec2(0.5, 0.5))
        imgui.set_next_window_size(imgui.ImVec2(width, min(520.0, display.y - 140.0)))
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
                 | imgui.WindowFlags_.no_nav_inputs)
        imgui.begin("##home", None, flags)
        imgui.text("Chronovox Reforged")
        imgui.text_disabled(f"Voxel model editor, version {__version__}")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 6))
        half = (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) / 2
        if imgui.button("New model   Ctrl+N", imgui.ImVec2(half, 40)):
            self.new_model()
        imgui.same_line()
        if imgui.button("Open model...   Ctrl+O", imgui.ImVec2(half, 40)):
            self._choose_file()
        imgui.dummy(imgui.ImVec2(0, 10))
        imgui.text("Recent models")
        if self.recent_files:
            imgui.same_line()
            for view in ("list", "grid"):
                if imgui.radio_button(f"{view.capitalize()}##recentview", self.recent_view == view):
                    self.recent_view = view
                if view == "list":
                    imgui.same_line()
        imgui.separator()
        if not self.recent_files:
            imgui.text_disabled("Models you open or save are listed here.")
            imgui.end()
            return
        now = time.perf_counter()
        if now - self._recent_checked > 2.0:           # a quick look at the disk, not on every frame
            self._recent_checked = now
            self._recent_exists = {p: Path(p).exists() for p in self.recent_files}
        imgui.begin_child("##recent", imgui.ImVec2(0, 0), 0, PANEL_FLAGS)
        chosen = self._draw_recent_grid() if self.recent_view == "grid" else self._draw_recent_list(width)
        imgui.end_child()
        imgui.end()
        # what was asked for is done after drawing, so the list is not changed while it is being drawn
        if self._recent_to_remove is not None:
            self.recent_files = remove_recent(self.recent_files, self._recent_to_remove)
            self._recent_to_remove = None
            self.status = "Removed from the recent list. The file itself was not touched."
        elif self._recent_drop_missing:
            gone = [p for p in self.recent_files if not Path(p).exists()]
            for path in gone:
                self.recent_files = remove_recent(self.recent_files, path)
            self._recent_drop_missing = False
            self.status = f"Removed {len(gone)} missing model{'s' if len(gone) != 1 else ''} from the recent list."
        elif chosen is not None or self._recent_to_open is not None:
            opening, self._recent_to_open = chosen or self._recent_to_open, None
            self.open_file(opening)

    def _recent_context_menu(self, path: str, item_id: str) -> None:
        """Right-click on a recent model: open it, find it on disk, copy its path, or take it off the list.

        Call right after drawing the entry's item. The menu cannot be dragged away from where it opened.
        """
        if imgui.is_item_hovered() and imgui.is_mouse_released(imgui.MouseButton_.right):
            imgui.open_popup(item_id)
        if not imgui.begin_popup(item_id, imgui.WindowFlags_.no_move):
            return
        exists = Path(path).exists()
        if imgui.menu_item("Open", "", False, exists)[0]:
            self._recent_to_open = path
        if imgui.menu_item(filemanager.label(), "", False, exists or Path(path).parent.is_dir())[0]:
            problem = filemanager.reveal(path)
            if problem:
                self.status = problem
        if imgui.menu_item("Copy path", "", False)[0]:
            imgui.set_clipboard_text(str(Path(path)))
            self.status = "Copied the path."
        imgui.separator()
        if imgui.menu_item("Remove from this list", "", False)[0]:
            self._recent_to_remove = path
        if any(not Path(p).exists() for p in self.recent_files):
            if imgui.menu_item("Remove all missing files from the list", "", False)[0]:
                self._recent_drop_missing = True
        imgui.end_popup()

    # -- the Explorer --
    EXPLORER_REFRESH_SECONDS = 5.0

    def add_explorer_root(self) -> None:
        from tkinter import filedialog
        root = self._tk_root()
        try:
            chosen = filedialog.askdirectory(title="Add a folder to the Explorer", initialdir=self.last_dir or None)
        finally:
            root.destroy()
        if not chosen:
            return
        chosen = str(Path(chosen))
        if not any(explorer.norm(chosen) == explorer.norm(r) for r in self.explorer_roots):
            self.explorer_roots.append(chosen)
            self.explorer_expanded.add(chosen)
        self.explorer_visible = True
        self._explorer_dirty = True

    def _open_model_paths(self) -> dict[str, bool]:
        """The files open in tabs (normalized path -> unsaved changes), for marking them in the Explorer."""
        found = {}
        for i, tab in enumerate(self.tabs):
            path, dirty = (self.path, self.dirty) if i == self.current else (tab.path, tab.dirty)
            if path is not None:
                found[explorer.norm(str(path))] = bool(dirty)
        return found

    def _refresh_explorer(self) -> None:
        now = time.monotonic()
        if not (self._explorer_dirty or now - self._explorer_scanned > self.EXPLORER_REFRESH_SECONDS):
            return
        if not self._explorer_dirty and not glfw.get_window_attrib(self.window, glfw.FOCUSED):
            return
        exts = explorer.MODEL_EXTENSIONS + (explorer.LEGACY_EXTENSIONS if self.explorer_csm else ())
        self._explorer_trees = {root: explorer.scan(root, extensions=exts) for root in self.explorer_roots}
        self._explorer_scanned, self._explorer_dirty = now, False

    EXPLORER_STRIP = 30.0             # the width of the sidebar when it is folded away
    EXPLORER_MIN_WIDTH = 170.0

    def _sidebar_width(self) -> float:
        """How much of the left edge the Explorer sidebar takes: the other windows stay to the right of it."""
        if self.preview_mode:
            return 0.0
        return self.explorer_width if self.explorer_visible else self.EXPLORER_STRIP

    def show_sidebar_tab(self, tab: str) -> None:
        """Open the sidebar (if it is folded) on one of its tabs."""
        self.explorer_visible = True
        self._sidebar_tab_request = tab

    def _draw_explorer_panel(self) -> None:
        """The Explorer sidebar: docked to the left edge under the tab bar, folded to a thin strip or opened to a tree."""
        display = imgui.get_io().display_size
        top = imgui.get_frame_height()
        height = display.y - self._bar_height() - top
        self.explorer_width = max(self.EXPLORER_MIN_WIDTH, min(self.explorer_width, display.x * 0.6))
        width = self._sidebar_width()
        if self.explorer_visible:
            self._refresh_explorer()
        imgui.set_next_window_pos(imgui.ImVec2(0, top))
        imgui.set_next_window_size(imgui.ImVec2(width, height))
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings
                 | imgui.WindowFlags_.no_nav_inputs | imgui.WindowFlags_.no_bring_to_front_on_focus)
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_min_size, imgui.ImVec2(1.0, 1.0))
        imgui.begin("##files", None, flags)
        if not self.explorer_visible:
            if imgui.button(">", imgui.ImVec2(-1, 0)):
                self.explorer_visible = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Show the Explorer sidebar (Ctrl+B)")
            imgui.end()
            imgui.pop_style_var(2)
            return
        if imgui.button("<"):
            self.explorer_visible = False
        if imgui.is_item_hovered():
            imgui.set_tooltip("Fold the sidebar away (Ctrl+B)")
        imgui.same_line()
        self.workspace.open = False
        if imgui.begin_tab_bar("##sidebar_tabs", imgui.TabBarFlags_.no_close_with_middle_mouse_button):
            for label, key in (("Explorer", "explorer"), ("Workspace", "workspace")):
                wanted = imgui.TabItemFlags_.set_selected if self._sidebar_tab_request == key else 0
                if imgui.begin_tab_item(label, None, wanted)[0]:
                    self.sidebar_tab = key
                    if key == "explorer":
                        self._draw_explorer_tab(width, top, height)
                    else:
                        self.workspace.open = True
                        self.workspace.draw()
                    imgui.end_tab_item()
            imgui.end_tab_bar()
        self._sidebar_tab_request = None
        self._draw_delete_confirm()
        self._draw_clone_dialog()
        # a thin handle on the right edge: drag it to make the sidebar wider or narrower
        imgui.set_cursor_screen_pos(imgui.ImVec2(width - 5.0, top))
        imgui.invisible_button("##explorer_resize", imgui.ImVec2(5.0, height))
        grabbed = imgui.is_item_active()
        if grabbed or imgui.is_item_hovered():
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
            edge = imgui.get_color_u32(imgui.Col_.separator_active if grabbed else imgui.Col_.separator_hovered)
            imgui.get_window_draw_list().add_rect_filled(imgui.ImVec2(width - 3.0, top), imgui.ImVec2(width, top + height), edge)
        if grabbed:
            self.explorer_width = max(self.EXPLORER_MIN_WIDTH, min(imgui.get_io().mouse_pos.x, display.x * 0.6))
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_delete_confirm(self) -> None:
        """Ask before a model goes to the Recycle Bin. Nothing is deleted without pressing Delete here."""
        title = "Delete model?"
        if self._delete_candidate is not None and not imgui.is_popup_open(title):
            imgui.open_popup(title)
        path = self._delete_candidate
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(imgui.ImVec2(display.x / 2, display.y / 3), imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.0))
        if not imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
            return
        if path is None:
            imgui.close_current_popup()
            imgui.end_popup()
            return
        imgui.text(Path(path).name)
        imgui.text_disabled(str(Path(path).parent))
        imgui.push_text_wrap_pos(imgui.get_font_size() * 24)
        imgui.text("It is moved to the Recycle Bin, so you can still get it back there.")
        if explorer.norm(path) in self._open_model_paths():
            imgui.text_colored(imgui.ImVec4(1.0, 0.75, 0.3, 1.0),
                               "It is open in a tab. The tab stays open, and saving it would create the file again.")
        imgui.pop_text_wrap_pos()
        if imgui.button("Delete", imgui.ImVec2(90, 0)):
            self._delete_model_file(path)
            self._delete_candidate = None
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(90, 0)) or imgui.is_key_pressed(imgui.Key.escape):
            self._delete_candidate = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _draw_clone_dialog(self) -> None:
        """Ask for the name of a copy of a model, which is made in the same folder."""
        title = "Clone model"
        if self._clone_source is not None and not imgui.is_popup_open(title):
            imgui.open_popup(title)
        source = self._clone_source
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(imgui.ImVec2(display.x / 2, display.y / 3), imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.0))
        if not imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
            return
        if source is None:
            imgui.close_current_popup()
            imgui.end_popup()
            return
        extension = Path(source).suffix
        imgui.text(Path(source).name)
        imgui.text_disabled(str(Path(source).parent))
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here()
        imgui.set_next_item_width(imgui.get_font_size() * 18)
        entered, self._clone_name = imgui.input_text(f"{extension}##clone_name", self._clone_name,
                                                      imgui.InputTextFlags_.enter_returns_true)
        path, problem = explorer.clone_target(source, self._clone_name)
        if problem and self._clone_name.strip():
            imgui.text_colored(imgui.ImVec4(1.0, 0.75, 0.3, 1.0), problem)
        if explorer.norm(source) in self._open_model_paths():
            imgui.text_disabled("The copy is made from the file on disk, without unsaved changes in its tab.")
        imgui.begin_disabled(path is None)
        make = imgui.button("Clone", imgui.ImVec2(90, 0)) or (entered and path is not None)
        imgui.end_disabled()
        if make:
            self._clone_model_file(source, path)
            self._clone_source = None
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(90, 0)) or imgui.is_key_pressed(imgui.Key.escape):
            self._clone_source = None
            imgui.close_current_popup()
        imgui.end_popup()

    def _clone_model_file(self, source: str, path: str) -> None:
        try:
            shutil.copy2(source, path)
        except OSError as exc:
            self.status = f"Not cloned: {exc.strerror or exc}"
            return
        self._explorer_dirty = True
        self._explorer_reveal = explorer.norm(path)
        self.status = f"Cloned {Path(source).name} to {Path(path).name}"

    def _delete_model_file(self, path: str) -> None:
        problem = trash.move_to_trash(path)
        if problem:
            self.status = f"Not deleted: {problem}"
            return
        self.recent_files = remove_recent(self.recent_files, path)
        self._explorer_dirty = True
        self.status = f"Moved {Path(path).name} to the Recycle Bin."

    def _draw_explorer_tab(self, width: float, top: float, height: float) -> None:
        """The Explorer tab: add folders, filter, and the tree of models."""
        if imgui.button("Add folder..."):
            self.add_explorer_root()
        imgui.same_line()
        if imgui.button("Refresh"):
            self._explorer_dirty = True
        self._draw_explorer_menu_button()
        imgui.set_next_item_width(-1)
        _, self.explorer_filter = imgui.input_text_with_hint("##explorer_filter", "Filter by name", self.explorer_filter)
        imgui.separator()
        if not self.explorer_roots:
            imgui.push_text_wrap_pos(0.0)
            imgui.text_disabled("Add the folder your models are in to browse everything under it, and open a "
                                "model with one click.")
            imgui.pop_text_wrap_pos()
        opened_files = self._open_model_paths()
        current = explorer.norm(str(self.path)) if self.path is not None else None
        if current != self._explorer_seen:            # another tab or file: show it in the tree
            self._explorer_seen = current
            self._explorer_reveal, self._explorer_reveal_tries = current, 0
        if self._explorer_reveal is not None:
            route = next((r for r in (explorer.route_to(tree, self._explorer_reveal) for tree in self._explorer_trees.values())
                          if r is not None), None)
            if route is not None:
                self.explorer_expanded.update(node.path for node in route[:-1])
                self._explorer_scroll_to = self._explorer_reveal
                self._explorer_reveal = None
            else:
                self._explorer_reveal_tries += 1      # not there (yet): the tree may be about to be read again
                if self._explorer_reveal_tries > 3:
                    self._explorer_reveal = None
        to_open = None
        gone = None
        if imgui.begin_child("##explorer_tree"):
            imgui.push_style_var(imgui.StyleVar_.indent_spacing, imgui.calc_text_size("  ").x)   # two spaces per level
            searching = bool(self.explorer_filter.strip())
            for root in self.explorer_roots:
                tree = self._explorer_trees.get(root)
                shown = explorer.filtered(tree, self.explorer_filter) if tree is not None else None
                if shown is None:
                    continue
                result = self._draw_explorer_node(shown, True, opened_files, current, searching, root)
                to_open = to_open or result[0]
                gone = gone or result[1]
            imgui.pop_style_var()
            blank = imgui.PopupFlags_.mouse_button_right | imgui.PopupFlags_.no_open_over_items
            if imgui.begin_popup_context_window("##explorer_blank_menu", blank):      # right click on empty space
                if imgui.menu_item("Add folder...", "", False)[0]:
                    self.add_explorer_root()
                if imgui.menu_item("Refresh", "", False)[0]:
                    self._explorer_dirty = True
                imgui.end_popup()
        imgui.end_child()
        if to_open:
            self._explorer_seen = explorer.norm(to_open)     # opened from here: it is in view already, so no scrolling
            self._explorer_reveal = None
            self.open_file(to_open)
        if gone:
            self.explorer_roots = [r for r in self.explorer_roots if explorer.norm(r) != explorer.norm(gone)]
            self._explorer_trees.pop(gone, None)

    def _draw_explorer_menu_button(self) -> None:
        """A three-dot button in the top right corner of the Explorer, opening its options."""
        side = imgui.get_frame_height()
        imgui.same_line(imgui.get_window_width() - imgui.get_style().window_padding.x - side)
        top_left = imgui.get_cursor_screen_pos()
        imgui.invisible_button("##explorer_options", imgui.ImVec2(side, side))
        draw = imgui.get_window_draw_list()
        hovered = imgui.is_item_hovered()
        if hovered or imgui.is_popup_open("##explorer_options_menu"):
            draw.add_rect_filled(top_left, imgui.ImVec2(top_left.x + side, top_left.y + side),
                                 imgui.get_color_u32(imgui.Col_.button_hovered), imgui.get_style().frame_rounding)
        dot = imgui.get_color_u32(imgui.Col_.text)
        for k in (-1, 0, 1):
            draw.add_circle_filled(imgui.ImVec2(top_left.x + side / 2 + k * side * 0.24, top_left.y + side / 2),
                                   max(side * 0.075, 1.5), dot)
        if hovered:
            imgui.set_tooltip("Options")
        if imgui.is_item_clicked(imgui.MouseButton_.left):
            imgui.open_popup("##explorer_options_menu")
        if imgui.begin_popup("##explorer_options_menu", imgui.WindowFlags_.no_move):
            if imgui.menu_item("Expand all", "", False)[0]:
                self.explorer_expanded = {path for tree in self._explorer_trees.values() for path in explorer.folders(tree)}
            if imgui.menu_item("Collapse all", "", False)[0]:
                self.explorer_expanded = set()
            imgui.separator()
            changed, self.explorer_csm = imgui.menu_item("Show legacy .csm files", "", self.explorer_csm)
            if changed:
                self._explorer_dirty = True
            imgui.end_popup()

    @staticmethod
    def _draw_model_icon(legacy: bool) -> None:
        """A small cube in the blank slot before a model's name: teal for .cvox, muted amber for legacy .csm."""
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        size = imgui.get_font_size()
        centre = imgui.ImVec2(low.x + imgui.get_style().frame_padding.x + size * 0.5, (low.y + high.y) / 2)
        r = size * 0.46
        shades = ((0.80, 0.62, 0.30), (0.62, 0.47, 0.22), (0.45, 0.34, 0.16)) if legacy else (
            (0.35, 0.85, 0.82), (0.20, 0.62, 0.62), (0.12, 0.42, 0.45))
        def pt(x, y):
            return imgui.ImVec2(centre.x + x * r, centre.y + y * r)
        top, upper_right, lower_right = pt(0, -1), pt(0.87, -0.5), pt(0.87, 0.5)
        bottom, lower_left, upper_left, middle = pt(0, 1), pt(-0.87, 0.5), pt(-0.87, -0.5), pt(0, 0)
        draw = imgui.get_window_draw_list()
        for quad, shade in (((top, upper_right, middle, upper_left), shades[0]),
                            ((upper_left, middle, bottom, lower_left), shades[1]),
                            ((middle, upper_right, lower_right, bottom), shades[2])):
            draw.add_quad_filled(*quad, imgui.get_color_u32(imgui.ImVec4(*shade, 1.0)))

    def _explorer_menu(self, node, is_root: bool, root: str):
        """The right-click menu of a row. Returns (path to open, root to remove)."""
        to_open = gone = None
        menu_id = f"##explorermenu{node.path}"
        if imgui.is_item_hovered() and imgui.is_mouse_released(imgui.MouseButton_.right):
            imgui.open_popup(menu_id)
        if imgui.begin_popup(menu_id, imgui.WindowFlags_.no_move):
            if not node.is_dir and imgui.menu_item("Open", "", False)[0]:
                to_open = node.path
            if imgui.menu_item(filemanager.label(), "", False)[0]:
                problem = filemanager.reveal(node.path)
                if problem:
                    self.status = problem
            if imgui.menu_item("Copy path", "", False)[0]:
                imgui.set_clipboard_text(node.path)
            if not node.is_dir:
                imgui.separator()
                if imgui.menu_item("Clone...", "", False)[0]:
                    self._clone_source = node.path
                    self._clone_name = explorer.default_clone_name(node.path)
                if imgui.menu_item("Delete...", "", False)[0]:
                    self._delete_candidate = node.path
            if is_root:
                imgui.separator()
                if imgui.menu_item("Remove from this panel", "", False)[0]:
                    gone = root
            imgui.end_popup()
        return to_open, gone

    def _draw_explorer_node(self, node, is_root: bool, opened_files: dict, current, searching: bool, root: str):
        """Draw one folder or model and what is under it. Returns (path to open, root to remove)."""
        to_open = gone = None
        key = explorer.norm(node.path)
        if node.is_dir:
            imgui.set_next_item_open(True if searching else node.path in self.explorer_expanded, imgui.Cond_.always)
            flags = imgui.TreeNodeFlags_.span_avail_width                  # a single click anywhere on the row toggles it
            is_open = imgui.tree_node_ex(f"{node.name}  ({node.files})##{node.path}", flags)
            if imgui.is_item_hovered():
                imgui.set_tooltip(node.path)
            if not searching:
                (self.explorer_expanded.add if is_open else self.explorer_expanded.discard)(node.path)
            to_open, gone = self._explorer_menu(node, is_root, root)
            if is_open:
                for child in node.children:
                    to_open = to_open or self._draw_explorer_node(child, False, opened_files, current, searching, root)[0]
                imgui.tree_pop()
            return to_open, gone
        is_current = key == current
        dirty = opened_files.get(key)
        flags = (imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open | imgui.TreeNodeFlags_.span_avail_width)
        if is_current:
            flags |= imgui.TreeNodeFlags_.selected
        elsewhere = key in opened_files and not is_current
        if elsewhere:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.55, 0.85, 1.0, 1.0))   # open in another tab
        imgui.tree_node_ex(node.name + (" *" if dirty else "") + f"##{node.path}", flags)
        if elsewhere:
            imgui.pop_style_color()
        self._draw_model_icon(node.name.lower().endswith(".csm"))
        if key == getattr(self, "_explorer_scroll_to", None):
            imgui.set_scroll_here_y(0.5)              # bring it to the middle of the sidebar
            self._explorer_scroll_to = None
        if imgui.is_item_clicked(imgui.MouseButton_.left):
            to_open = node.path
        if imgui.is_item_hovered():
            imgui.set_tooltip(node.path)
        return to_open or self._explorer_menu(node, False, root)[0], gone

    def _draw_recent_list(self, width: float) -> str | None:
        chosen = None
        if imgui.begin_table("##recent_table", 2, imgui.TableFlags_.sizing_fixed_fit):
            imgui.table_setup_column("name", imgui.TableColumnFlags_.width_fixed, width * 0.4)
            imgui.table_setup_column("folder", imgui.TableColumnFlags_.width_stretch)
            for i, path in enumerate(self.recent_files):
                missing = not self._recent_exists.get(path, True)
                imgui.table_next_row()
                imgui.table_next_column()
                label = Path(path).name + ("  (missing)" if missing else "")
                menu_open = imgui.is_popup_open(f"##recentmenu{i}")                # keep the row lit while its menu is up
                if imgui.selectable(f"{label}##recent{i}", menu_open, imgui.SelectableFlags_.span_all_columns)[0]:
                    chosen = path
                if imgui.is_item_hovered():
                    imgui.set_tooltip(path)
                self._recent_context_menu(path, f"##recentmenu{i}")
                imgui.table_next_column()
                imgui.text_disabled(str(Path(path).parent))
            imgui.end_table()
        return chosen

    def _draw_recent_grid(self) -> str | None:
        """The recent models as cards: a picture of the model with its name underneath."""
        chosen = None
        style = imgui.get_style()
        card = float(PREVIEW_DISPLAY_SIZE) + 2 * style.frame_padding.x       # the button around the picture
        columns = max(int(imgui.get_content_region_avail().x // (card + style.item_spacing.x)), 1)
        measure = lambda text: imgui.calc_text_size(text).x
        budget = [1]                                    # at most one picture is drawn from scratch per frame
        for stale in [p for p in self._previews if p not in self.recent_files]:
            texture = self._previews.pop(stale)[1]
            if texture is not None:
                texture.release()
        for i, path in enumerate(self.recent_files):
            if i % columns:
                imgui.same_line()
            missing = not self._recent_exists.get(path, True)
            imgui.begin_group()
            texture = None if missing else self._preview(path, budget)
            image_size = imgui.ImVec2(PREVIEW_DISPLAY_SIZE, PREVIEW_DISPLAY_SIZE)
            menu_open = imgui.is_popup_open(f"##cardmenu{i}")
            if menu_open:                                       # the card keeps its hovered look while its menu is up
                imgui.push_style_color(imgui.Col_.button, imgui.get_style_color_vec4(imgui.Col_.button_hovered))
            if texture is not None:
                # OpenGL pictures start at the bottom row, so the picture is drawn upside down to turn it over.
                clicked = imgui.image_button(f"##card{i}", imgui.ImTextureRef(texture.glo), image_size,
                                             imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))
            else:
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.14, 0.15, 0.17, 1))
                clicked = imgui.button(("(missing)" if missing else "...") + f"##card{i}",
                                       imgui.ImVec2(card, PREVIEW_DISPLAY_SIZE + 2 * style.frame_padding.y))
                imgui.pop_style_color()
            if menu_open:
                imgui.pop_style_color()
            if clicked:
                chosen = path
            if imgui.is_item_hovered() and not menu_open:
                imgui.set_tooltip(path)
            self._recent_context_menu(path, f"##cardmenu{i}")
            name = fit_text(Path(path).stem, card, measure)
            (imgui.text_disabled if missing else imgui.text)(name)
            imgui.end_group()
        return chosen

    def _draw_preview_hint(self) -> None:
        """A short reminder of how to leave picture mode, fading out after a few seconds."""
        age = time.perf_counter() - self._preview_since
        if age > 4.0:
            return
        alpha = 1.0 if age < 2.5 else (4.0 - age) / 1.5
        text = "Preview mode. Press P or Esc to leave."
        display = imgui.get_io().display_size
        size = imgui.calc_text_size(text)
        imgui.get_foreground_draw_list().add_text(
            imgui.ImVec2((display.x - size.x) / 2, display.y - size.y - 24),
            imgui.get_color_u32(imgui.ImVec4(1, 1, 1, 0.8 * alpha)), text)

    def _draw_ui(self) -> None:
        if self.preview_mode:
            self._draw_preview_hint()
            self._reset_layout = False
            return
        self._draw_menu()
        self._draw_tab_bar()
        self._draw_info_bar()
        self._draw_explorer_panel()
        self._draw_keybinds_window()
        if self.model is None:
            self._draw_home()
        else:
            self._draw_axis_gizmo()
            self._draw_right_dock()
            self._draw_mode_switch()
            self._draw_tool_strip()
        self._draw_new_matrix_window()
        self._reset_layout = False

    # -- frame --
    def _draw_lines(self, view_proj: glm.mat4, transform: glm.mat4, color, alpha: float = 1.0) -> None:
        self.flat["u_view_proj"].write(view_proj.to_bytes())
        self.flat["u_model"].write(transform.to_bytes())
        self.flat["u_color"].value = color
        self.flat["u_alpha"].value = alpha
        self.cube_vao.render(moderngl.LINES)

    def _draw_face(self, view_proj: glm.mat4, cell, axis: int, side: int, color) -> None:
        """Tint one face of a cell of the matrix the cursor is over, with an edge line, so it is clear what a click builds on."""
        item = self._hover_item()
        face = glm.mat4(1)
        for column, direction in enumerate(((axis + 1) % 3, (axis + 2) % 3, axis)):
            vector = [0.0, 0.0, 0.0, 0.0]
            vector[direction] = 1.0
            face[column] = glm.vec4(*vector)
        origin = [float(c) for c in cell]
        origin[axis] += 1.0 if side > 0 else 0.0
        face[3] = glm.vec4(*origin, 1.0)
        transform = item.transform * face
        ctx = self.ctx
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.screen.depth_mask = False
        self.flat["u_view_proj"].write(view_proj.to_bytes())
        self.flat["u_model"].write(transform.to_bytes())
        self.flat["u_color"].value = color
        self.flat["u_alpha"].value = HOVER_FACE_ALPHA
        self.face_fill_vao.render(moderngl.TRIANGLES)
        ctx.screen.depth_mask = True
        self.flat["u_alpha"].value = HOVER_FACE_EDGE_ALPHA
        self.face_edge_vao.render(moderngl.LINES)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.CULL_FACE)

    def _draw_grid(self, view_proj: glm.mat4) -> None:
        """Ground grid on y = 0 around the camera focus, with the X and Z axes as colored lines.

        Depth-tested, so voxels hide it. The scale is fixed: see ``chronovox.grid``.
        """
        cam, ctx = self.camera, self.ctx
        lines, line_colors = ground_grid.grid_lines(cam.target.x, cam.target.z)
        self.grid_vbo.write(np.hstack((lines, line_colors)).astype(np.float32).tobytes())

        # Lift the grid a hair toward the side the camera is on, so it wins against voxel faces that lie exactly
        # on y = 0. It must be a world-space shift: a depth-buffer nudge grows with distance and, once zoomed
        # out, would lift the grid above voxels that sit on it.
        lift = 0.01 + 5e-5 * cam.distance
        prog = self.grid_prog
        prog["u_lift"].value = lift if cam.eye().y >= 0 else -lift
        prog["u_view_proj"].write(view_proj.to_bytes())
        prog["u_center"].value = tuple(cam.target)
        prog["u_radius"].value = float(ground_grid.RADIUS)
        ctx.enable(moderngl.BLEND)
        ctx.screen.depth_mask = False   # lines are see-through, so they must not hide what is drawn later
        ctx.depth_func = "<="
        prog["u_fade"].value = 1.0
        self.grid_vao.render(moderngl.LINES, vertices=len(lines), first=0)
        ctx.depth_func = "<"
        ctx.screen.depth_mask = True
        ctx.disable(moderngl.BLEND)

    def _draw_overlay(self, view_proj: glm.mat4, segments: np.ndarray, colors: np.ndarray) -> None:
        """Draw colored world-space line segments on top of everything."""
        if not len(segments):
            return
        n = min(len(segments), self.overlay_vbo.size // (2 * 7 * 4))
        verts = np.hstack((segments[:n].reshape(-1, 3), np.repeat(colors[:n], 2, axis=0)))
        self.overlay_vbo.write(verts.astype(np.float32).tobytes())
        prog = self.grid_prog
        prog["u_lift"].value = 0.0
        prog["u_view_proj"].write(view_proj.to_bytes())
        prog["u_fade"].value = 0.0
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.overlay_vao.render(moderngl.LINES, vertices=n * 2)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

    def _draw_overlay_triangles(self, view_proj: glm.mat4, triangles: np.ndarray, colors: np.ndarray) -> None:
        """Draw translucent colored world-space triangles on top of everything (both sides)."""
        if not len(triangles):
            return
        n = min(len(triangles), self.overlay_vbo.size // (3 * 7 * 4))
        verts = np.hstack((triangles[:n].reshape(-1, 3), np.repeat(colors[:n], 3, axis=0)))
        self.overlay_vbo.write(verts.astype(np.float32).tobytes())
        prog = self.grid_prog
        prog["u_lift"].value = 0.0
        prog["u_view_proj"].write(view_proj.to_bytes())
        prog["u_fade"].value = 0.0
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.overlay_vao.render(moderngl.TRIANGLES, vertices=n * 3)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)

    def render_scene(self, width: int, height: int) -> None:
        ctx = self.ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, width, height)
        ctx.clear(*BACKGROUND, 1.0)
        if self.model is None:
            return
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.front_face = "cw"  # faces wind counter-clockwise in right-handed terms; view is left-handed
        aspect = width / max(height, 1)
        view_proj = self.camera.projection(aspect) * self.camera.view()
        self.prog["u_ao_strength"].value = 0.5 if self.ambient_occlusion else 0.0
        for item in self.items:
            if not item.visible or item.vao is None:
                continue
            self.prog["u_view_proj"].write(view_proj.to_bytes())
            self.prog["u_model"].write(item.transform.to_bytes())
            item.vao.render()

        if self.preview_mode:
            return                                 # just the model: no grid, outlines, tinted faces or handles
        if self.show_grid:
            self._draw_grid(view_proj)

        # Outlines are depth-tested, so voxels in front of them hide them. They often lie exactly on a voxel face,
        # so each is pulled a hair toward the camera (in world units, growing gently with distance).
        self.flat["u_eye"].value = tuple(self.camera.eye())
        self.flat["u_pull"].value = 0.005 + 1e-4 * self.camera.distance
        active = self.active_item()
        selected = self.selected_indices()
        if self.show_inactive_outlines:
            # Faint outlines so empty or hard-to-see matrices can still be found. The selected ones are left out:
            # their own outline is drawn below, and a second line at the same depth would not show.
            ctx.enable(moderngl.BLEND)
            for index, item in enumerate(self.items):
                if item is not active and item.visible and index not in selected:
                    sx, sy, sz = item.matrix.size
                    self._draw_lines(view_proj, item.transform * glm.scale(glm.vec3(sx, sy, sz)),
                                     BOUNDS_COLOR, INACTIVE_BOUNDS_ALPHA)
            ctx.disable(moderngl.BLEND)
        for index in selected:                             # the other matrices of a Ctrl+click selection
            item = self.items[index]
            if item is not active and item.visible:
                sx, sy, sz = item.matrix.size
                self._draw_lines(view_proj, item.transform * glm.scale(glm.vec3(sx, sy, sz)), SELECTED_BOUNDS_COLOR)
        if active is not None and active.visible:
            sx, sy, sz = active.matrix.size
            # Among several selected matrices the active (primary) one stands out in white.
            self._draw_lines(view_proj, active.transform * glm.scale(glm.vec3(sx, sy, sz)),
                             PRIMARY_BOUNDS_COLOR if len(selected) > 1 else BOUNDS_COLOR)
            count, box = self.selection_info() if self.selection_visible() else (0, None)
            if box is not None:                          # the box around the selected voxels
                lo, hi = box
                extent = glm.vec3(*(h - l + 1 for l, h in zip(lo, hi)))
                self._draw_lines(view_proj, active.transform * glm.translate(glm.vec3(*lo)) * glm.scale(extent),
                                 SELECTION_OUTLINE)
            if self.mode is Mode.MATRIX:
                eye = np.array(self.camera.eye(), float)
                anchor = self._gizmo_group()[1]
                triangles, fill_colors = self.matrix_tools.fills(active.matrix, eye, anchor)
                self._draw_overlay_triangles(view_proj, triangles, fill_colors)
                segments, colors = self.matrix_tools.overlay(active.matrix, eye, anchor)
                self._draw_overlay(view_proj, segments, colors)
            elif self.sel_drag is not None and self.sel_drag.kind == "box":
                lo = tuple(min(a, b) for a, b in zip(self.sel_drag.start, self.sel_drag.end))
                hi = tuple(max(a, b) for a, b in zip(self.sel_drag.start, self.sel_drag.end))
                extent = glm.vec3(*(h - l + 1 for l, h in zip(lo, hi)))
                self._draw_lines(view_proj, active.transform * glm.translate(glm.vec3(*lo)) * glm.scale(extent),
                                 SELECTION_OUTLINE)
            elif self.box is not None:
                lo, hi = self.box.corners()
                extent = glm.vec3(*(h - l + 1 for l, h in zip(lo, hi)))
                self._draw_lines(view_proj, active.transform * glm.translate(glm.vec3(*lo))
                                 * glm.scale(extent), TOOL_COLORS[self.box.tool])
            else:
                self._draw_hover(view_proj)
        elif self.mode is Mode.VOXEL:
            self._draw_hover(view_proj)

    def _draw_hover(self, view_proj: glm.mat4) -> None:
        """The outline of the cell the tool would act on, drawn on the matrix it is in."""
        item = self._hover_item()
        if self.hover is None or item is None:
            return
        self._draw_lines(view_proj, item.transform * glm.translate(glm.vec3(*self.hover)), TOOL_COLORS[self.tool])
        for cell in symmetric_cells(self.hover, item.matrix.size, tuple(self.mirror), self.radial)[1:]:
            self._draw_lines(view_proj, item.transform * glm.translate(glm.vec3(*cell)), TOOL_COLORS[self.tool], 0.5)
        if self.hover_face is not None and self.tool is Tool.PLACE:
            self._draw_face(view_proj, self.hover, *self.hover_face, HOVER_FACE_COLOR)

    def _update_title(self) -> None:
        name = self.path.name if self.path else "untitled"
        title = ("Chronovox Reforged" if self.model is None
                 else f"{'*' if self.dirty else ''}{name} - Chronovox Reforged")
        if title != self._title:
            self._title = title
            glfw.set_window_title(self.window, title)

    def frame(self, swap: bool = True) -> None:
        if self._swallow_press:
            self._swallow_press -= 1
        glfw.poll_events()
        self.imgui_impl.process_inputs()
        now = time.perf_counter()
        dt = min(now - self._last_time, 0.1)
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._last_time, 1e-6)
        self._last_time = now
        if glfw.get_key(self.window, glfw.KEY_HOME) == glfw.PRESS and self.model \
                and not imgui.get_io().want_text_input:
            self.camera.frame(*model_bounds(self.model))
        self._track_window()
        self._open_dropped()
        if now - self._last_pref_check > PREFERENCE_CHECK_SECONDS:
            self._last_pref_check = now
            self.save_preferences()
        self.workspace.poll()
        self._handle_shortcuts()
        self._handle_camera_input(dt, glfw.get_window_size(self.window)[1])
        self._handle_movement(dt)

        fb_size = glfw.get_framebuffer_size(self.window)
        imgui.new_frame()
        self._draw_ui()
        if self.preview_mode:                      # nothing can be edited or picked in picture mode
            self.hover = None
            self._lmb_was_down = self._mouse_down(glfw.MOUSE_BUTTON_LEFT)
        else:
            self._update_hover(fb_size)
            if self.mode is Mode.VOXEL:
                self._handle_editing_input(fb_size)
            else:
                self._handle_matrix_input(fb_size)
        imgui.render()
        self._update_title()

        self._sync_highlights()
        self.render_scene(*fb_size)
        self.imgui_impl.render(imgui.get_draw_data())
        if swap:
            glfw.swap_buffers(self.window)

    def run(self) -> None:
        while True:
            if glfw.window_should_close(self.window):
                if self._confirm_discard():
                    break
                glfw.set_window_should_close(self.window, False)
            self.frame()
        self.close()

    def screenshot(self, path: str | Path, frames: int = 3) -> None:
        """Render the scene and UI to a PNG."""
        for _ in range(frames):
            self.frame()
        self.frame(swap=False)
        width, height = glfw.get_framebuffer_size(self.window)
        raw = self.ctx.screen.read(viewport=(0, 0, width, height), components=3)
        gl.glGetError()  # moderngl's read leaves a sticky error that PyOpenGL (imgui backend) would raise later
        write_png(path, raw, width, height)

    def close(self) -> None:
        self.save_preferences()
        for item in self.items:
            item.release()
        self.items = []
        self.workspace.shutdown()
        for _, texture in self._previews.values():
            if texture is not None:
                texture.release()
        self.cube_vao.release()
        self.cube_vbo.release()
        self.grid_vao.release()
        self.grid_vbo.release()
        self.overlay_vao.release()
        self.overlay_vbo.release()
        for buffer in (self.face_fill_vao, self.face_fill_vbo, self.face_edge_vao, self.face_edge_vbo):
            buffer.release()
        self.imgui_impl.shutdown()
        glfw.destroy_window(self.window)
        glfw.terminate()


def write_png(path: str | Path, rgb: bytes, width: int, height: int) -> None:
    """Minimal PNG writer; ``rgb`` is bottom-up as returned by OpenGL."""
    stride = width * 3
    rows = [b"\0" + rgb[y * stride:(y + 1) * stride] for y in range(height - 1, -1, -1)]

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"".join(rows)))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)
