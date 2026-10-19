"""Ground grid on the y = 0 plane, as line lists. Pure numpy: no GL, UI or file imports.

The grid has a fixed scale: a light line at every whole voxel and a brighter one every
``MAJOR_EVERY`` voxels, whatever the zoom. It covers a square of ``RADIUS`` voxels either side of the
camera focus (the point the camera orbits, not the camera's own position) and fades out toward its
edge. The X and Z axes (z = 0 and x = 0) are drawn in the gizmo colors, so the world origin is easy to
find where they cross.
"""
from __future__ import annotations

import numpy as np

RADIUS = 128        # voxels from the focus to the edge of the grid
MAJOR_EVERY = 8     # every this many voxels, a brighter line

MINOR_COLOR = (0.60, 0.66, 0.76, 0.16)
MAJOR_COLOR = (0.66, 0.72, 0.84, 0.34)
X_AXIS_COLOR = (0.93, 0.33, 0.33, 0.85)
Z_AXIS_COLOR = (0.38, 0.58, 1.00, 0.85)

VERTEX_COUNT = 2 * 2 * (2 * RADIUS + 1)     # two vertices per line, a line per voxel, in each direction


def grid_lines(focus_x: float, focus_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Line list (positions, RGBA colors) for the grid around a focus point.

    The grid is centered on the focus snapped to a whole voxel, so lines always sit at whole
    world coordinates and the pattern stays put while the focus moves across it. The lines through
    x = 0 and z = 0 are the colored axes, present only if the origin is within the grid.
    """
    cx, cz = int(round(focus_x)), int(round(focus_z))
    ks = np.arange(-RADIUS, RADIUS + 1)
    lo_x, hi_x, lo_z, hi_z = cx - RADIUS, cx + RADIUS, cz - RADIUS, cz + RADIUS

    def colors_for(coords, axis_color):
        colors = np.tile(np.array(MINOR_COLOR, np.float32), (len(coords), 1))
        colors[coords % MAJOR_EVERY == 0] = MAJOR_COLOR
        colors[coords == 0] = axis_color
        return colors

    xs, zs = cx + ks, cz + ks
    along_z = np.zeros((len(xs), 2, 3), np.float32)       # lines of constant x, running along Z
    along_z[:, :, 0] = xs[:, None]
    along_z[:, 0, 2], along_z[:, 1, 2] = lo_z, hi_z
    along_x = np.zeros((len(zs), 2, 3), np.float32)       # lines of constant z, running along X
    along_x[:, :, 2] = zs[:, None]
    along_x[:, 0, 0], along_x[:, 1, 0] = lo_x, hi_x

    positions = np.concatenate((along_z, along_x)).reshape(-1, 3)
    colors = np.concatenate((colors_for(xs, Z_AXIS_COLOR), colors_for(zs, X_AXIS_COLOR)))
    return positions, np.repeat(colors, 2, axis=0)
