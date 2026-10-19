"""Turn a voxel grid into renderable faces. Pure numpy: no GL, UI or file imports.

Hidden faces (between two solid voxels) are culled. Each face vertex also gets an
ambient-occlusion level from the voxels around it. Greedy meshing comes in the
performance pass (milestone 8).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AO_LEVELS = 3  # 0 = fully occluded corner ... 3 = open

# Highlighted voxels (the selection) keep their own color, pulled part of the way toward this one.
HIGHLIGHT_RGB = np.array([110, 175, 255], np.float32)
HIGHLIGHT_STRENGTH = 0.35


def highlighted(colors: np.ndarray, strength: float = HIGHLIGHT_STRENGTH) -> np.ndarray:
    """RGBA colors (N, 4) as a slightly highlighted version of themselves; alpha is unchanged."""
    out = colors.copy()
    rgb = colors[:, :3].astype(np.float32) * (1.0 - strength) + HIGHLIGHT_RGB * strength
    out[:, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out


@dataclass
class Mesh:
    """Four vertices per face, drawn as two triangles via ``indices``."""

    positions: np.ndarray  # (4F, 3) float32, matrix-local voxel units
    normals: np.ndarray    # (4F, 3) float32
    colors: np.ndarray     # (4F, 4) uint8 RGBA
    ao: np.ndarray         # (4F,) uint8, 0-3 scaled to 0-255 (level * 85)
    indices: np.ndarray    # (6F,) uint32

    @property
    def face_count(self) -> int:
        return len(self.positions) // 4

    def ao_levels(self) -> np.ndarray:
        """Occlusion per vertex as the integers 0-3."""
        return (self.ao // 85).astype(np.uint8)


# For each face direction: (normal xyz, u xyz, v xyz).
# u x v = normal, so corners base, base+u, base+u+v, base+v wind consistently.
_FACES = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
)

# Corner k sits at base + (a*u + b*v); its AO neighbours lie on the (su, sv) side of the face.
_CORNERS = ((0, 0, -1, -1), (1, 0, +1, -1), (1, 1, +1, +1), (0, 1, -1, +1))

_PAD = 2  # AO looks up to two cells beyond the grid edge
_QUAD = np.array([0, 1, 2, 0, 2, 3], np.uint32)
_QUAD_FLIPPED = np.array([1, 2, 3, 1, 3, 0], np.uint32)  # other diagonal, same winding


def build_mesh(voxels: np.ndarray, palette: np.ndarray, highlight: np.ndarray | None = None) -> Mesh:
    """Build a face-culled, ambient-occluded mesh for a (z, y, x) grid of palette indices.

    Palette row k is index k + 1. Any non-zero index counts as solid, for culling
    and for occlusion. Voxels marked in the boolean grid ``highlight`` get slightly highlighted colors; the
    shape of the mesh is the same either way.
    """
    depth, height, width = voxels.shape
    solid = voxels != 0
    padded = np.pad(solid, _PAD).astype(np.uint8)
    palette = np.asarray(palette, dtype=np.uint8).reshape(-1, 4)
    colors_lut = np.vstack(([0, 0, 0, 0], palette)).astype(np.uint8)

    def shifted(dx: int, dy: int, dz: int) -> np.ndarray:
        """The solid grid as seen from one cell over, for every cell at once."""
        return padded[_PAD + dz:_PAD + dz + depth, _PAD + dy:_PAD + dy + height,
                      _PAD + dx:_PAD + dx + width]

    pos_parts, nrm_parts, col_parts, ao_parts, idx_parts = [], [], [], [], []
    face_offset = 0
    for normal, u, v in _FACES:
        zs, ys, xs = np.nonzero(solid & (shifted(*normal) == 0))
        n = len(zs)
        if n == 0:
            continue

        def occupied(dx: int, dy: int, dz: int) -> np.ndarray:
            return padded[zs + _PAD + dz, ys + _PAD + dy, xs + _PAD + dx].astype(np.int8)

        ao = np.empty((n, 4), np.uint8)
        for k, (a, b, su, sv) in enumerate(_CORNERS):
            su_off = tuple(normal[i] + su * u[i] for i in range(3))
            sv_off = tuple(normal[i] + sv * v[i] for i in range(3))
            corner_off = tuple(normal[i] + su * u[i] + sv * v[i] for i in range(3))
            side1, side2, corner = occupied(*su_off), occupied(*sv_off), occupied(*corner_off)
            ao[:, k] = np.where((side1 == 1) & (side2 == 1), 0, AO_LEVELS - (side1 + side2 + corner))

        base = np.stack((xs, ys, zs), axis=1).astype(np.float32)
        if any(c > 0 for c in normal):
            base += np.array(normal, np.float32)  # face sits on the far side
        u_vec, v_vec = np.array(u, np.float32), np.array(v, np.float32)
        corners = np.stack((base, base + u_vec, base + u_vec + v_vec, base + v_vec), axis=1)
        pos_parts.append(corners.reshape(-1, 3))
        nrm_parts.append(np.broadcast_to(np.array(normal, np.float32), (n * 4, 3)))
        face_colors = colors_lut[voxels[zs, ys, xs]]
        if highlight is not None:
            marked = highlight[zs, ys, xs]
            if marked.any():
                face_colors[marked] = highlighted(face_colors[marked])
        col_parts.append(np.repeat(face_colors, 4, axis=0))
        ao_parts.append((ao * 85).reshape(-1))

        # Split the quad along the diagonal whose ends are most alike, or the AO
        # gradient shows a visible crease.
        ao_i = ao.astype(np.int16)
        flip = ao_i[:, 0] + ao_i[:, 2] > ao_i[:, 1] + ao_i[:, 3]
        pattern = np.where(flip[:, None], _QUAD_FLIPPED, _QUAD)
        first = (np.arange(n, dtype=np.uint32) + face_offset) * 4
        idx_parts.append((first[:, None] + pattern).reshape(-1))
        face_offset += n

    if not pos_parts:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                    np.zeros((0, 4), np.uint8), np.zeros(0, np.uint8), np.zeros(0, np.uint32))

    return Mesh(np.concatenate(pos_parts), np.concatenate(nrm_parts), np.concatenate(col_parts),
                np.concatenate(ao_parts), np.concatenate(idx_parts).astype(np.uint32))
