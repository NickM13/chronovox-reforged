"""How bright each voxel face is, by the direction it faces. Pure math: no GL, UI or file imports.

Brightness is linear in the face normal: ``BASE + WEIGHTS . normal``. The weights are chosen so that
all six axis directions come out at different brightness, and any two that can meet at a corner differ
by at least ``MIN_GAP``. A single directional light would leave the three faces turned away from it
all at the same ambient level, which makes corners hard to read.

The top is the brightest and the bottom the darkest, over about the same range as before (0.32 to 0.88).
Of the sides, +X and -Z (toward the old key light, from the upper right and behind) are lighter and -X and
+Z darker, as before.
"""
from __future__ import annotations

BASE = 0.60
WEIGHTS = (0.14, 0.28, -0.05)      # x, y, z

MIN_GAP = 0.08                      # the least difference between faces of different axes


def face_brightness(normal) -> float:
    """Brightness factor (about 0.32 to 0.88) of a face with this unit normal, before ambient occlusion."""
    return BASE + sum(w * n for w, n in zip(WEIGHTS, normal))
