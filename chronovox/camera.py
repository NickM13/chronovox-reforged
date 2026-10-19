"""Scene camera with Unity-editor-style controls. No GL, UI or file imports.

The camera looks at a focus point (``target``) from ``distance`` away. Rotating
(orbit) and scrolling never move the focus; panning and WASD do, carrying
the camera along with it. Coordinates are Y-up, left-handed.
"""
from __future__ import annotations

import math

from pyglm import glm

FOV = math.radians(60)  # Unity's default scene-view field of view
PITCH_LIMIT = math.radians(89)
LOOK_SENSITIVITY = 0.005


class Camera:
    def __init__(self) -> None:
        self.target = glm.vec3(0)
        self.distance = 10.0
        self.yaw = math.radians(35)
        self.pitch = math.radians(25)  # elevation of the eye above the pivot
        self.fly_speed = 10.0          # units per second

    # -- state --
    def frame(self, center: glm.vec3, radius: float) -> None:
        self.target = glm.vec3(center)
        self.distance = max(radius, 0.5) / math.sin(FOV / 2) * 1.1
        self.fly_speed = max(radius, 1.0)

    def focus_on(self, point: glm.vec3) -> None:
        """Move the focus (the point the camera orbits) without changing the view angle or distance."""
        self.target = glm.vec3(point)

    def _offset(self) -> glm.vec3:
        """Unit vector from pivot to eye."""
        cp = math.cos(self.pitch)
        return glm.vec3(cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw))

    def eye(self) -> glm.vec3:
        return self.target + self._offset() * self.distance

    def view(self) -> glm.mat4:
        return glm.lookAtLH(self.eye(), self.target, glm.vec3(0, 1, 0))

    def projection(self, aspect: float) -> glm.mat4:
        return glm.perspectiveLH_NO(FOV, aspect, self.near(), self.far())

    def near(self) -> float:
        return max(self.distance * 0.01, 0.05)

    def far(self) -> float:
        return max(2000.0, self.distance * 20)

    def axes(self) -> tuple[glm.vec3, glm.vec3, glm.vec3]:
        """Camera right, up and forward in world space."""
        view = self.view()
        right = glm.vec3(view[0][0], view[1][0], view[2][0])
        up = glm.vec3(view[0][1], view[1][1], view[2][1])
        forward = glm.vec3(view[0][2], view[1][2], view[2][2])
        return right, up, forward

    def _clamp_pitch(self) -> None:
        self.pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, self.pitch))

    # -- Unity controls --
    def orbit(self, dx: float, dy: float) -> None:
        """Alt + LMB or RMB drag: swing around the focus; the focus itself never moves.

        Dragging right turns the view right and dragging down turns it down.
        """
        self.yaw += dx * LOOK_SENSITIVITY
        self.pitch += dy * LOOK_SENSITIVITY
        self._clamp_pitch()

    def pan(self, dx: float, dy: float, viewport_height: float) -> None:
        """MMB drag: the point under the cursor stays under the cursor."""
        right, up, _ = self.axes()
        per_pixel = 2 * self.distance * math.tan(FOV / 2) / max(viewport_height, 1)
        self.target += (-right * dx + up * dy) * per_pixel

    def zoom_drag(self, dx: float, dy: float) -> None:
        """Alt + RMB drag: dolly toward or away from the pivot."""
        self.distance = max(0.05, self.distance * math.exp(-(dx - dy) * 0.005))

    def scroll(self, steps: float) -> None:
        """Mouse wheel: dolly toward the pivot, proportional to distance."""
        self.distance = max(0.05, self.distance * (0.85 ** steps))

    def fly_move(self, right: float, up: float, forward: float, dt: float, fast: bool) -> None:
        """WASD/QE: move the focus, and the camera with it. Axes are -1, 0 or 1; Q/E move along world Y."""
        r, _, f = self.axes()
        move = r * right + f * forward + glm.vec3(0, up, 0)
        if glm.length(move) == 0:
            return
        speed = self.fly_speed * (3.0 if fast else 1.0)
        self.target += glm.normalize(move) * speed * dt

    def gizmo_axes(self) -> list[tuple[str, int, float, float, float]]:
        """The six world axis directions as the camera sees them, for an axis gizmo.

        Each entry is (axis name, sign, x, y, depth): x is to the right and y is up on
        screen, and depth is positive away from the viewer. Sorted far to near.
        """
        right, up, forward = self.axes()
        out = []
        for index, name in enumerate("XYZ"):
            for sign in (1, -1):
                d = glm.vec3(0)
                d[index] = sign
                out.append((name, sign, glm.dot(d, right), glm.dot(d, up), glm.dot(d, forward)))
        return sorted(out, key=lambda e: -e[4])
