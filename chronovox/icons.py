"""Picture icons from assets/icons/<name>.png, drawn crisp (no smoothing) because they are pixel art.

Drop a PNG named after an icon into that folder to replace it. A name with no file makes the caller fall back to
its own drawing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import moderngl
import numpy as np

from . import png


def icon_folder() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "assets" / "icons"


class IconSet:
    def __init__(self, ctx: moderngl.Context, folder: Path | None = None) -> None:
        self.ctx = ctx
        self.folder = folder or icon_folder()
        self._cache: dict[str, tuple[moderngl.Texture, int, int] | None] = {}

    def get(self, name: str):
        """(texture, width, height) of an icon, or None if there is no such file."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str):
        path = self.folder / f"{name}.png"
        try:
            pixels = png.read_png(path)
        except (OSError, ValueError, KeyError):
            return None
        height, width = pixels.shape[:2]
        texture = self.ctx.texture((width, height), 4, np.ascontiguousarray(pixels, np.uint8).tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)           # pixel art stays sharp
        return texture, width, height

    def release(self) -> None:
        for entry in self._cache.values():
            if entry is not None:
                entry[0].release()
        self._cache.clear()
