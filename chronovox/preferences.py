"""User preferences, stored as JSON in the user's config folder. No GL, UI or file-format imports.

Everything here is forgiving on purpose: a missing, damaged or hand-edited file never stops the
editor from starting. Each field is checked on its own and falls back to its default if it is
missing or has a bad value, and unknown fields are ignored.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 1
MAX_RECENT = 100   # how many recently opened models are remembered

MIN_WINDOW = (480, 320)
MAX_WINDOW = (16384, 16384)


def default_path() -> Path:
    """Where preferences live: %APPDATA%\\ChronovoxReforged on Windows, ~/.config elsewhere."""
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "ChronovoxReforged" / "preferences.json"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "chronovox-reforged" / "preferences.json"


# --- field checks: each returns the cleaned value, or the default if the input is no good ---

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bool(value, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _number_list(value, count: int, default):
    if isinstance(value, (list, tuple)) and len(value) == count and all(_is_number(v) for v in value):
        return [float(v) for v in value]
    return default


def _int_in(value, low: int, high: int, default):
    if _is_number(value) and low <= value <= high:
        return int(value)
    return default


def _section(data, key: str) -> dict:
    value = data.get(key) if isinstance(data, dict) else None
    return value if isinstance(value, dict) else {}


@dataclass
class Preferences:
    # window: the size and position it has when not maximized, and whether it is maximized
    window_width: int = 1280
    window_height: int = 800
    window_x: int | None = None      # None: let the system place it
    window_y: int | None = None
    maximized: bool = False
    # view toggles
    ambient_occlusion: bool = True
    show_grid: bool = True
    show_inactive_outlines: bool = True
    live_box_preview: bool = True
    # panels: None means "use the default layout"
    model_panel: list[float] | None = None    # x, y, width, height
    tools_panel: list[float] | None = None    # gap to the right edge, y, width, height
    # other
    brush_color: list[int] = field(default_factory=lambda: [89, 153, 242])
    last_dir: str = ""
    bulk_resolution: int = 1         # the resolution the "set all" box starts with
    expand_on_edge: bool = True      # placing on a face at the edge of a matrix grows the matrix by one cell to take it
    recent_files: list[str] = field(default_factory=list)   # most recent first
    recent_view: str = "list"        # how the home page shows them: "list" or "grid"
    workspace_url: str = ""          # the shared workspace server the login form starts with
    explorer_csm: bool = False       # the Files panel also lists legacy .csm models
    explorer_width: float = 260.0    # how wide the Files sidebar is when open
    right_dock_width: float = 340.0  # the docked Model and Tools panels on the right
    right_dock_split: float = 0.4    # how much of their height the Model panel gets
    explorer_visible: bool = True    # the Files panel
    explorer_roots: list[str] = field(default_factory=list)      # the folders it shows
    explorer_expanded: list[str] = field(default_factory=list)   # the folders opened in it

    # -- conversion --
    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "window": {"width": self.window_width, "height": self.window_height,
                       "x": self.window_x, "y": self.window_y, "maximized": self.maximized},
            "view": {"ambient_occlusion": self.ambient_occlusion, "show_grid": self.show_grid,
                     "show_inactive_outlines": self.show_inactive_outlines,
                     "live_box_preview": self.live_box_preview},
            "panels": {"model": self.model_panel, "tools": self.tools_panel},
            "brush_color": self.brush_color,
            "last_dir": self.last_dir,
            "bulk_resolution": self.bulk_resolution,
            "expand_on_edge": self.expand_on_edge,
            "recent_files": self.recent_files,
            "recent_view": self.recent_view,
            "workspace_url": self.workspace_url,
            "explorer_width": self.explorer_width,
            "right_dock": {"width": self.right_dock_width, "split": self.right_dock_split},
            "explorer_visible": self.explorer_visible,
            "explorer_csm": self.explorer_csm,
            "explorer_roots": self.explorer_roots,
            "explorer_expanded": self.explorer_expanded,
        }

    @classmethod
    def from_dict(cls, data) -> "Preferences":
        """Build preferences from parsed JSON, replacing anything unusable with its default."""
        d = cls()
        window, view, panels = _section(data, "window"), _section(data, "view"), _section(data, "panels")
        d.window_width = _int_in(window.get("width"), MIN_WINDOW[0], MAX_WINDOW[0], d.window_width)
        d.window_height = _int_in(window.get("height"), MIN_WINDOW[1], MAX_WINDOW[1], d.window_height)
        x, y = window.get("x"), window.get("y")
        if _is_number(x) and _is_number(y) and abs(x) < 1e6 and abs(y) < 1e6:
            d.window_x, d.window_y = int(x), int(y)
        d.maximized = _bool(window.get("maximized"), d.maximized)
        for name in ("ambient_occlusion", "show_grid", "show_inactive_outlines", "live_box_preview"):
            setattr(d, name, _bool(view.get(name), getattr(d, name)))
        d.model_panel = _number_list(panels.get("model"), 4, None)
        d.tools_panel = _number_list(panels.get("tools"), 4, None)
        color = data.get("brush_color") if isinstance(data, dict) else None
        if isinstance(color, (list, tuple)) and len(color) == 3 and all(
                _is_number(c) and 0 <= c <= 255 for c in color):
            d.brush_color = [int(c) for c in color]
        last_dir = data.get("last_dir") if isinstance(data, dict) else None
        d.last_dir = last_dir if isinstance(last_dir, str) else ""
        d.expand_on_edge = _bool(data.get("expand_on_edge") if isinstance(data, dict) else None, d.expand_on_edge)
        d.bulk_resolution = _int_in(data.get("bulk_resolution") if isinstance(data, dict) else None, 1, 8, d.bulk_resolution)
        recent = data.get("recent_files") if isinstance(data, dict) else None
        if isinstance(recent, list):
            d.recent_files = [p for p in recent if isinstance(p, str) and p][:MAX_RECENT]
        view = data.get("recent_view") if isinstance(data, dict) else None
        d.recent_view = view if view in ("list", "grid") else d.recent_view
        url = data.get("workspace_url") if isinstance(data, dict) else None
        d.workspace_url = url.strip()[:300] if isinstance(url, str) else ""
        d.explorer_visible = _bool(data.get("explorer_visible") if isinstance(data, dict) else None, d.explorer_visible)
        width = data.get("explorer_width") if isinstance(data, dict) else None
        if _is_number(width) and 100 <= width <= 4000:
            d.explorer_width = float(width)
        dock = _section(data, "right_dock")
        if _is_number(dock.get("width")) and 150 <= dock["width"] <= 4000:
            d.right_dock_width = float(dock["width"])
        if _is_number(dock.get("split")) and 0.05 <= dock["split"] <= 0.95:
            d.right_dock_split = float(dock["split"])
        d.explorer_csm = _bool(data.get("explorer_csm") if isinstance(data, dict) else None, d.explorer_csm)
        for name in ("explorer_roots", "explorer_expanded"):
            value = data.get(name) if isinstance(data, dict) else None
            if isinstance(value, list):
                setattr(d, name, [p for p in value if isinstance(p, str) and p][:2000])
        return d

    # -- files --
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Preferences":
        """Read preferences; defaults if the file is missing, unreadable or not valid JSON."""
        try:
            data = json.loads(Path(path or default_path()).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        return cls.from_dict(data)

    def save(self, path: str | Path | None = None) -> bool:
        """Write the file (replacing it in one step so a crash cannot leave half a file).

        Returns False instead of raising if the folder cannot be written.
        """
        target = Path(path or default_path())
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False,
                                             suffix=".tmp") as handle:
                json.dump(self.to_dict(), handle, indent=2)
                temp = Path(handle.name)
            os.replace(temp, target)
            return True
        except OSError:
            return False

