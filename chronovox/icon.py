"""The application icon: finding it, making the sizes the window needs, and the Windows .ico for the .exe."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

from . import png

APP_ID = "NickMead.ChronovoxReforged"
WINDOW_ICON_SIZES = (16, 32, 48, 64, 128)       # the sizes handed to the window; the system picks the best fit
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def resource_path(*parts: str) -> Path:
    """A file shipped with the program: next to the source, or inside the bundle of a built .exe."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base.joinpath(*parts)


def master_picture() -> "png.np.ndarray":
    """The 256 pixel icon as an RGBA array."""
    return png.read_png(resource_path("assets", "icon_256.png"))


def window_icons(sizes=WINDOW_ICON_SIZES) -> list[tuple[int, int, list]]:
    """The icon at several sizes in the form ``glfw.set_window_icon`` takes: (width, height, rows of RGBA pixels)."""
    picture = master_picture()
    return [(s, s, png.resize(picture, s).tolist()) for s in sizes]


def make_ico(sizes=ICO_SIZES) -> bytes:
    """A Windows .ico file holding the icon at several sizes (each stored as a PNG, which Windows Vista and up read)."""
    picture = master_picture()
    images = [png.encode_png(picture if s == picture.shape[0] else png.resize(picture, s)) for s in sizes]
    out = bytearray(struct.pack("<HHH", 0, 1, len(sizes)))
    offset = 6 + 16 * len(sizes)
    for size, data in zip(sizes, images):
        out += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)   # 256 is stored as 0
        offset += len(data)
    return bytes(out) + b"".join(images)


def set_app_id() -> None:
    """Tell Windows this is its own program, so its taskbar button shows our icon and is not grouped with python.exe."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError):
        pass
