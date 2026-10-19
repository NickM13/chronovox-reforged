"""Text for the info bar along the bottom of the window. Plain strings only: no ImGui, so it is easy to test."""
from __future__ import annotations

from .model import POSITION_SCALE, Matrix


def number(value: float) -> str:
    """A number without trailing zeros: 2.5, -3, 0.0078125."""
    return f"{value:.7f}".rstrip("0").rstrip(".") or "0"


def matrix_summary(matrix: Matrix) -> str:
    """The active matrix in one short line: name, size in cells, resolution and position."""
    sx, sy, sz = matrix.size
    position = ", ".join(number(p / POSITION_SCALE) for p in matrix.position)
    return f"{matrix.name}  |  {sx}x{sy}x{sz}  res {matrix.resolution}  |  pos {position}"


def fit_text(text: str, width: float, measure) -> str:
    """Shorten ``text`` with "..." until ``measure(text)`` is at most ``width``. Cuts from the end."""
    if measure(text) <= width:
        return text
    low, high = 0, len(text)
    while low < high:                      # the longest prefix that still fits with the dots
        mid = (low + high + 1) // 2
        if measure(text[:mid] + "...") <= width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + "..." if low else ""
