"""The toolbox: a column of buttons that each show a small drawn icon, with the name and key on hover."""
from __future__ import annotations

import math
from dataclasses import dataclass

from imgui_bundle import imgui

ACCENT = (0.05, 0.45, 1.0)            # the selected tool: a strong blue that shows from the corner of the eye
IDLE = (0.16, 0.17, 0.19)             # the others: a dark grey with no color of its own
IDLE_HOVER = (0.27, 0.29, 0.34)


def push_selected_colors(selected: bool) -> int:
    """Color the next button: strong blue if it is the chosen one, dark grey if not. Returns how many colors to pop."""
    base, hover, down = (ACCENT, (ACCENT[0] + .08, ACCENT[1] + .08, ACCENT[2], ), (ACCENT[0] - .03, ACCENT[1] - .05, ACCENT[2] - .1))         if selected else (IDLE, IDLE_HOVER, (0.34, 0.36, 0.42))
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*base, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(*hover, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(*down, 1.0))
    return 3
GRASS, BRICK, SKY, SPARK = (0.35, 0.85, 0.45), (0.95, 0.35, 0.30), (0.40, 0.65, 1.0), (1.0, 0.85, 0.30)
STEEL = (0.78, 0.81, 0.86)


@dataclass(frozen=True)
class ToolButton:
    key: object          # what is returned when the button is clicked
    label: str
    icon: str            # which drawing to use, see ``draw_icon``
    shortcut: str        # the key that picks it, as shown on hover
    tip: str             # shown on hover


def _col(color, alpha: float = 1.0) -> int:
    return imgui.get_color_u32(imgui.ImVec4(color[0], color[1], color[2], alpha))


def _pt(x: float, y: float) -> imgui.ImVec2:
    return imgui.ImVec2(x, y)


def _cube(draw, cx: float, cy: float, r: float, shades) -> None:
    """An isometric cube: top, left and right faces in three shades."""
    def at(x, y):
        return _pt(cx + x * r, cy + y * r)
    top, ur, lr, bottom, ll, ul, mid = at(0, -1), at(.87, -.5), at(.87, .5), at(0, 1), at(-.87, .5), at(-.87, -.5), at(0, 0)
    draw.add_quad_filled(top, ur, mid, ul, _col(shades[0]))
    draw.add_quad_filled(ul, mid, bottom, ll, _col(shades[1]))
    draw.add_quad_filled(mid, ur, lr, bottom, _col(shades[2]))


def _dashed_rect(draw, x0, y0, x1, y1, color, thickness=1.5, dash=3.0) -> None:
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = math.hypot(bx - ax, by - ay)
        steps = max(int(length / dash), 1)
        for k in range(0, steps, 2):
            a, b = k / steps, min((k + 1) / steps, 1.0)
            draw.add_line(_pt(ax + (bx - ax) * a, ay + (by - ay) * a), _pt(ax + (bx - ax) * b, ay + (by - ay) * b),
                          color, thickness)


def _arrow_head(draw, tip, direction, size, color) -> None:
    dx, dy = direction
    px, py = -dy, dx
    base = (tip[0] - dx * size, tip[1] - dy * size)
    draw.add_triangle_filled(_pt(*tip), _pt(base[0] + px * size * .6, base[1] + py * size * .6),
                             _pt(base[0] - px * size * .6, base[1] - py * size * .6), color)


def draw_icon(draw, kind: str, cx: float, cy: float, r: float, brush=(0.35, 0.60, 0.95)) -> None:
    """Draw an icon of half-size ``r`` around (cx, cy). ``brush`` is the current brush color, shown on color tools."""
    thick = max(r * 0.22, 1.5)
    if kind in ("place", "erase"):
        _cube(draw, cx - r * .15, cy + r * .2, r * .78, ((.62, .66, .74), (.45, .48, .56), (.34, .36, .43)))
        mark = GRASS if kind == "place" else BRICK
        mx, my, m = cx + r * .62, cy - r * .62, r * .38
        if kind == "place":
            draw.add_line(_pt(mx - m, my), _pt(mx + m, my), _col(mark), thick)
            draw.add_line(_pt(mx, my - m), _pt(mx, my + m), _col(mark), thick)
        else:
            draw.add_line(_pt(mx - m, my - m), _pt(mx + m, my + m), _col(mark), thick)
            draw.add_line(_pt(mx - m, my + m), _pt(mx + m, my - m), _col(mark), thick)
    elif kind == "paint":
        draw.add_line(_pt(cx - r * .55, cy + r * .55), _pt(cx + r * .75, cy - r * .75), _col(STEEL), thick * 1.3)
        draw.add_circle_filled(_pt(cx - r * .7, cy + r * .7), r * .42, _col(brush))
    elif kind == "pick":
        draw.add_line(_pt(cx - r * .65, cy + r * .65), _pt(cx + r * .25, cy - r * .25), _col(STEEL), thick)
        draw.add_circle_filled(_pt(cx + r * .5, cy - r * .5), r * .4, _col(STEEL))
        draw.add_circle_filled(_pt(cx - r * .75, cy + r * .75), r * .2, _col(brush))
    elif kind == "select_box":
        _dashed_rect(draw, cx - r * .8, cy - r * .7, cx + r * .8, cy + r * .7, _col(SPARK), thick * .8)
    elif kind == "wand":
        draw.add_line(_pt(cx - r * .8, cy + r * .8), _pt(cx + r * .1, cy - r * .1), _col(STEEL), thick * 1.2)
        sx, sy, s = cx + r * .45, cy - r * .45, r * .5
        for dx, dy in ((1, 0), (0, 1), (.7, .7), (.7, -.7)):
            draw.add_line(_pt(sx - dx * s, sy - dy * s), _pt(sx + dx * s, sy + dy * s), _col(SPARK), thick * .7)
    elif kind in ("fill_face", "fill_replace"):
        body = _col((.62, .66, .74))
        draw.add_quad_filled(_pt(cx - r * .65, cy - r * .15), _pt(cx + r * .45, cy - r * .15),
                             _pt(cx + r * .3, cy + r * .8), _pt(cx - r * .5, cy + r * .8), body)
        draw.add_line(_pt(cx - r * .65, cy - r * .15), _pt(cx, cy - r * .85), _col(STEEL), thick * .8)
        draw.add_line(_pt(cx, cy - r * .85), _pt(cx + r * .45, cy - r * .15), _col(STEEL), thick * .8)
        mark = {"fill_face": GRASS, "fill_replace": brush}[kind]
        draw.add_circle_filled(_pt(cx + r * .72, cy + r * .55), r * .3, _col(mark))
    elif kind == "move":
        for dx, dy in ((1, 0), (0, 1)):
            draw.add_line(_pt(cx - dx * r, cy - dy * r), _pt(cx + dx * r, cy + dy * r), _col(STEEL), thick * .8)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            _arrow_head(draw, (cx + dx * r, cy + dy * r), (dx, dy), r * .45, _col(STEEL))
    elif kind == "resize":
        draw.add_rect(_pt(cx - r * .55, cy - r * .55), _pt(cx + r * .55, cy + r * .55), _col(STEEL), thickness=thick * .8)
        for sx, sy in ((1, 1), (-1, -1)):
            tip = (cx + sx * r * .95, cy + sy * r * .95)
            draw.add_line(_pt(cx + sx * r * .35, cy + sy * r * .35), _pt(*tip), _col(SKY), thick * .8)
            _arrow_head(draw, tip, (sx * .7, sy * .7), r * .5, _col(SKY))
    elif kind == "rotate":
        radius = r * .75
        points = [_pt(cx + math.cos(a) * radius, cy - math.sin(a) * radius)
                  for a in [math.radians(40 + 280 * k / 14) for k in range(15)]]
        for a, b in zip(points, points[1:]):
            draw.add_line(a, b, _col(STEEL), thick * .9)
        end = math.radians(320)
        direction = (-math.sin(end), -math.cos(end))                            # along the arc at its end (unit length)
        tx, ty = cx + math.cos(end) * radius, cy - math.sin(end) * radius
        _arrow_head(draw, (tx + direction[0] * r * .25, ty + direction[1] * r * .25), direction, r * .5, _col(STEEL))


def tool_strip(buttons: list[ToolButton], selected, brush=(0.35, 0.60, 0.95), size: float = 34.0):
    """A column of icon-only buttons. Hovering one shows its name and key. Returns the key clicked this frame, or None."""
    clicked = None
    draw = imgui.get_window_draw_list()
    for button in buttons:
        count = push_selected_colors(button.key == selected)
        if imgui.button(f"##strip_{button.icon}", imgui.ImVec2(size, size)):
            clicked = button.key
        imgui.pop_style_color(count)
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_icon(draw, button.icon, (low.x + high.x) / 2, (low.y + high.y) / 2, size * 0.34, brush)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{button.label} ({button.shortcut})")
    return clicked
