"""Number fields that step with the Up and Down arrow keys. Dear ImGui drawing only: no model or file imports.

ImGui's own number inputs ignore the arrow keys. These wrap them: while a field is being edited, Up and Down
add or subtract a step (hold to repeat) and the new value is reported like any other change, so it applies
at once. Shift takes the field's other step (a bigger one for whole numbers, a fine one for positions).
After a step the text being edited is reloaded from the new value, so the field shows it at once and the
next press carries on from it.
"""
from __future__ import annotations

from imgui_bundle import imgui


def shift_down() -> bool:
    """Whether a Shift key is held, by whichever way ImGui reports it (the modifier flag or the keys themselves)."""
    return (imgui.get_io().key_shift or imgui.is_key_down(imgui.Key.left_shift)
            or imgui.is_key_down(imgui.Key.right_shift))


def ctrl_down() -> bool:
    return (imgui.get_io().key_ctrl or imgui.is_key_down(imgui.Key.left_ctrl)
            or imgui.is_key_down(imgui.Key.right_ctrl))


class NumberFields:
    def number(self, key: str, label: str, value, *, step: float = 1.0, shift_step: float | None = None,
               is_float: bool = False, fmt: str = "%.10g", flags: int = 0, width: float | None = None,
               buttons: bool = False) -> tuple[bool, float]:
        """One number field. Returns (changed, value), where a step by the arrow keys counts as a change.

        ``key`` names the field and must be unique among the fields on screen. With ``buttons`` the field
        also has ImGui's - and + buttons.
        """
        if width is not None:
            imgui.set_next_item_width(width)
        if is_float:
            changed, result = imgui.input_float(label, float(value), step if buttons else 0.0, 0.0, fmt, flags)
        else:
            changed, result = imgui.input_int(label, int(value), int(step) if buttons else 0, 0, flags)
        if not changed and imgui.is_item_active():
            up, down = imgui.is_key_pressed(imgui.Key.up_arrow), imgui.is_key_pressed(imgui.Key.down_arrow)
            if up != down:
                big = shift_down()
                amount = (shift_step if shift_step is not None else step * 10) if big else step
                result = value + amount if up else value - amount
                result = float(result) if is_float else int(round(result))
                changed = True
                # While a field is being edited ImGui keeps its own text and ignores the value, so ask it to
                # reload the text from the value on the next frame, when the caller has stored the new one.
                state = imgui.internal.get_input_text_state(imgui.get_item_id())
                if state is not None:
                    state.reload_user_buf_and_select_all()
        return changed, result

    def row(self, key: str, label: str, values, *, step: float = 1.0, shift_step: float | None = None,
            is_float: bool = False, fmt: str = "%.10g", flags: int = 0) -> tuple[bool, list]:
        """Several number fields side by side with one label, like ImGui's input_float3 / input_int3.

        Returns (changed, values): ``values`` is the whole list, with the changed field replaced. If more
        than one field reports a change in the same frame, the first one wins.
        """
        count = len(values)
        spacing = imgui.get_style().item_inner_spacing.x
        label_width = imgui.calc_text_size(label.split("##")[0]).x + spacing
        width = max((imgui.get_content_region_avail().x - label_width - spacing * (count - 1)) / count, 20.0)
        out, changed = list(values), False
        for i, value in enumerate(values):
            if i:
                imgui.same_line(0.0, spacing)
            imgui.push_id(i)
            did, new = self.number(f"{key}.{i}", f"##{label}", value, step=step, shift_step=shift_step,
                                   is_float=is_float, fmt=fmt, flags=flags, width=width)
            imgui.pop_id()
            if did and not changed:
                out[i], changed = new, True
        imgui.same_line(0.0, spacing)
        imgui.text(label.split("##")[0])
        return changed, out
