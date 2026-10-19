"""Matrix naming conventions used by the game (Aetheria). No GL, UI or file imports.

The suffix of a matrix name controls how the game treats it. The editor does not
act on these; it only describes them so a name can be checked at a glance.
"""
from __future__ import annotations


def name_behaviour(name: str) -> str:
    """What the game does with a matrix of this name."""
    if name.endswith("+"):
        return "invisible"
    if name.endswith("-box"):
        return "box collision"
    if name.endswith("-"):
        return "visible, no hitbox"
    return "visible, mesh hitbox"


GAME_SUFFIXES = ("-box", "+", "-")


def split_suffix(name: str) -> tuple[str, str]:
    """Split a name into (stem, game suffix). The suffix is "" if the name ends in none of them."""
    for suffix in GAME_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    return name, ""


def variant_name(name: str, taken) -> str:
    """A name for a copy of ``name`` that is not in ``taken``: ``Arm`` -> ``Arm2``, ``Arm2`` -> ``Arm3``.

    The number goes before the game suffix, so the copy behaves like the original in the game:
    ``Torso-box`` -> ``Torso2-box``, ``Shield+`` -> ``Shield2+``. No spaces are added.
    """
    stem, suffix = split_suffix(name)
    stem = stem.rstrip("0123456789")
    number = 2
    while f"{stem}{number}{suffix}" in taken:
        number += 1
    return f"{stem}{number}{suffix}"
