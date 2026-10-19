import itertools

import pytest

from chronovox import shading
from chronovox.shading import face_brightness

NORMALS = {
    "+X": (1, 0, 0), "-X": (-1, 0, 0),
    "+Y": (0, 1, 0), "-Y": (0, -1, 0),
    "+Z": (0, 0, 1), "-Z": (0, 0, -1),
}


def test_all_six_faces_have_different_brightness():
    values = [face_brightness(n) for n in NORMALS.values()]
    assert len(set(round(v, 6) for v in values)) == 6


def test_faces_that_can_meet_at_a_corner_differ_by_a_visible_amount():
    axis_of = lambda name: name[1]
    for a, b in itertools.combinations(NORMALS, 2):
        if axis_of(a) != axis_of(b):                    # opposite faces of one axis are never seen together
            gap = abs(face_brightness(NORMALS[a]) - face_brightness(NORMALS[b]))
            assert gap >= shading.MIN_GAP - 1e-9, (a, b, gap)


def test_every_corner_shows_three_different_shades():
    for sx, sy, sz in itertools.product("+-", repeat=3):
        faces = (face_brightness(NORMALS[f"{sx}X"]), face_brightness(NORMALS[f"{sy}Y"]),
                 face_brightness(NORMALS[f"{sz}Z"]))
        assert len(set(round(f, 6) for f in faces)) == 3


def test_the_top_is_brightest_and_the_bottom_darkest_and_nothing_goes_black_or_blows_out():
    values = {name: face_brightness(n) for name, n in NORMALS.items()}
    assert max(values, key=values.get) == "+Y" and min(values, key=values.get) == "-Y"
    assert 0.3 < min(values.values()) and max(values.values()) < 0.95


def test_the_old_light_direction_is_kept_the_lit_sides_stay_lighter():
    assert face_brightness(NORMALS["+X"]) > face_brightness(NORMALS["-X"])
    assert face_brightness(NORMALS["-Z"]) > face_brightness(NORMALS["+Z"])


def test_shading_changes_smoothly_for_tilted_faces():
    tilted = (0.0, 0.7071, 0.7071)                      # a face between +Y and +Z
    assert face_brightness(NORMALS["+Z"]) < face_brightness(tilted) < face_brightness(NORMALS["+Y"])
