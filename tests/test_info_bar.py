import numpy as np

from chronovox.info_bar import fit_text, matrix_summary, number
from chronovox.model import POSITION_SCALE, Matrix


def test_number_has_no_trailing_zeros():
    assert number(2.5) == "2.5" and number(-3.0) == "-3" and number(0.0) == "0" and number(1 / 128) == "0.0078125"


def test_summary_is_name_size_resolution_and_position():
    m = Matrix("LeftArm-box", np.ones((4, 6, 2), np.uint16), (POSITION_SCALE * 5 // 2, 0, -POSITION_SCALE), resolution=3)
    assert matrix_summary(m) == "LeftArm-box  |  2x6x4  res 3  |  pos 2.5, 0, -1"      # voxels are (z, y, x), size is (x, y, z)


def test_fit_text_leaves_short_text_alone_and_cuts_long_text_with_dots():
    measure = len
    assert fit_text("short", 10, measure) == "short"
    assert fit_text("a rather long message", 10, measure) == "a rathe..."
    assert fit_text("anything", 2, measure) == ""
