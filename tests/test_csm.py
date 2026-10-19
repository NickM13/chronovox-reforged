import struct

import numpy as np
import pytest

from chronovox import csm, cvox
from chronovox.csm import CsmError
from chronovox.model import POSITION_SCALE


# --- a tiny .csm v4 writer, used only to build test files -------------------------

def pstr(text: str) -> bytes:
    raw = text.encode("ascii")
    return struct.pack("<i", len(raw)) + raw


def runs(colors) -> bytes:
    """RLE-encode a sequence of (r, g, b, a) in file order, splitting runs at 255."""
    out, i = bytearray(), 0
    colors = list(colors)
    while i < len(colors):
        j = i
        while j + 1 < len(colors) and colors[j + 1] == colors[i] and j + 1 - i < 255:
            j += 1
        out += bytes([j - i + 1, *colors[i]])
        i = j + 1
    return bytes(out)


def matrix_block(name, position, size, colors, parent="") -> bytes:
    return pstr(name) + pstr(parent) + struct.pack("<3h", *position) + struct.pack("<3h", *size) + runs(colors)


def csm_file(blocks, version=4) -> bytes:
    return b".CSM" + struct.pack("<3i", version, 0, len(blocks)) + b"".join(blocks)


def file_order(size, color_of):
    """Colors for every voxel in .csm order: z fastest, then y, then x outermost."""
    sx, sy, sz = size
    return [color_of(x, y, z) for x in range(sx) for y in range(sy) for z in range(sz)]


RED, GREEN, BLUE, CLEAR = (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (0, 0, 0, 0)


# --- layout, mirroring and positions -------------------------------------------------

def test_voxels_are_transposed_from_z_fastest_and_mirrored_in_x():
    size = (2, 3, 4)
    # every voxel gets its own color, encoding its file coordinates
    color_of = lambda x, y, z: (x + 1, y + 1, z + 1, 255)
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), size, file_order(size, color_of))]))
    m = model.matrices[0]
    assert m.voxels.shape == (4, 3, 2) and m.size == size
    for x in range(2):
        for y in range(3):
            for z in range(4):
                index = m.voxels[z, y, 1 - x]                    # mirrored in x
                assert tuple(model.palette[index - 1][:3]) == (x + 1, y + 1, z + 1)


def test_position_is_mirrored_in_x_and_converted_to_128ths():
    # -35 tenths = -3.5 voxels; size 4: mirrored = 3.5 - 4 = -0.5 voxel = -64/128
    model = csm.loads(csm_file([matrix_block("m", (-35, 25, -5), (4, 2, 2), [RED] * 16)]))
    assert model.matrices[0].position == (-64, round(2.5 * 128), round(-0.5 * 128))


def test_position_zero_with_size_16_puts_the_corner_at_minus_16_voxels():
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (16, 1, 1), [RED] * 16)]))
    assert model.matrices[0].position[0] == -16 * POSITION_SCALE


def test_position_rounding_stays_within_a_quarter_of_a_step():
    for tenths in range(-30, 31):
        model = csm.loads(csm_file([matrix_block("m", (tenths, tenths, 0), (1, 1, 1), [RED])]))
        pos = model.matrices[0].position
        assert abs(pos[1] - tenths / 10 * POSITION_SCALE) <= 0.5
        assert abs(pos[0] - (-tenths / 10 - 1) * POSITION_SCALE) <= 0.5


def test_rotation_is_identity_and_the_pivot_is_the_matrix_center():
    m = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (4, 6, 2), [RED] * 48)])).matrices[0]
    assert m.rotation == (0.0, 0.0, 0.0, 1.0) and m.pivot == (4 * 64, 6 * 64, 2 * 64)


def test_names_are_kept_and_the_parent_is_discarded():
    model = csm.loads(csm_file([matrix_block("LeftArm", (0, 0, 0), (1, 1, 1), [RED], parent="Torso"),
                                matrix_block("Head+", (0, 0, 0), (1, 1, 1), [RED], parent="")]))
    assert [m.name for m in model.matrices] == ["LeftArm", "Head+"]


# --- colors --------------------------------------------------------------------------

def test_one_palette_is_shared_across_matrices_in_order_of_first_appearance():
    a = matrix_block("a", (0, 0, 0), (2, 1, 1), [BLUE, RED])
    b = matrix_block("b", (0, 0, 0), (2, 1, 1), [RED, GREEN])
    model = csm.loads(csm_file([a, b]))
    assert [tuple(c) for c in model.palette] == [BLUE, RED, GREEN]
    # a file order is x-outermost then mirrored, so x=0 BLUE lands at x=1
    assert model.matrices[0].voxels[0, 0].tolist() == [2, 1]
    assert model.matrices[1].voxels[0, 0].tolist() == [3, 2]


def test_alpha_zero_is_empty_whatever_its_rgb_and_never_enters_the_palette():
    ghost = (12, 34, 56, 0)
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (3, 1, 1), [ghost, RED, CLEAR])]))
    assert model.matrices[0].voxels[0, 0].tolist() == [0, 1, 0]
    assert [tuple(c) for c in model.palette] == [RED]


def test_any_non_zero_alpha_just_means_solid_and_comes_out_fully_opaque():
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (3, 1, 1), [(10, 20, 30, 1), (40, 50, 60, 128), (70, 80, 90, 255)])]))
    assert [tuple(int(v) for v in c) for c in model.palette] == [(10, 20, 30, 255), (40, 50, 60, 255), (70, 80, 90, 255)]
    assert (model.matrices[0].voxels > 0).all()


def test_colors_that_differ_only_in_alpha_are_one_color():
    colors = [(10, 20, 30, 1), (10, 20, 30, 255), (10, 20, 30, 128), (5, 5, 5, 1)]
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (4, 1, 1), colors)]))
    assert len(model.palette) == 2
    assert model.matrices[0].voxels[0, 0].tolist() == [2, 1, 1, 1]          # x is mirrored: the lone second color comes first


def test_the_alpha_that_old_models_use_for_solid_leaves_no_trace_in_the_palette():
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (2, 1, 1), [(200, 60, 60, 1), (60, 200, 60, 1)])]))
    assert (model.palette[:, 3] == 255).all()


def test_an_all_empty_model_has_an_empty_palette():
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (2, 2, 2), [CLEAR] * 8)]))
    assert len(model.palette) == 0 and not model.matrices[0].voxels.any()


def test_too_many_colors_is_an_error_not_a_quantize():
    colors = [(i & 255, (i >> 8) & 255, 7, 255) for i in range(65536)]
    with pytest.raises(CsmError, match="distinct colors"):
        csm.loads(csm_file([matrix_block("m", (0, 0, 0), (256, 256, 1), colors)]))


def test_the_maximum_number_of_colors_is_accepted():
    colors = [(i & 255, (i >> 8) & 255, 7, 255) for i in range(65535)] + [CLEAR]
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (256, 256, 1), colors)]))
    assert len(model.palette) == 65535


# --- run-length handling ---------------------------------------------------------------

def test_runs_longer_than_255_are_stored_as_several_runs():
    model = csm.loads(csm_file([matrix_block("m", (0, 0, 0), (20, 20, 20), [RED] * 8000)]))
    assert (model.matrices[0].voxels == 1).all()


def test_matrices_read_one_after_another_without_bleeding_into_each_other():
    blocks = [matrix_block(f"m{i}", (i, 0, 0), (3, 3, 3), [(i + 1, 0, 0, 255)] * 27) for i in range(5)]
    model = csm.loads(csm_file(blocks))
    assert [int(m.voxels.max()) for m in model.matrices] == [1, 2, 3, 4, 5]


# --- conversion ----------------------------------------------------------------------------

def test_an_imported_model_round_trips_through_cvox():
    size = (5, 4, 3)
    color_of = lambda x, y, z: [CLEAR, RED, GREEN, BLUE][(x + 2 * y + 3 * z) % 4]
    model = csm.loads(csm_file([matrix_block("A", (7, -3, 12), size, file_order(size, color_of)),
                                matrix_block("B-box", (0, 0, 0), (2, 2, 2), [BLUE] * 8)]))
    back = cvox.loads(cvox.dumps(model, compress=True))
    assert np.array_equal(back.palette, model.palette)
    for a, b in zip(model.matrices, back.matrices):
        assert a.name == b.name and a.position == b.position and np.array_equal(a.voxels, b.voxels)


# --- errors ----------------------------------------------------------------------------------

GOOD = csm_file([matrix_block("m", (0, 0, 0), (2, 2, 2), [RED] * 8)])


def test_bad_format_type_and_unsupported_versions_are_rejected():
    with pytest.raises(CsmError, match="not a .csm"):
        csm.loads(b"XXXX" + GOOD[4:])
    for version in (3, 5, 0):
        with pytest.raises(CsmError, match=f"version {version}"):
            csm.loads(csm_file([], version=version))


def test_every_truncation_is_an_error_not_a_crash():
    for cut in range(len(GOOD)):
        with pytest.raises(CsmError):
            csm.loads(GOOD[:cut])


def test_non_ascii_or_oversized_names_are_rejected():
    bad = struct.pack("<i", 2) + "é".encode("utf-8")
    block = bad + pstr("") + struct.pack("<3h", 0, 0, 0) + struct.pack("<3h", 1, 1, 1) + runs([RED])
    with pytest.raises(CsmError, match="ASCII"):
        csm.loads(csm_file([block]))
    with pytest.raises(CsmError, match="longer than"):
        csm.loads(csm_file([matrix_block("x" * 256, (0, 0, 0), (1, 1, 1), [RED])]))
    block = struct.pack("<i", -5) + pstr("") + struct.pack("<3h", 0, 0, 0) + struct.pack("<3h", 1, 1, 1) + runs([RED])
    with pytest.raises(CsmError, match="negative"):
        csm.loads(csm_file([block]))


def test_bad_sizes_are_rejected():
    for size in ((0, 1, 1), (1, -2, 1)):
        block = pstr("m") + pstr("") + struct.pack("<3h", 0, 0, 0) + struct.pack("<3h", *size)
        with pytest.raises(CsmError, match="size"):
            csm.loads(csm_file([block]))


def test_runs_that_overshoot_or_stop_short_are_rejected():
    too_much = pstr("m") + pstr("") + struct.pack("<3h", 0, 0, 0) + struct.pack("<3h", 2, 1, 1) + bytes([3, *RED])
    with pytest.raises(CsmError, match="overshoot"):
        csm.loads(csm_file([too_much]))
    too_little = pstr("m") + pstr("") + struct.pack("<3h", 0, 0, 0) + struct.pack("<3h", 2, 2, 2) + bytes([3, *RED])
    with pytest.raises(CsmError, match="ends early"):
        csm.loads(csm_file([too_little]))
    zero = pstr("m") + pstr("") + struct.pack("<3h", 0, 0, 0) + struct.pack("<3h", 2, 1, 1) + bytes([0, *RED, 2, *RED])
    with pytest.raises(CsmError, match="zero"):
        csm.loads(csm_file([zero]))


def test_a_negative_or_huge_matrix_count_is_rejected():
    for count in (-1, 70000):
        with pytest.raises(CsmError, match="matrix count"):
            csm.loads(b".CSM" + struct.pack("<3i", 4, 0, count))


def test_an_empty_model_is_fine():
    model = csm.loads(csm_file([]))
    assert model.matrices == [] and len(model.palette) == 0


def test_load_reads_a_file(tmp_path):
    path = tmp_path / "x.csm"
    path.write_bytes(GOOD)
    assert csm.load(path).matrices[0].name == "m"
