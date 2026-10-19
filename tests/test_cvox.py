import struct

import numpy as np
import pytest

from chronovox import cvox
from chronovox.cvox import CvoxError
from chronovox.model import Matrix, Model


def make_model(colors: int, seed: int = 0) -> Model:
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 4), dtype=np.uint8)
    a = rng.integers(0, colors + 1, size=(4, 5, 6)).astype(np.uint16)
    b = np.zeros((3, 3, 3), np.uint16)
    b[1, 1, 1] = colors
    return Model(palette, [
        Matrix("noise", a, position=(-128, 64, 1_000_000), rotation=(0.0, 0.5, 0.0, 0.5)),
        Matrix("single", b, position=(0, 0, 0), pivot=(5, -6, 7)),
    ])


def assert_same(a: Model, b: Model):
    assert np.array_equal(a.palette, b.palette)
    assert len(a.matrices) == len(b.matrices)
    for x, y in zip(a.matrices, b.matrices):
        assert x.name == y.name
        assert tuple(x.position) == tuple(y.position)
        assert tuple(x.pivot) == tuple(y.pivot)
        assert np.allclose(x.rotation, y.rotation)
        assert np.array_equal(x.voxels, y.voxels)


@pytest.mark.parametrize("colors", [0, 1, 255, 256, 1000, 65535])
@pytest.mark.parametrize("compress", [False, True])
def test_round_trip(colors, compress):
    if colors:
        model = make_model(colors)
    else:
        model = Model(np.zeros((0, 4), np.uint8),
                      [Matrix("empty", np.zeros((2, 2, 2), np.uint16))])
    assert_same(model, cvox.loads(cvox.dumps(model, compress)))


def test_round_trip_file(tmp_path):
    model = make_model(10)
    cvox.save(model, tmp_path / "m.cvox", compress=True)
    assert_same(model, cvox.load(tmp_path / "m.cvox"))


def test_index_width():
    vox = np.full((1, 1, 1), 7, np.uint16)
    assert len(cvox.encode_voxels(vox, 1)) == 2
    assert len(cvox.encode_voxels(vox, 2)) == 3


def test_long_run_is_single_varint_run():
    vox = np.zeros((10, 100, 200), np.uint16)  # 200,000 voxels
    assert cvox.encode_voxels(vox, 1) == bytes([0xC0, 0x9A, 0x0C, 0])


def test_runs_cross_row_boundaries_and_merge():
    assert cvox.encode_voxels(np.ones((2, 2, 2), np.uint16), 1) == bytes([8, 1])


def test_x_fastest_order():
    vox = np.zeros((1, 1, 3), np.uint16)  # shape (z, y, x)
    vox[0, 0, 1] = 1
    assert cvox.encode_voxels(vox, 1) == bytes([1, 0, 1, 1, 1, 0])


def test_layout_of_header_and_block():
    model = Model(np.array([[1, 2, 3, 4]], np.uint8),
                  [Matrix("a", np.ones((1, 1, 1), np.uint16), position=(128, 0, 0),
                          pivot=(64, 64, 64))])
    blob = cvox.dumps(model)
    assert blob[:4] == b"CVOX"
    assert struct.unpack_from("<HHH", blob, 4) == (1, 0, 1)
    assert blob[10:14] == bytes([1, 2, 3, 4])
    assert struct.unpack_from("<H", blob, 14) == (1,)
    (block_len,) = struct.unpack_from("<I", blob, 16)
    assert len(blob) == 20 + block_len
    assert block_len == 1 + 2 + 12 + 6 + 12 + 16 + 4 + 2       # the first byte is the resolution


def test_default_pivot_is_center():
    assert Matrix("m", np.zeros((2, 4, 6), np.uint16)).pivot == (6 * 64, 4 * 64, 2 * 64)


def test_compression_shrinks_cross_row_patterns():
    rng = np.random.default_rng(1)
    row = rng.integers(0, 2, size=(1, 1, 64)).astype(np.uint16)
    vox = np.tile(row, (40, 40, 1))  # identical rows: RLE can't merge them, zlib can
    model = Model(np.array([[1, 1, 1, 255]], np.uint8), [Matrix("m", vox)])
    assert len(cvox.dumps(model, True)) < len(cvox.dumps(model, False))


# --- writer rejections --------------------------------------------------------

def one(**kw):
    kw.setdefault("voxels", np.ones((1, 1, 1), np.uint16))
    return Model(np.array([[0, 0, 0, 255]], np.uint8), [Matrix("m", **kw)])


def test_writer_rejects_non_ascii_name():
    m = one()
    m.matrices[0].name = "café"
    with pytest.raises(CvoxError):
        cvox.dumps(m)


def test_writer_name_length_limit():
    m = one()
    m.matrices[0].name = "x" * 256
    with pytest.raises(CvoxError):
        cvox.dumps(m)
    m.matrices[0].name = "x" * 255
    cvox.dumps(m)


def test_writer_rejects_index_beyond_palette():
    with pytest.raises(CvoxError):
        cvox.dumps(one(voxels=np.full((1, 1, 1), 2, np.uint16)))


def test_writer_rejects_position_out_of_range():
    with pytest.raises(CvoxError):
        cvox.dumps(one(position=(2**31, 0, 0)))


# --- reader validation --------------------------------------------------------

GOOD = cvox.dumps(make_model(3))


def first_block_offset(blob: bytes) -> int:
    pal = struct.unpack_from("<H", blob, 8)[0]
    return 10 + 4 * pal + 2


def test_reader_rejects_bad_magic():
    with pytest.raises(CvoxError):
        cvox.loads(b"XXXX" + GOOD[4:])


def test_reader_rejects_future_version():
    with pytest.raises(CvoxError):
        cvox.loads(GOOD[:4] + struct.pack("<H", 2) + GOOD[6:])


def test_reader_rejects_unknown_flags():
    with pytest.raises(CvoxError):
        cvox.loads(GOOD[:6] + struct.pack("<H", 2) + GOOD[8:])


def test_reader_rejects_truncation_everywhere():
    for cut in range(len(GOOD)):
        with pytest.raises(CvoxError):
            cvox.loads(GOOD[:cut])


def test_reader_rejects_index_over_palette():
    model = Model(np.array([[1, 1, 1, 1]], np.uint8),
                  [Matrix("m", np.ones((1, 1, 2), np.uint16))])
    blob = bytearray(cvox.dumps(model))
    blob[-1] = 2  # last run's index -> 2 > palette count 1
    with pytest.raises(CvoxError):
        cvox.loads(bytes(blob))


def test_reader_rejects_wrong_run_total():
    model = Model(np.array([[1, 1, 1, 1]], np.uint8),
                  [Matrix("m", np.ones((1, 1, 2), np.uint16))])
    blob = bytearray(cvox.dumps(model))
    blob[-2] = 1  # run count 2 -> 1; sums to 1, expected 2
    with pytest.raises(CvoxError):
        cvox.loads(bytes(blob))


def test_reader_rejects_block_length_past_eof():
    blob = bytearray(GOOD)
    struct.pack_into("<I", blob, first_block_offset(blob), 0xFFFFFFF0)
    with pytest.raises(CvoxError):
        cvox.loads(bytes(blob))


def test_reader_rejects_bad_zlib():
    blob = bytearray(cvox.dumps(make_model(3), compress=True))
    blob[-3:] = b"\xff\xff\xff"
    with pytest.raises(CvoxError):
        cvox.loads(bytes(blob))


def test_reader_skips_trailing_bytes_in_block():
    model = make_model(3)
    blob = cvox.dumps(model)
    off = first_block_offset(blob)
    (length,) = struct.unpack_from("<I", blob, off)
    padded = (blob[:off] + struct.pack("<I", length + 3) + blob[off + 4:off + 4 + length]
              + b"\0\0\0" + blob[off + 4 + length:])
    assert_same(model, cvox.loads(padded))


# --- resolution (the first byte of each matrix block) ------------------------------------------

def resolution_model():
    return Model(np.array([[1, 2, 3, 255]], np.uint8),
                 [Matrix("leaves", np.ones((2, 2, 2), np.uint16), resolution=1),
                  Matrix("twig", np.ones((1, 1, 1), np.uint16), resolution=4),
                  Matrix("fine", np.ones((1, 1, 1), np.uint16), resolution=8)])


@pytest.mark.parametrize("compress", [False, True])
def test_resolutions_round_trip(compress):
    back = cvox.loads(cvox.dumps(resolution_model(), compress))
    assert [m.resolution for m in back.matrices] == [1, 4, 8]
    assert [m.cell_size for m in back.matrices] == [1.0, 0.125, 1 / 128]


def test_the_resolution_is_the_first_byte_after_the_block_length():
    model = Model(np.array([[1, 2, 3, 255]], np.uint8), [Matrix("m", np.ones((1, 1, 2), np.uint16), resolution=5)])
    blob = cvox.dumps(model)
    off = first_block_offset(blob)
    assert blob[off + 4] == 5 and blob[off + 5] == 1 and blob[off + 6:off + 7] == b"m"     # then the name: length, text


def test_reader_rejects_a_resolution_outside_the_range():
    blob = bytearray(cvox.dumps(resolution_model()))
    off = first_block_offset(blob)
    for bad in (0, 9, 255):
        blob[off + 4] = bad
        with pytest.raises(CvoxError, match="resolution"):
            cvox.loads(bytes(blob))


def test_writer_rejects_a_resolution_outside_the_range():
    for bad in (0, 9, -1):
        model = Model(np.array([[1, 2, 3, 255]], np.uint8), [Matrix("m", np.ones((1, 1, 1), np.uint16), resolution=bad)])
        with pytest.raises(CvoxError):
            cvox.dumps(model)


def test_a_truncated_block_cannot_hide_a_missing_resolution():
    blob = cvox.dumps(resolution_model())
    off = first_block_offset(blob)
    broken = blob[:off] + struct.pack("<I", 0) + blob[off + 4:]       # a block of length 0 has no resolution byte
    with pytest.raises(CvoxError):
        cvox.loads(broken)
