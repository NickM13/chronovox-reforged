import struct
import zlib

import numpy as np
import pytest

from chronovox import icon, png


def synthetic(width=7, height=5):
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (height, width, 4), dtype=np.uint8)


def encode_with_filter(rgba, kind):
    """A PNG whose every row uses one filter type, written the slow plain way, to test the reader against."""
    height, width, channels = rgba.shape
    stride = width * channels
    rows, previous = [], [0] * stride
    for y in range(height):
        cur = rgba[y].reshape(-1).tolist()
        out = []
        for i, value in enumerate(cur):
            left = cur[i - channels] if i >= channels else 0
            up = previous[i]
            upper_left = previous[i - channels] if i >= channels else 0
            if kind == 0:
                predictor = 0
            elif kind == 1:
                predictor = left
            elif kind == 2:
                predictor = up
            elif kind == 3:
                predictor = (left + up) // 2
            else:
                estimate = left + up - upper_left
                da, db, dc = abs(estimate - left), abs(estimate - up), abs(estimate - upper_left)
                predictor = left if da <= db and da <= dc else (up if db <= dc else upper_left)
            out.append((value - predictor) & 255)
        rows.append(bytes([kind]) + bytes(out))
        previous = cur

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (png.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def test_write_then_read_gives_back_the_same_pixels():
    image = synthetic()
    assert np.array_equal(png.decode_png(png.encode_png(image)), image)


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4])
def test_the_reader_undoes_every_row_filter(kind):
    image = synthetic()
    assert np.array_equal(png.decode_png(encode_with_filter(image, kind)), image)


def test_rgb_png_files_get_full_alpha_and_other_formats_are_refused():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rows = b"".join(b"\0" + rgb[y].tobytes() for y in range(2))

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    data = (png.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    decoded = png.decode_png(data)
    assert np.array_equal(decoded[..., :3], rgb) and (decoded[..., 3] == 255).all()
    with pytest.raises(png.PngError):
        png.decode_png(b"not a png")
    with pytest.raises(png.PngError, match="8 bit"):
        png.decode_png(png.SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 16, 6, 0, 0, 0))
                       + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b""))


def test_resizing_keeps_a_flat_color_and_the_size():
    flat = np.tile(np.array([40, 120, 200, 255], np.uint8), (30, 30, 1))
    small = png.resize(flat, 7)
    assert small.shape == (7, 7, 4) and (small == flat[0, 0]).all()


def test_resizing_averages_and_does_not_darken_transparent_edges():
    image = np.zeros((4, 4, 4), np.uint8)
    image[:, :2] = (255, 0, 0, 255)                          # left half opaque red, right half fully transparent
    half = png.resize(image, 2)
    assert (half[:, 0] == (255, 0, 0, 255)).all() and (half[:, 1, 3] == 0).all()
    one = png.resize(image, 1)
    assert one[0, 0, 3] in (127, 128) and one[0, 0, 0] == 255      # half as opaque, but still pure red


def test_the_shipped_icon_loads_and_has_a_transparent_corner_and_an_opaque_middle():
    picture = icon.master_picture()
    assert picture.shape == (256, 256, 4)
    assert picture[0, 0, 3] == 0 and picture[:, :, 3].max() == 255


def test_window_icons_come_in_every_requested_size_as_rows_of_rgba():
    icons = icon.window_icons((16, 32))
    assert [(w, h) for w, h, _ in icons] == [(16, 16), (32, 32)]
    w, h, pixels = icons[0]
    assert len(pixels) == h and len(pixels[0]) == w and len(pixels[0][0]) == 4


def test_the_ico_file_lists_every_size_and_each_entry_is_a_png_of_that_size():
    data = icon.make_ico((16, 32, 256))
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind, count) == (0, 1, 3)
    for n, size in enumerate((16, 32, 256)):
        width, height, _, _, planes, bits, length, offset = struct.unpack("<BBBBHHII", data[6 + 16 * n:22 + 16 * n])
        assert (width, height) == (size % 256, size % 256) and planes == 1 and bits == 32
        assert png.decode_png(data[offset:offset + length]).shape[:2] == (size, size)
