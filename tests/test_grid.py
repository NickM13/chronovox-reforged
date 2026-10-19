import numpy as np

from chronovox import grid


def rnd(color):
    return tuple(round(float(v), 2) for v in color)


def lines_of(positions):
    """Split a line list into (the lines along Z, the lines along X), each as (n, 2, 3) endpoint pairs."""
    pairs = positions.reshape(-1, 2, 3)
    along_z = pairs[pairs[:, 0, 0] == pairs[:, 1, 0]]
    along_x = pairs[pairs[:, 0, 2] == pairs[:, 1, 2]]
    return along_z, along_x


def test_the_grid_has_a_fixed_size_and_lies_on_the_ground_plane():
    pos, col = grid.grid_lines(0.0, 0.0)
    assert len(pos) == len(col) == grid.VERTEX_COUNT
    assert (pos[:, 1] == 0).all()


def test_there_is_one_line_per_voxel_in_each_direction_out_to_a_radius_of_128():
    assert grid.RADIUS == 128 and grid.MAJOR_EVERY == 8
    along_z, along_x = lines_of(grid.grid_lines(0.0, 0.0)[0])
    assert along_z[:, 0, 0].tolist() == list(range(-128, 129))      # a line at every whole voxel
    assert along_x[:, 0, 2].tolist() == list(range(-128, 129))
    assert along_z[:, :, 2].min() == -128 and along_z[:, :, 2].max() == 128
    assert along_x[:, :, 0].min() == -128 and along_x[:, :, 0].max() == 128


def test_the_grid_follows_the_focus_in_whole_voxels_and_keeps_the_same_size():
    pos, _ = grid.grid_lines(40.3, -17.8)                            # focus snaps to (40, -18)
    along_z, along_x = lines_of(pos)
    assert along_z[:, 0, 0].min() == 40 - 128 and along_z[:, 0, 0].max() == 40 + 128
    assert along_x[:, 0, 2].min() == -18 - 128 and along_x[:, 0, 2].max() == -18 + 128
    assert len(pos) == grid.VERTEX_COUNT
    assert (along_z[:, 0, 0] % 1 == 0).all() and (along_x[:, 0, 2] % 1 == 0).all()


def test_spacing_never_depends_on_anything_but_whole_voxels():
    for focus in [(0.0, 0.0), (0.4, 0.6), (1000.5, -3000.2), (-0.5, 0.5)]:
        along_z, _ = lines_of(grid.grid_lines(*focus)[0])
        assert (np.diff(along_z[:, 0, 0]) == 1).all()


def test_axes_are_colored_when_the_origin_is_in_range_and_absent_otherwise():
    _, col = grid.grid_lines(2.0, 3.0)
    colors = {rnd(c) for c in col}
    assert rnd(grid.X_AXIS_COLOR) in colors and rnd(grid.Z_AXIS_COLOR) in colors
    _, far = grid.grid_lines(10_000.0, 10_000.0)
    far_colors = {rnd(c) for c in far}
    assert rnd(grid.X_AXIS_COLOR) not in far_colors and rnd(grid.Z_AXIS_COLOR) not in far_colors


def test_the_axes_are_the_lines_through_zero_and_match_the_gizmo_colors():
    pos, col = grid.grid_lines(0.0, 0.0)
    pairs, colors = pos.reshape(-1, 2, 3), col.reshape(-1, 2, 4)
    for pair, color in zip(pairs, colors):
        if pair[0, 0] == pair[1, 0] == 0:                  # the line along Z, at x = 0
            assert rnd(color[0]) == rnd(grid.Z_AXIS_COLOR)
        if pair[0, 2] == pair[1, 2] == 0:                  # the line along X, at z = 0
            assert rnd(color[0]) == rnd(grid.X_AXIS_COLOR)


def test_every_eighth_line_is_a_major_line_and_the_rest_are_light():
    for focus in [(0.0, 0.0), (100.0, 100.0), (-37.0, 61.0)]:
        pos, col = grid.grid_lines(*focus)
        for p, c in zip(pos.reshape(-1, 2, 3), col.reshape(-1, 2, 4)[:, 0]):
            k = p[0, 0] if p[0, 0] == p[1, 0] else p[0, 2]
            if k == 0:
                continue                                        # the colored axes
            expected = grid.MAJOR_COLOR if k % grid.MAJOR_EVERY == 0 else grid.MINOR_COLOR
            assert rnd(c) == rnd(expected)


def test_both_ends_of_a_line_share_a_color():
    _, col = grid.grid_lines(5.0, 5.0)
    pairs = col.reshape(-1, 2, 4)
    assert (pairs[:, 0] == pairs[:, 1]).all()
