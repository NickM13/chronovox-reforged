import numpy as np

from chronovox.mesh import build_mesh

PALETTE = np.array([[255, 0, 0, 255], [0, 255, 0, 255]], np.uint8)


def test_empty_grid_has_no_faces():
    assert build_mesh(np.zeros((2, 2, 2), np.uint16), PALETTE).face_count == 0


def test_single_voxel_has_six_faces_inside_unit_cube():
    vox = np.zeros((3, 3, 3), np.uint16)
    vox[1, 1, 1] = 1
    mesh = build_mesh(vox, PALETTE)
    assert mesh.face_count == 6
    assert mesh.positions.min() == 1 and mesh.positions.max() == 2
    assert len(mesh.indices) == 36


def test_adjacent_voxels_hide_shared_faces():
    vox = np.zeros((1, 1, 2), np.uint16)
    vox[0, 0, :] = 1
    assert build_mesh(vox, PALETTE).face_count == 10


def test_solid_cube_only_has_surface_faces():
    assert build_mesh(np.ones((3, 3, 3), np.uint16), PALETTE).face_count == 6 * 9


def test_normals_point_away_from_voxel():
    vox = np.zeros((3, 3, 3), np.uint16)
    vox[1, 1, 1] = 1
    mesh = build_mesh(vox, PALETTE)
    centers = mesh.positions.reshape(-1, 4, 3).mean(axis=1)
    normals = mesh.normals.reshape(-1, 4, 3)[:, 0]
    assert np.allclose(centers - 1.5, normals * 0.5)


def test_winding_matches_normal():
    vox = np.ones((1, 1, 1), np.uint16)
    mesh = build_mesh(vox, PALETTE)
    p = mesh.positions.reshape(-1, 4, 3)
    n = mesh.normals.reshape(-1, 4, 3)[:, 0]
    assert np.allclose(np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]), n)


def test_colors_come_from_palette():
    vox = np.zeros((1, 1, 2), np.uint16)
    vox[0, 0] = [1, 2]
    mesh = build_mesh(vox, PALETTE)
    colors = {tuple(c) for c in mesh.colors}
    assert colors == {(255, 0, 0, 255), (0, 255, 0, 255)}


# --- ambient occlusion ----------------------------------------------------------

def face_ao(mesh, normal):
    """AO levels (4 per face) for faces whose normal is the given one."""
    n = mesh.normals.reshape(-1, 4, 3)[:, 0]
    sel = np.all(n == np.array(normal, np.float32), axis=1)
    return mesh.ao_levels().reshape(-1, 4)[sel], mesh.positions.reshape(-1, 4, 3)[sel]


def test_lone_voxel_is_fully_lit():
    vox = np.zeros((3, 3, 3), np.uint16)
    vox[1, 1, 1] = 1
    assert (build_mesh(vox, PALETTE).ao_levels() == 3).all()


def test_wall_next_to_a_top_face_darkens_the_adjacent_corners():
    vox = np.zeros((1, 2, 2), np.uint16)  # shape (z, y, x)
    vox[0, 0, 0] = 1  # A at (0,0,0)
    vox[0, 1, 1] = 1  # B at (1,1,0): beside the cell above A
    levels, pos = face_ao(build_mesh(vox, PALETTE), (0, 1, 0))
    a_top = [(lv, p) for lv, p in zip(levels, pos) if p[:, 1].min() == 1 and p[:, 0].max() == 1][0]
    for level, corner in zip(*a_top):
        assert level == (2 if corner[0] == 1 else 3)  # corners touching B's side are darker


def test_two_sides_make_a_fully_occluded_corner():
    vox = np.zeros((2, 2, 2), np.uint16)  # shape (z, y, x)
    vox[0, 0, 0] = 1   # A at (0,0,0)
    vox[0, 1, 1] = 1   # (x=1, y=1, z=0): beside the cell above A, toward +x
    vox[1, 1, 0] = 1   # (x=0, y=1, z=1): beside the cell above A, toward +z
    levels, pos = face_ao(build_mesh(vox, PALETTE), (0, 1, 0))
    top = [(lv, p) for lv, p in zip(levels, pos) if p[:, 1].min() == 1 and p[:, 0].max() == 1
           and p[:, 2].max() == 1 and (p[:, 0] <= 1).all() and (p[:, 2] <= 1).all()][0]
    by_corner = {tuple(c[[0, 2]]): int(lv) for lv, c in zip(*top)}
    assert by_corner[(1.0, 1.0)] == 0       # both sides solid
    assert by_corner[(0.0, 0.0)] == 3       # open corner


def test_every_triangle_faces_along_its_normal_even_when_the_diagonal_flips():
    rng = np.random.default_rng(7)
    vox = (rng.random((6, 6, 6)) < 0.4).astype(np.uint16)
    mesh = build_mesh(vox, PALETTE)
    p, n = mesh.positions, mesh.normals
    tri = mesh.indices.reshape(-1, 3)
    geometric = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    assert (np.einsum("ij,ij->i", geometric, n[tri[:, 0]]) > 0).all()
    assert len(tri) == 2 * mesh.face_count


def test_quads_split_along_the_diagonal_with_the_closer_ao():
    rng = np.random.default_rng(11)
    vox = (rng.random((8, 8, 8)) < 0.35).astype(np.uint16)
    mesh = build_mesh(vox, PALETTE)
    ao = mesh.ao_levels().reshape(-1, 4).astype(int)
    patterns = (mesh.indices.reshape(-1, 6) - mesh.indices.reshape(-1, 6)[:, :1] * 0)
    patterns = patterns - (np.arange(mesh.face_count)[:, None] * 4)
    flipped = ao[:, 0] + ao[:, 2] > ao[:, 1] + ao[:, 3]
    assert flipped.any() and (~flipped).any()
    assert (patterns[flipped] == [1, 2, 3, 1, 3, 0]).all()
    assert (patterns[~flipped] == [0, 1, 2, 0, 2, 3]).all()


def test_ao_does_not_change_culling():
    assert build_mesh(np.ones((3, 3, 3), np.uint16), PALETTE).face_count == 54


# --- highlighted voxels (the selection) --------------------------------------------------------------

def test_a_highlight_changes_only_the_colors_of_the_marked_voxels_and_not_the_shape():
    from chronovox.mesh import HIGHLIGHT_RGB, HIGHLIGHT_STRENGTH, build_mesh
    voxels = np.zeros((3, 3, 3), np.uint16)
    voxels[1, 1, 0], voxels[1, 1, 2] = 1, 1                          # two separate voxels of the same color
    palette = np.array([[200, 40, 40, 255]], np.uint8)
    marked = np.zeros(voxels.shape, bool)
    marked[1, 1, 0] = True
    plain, lit = build_mesh(voxels, palette), build_mesh(voxels, palette, marked)
    assert np.array_equal(plain.positions, lit.positions) and np.array_equal(plain.indices, lit.indices)
    assert np.array_equal(plain.ao, lit.ao) and np.array_equal(plain.normals, lit.normals)
    changed = (plain.colors != lit.colors).any(axis=1)
    near_x0 = plain.positions[:, 0] < 1.5                              # the faces of the voxel at x = 0
    assert changed.sum() == 24 and (changed == near_x0).all()          # all 6 faces (4 vertices each) of the marked one
    expected = np.rint(np.array([200, 40, 40]) * (1 - HIGHLIGHT_STRENGTH) + HIGHLIGHT_RGB * HIGHLIGHT_STRENGTH)
    assert (lit.colors[changed][:, :3] == expected).all() and (lit.colors[changed][:, 3] == 255).all()


def test_the_highlighted_color_is_a_milder_version_of_the_original_than_the_highlight_itself():
    from chronovox.mesh import HIGHLIGHT_RGB, highlighted
    original = np.array([[10, 200, 90, 255], [255, 255, 255, 255]], np.uint8)
    lit = highlighted(original)
    for before, after in zip(original, lit):
        for channel in range(3):
            low, high = sorted((int(before[channel]), int(HIGHLIGHT_RGB[channel])))
            assert low <= after[channel] <= high                          # between the original and the highlight color
        assert abs(int(after[1]) - int(before[1])) < abs(int(HIGHLIGHT_RGB[1]) - int(before[1]))   # and nearer the original
    assert not np.array_equal(lit, original)


def test_a_highlight_that_marks_nothing_visible_changes_nothing():
    from chronovox.mesh import build_mesh
    voxels = np.zeros((2, 2, 2), np.uint16)
    voxels[0, 0, 0] = 1
    palette = np.array([[1, 2, 3, 255]], np.uint8)
    nothing = np.zeros(voxels.shape, bool)
    assert np.array_equal(build_mesh(voxels, palette).colors, build_mesh(voxels, palette, nothing).colors)
