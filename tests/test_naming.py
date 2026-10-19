from chronovox.naming import name_behaviour


def test_plus_means_invisible():
    assert name_behaviour("Shield+") == "invisible"


def test_minus_means_visible_without_a_hitbox():
    assert name_behaviour("Wing-") == "visible, no hitbox"


def test_minus_box_means_box_collision():
    assert name_behaviour("Torso-box") == "box collision"


def test_anything_else_is_a_visible_mesh_hitbox():
    for name in ("LeftArm", "Matrix", "box", "a-b", "Plus+Arm", "x-boxy"):
        assert name_behaviour(name) == "visible, mesh hitbox"
