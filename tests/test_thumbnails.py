import numpy as np

from chronovox import thumbnails as t


def picture(value):
    return np.full((t.SIZE, t.SIZE, 4), value, np.uint8)


def test_a_stored_preview_comes_back_for_the_same_file_state(tmp_path):
    model = tmp_path / "Dragon.cvox"
    model.write_bytes(b"abc")
    key = t.file_key(model)
    assert t.load(tmp_path / "cache", model, key) is None
    t.store(tmp_path / "cache", model, key, picture(7))
    assert (t.load(tmp_path / "cache", model, key) == 7).all()


def test_a_changed_file_misses_the_cache_and_the_old_preview_is_replaced(tmp_path):
    model = tmp_path / "Dragon.cvox"
    model.write_bytes(b"abc")
    cache = tmp_path / "cache"
    old = t.file_key(model)
    t.store(cache, model, old, picture(1))
    model.write_bytes(b"abcdef")                           # saved again: another size (and time)
    new = t.file_key(model)
    assert new != old and t.load(cache, model, new) is None
    t.store(cache, model, new, picture(2))
    assert t.load(cache, model, old) is None and (t.load(cache, model, new) == 2).all()
    assert len(list(cache.glob("*.npy"))) == 1


def test_previews_of_different_files_do_not_clash(tmp_path):
    a, b = tmp_path / "a.cvox", tmp_path / "b.cvox"
    a.write_bytes(b"1")
    b.write_bytes(b"1")
    t.store(tmp_path / "c", a, t.file_key(a), picture(1))
    t.store(tmp_path / "c", b, t.file_key(b), picture(2))
    assert (t.load(tmp_path / "c", a, t.file_key(a)) == 1).all()
    assert (t.load(tmp_path / "c", b, t.file_key(b)) == 2).all()


def test_prune_removes_previews_of_files_no_longer_listed(tmp_path):
    a, b = tmp_path / "a.cvox", tmp_path / "b.cvox"
    a.write_bytes(b"1")
    b.write_bytes(b"1")
    t.store(tmp_path / "c", a, t.file_key(a), picture(1))
    t.store(tmp_path / "c", b, t.file_key(b), picture(2))
    t.prune(tmp_path / "c", [str(a)])
    assert t.load(tmp_path / "c", a, t.file_key(a)) is not None and t.load(tmp_path / "c", b, t.file_key(b)) is None


def test_a_missing_file_has_no_key_and_a_damaged_preview_is_ignored(tmp_path):
    assert t.file_key(tmp_path / "nope.cvox") is None
    model = tmp_path / "a.cvox"
    model.write_bytes(b"1")
    key = t.file_key(model)
    (tmp_path / "c").mkdir()
    t.cache_file(tmp_path / "c", model, key).write_bytes(b"not a numpy file")
    assert t.load(tmp_path / "c", model, key) is None
    t.store(tmp_path / "c", model, key, np.zeros((3, 3, 4), np.uint8))        # wrong shape
    assert t.load(tmp_path / "c", model, key) is None
    assert t.load(None, model, key) is None
