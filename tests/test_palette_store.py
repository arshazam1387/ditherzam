import numpy as np
import pytest

from ditherzam.color.palette import Palette
from ditherzam.color.palette_store import PaletteStore


def _store(tmp_path):
    return PaletteStore(user_dir=tmp_path / "palettes")


def test_list_includes_builtins(tmp_path):
    s = _store(tmp_path)
    names = s.list()
    for b in ("grayscale", "gameboy", "cga", "pico8", "sepia"):
        assert b in names


def test_get_builtin_returns_copy(tmp_path):
    s = _store(tmp_path)
    a = s.get("gameboy")
    b = s.get("gameboy")
    assert a.colors is not b.colors           # independent arrays
    a.colors[0, 0] = 123.0
    assert s.get("gameboy").colors[0, 0] != 123.0   # source not mutated


def test_get_unknown_raises(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(KeyError):
        s.get("no-such-palette")


def test_save_then_get_roundtrip(tmp_path):
    s = _store(tmp_path)
    p = Palette.from_list("mine", [[1, 2, 3], [4, 5, 6]])
    s.save(p)
    assert s.is_user("mine")
    got = s.get("mine")
    np.testing.assert_array_equal(got.colors, p.colors)
    assert "mine" in s.list()


def test_user_file_shadows_builtin(tmp_path):
    s = _store(tmp_path)
    fork = Palette.from_list("gameboy", [[0, 0, 0], [255, 255, 255]])
    s.save(fork)
    assert s.is_user("gameboy") and s.is_builtin("gameboy")
    np.testing.assert_array_equal(s.get("gameboy").colors, fork.colors)
    assert s.list().count("gameboy") == 1     # de-duplicated


def test_delete_reveals_builtin_again(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("gameboy", [[0, 0, 0], [255, 255, 255]]))
    s.delete("gameboy")
    assert not s.is_user("gameboy")
    assert s.get("gameboy").colors.shape[0] > 2   # original builtin restored


def test_reset_to_builtin_returns_builtin_and_drops_fork(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("gameboy", [[0, 0, 0], [255, 255, 255]]))
    restored = s.reset_to_builtin("gameboy")
    assert not s.is_user("gameboy")
    assert restored.colors.shape[0] > 2


def test_reset_to_builtin_without_builtin_raises(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("customonly", [[1, 1, 1]]))
    with pytest.raises(KeyError):
        s.reset_to_builtin("customonly")


def test_user_dir_not_created_until_save(tmp_path):
    d = tmp_path / "palettes"
    s = PaletteStore(user_dir=d)
    s.list()                     # read-only ops must not create the dir
    assert not d.exists()
    s.save(Palette.from_list("x", [[0, 0, 0]]))
    assert d.exists()


def test_get_carries_category(tmp_path):
    s = _store(tmp_path)
    assert s.get("gameboy").category == "retro"


def test_list_by_category_groups_builtins(tmp_path):
    s = _store(tmp_path)
    cats = s.list_by_category()
    assert set(cats["retro"]) >= {"gameboy", "pico8", "cga"}
    assert set(cats["mono"]) >= {"grayscale", "sepia"}


def test_list_by_category_uncategorized_last(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("loner", [[1, 1, 1]]))   # empty category
    cats = s.list_by_category()
    assert "loner" in cats["uncategorized"]
    assert list(cats.keys())[-1] == "uncategorized"


def test_user_category_wins_for_shadowed_name(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("gameboy", [[0, 0, 0]], category="favourites"))
    cats = s.list_by_category()
    assert "gameboy" in cats["favourites"]
    assert "gameboy" not in cats.get("retro", [])
