import numpy as np
import pytest

pytest.importorskip("PySide6")

from ditherzam.color.palette import Palette


def _strip():
    from ditherzam.ui.palette_editor import SwatchStrip
    s = SwatchStrip()
    s.set_palette(Palette.from_list("t", [[0, 0, 0], [128, 128, 128], [255, 255, 255]]))
    return s


def test_set_palette_adopts_copy(qapp_fixture):
    p = Palette.from_list("t", [[10, 20, 30], [40, 50, 60]])
    from ditherzam.ui.palette_editor import SwatchStrip
    s = SwatchStrip()
    s.set_palette(p)
    p.colors[0, 0] = 200
    assert s.palette().colors[0, 0] == 10          # independent from the source


def test_recolor_updates_and_emits(qapp_fixture):
    s = _strip()
    seen = []
    s.edited.connect(lambda pal: seen.append(pal))
    s.set_swatch_color(1, (10, 20, 30))
    np.testing.assert_array_equal(s.palette().colors[1], [10, 20, 30])
    assert len(seen) == 1


def test_add_swatch_appends(qapp_fixture):
    s = _strip()
    s.add_swatch()
    assert s.palette().colors.shape[0] == 4


def test_remove_swatch_respects_minimum(qapp_fixture):
    s = _strip()
    s.remove_swatch(0)
    s.remove_swatch(0)
    assert s.palette().colors.shape[0] == 1
    s.remove_swatch(0)                              # would drop to 0 -> no-op
    assert s.palette().colors.shape[0] == 1


def test_locked_indices_survive_shuffle(qapp_fixture):
    s = _strip()
    s.toggle_lock(1)
    assert s.locked() == {1}
    before = s.palette().colors[1].copy()
    s.shuffle(rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s.palette().colors[1], before)


def test_remove_reindexes_locks(qapp_fixture):
    s = _strip()
    s.toggle_lock(2)
    s.remove_swatch(0)                              # index 2 -> now index 1
    assert s.locked() == {1}


def test_move_swatch_reorders_colors(qapp_fixture):
    s = _strip()   # colors [[0,0,0],[128,128,128],[255,255,255]]
    s.move_swatch(0, 2)
    np.testing.assert_array_equal(s.palette().colors[2], [0, 0, 0])
    np.testing.assert_array_equal(s.palette().colors[0], [128, 128, 128])


def test_move_swatch_emits_edited(qapp_fixture):
    s = _strip()
    seen = []
    s.edited.connect(lambda pal: seen.append(pal))
    s.move_swatch(2, 0)
    assert len(seen) == 1


def test_move_swatch_remaps_locks(qapp_fixture):
    s = _strip()
    s.toggle_lock(0)              # lock the first swatch
    s.move_swatch(0, 2)          # it moves to index 2
    assert s.locked() == {2}


def test_move_swatch_noop_same_index(qapp_fixture):
    s = _strip()
    before = s.palette().colors.copy()
    s.move_swatch(1, 1)
    np.testing.assert_array_equal(s.palette().colors, before)


def test_set_palette_preserves_category(qapp_fixture):
    from ditherzam.ui.palette_editor import SwatchStrip
    s = SwatchStrip()
    s.set_palette(Palette.from_list("t", [[1, 2, 3], [4, 5, 6]], category="retro"))
    assert s.palette().category == "retro"
