import numpy as np
import pytest

pytest.importorskip("PySide6")


def _win(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    return ImageEditor()


def test_current_palette_is_working_palette(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    win.panel.swatch_strip.set_swatch_color(0, (5, 6, 7))
    pal = win._current_palette()
    np.testing.assert_array_equal(pal.colors[0], [5, 6, 7])


def test_current_palette_none_when_off(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "off"
    assert win._current_palette() is None


def test_preview_palette_overrides_current(qapp_fixture):
    from ditherzam.color.palette import Palette
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    win._on_palette_preview(Palette.from_list("temp", [[9, 9, 9]]))
    assert win._current_palette().name == "temp"
    assert win.panel.working_palette.name == "gameboy"   # unchanged


def test_preview_none_reverts(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    win._on_palette_preview(None)
    assert win._current_palette().name == "gameboy"


def test_preview_ignored_when_disabled(qapp_fixture):
    from ditherzam.color.palette import Palette
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    win.panel.state["palette_preview"] = False
    win._on_palette_preview(Palette.from_list("temp", [[9, 9, 9]]))
    assert win._current_palette().name == "gameboy"


def test_load_array_retains_rgb(qapp_fixture):
    win = _win(qapp_fixture)
    rgb = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    gray = rgb.mean(axis=2).astype(np.float32)
    win.load_array(gray, rgb)
    assert win._base_rgb is not None
    assert win._base_rgb.shape == (8, 8, 3)


def test_from_image_request_sets_working_palette(qapp_fixture):
    win = _win(qapp_fixture)
    rgb = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    win.load_array(rgb.mean(axis=2).astype(np.float32), rgb)
    win.panel.state["extract_unit"] = "k"
    win.panel.extract_slider.setValue(6)
    win._on_from_image_requested()
    assert win.panel.working_palette.colors.shape == (6, 3)
    assert win.panel.working_palette.name == "from image"


def test_from_image_no_image_is_noop(qapp_fixture):
    win = _win(qapp_fixture)
    win._base_rgb = None
    win._on_from_image_requested()      # must not raise
