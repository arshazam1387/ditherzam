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


def test_color_off_has_no_engine(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "off"

    assert win._current_color_engine() is None


def test_fresh_engines_share_editor_owned_context(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")

    first = win._current_color_engine()
    second = win._current_color_engine()

    assert first is not second
    assert first.context_cache is win._color_context_cache
    assert second.context_cache is win._color_context_cache
    assert first.context is second.context


def test_same_name_palette_edit_misses_editor_context(qapp_fixture):
    from ditherzam.color.palette import Palette

    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    original = Palette.from_list("same", [[0, 0, 0], [255, 255, 255]])
    edited = Palette.from_list("same", [[1, 0, 0], [255, 255, 255]])

    win.panel.set_working_palette(original)
    original_context = win._current_color_engine().context
    win.panel.set_working_palette(edited)
    edited_context = win._current_color_engine().context

    assert edited_context is not original_context


def test_palette_hover_then_revert_restores_cached_context(qapp_fixture):
    from ditherzam.color.palette import Palette

    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    original_context = win._current_color_engine().context

    win._on_palette_preview(Palette.from_list("hover", [[9, 9, 9], [99, 99, 99]]))
    hover_context = win._current_color_engine().context
    win._on_palette_preview(None)
    restored_context = win._current_color_engine().context

    assert hover_context is not original_context
    assert restored_context is original_context


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
    assert win.panel.state["color_mode"] == "source"
    assert win._current_color_engine().source_dither == 100


def test_from_image_no_image_is_noop(qapp_fixture):
    win = _win(qapp_fixture)
    win._base_rgb = None
    win._on_from_image_requested()      # must not raise
