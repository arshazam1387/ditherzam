import numpy as np
import pytest

pytest.importorskip("PySide6")

from ditherzam.color.palette import Palette
from ditherzam.color.palette_store import PaletteStore


def _panel(tmp_path):
    from ditherzam.ui.controls import ControlPanel
    return ControlPanel(store=PaletteStore(user_dir=tmp_path / "pal"))


def test_combo_populated_from_store(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    items = [panel.palette_combo.itemText(i) for i in range(panel.palette_combo.count())]
    assert "gameboy" in items and "pico8" in items


def test_new_state_defaults(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    assert panel.state["palette_autosave"] is False
    assert panel.state["extract_unit"] == "k"


def test_selecting_palette_sets_working_copy(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    assert panel.working_palette.name == "gameboy"
    assert panel.swatch_strip.palette().name == "gameboy"


def test_swatch_edit_updates_working_and_emits_changed(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.swatch_strip.set_swatch_color(0, (7, 8, 9))
    np.testing.assert_array_equal(panel.working_palette.colors[0], [7, 8, 9])
    assert seen                                  # changed fired


def test_save_palette_persists_and_refreshes_combo(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    panel.swatch_strip.set_swatch_color(0, (1, 2, 3))
    panel._on_save_palette()
    assert panel.store.is_user("gameboy")
    # combo still holds a single "gameboy" entry, still selected
    items = [panel.palette_combo.itemText(i) for i in range(panel.palette_combo.count())]
    assert items.count("gameboy") == 1
    assert panel.palette_combo.currentText() == "gameboy"


def test_autosave_writes_on_edit_when_enabled(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    panel.state["palette_autosave"] = True
    panel.swatch_strip.set_swatch_color(0, (1, 2, 3))
    assert panel.store.is_user("gameboy")


def test_reset_to_builtin_drops_fork(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    panel.swatch_strip.set_swatch_color(0, (1, 2, 3))
    panel._on_save_palette()
    panel._on_reset_palette()
    assert not panel.store.is_user("gameboy")


def test_set_working_palette_pushes_to_strip(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.set_working_palette(Palette.from_list("from image", [[9, 9, 9], [1, 1, 1]]))
    assert panel.working_palette.name == "from image"
    assert panel.swatch_strip.palette().colors.shape[0] == 2
    assert seen


def test_set_working_palette_syncs_combo(qapp_fixture, tmp_path):
    from ditherzam.color.palette import Palette
    panel = _panel(tmp_path)
    panel.set_working_palette(Palette.from_list("from image", [[1, 1, 1], [2, 2, 2]]))
    assert panel.palette_combo.currentText() == "from image"
    assert panel.palette_combo.findText("from image") >= 0


def test_autosave_toggle_widget_sets_state(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.autosave_toggle.setChecked(True)
    assert panel.state["palette_autosave"] is True
    panel.autosave_toggle.setChecked(False)
    assert panel.state["palette_autosave"] is False


def test_extract_unit_widget_switches_range(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.extract_unit_combo.setCurrentText("%")
    assert panel.state["extract_unit"] == "pct"
    assert panel.extract_slider.maximum() == 100
    panel.extract_unit_combo.setCurrentText("k")
    assert panel.state["extract_unit"] == "k"
    assert panel.extract_slider.maximum() == 64
