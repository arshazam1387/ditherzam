import pytest

pytest.importorskip("PySide6")

from ditherzam.color.palette_store import PaletteStore


def _picker(tmp_path):
    from ditherzam.ui.palette_picker import PalettePicker
    p = PalettePicker()
    p.populate(PaletteStore(user_dir=tmp_path / "pal"))
    return p


def _palette_items(picker):
    from PySide6.QtCore import Qt
    out = []
    for i in range(picker.topLevelItemCount()):
        top = picker.topLevelItem(i)
        for j in range(top.childCount()):
            out.append(top.child(j))
    return out


def test_headers_are_categories(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    headers = [picker.topLevelItem(i).text(0) for i in range(picker.topLevelItemCount())]
    assert "retro" in headers and "mono" in headers


def test_palette_rows_present(qapp_fixture, tmp_path):
    from PySide6.QtCore import Qt
    picker = _picker(tmp_path)
    names = [it.data(0, Qt.ItemDataRole.UserRole) for it in _palette_items(picker)]
    assert "gameboy" in names and "grayscale" in names


def test_picker_is_enlarged(qapp_fixture, tmp_path):
    # Palette section made bigger: taller list + larger swatches.
    picker = _picker(tmp_path)
    assert picker.minimumHeight() >= 240
    assert picker.iconSize().width() >= 96


def test_added_palettes_and_category_visible(qapp_fixture, tmp_path):
    from PySide6.QtCore import Qt
    picker = _picker(tmp_path)
    headers = [picker.topLevelItem(i).text(0) for i in range(picker.topLevelItemCount())]
    assert "cool" in headers
    names = [it.data(0, Qt.ItemDataRole.UserRole) for it in _palette_items(picker)]
    for n in ("c64", "zxspectrum", "nord", "solarized", "greencrt", "ambercrt"):
        assert n in names


def test_click_palette_emits_selected(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.selected.connect(seen.append)
    item = next(it for it in _palette_items(picker)
                if it.data(0, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ItemDataRole.UserRole) == "gameboy")
    picker._on_item_clicked(item, 0)
    assert seen == ["gameboy"]


def test_header_click_does_not_emit_selected(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.selected.connect(seen.append)
    picker._on_item_clicked(picker.topLevelItem(0), 0)
    assert seen == []


def test_hover_emits_preview_when_enabled(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.preview.connect(seen.append)
    item = next(it for it in _palette_items(picker)
                if it.data(0, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ItemDataRole.UserRole) == "gameboy")
    picker._on_item_entered(item, 0)
    assert seen and seen[-1] is not None and seen[-1].name == "gameboy"


def test_hover_suppressed_when_disabled(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    picker.set_preview_enabled(False)
    seen = []
    picker.preview.connect(seen.append)
    item = _palette_items(picker)[0]
    picker._on_item_entered(item, 0)
    assert seen == []


def test_select_does_not_emit(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen_sel, seen_prev = [], []
    picker.selected.connect(seen_sel.append)
    picker.preview.connect(seen_prev.append)
    picker.select("gameboy")
    assert seen_sel == [] and seen_prev == []


def test_wheel_cycle_previews_only_when_on(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.preview.connect(seen.append)
    picker.set_wheel_cycle(True)
    picker.select("gameboy")
    picker._cycle(1)                     # step to next palette
    assert seen and seen[-1] is not None
