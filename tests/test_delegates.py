import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt


def test_populate_marks_headers_nonselectable(qapp_fixture):
    from ditherzam.ui.widgets import NoScrollComboBox
    from ditherzam.ui.delegates import populate_dither_combo, HEADER_ROLE

    combo = NoScrollComboBox()
    by_cat = {"Default": ["None"], "Error Diffusion": ["Floyd-Steinberg", "Atkinson"]}
    populate_dither_combo(combo, by_cat)

    model = combo.model()
    assert combo.count() == 5                       # 2 headers + 3 styles

    header_idx = model.index(0, 0)                  # "Default"
    assert header_idx.data(HEADER_ROLE) is True
    assert not (header_idx.flags() & Qt.ItemFlag.ItemIsSelectable)

    style_idx = model.index(1, 0)                   # "None"
    assert style_idx.data(Qt.ItemDataRole.UserRole) == "None"
    assert style_idx.flags() & Qt.ItemFlag.ItemIsSelectable


def test_delegate_constructs(qapp_fixture):
    from ditherzam.ui.delegates import DitherStyleDelegate
    assert DitherStyleDelegate() is not None
