# tests/test_glow_tab.py
import pytest

pytest.importorskip("PySide6")


def test_editor_has_glow_tab(qapp_fixture):
    from PySide6.QtWidgets import QTabWidget
    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.ui.glow_panel import GlowPanel
    ed = ImageEditor()
    assert isinstance(ed.tabs, QTabWidget)
    titles = [ed.tabs.tabText(i) for i in range(ed.tabs.count())]
    assert "Editor" in titles and "Glow" in titles
    assert isinstance(ed.glow_panel, GlowPanel)


def test_epsilon_glow_removed_from_name_only_effects(qapp_fixture):
    from ditherzam.ui.controls import _EFFECTS
    assert "Epsilon Glow" not in _EFFECTS
