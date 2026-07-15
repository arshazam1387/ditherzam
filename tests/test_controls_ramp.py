import pytest

from ditherzam.color.ramp import RAMP_MODES

pytest.importorskip("PySide6")


def test_panel_state_has_depth_and_mapping_defaults(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    panel = ControlPanel()
    assert panel.state["depth"] == 2
    assert panel.state["color_mapping"] == "match"


def test_ramp_is_a_selectable_color_mode(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    panel = ControlPanel()
    items = [panel.mode_combo.itemText(i) for i in range(panel.mode_combo.count())]
    assert "ramp" in items


def test_mapping_combo_lists_all_ramp_modes(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    panel = ControlPanel()
    items = [panel.mapping_combo.itemText(i) for i in range(panel.mapping_combo.count())]
    for mode in RAMP_MODES:
        assert mode in items


def test_depth_slider_updates_state_and_display(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    panel = ControlPanel()
    panel.depth_slider.setValue(32)
    assert panel.state["depth"] == 32
    assert panel._spins["depth"].text() == "32"
