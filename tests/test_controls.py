import pytest

pytest.importorskip("PySide6")


def test_control_panel_state_defaults(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    st = p.state
    assert st["contrast"] == 50 and st["scale"] == 5 and st["style"] == "None"
    assert st["invert"] is False and st["preview_disabled"] is False
    assert st["saturation"] == 50 and st["params"] == {}


def test_slider_edit_updates_state_and_emits(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    seen = []
    p.changed.connect(lambda: seen.append(True))
    p.contrast_slider.setValue(70)
    assert p.state["contrast"] == 70
    assert seen                       # changed fired


def test_set_style_updates_state(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    p.set_registry_categories({"Default": ["None"], "Error Diffusion": ["Atkinson"]})
    p.set_style("Atkinson")
    assert p.state["style"] == "Atkinson"


def test_slider_updates_its_number_display(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    # adjustment spin shows round(value/100 * max_display); contrast max_display=250
    p.contrast_slider.setValue(70)
    assert p._spins["contrast"].text() == str(round(70 / 100 * 250))   # "175"
    # saturation spin is a plain 0..100 display
    p.saturation_slider.setValue(80)
    assert p.saturation_spin.text() == "80"
    # scale slider (1..20) shows its own value
    p.scale_slider.setValue(12)
    assert p.scale_spin.text() == "12"


def test_invert_and_preview_toggles(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    p.invert_toggle.setChecked(True)
    p.preview_toggle.setChecked(True)
    assert p.state["invert"] is True
    assert p.state["preview_disabled"] is True
