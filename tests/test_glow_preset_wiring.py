import pytest

pytest.importorskip("PySide6")


def test_enabled_glow_appears_in_effect_stack(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    ed = ImageEditor()
    ed.glow_panel.enable_toggle.setChecked(True)
    ed.glow_panel._sliders["glow_threshold"].setValue(120)
    stack = ed._current_effect_stack()
    names = [n for n, _ in stack.items]
    assert "Epsilon Glow" in names
    params = dict(stack.items)["Epsilon Glow"]
    assert params["threshold"] == 120.0


def test_disabled_glow_absent_from_stack(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    ed = ImageEditor()
    assert ed.glow_panel.state["glow_enabled"] is False
    stack = ed._current_effect_stack()
    names = [] if stack is None else [n for n, _ in stack.items]
    assert "Epsilon Glow" not in names


def test_apply_preset_routes_glow_into_tab(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    ed = ImageEditor()
    settings = ed._collect_settings()
    effects = [("Epsilon Glow", {"threshold": 100.0, "smoothing": 20.0,
                                  "radius": 30.0, "intensity": 1.4, "epsilon": 0.6,
                                  "falloff": 0.3, "distance_scale": 1.0, "aspect": 1.0})]
    ed._apply_preset(settings, None, effects)
    assert ed.glow_panel.state["glow_enabled"] is True
    assert ed.glow_panel.state["glow_threshold"] == 100
    # not added to the name-only effects list
    assert "Epsilon Glow" not in ed.panel.state.get("effects", [])
