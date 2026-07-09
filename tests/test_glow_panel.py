import pytest

pytest.importorskip("PySide6")


def test_glow_panel_has_eight_sliders_and_toggle(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    p = GlowPanel()
    keys = ("glow_threshold", "glow_smoothing", "glow_radius", "glow_intensity",
            "glow_epsilon", "glow_falloff", "glow_distance", "glow_aspect")
    for k in keys:
        assert k in p._sliders
    assert p.state["glow_enabled"] is False


def test_glow_slider_edit_updates_state_and_emits(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    p = GlowPanel()
    seen = []
    p.changed.connect(lambda: seen.append(True))
    p._sliders["glow_threshold"].setValue(120)
    assert p.state["glow_threshold"] == 120
    assert seen


def test_glow_enable_toggle_updates_state(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    p = GlowPanel()
    p.enable_toggle.setChecked(True)
    assert p.state["glow_enabled"] is True


def test_glow_reset_all_restores_defaults(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    from ditherzam.effects.glow_params import GLOW_DEFAULTS
    p = GlowPanel()
    p._sliders["glow_radius"].setValue(200)
    p.reset_all()
    assert p.state["glow_radius"] == GLOW_DEFAULTS["glow_radius"]
