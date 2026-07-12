import pytest

pytest.importorskip("PySide6")


def test_control_panel_state_defaults(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    st = p.state
    assert st["contrast"] == 50 and st["scale"] == 5 and st["style"] == "None"
    assert st["invert"] is False and st["preview_disabled"] is False
    assert st["saturation"] == 50 and st["params"] == {}
    assert "smart_mask" not in st
    assert "mask" not in st["params"]
    assert p.smart_mask_panel.settings.enabled is False


def test_smart_mask_overlay_does_not_emit_creative_changed(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    changed = []
    p.changed.connect(lambda: changed.append(True))
    p.smart_mask_panel.enabled_check.click()
    p.smart_mask_panel.overlay_check.click()
    assert changed == []


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


def test_dither_search_button_filters_and_restores_without_changing_style(qapp_fixture):
    from PySide6.QtCore import Qt
    from ditherzam.ui.controls import ControlPanel

    p = ControlPanel()
    p.set_registry_categories({
        "Default": ["None"],
        "Error Diffusion": ["Atkinson", "Floyd-Steinberg"],
        "Patterned": ["Crosshatch"],
    })
    p.set_style("Atkinson")

    p.dither_search_btn.click()
    assert p.dither_search.isVisibleTo(p)
    p.dither_search.setText("floyd")
    choices = [p.dither_combo.itemData(i, Qt.ItemDataRole.UserRole)
               for i in range(p.dither_combo.count())]
    assert "Floyd-Steinberg" in choices
    assert "Atkinson" not in choices
    assert p.state["style"] == "Atkinson"

    p.dither_search_btn.click()
    restored = [p.dither_combo.itemData(i, Qt.ItemDataRole.UserRole)
                for i in range(p.dither_combo.count())]
    assert "Atkinson" in restored and "Crosshatch" in restored
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
    p._sliders["luminance_threshold"].setValue(63)
    assert p.state["luminance_threshold"] == 63
    assert p._spins["luminance_threshold"].text() == "63"


def test_invert_and_preview_toggles(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    p.invert_toggle.setChecked(True)
    p.preview_toggle.setChecked(True)
    assert p.state["invert"] is True
    assert p.state["preview_disabled"] is True


def test_style_specific_parameter_controls_are_built_and_update_state(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.controls import ControlPanel

    p = ControlPanel()
    p.set_registry(registry)
    p.set_style("Uniform Modulation Y")

    assert set(p.param_sliders) >= {
        "dither_parameter_slider", "smoothing_factor_slider",
        "bleed_fraction_slider",
    }
    assert p.state["params"]["dither_parameter_slider"] == 2
    p.param_sliders["bleed_fraction_slider"].setValue(72)
    assert p.state["params"]["bleed_fraction_slider"] == 72
    assert p.param_value_labels["bleed_fraction_slider"].text() == "72"


def test_style_parameter_values_survive_switching_styles(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.controls import ControlPanel

    p = ControlPanel()
    p.set_registry(registry)
    p.set_style("Displace Contour")
    p.param_sliders["line_mode_slider"].setValue(3)
    p.set_style("Floyd-Steinberg")
    assert "creative_mix" in p.param_sliders
    p.set_style("Displace Contour")
    assert p.param_sliders["line_mode_slider"].value() == 3


def test_shared_legacy_parameter_keys_are_namespaced_per_style(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.controls import ControlPanel

    p = ControlPanel()
    p.set_registry(registry)
    p.set_style("Flow Hatch")
    p.param_sliders["dither_parameter_slider"].setValue(19)
    p.set_style("Hex Bayer")
    assert p.param_sliders["dither_parameter_slider"].value() == 3
    p.param_sliders["dither_parameter_slider"].setValue(11)
    p.set_style("Flow Hatch")
    assert p.param_sliders["dither_parameter_slider"].value() == 19


def test_preset_parameter_value_rehydrates_dynamic_control(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.controls import ControlPanel

    p = ControlPanel()
    p.set_registry(registry)
    p.set_style("Sine Wave Modulation", {
        "wave_frequency_slider": 17, "wave_threshold_slider": 23,
    })
    assert p.param_sliders["wave_frequency_slider"].value() == 17
    assert p.param_sliders["wave_threshold_slider"].value() == 23
