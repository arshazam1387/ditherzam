def test_editor_wires_look_composer_dock(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    assert editor.composition_dock.windowTitle() == "Look Composer"
    assert editor.composition_panel.capture_btn.text() == "Capture Current Look"
    assert editor.composition_controller.panel is editor.composition_panel


def test_current_composition_preset_includes_smart_mask(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    preset = editor._current_composition_preset()
    assert set(preset["smart_mask"]) == {
        "enabled", "target", "sensitivity", "feather_px",
        "expansion_px", "invert", "outside", "bake_fill",
    }
