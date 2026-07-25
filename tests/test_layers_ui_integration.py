def test_editor_wires_spatial_layers_dock(qapp_fixture):
    from PySide6.QtCore import Qt

    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    assert editor.layers_dock.windowTitle() == "Layers"
    assert editor.dockWidgetArea(
        editor.layers_dock
    ) == Qt.DockWidgetArea.RightDockWidgetArea
    assert editor.layers_controller.panel is editor.layers_panel
    assert editor.composition_dock.windowTitle() == "Look Composer"


def test_layer_preset_provider_includes_smart_mask(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    preset = editor.layers_controller._preset_provider()
    assert set(preset["smart_mask"]) == {
        "enabled", "target", "sensitivity", "feather_px",
        "expansion_px", "invert", "outside", "bake_fill",
    }


def test_layer_frame_sink_retires_editor_work(qapp_fixture, monkeypatch):
    import numpy as np

    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((2, 2), dtype=np.float32)
    editor.load_array(gray)
    invalidated = []
    monkeypatch.setattr(
        editor._scheduler, "invalidate", lambda: invalidated.append(True))
    editor._show_composition_frame(np.zeros((2, 2, 4), dtype=np.uint8))
    assert invalidated == [True]


def test_layer_shutdown_is_called_on_close(qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    shutdown = []
    monkeypatch.setattr(
        editor.layers_controller, "shutdown", lambda: shutdown.append(True))
    editor.close()
    qapp_fixture.processEvents()
    assert shutdown


def test_layer_preset_application_does_not_render_single_look(
        qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    rendered = []
    monkeypatch.setattr(editor, "render_now", lambda: rendered.append(True))
    editor._apply_layer_preset(editor._current_composition_preset())
    assert rendered == []


def test_active_layer_edit_routes_away_from_editor_scheduler(
        qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    import numpy as np
    gray = np.zeros((2, 2), dtype=np.float32)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    editor.layers_controller.open_document(gray, rgba)
    routed = []
    ordinary = []
    monkeypatch.setattr(
        editor.layers_controller, "update_active_from_editor",
        lambda: routed.append(True) or True)
    monkeypatch.setattr(
        editor._debounce, "start", lambda *_args: ordinary.append(True))
    editor.schedule_render()
    assert routed == [True]
    assert ordinary == []


def test_loading_source_creates_one_selected_base_layer(qapp_fixture):
    import numpy as np

    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    editor.load_array(np.zeros((4, 6), dtype=np.float32))

    assert len(editor.layers_controller.stack.layers) == 1
    assert editor.layers_controller.stack.layers[0].name == "Layer 1"
    assert editor.layers_controller.active_index == 0


def test_replacing_source_resets_existing_stack(qapp_fixture):
    import numpy as np

    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    editor.load_array(np.zeros((4, 6), dtype=np.float32))
    editor.layers_controller.add_layer()
    assert len(editor.layers_controller.stack.layers) == 2

    editor.load_array(np.ones((3, 5), dtype=np.float32))

    assert len(editor.layers_controller.stack.layers) == 1
    assert editor.layers_controller.stack.layers[0].name == "Layer 1"
