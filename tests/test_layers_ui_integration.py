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


def test_active_layer_edit_schedules_proxy_then_settled_layer_previews(
        qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    import numpy as np
    gray = np.zeros((2, 2), dtype=np.float32)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    editor.layers_controller.open_document(gray, rgba)
    timers = []
    monkeypatch.setattr(editor._debounce, "start",
                        lambda delay: timers.append(("proxy", delay)))
    monkeypatch.setattr(editor._settle, "start",
                        lambda delay: timers.append(("settle", delay)))

    editor.schedule_render()

    assert timers == [
        ("proxy", editor._debounce_ms),
        ("settle", editor._settle_ms),
    ]


def test_layer_preview_ticks_use_selected_layer_then_document_composite(
        qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor(proxy_max_side=640)
    import numpy as np
    gray = np.zeros((800, 1200), dtype=np.float32)
    rgba = np.zeros((800, 1200, 4), dtype=np.uint8)
    caps = []
    monkeypatch.setattr(
        editor.layers_controller, "update_active_from_editor",
        lambda cap=None, **kwargs:
        caps.append(("update", cap, kwargs)) or True)
    monkeypatch.setattr(
        editor.layers_controller, "request_preview",
        lambda cap=None, **_kwargs: caps.append(("preview", cap)))
    editor.layers_controller.open_document(gray, rgba)
    caps.clear()
    launched = []
    monkeypatch.setattr(editor, "_launch_worker", launched.append)

    editor._do_render()
    editor._do_full_render()

    assert caps == [
        ("update", None, {"request_preview": False}),
        ("preview", editor._layer_policy_cap()),
    ]
    assert len(launched) == 1
    from ditherzam.ui.render_request import RenderKind
    assert launched[0].kind is RenderKind.DRAG
    assert launched[0].target_max_side == 640
    assert launched[0].layer_preview_context is not None


def test_layer_proxy_publication_keeps_settled_preview_timer_alive(
        qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    import numpy as np
    gray = np.zeros((8, 12), dtype=np.float32)
    rgba = np.zeros((8, 12, 4), dtype=np.uint8)
    editor.layers_controller.open_document(gray, rgba)
    editor._settle.start(10_000)

    editor._show_layer_frame(rgba)

    assert editor._settle.isActive()


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
