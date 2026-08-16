import numpy as np


def test_panel_exposes_place_blank_and_zero_layer_document_state(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    assert panel.new_blank_btn.text() == "New Blank"
    assert panel.place_image_btn.text() == "Place Image"
    assert panel.empty_label.text() == (
        "No layers. Place an image or create a blank layer.")
    panel.set_document_available(True)
    assert panel.export_btn.isEnabled()
    panel.set_layers(
        [("a", "Layer 1", True, 100, "normal")], selected=0)
    assert panel.delete_btn.isEnabled()


def test_selecting_layer_swaps_editor_source_without_single_render(
        qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray_a = np.full((3, 4), 10, np.float32)
    rgba_a = np.full((3, 4, 4), 10, np.uint8)
    rgba_a[..., 3] = 255
    gray_b = np.full((3, 4), 90, np.float32)
    rgba_b = np.full((3, 4, 4), 90, np.uint8)
    rgba_b[..., 3] = 255
    editor.layers_controller.open_document(gray_a, rgba_a)
    editor.layers_controller.place_source(gray_b, rgba_b)
    painted = []
    monkeypatch.setattr(editor, "render_now", lambda: painted.append(True))
    editor.layers_controller.activate_layer(0)
    assert np.array_equal(editor._base_gray, gray_a)
    assert np.array_equal(editor._base_rgba, rgba_a)
    assert painted == []


def test_decoded_image_opens_then_places_without_replacing_document(
        qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray_a = np.full((3, 4), 10, np.float32)
    rgba_a = np.full((3, 4, 4), 10, np.uint8)
    rgba_a[..., 3] = 255
    gray_b = np.full((3, 4), 90, np.float32)
    rgba_b = np.full((3, 4, 4), 90, np.uint8)
    rgba_b[..., 3] = 255
    editor._accept_decoded_layer_source(gray_a, rgba_a, intent="open")
    first = editor.layers_controller.document.layers[0]
    editor._accept_decoded_layer_source(gray_b, rgba_b, intent="place")
    assert len(editor.layers_controller.document.layers) == 2
    assert editor.layers_controller.document.layers[0] is first
    assert np.all(
        editor.layers_controller.document.layers[1].source.rgba[..., :3] == 90)
    assert np.array_equal(editor._base_gray, gray_b)
    assert np.array_equal(editor._base_rgba, rgba_b)


def test_layer_activation_restores_complete_color_and_depth_state(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((3, 4), np.float32)
    rgba = np.zeros((3, 4, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    preset = editor._current_composition_preset()
    preset["dither"]["depth"] = 17
    preset["dither"]["color_mapping"] = "reverse"
    preset["color"] = {
        "mode": "source",
        "source_dither": 31,
        "source_dither_brighten": True,
        "palette": {
            "name": "layer-colors",
            "category": "test",
            "colors": [[0, 0, 0], [255, 255, 255]],
        },
    }
    editor._apply_layer_preset(preset)
    assert editor.panel.depth_slider.value() == 17
    assert editor.panel.mapping_combo.currentText() == "reverse"
    assert editor.panel.mode_combo.currentText() == "source"
    assert editor.panel.source_dither_slider.value() == 31
    assert editor.panel.source_dither_brighten_check.isChecked()


def test_layer_preview_geometry_uses_document_canvas_not_active_source(
        qapp_fixture, monkeypatch):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    canvas_gray = np.zeros((80, 100), np.float32)
    canvas_rgba = np.zeros((80, 100, 4), np.uint8)
    canvas_rgba[..., 3] = 255
    placed_gray = np.zeros((20, 200), np.float32)
    placed_rgba = np.zeros((20, 200, 4), np.uint8)
    placed_rgba[..., 3] = 255
    editor.layers_controller.open_document(canvas_gray, canvas_rgba)
    editor.layers_controller.place_source(placed_gray, placed_rgba)
    calls = []
    monkeypatch.setattr(
        editor.viewport, "set_pixmap",
        lambda pixmap, logical_size=None, refit=True:
        calls.append((logical_size, refit)),
    )
    editor._show_layer_frame(np.zeros((80, 100, 4), np.uint8))
    assert calls[-1][0] == (100, 80)
    assert editor._layer_reference_size() == (100, 80)


def test_canvas_drag_moves_only_active_layer(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor.layers_controller.new_blank_layer()
    untouched = editor.layers_controller.document.layers[0]
    editor._begin_layer_drag()
    editor._drag_active_layer(3.2, -1.7)
    assert editor.layers_controller.active_geometry() == (3, -2, 12, 10)
    assert editor.layers_controller.document.layers[0] is untouched


def test_canvas_drag_does_not_switch_targets_mid_gesture(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor.layers_controller.new_blank_layer()
    upper_before = editor.layers_controller.document.layers[1]
    editor._begin_layer_drag()
    editor.layers_controller.activate_layer(0)
    editor._drag_active_layer(5, 4)
    assert editor.layers_controller.document.layers[0].transform.x == 0
    assert editor.layers_controller.document.layers[0].transform.y == 0
    assert editor.layers_controller.document.layers[1] is upper_before


def test_drag_target_syncs_before_async_preview_finishes(qapp_fixture):
    from PySide6.QtCore import QRectF
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor._start_layer_transform(0)
    editor.layers_controller.move_active_layer_to(7, -3)
    assert editor.viewport._layer_drag_rect == QRectF(7, -3, 12, 10)


def test_transform_takes_canvas_input_from_masking_tools(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor.viewport.set_mask_brush_mode(True, 32)
    editor.viewport.set_selection_tool("rectangle")
    editor.viewport.set_gradient_tool("linear")

    editor._start_layer_transform(0)

    assert editor._layer_transform_active()
    assert editor.viewport._mask_brush_mode is False
    assert editor.viewport._selection_tool is None
    assert editor.viewport._gradient_tool is None
    assert editor.viewport._layer_drag_rect is not None


def test_transform_mode_cancel_restores_geometry_and_confirm_keeps_it(
        qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor._start_layer_transform(0)
    editor._begin_layer_drag()
    editor._drag_active_layer(5, 4)
    editor.layers_panel.lock_aspect_check.setChecked(False)
    editor._begin_layer_resize("se")
    editor._resize_active_layer("se", 7, 6)
    assert editor.layers_controller.active_geometry() == (5, 4, 19, 16)
    editor._cancel_layer_transform()
    assert editor.layers_controller.active_geometry() == (0, 0, 12, 10)
    assert editor.viewport._layer_drag_rect is None

    editor._start_layer_transform(0)
    editor._begin_layer_drag()
    editor._drag_active_layer(3, -2)
    editor._confirm_layer_transform()
    assert editor.layers_controller.active_geometry() == (3, -2, 12, 10)
    assert editor.viewport._layer_drag_rect is None


def test_main_raster_export_uses_complete_layer_stack(
        qapp_fixture, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor.layers_controller.new_blank_layer()
    destination = tmp_path / "stack.png"
    exported = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *_args: (str(destination), "")))
    monkeypatch.setattr(
        editor.layers_controller, "export_current",
        lambda path: exported.append(path) or destination)
    monkeypatch.setattr(
        editor, "_rendered_rgb",
        lambda: (_ for _ in ()).throw(
            AssertionError("active-layer renderer must not export a stack")))

    editor._on_export_raster("PNG Images (*.png)", ".png")

    assert exported == [str(destination)]


def test_active_layer_mask_overlay_changes_document_owned_preview(
        qapp_fixture):
    from ditherzam.masking.contracts import (
        InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity,
    )
    from ditherzam.masking.settings import SmartMaskSettings
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.full((2, 2), 80, np.float32)
    rgba = np.full((2, 2, 4), 80, np.uint8)
    rgba[..., 3] = 255
    source = source_identity(rgba)
    identity = InferenceIdentity(
        source, ModelIdentity("test", "1", "a" * 64), "test", "primary")
    probability = ProbabilityMap(
        identity, np.array([[1.0, 0.0], [1.0, 0.0]], np.float32))
    editor.layers_controller.open_document(gray, rgba, probability)
    editor.panel.smart_mask_panel.set_settings(
        SmartMaskSettings(enabled=True, feather_px=0))
    composite = np.full((2, 2, 4), 80, np.uint8)
    composite[..., 3] = 255

    editor.panel.smart_mask_panel.overlay_check.setChecked(False)
    editor._show_layer_frame(composite)
    without_overlay = editor.last_qimage.copy()
    editor.panel.smart_mask_panel.overlay_check.setChecked(True)
    editor._show_layer_frame(composite)
    with_overlay = editor.last_qimage.copy()

    assert with_overlay != without_overlay


def test_active_layer_overlay_is_scoped_to_transformed_layer(qapp_fixture):
    from ditherzam.masking.contracts import (
        InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity,
    )
    from ditherzam.masking.settings import SmartMaskSettings
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    base_gray = np.zeros((4, 6), np.float32)
    base_rgba = np.zeros((4, 6, 4), np.uint8)
    base_rgba[..., 3] = 255
    masked_gray = np.full((2, 2), 80, np.float32)
    masked_rgba = np.full((2, 2, 4), 80, np.uint8)
    masked_rgba[..., 3] = 255
    source = source_identity(masked_rgba)
    probability = ProbabilityMap(
        InferenceIdentity(
            source, ModelIdentity("test", "1", "b" * 64), "test", "primary"),
        np.ones((2, 2), np.float32),
    )
    editor.layers_controller.open_document(base_gray, base_rgba)
    editor.layers_controller.place_source(masked_gray, masked_rgba, probability)
    editor.layers_controller.move_active_layer_to(3, 1)
    editor.panel.smart_mask_panel.set_settings(
        SmartMaskSettings(enabled=True, feather_px=0))
    editor.panel.smart_mask_panel.overlay_check.setChecked(True)
    composite = np.full((4, 6, 4), 80, np.uint8)
    composite[..., 3] = 255

    overlaid = editor._apply_active_layer_mask_overlay(composite)

    assert np.array_equal(overlaid[0, 0], composite[0, 0])
    assert not np.array_equal(overlaid[1, 3], composite[1, 3])
    assert np.array_equal(overlaid[3, 5], composite[3, 5])


def test_active_layer_overlay_does_not_tint_lower_only_pixels(qapp_fixture):
    from ditherzam.masking.contracts import (
        InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity,
    )
    from ditherzam.masking.settings import SmartMaskSettings
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    bottom_gray = np.zeros((3, 4), np.float32)
    bottom_rgba = np.full((3, 4, 4), [20, 40, 60, 255], np.uint8)
    active_gray = np.full((2, 2), 80, np.float32)
    active_rgba = np.full((2, 2, 4), [80, 80, 80, 255], np.uint8)
    active_rgba[0, 0, 3] = 0
    source = source_identity(active_rgba)
    probability = ProbabilityMap(
        InferenceIdentity(
            source, ModelIdentity("test", "1", "c" * 64), "test", "primary"),
        np.ones((2, 2), np.float32),
    )
    editor.layers_controller.open_document(bottom_gray, bottom_rgba)
    editor.layers_controller.place_source(active_gray, active_rgba, probability)
    editor.panel.smart_mask_panel.set_settings(
        SmartMaskSettings(enabled=True, feather_px=0))
    editor.panel.smart_mask_panel.overlay_check.setChecked(True)
    composite = bottom_rgba.copy()

    visible = editor._apply_active_layer_mask_overlay(composite)
    assert np.array_equal(visible[0, 1], composite[0, 1])
    assert not np.array_equal(visible[0, 2], composite[0, 2])

    editor.layers_controller.set_opacity(1, 0)
    assert np.array_equal(
        editor._apply_active_layer_mask_overlay(composite), composite)

    editor.layers_controller.set_opacity(1, 100)
    editor.layers_controller.set_visibility(1, False)
    assert np.array_equal(
        editor._apply_active_layer_mask_overlay(composite), composite)


def test_overlay_toggle_does_not_change_exact_layer_export(
        qapp_fixture, tmp_path):
    from ditherzam.masking.contracts import (
        InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity,
    )
    from ditherzam.masking.settings import SmartMaskSettings
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.full((2, 3), 80, np.float32)
    rgba = np.full((2, 3, 4), 80, np.uint8)
    rgba[..., 3] = 255
    source = source_identity(rgba)
    probability = ProbabilityMap(
        InferenceIdentity(
            source, ModelIdentity("test", "1", "d" * 64), "test", "primary"),
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], np.float32),
    )
    editor.layers_controller.open_document(gray, rgba, probability)
    editor.panel.smart_mask_panel.set_settings(
        SmartMaskSettings(enabled=True, feather_px=0))
    editor.layers_controller.update_active_from_editor()

    editor.panel.smart_mask_panel.overlay_check.setChecked(False)
    first = editor.layers_controller.export_current(tmp_path / "off.png")
    editor.panel.smart_mask_panel.overlay_check.setChecked(True)
    second = editor.layers_controller.export_current(tmp_path / "on.png")

    assert first.read_bytes() == second.read_bytes()


def test_overlay_publication_does_not_mutate_thumbnail_cache(qapp_fixture):
    from PySide6.QtGui import QPixmap
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.full((2, 3), 80, np.float32)
    rgba = np.full((2, 3, 4), 80, np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    layer = editor.layers_controller.document.layers[0]
    key = (
        layer.id,
        layer.source.source_identity,
        layer.look.signature,
        layer.mask_revision,
        layer.transform,
    )
    thumbnail = QPixmap(3, 2)
    thumbnail.fill()
    editor.layers_controller._thumbnail_cache[key] = thumbnail
    before = thumbnail.cacheKey()
    composite = rgba.copy()

    editor.panel.smart_mask_panel.overlay_check.setChecked(False)
    editor._show_layer_frame(composite)
    editor.panel.smart_mask_panel.overlay_check.setChecked(True)
    editor._show_layer_frame(composite)

    assert editor.layers_controller._thumbnail_cache[key] is thumbnail
    assert thumbnail.cacheKey() == before


def test_transform_blocks_decoded_place_and_keeps_transaction(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor._start_layer_transform(0)
    editor.layers_controller.move_active_layer_to(4, 3)

    editor._accept_decoded_layer_source(gray, rgba, intent="place")

    assert len(editor.layers_controller.document.layers) == 1
    assert editor.layers_controller.active_geometry() == (4, 3, 12, 10)
    assert editor._layer_transform_session is not None
    assert editor.layers_panel.status_label.text() == (
        "Confirm or cancel the active layer transform first.")


def test_transform_blocks_public_source_replacement_and_reentry(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    original_id = editor.layers_controller.document.layers[0].id
    editor._start_layer_transform(0)
    editor.layers_controller.move_active_layer_to(4, 3)

    editor.load_array(np.full((5, 6), 90, np.float32))
    editor._start_layer_transform(0)

    assert editor.layers_controller.document.layers[0].id == original_id
    assert editor.layers_controller.active_geometry() == (4, 3, 12, 10)
    editor._cancel_layer_transform()
    assert editor.layers_controller.active_geometry() == (0, 0, 12, 10)


def test_edit_menu_history_actions_use_standard_shortcuts_and_dynamic_labels(
        qapp_fixture, monkeypatch
):
    from PySide6.QtGui import QKeySequence
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    monkeypatch.setattr(editor.layers_controller, "request_preview", lambda: None)
    gray = np.zeros((3, 4), np.float32)
    rgba = np.zeros((3, 4, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    assert not editor.undo_action.isEnabled()
    editor.layers_controller.set_name(0, "Ink")
    assert editor.undo_action.text() == "Undo Rename Layer"
    assert editor.undo_action.shortcut().matches(
        QKeySequence.StandardKey.Undo)
    editor.undo_action.trigger()
    assert editor.layers_controller.document.layers[0].name == "Layer 1"
    assert editor.redo_action.text() == "Redo Rename Layer"
    assert editor.redo_action.shortcut().matches(
        QKeySequence.StandardKey.Redo)


def test_transform_confirm_is_one_history_entry_and_actions_are_blocked(
        qapp_fixture, monkeypatch
):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    monkeypatch.setattr(editor.layers_controller, "request_preview", lambda: None)
    gray = np.zeros((3, 4), np.float32)
    rgba = np.zeros((3, 4, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor._start_layer_transform(0)
    editor.layers_controller.move_active_layer_to(4, 5)
    editor.layers_controller.move_active_layer_to(8, 9)
    assert not editor.undo_action.isEnabled()
    editor._confirm_layer_transform()
    assert editor.undo_action.text() == "Undo Transform Layer"
    assert editor.layers_controller.undo()
    assert editor.layers_controller.active_geometry() == (0, 0, 4, 3)


def test_transform_disables_editor_mutations_until_cancel(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)

    editor._start_layer_transform(0)
    assert not editor.tabs.isEnabled()
    editor.layers_controller.move_active_layer_to(8, -2)
    editor._cancel_layer_transform()

    assert editor.tabs.isEnabled()
    assert editor.layers_controller.active_geometry() == (0, 0, 12, 10)


def test_keyboard_nudge_stays_inside_cancel_snapshot(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor._start_layer_transform(0)

    editor.viewport.layer_nudge_requested.emit(10, -1)
    assert editor.layers_controller.active_geometry() == (10, -1, 12, 10)
    editor.viewport.layer_transform_cancel_requested.emit()

    assert editor.layers_controller.active_geometry() == (0, 0, 12, 10)


def test_edge_handle_resizes_one_axis_when_aspect_unlocked(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    gray = np.zeros((10, 12), np.float32)
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 3] = 255
    editor.layers_controller.open_document(gray, rgba)
    editor._start_layer_transform(0)
    editor.layers_panel.lock_aspect_check.setChecked(False)

    editor._begin_layer_resize("e")
    editor._resize_active_layer("e", 5, 7)
    editor._finish_layer_resize()

    assert editor.layers_controller.active_geometry() == (0, 0, 17, 10)


def test_smart_mask_scope_tracks_active_layer_and_zero_layer_state(
        qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor()
    assert editor.panel.smart_mask_panel.scope_label.text() == (
        "Masking: No active layer")
    gray = np.zeros((3, 4), np.float32)
    rgba = np.zeros((3, 4, 4), np.uint8)
    rgba[..., 3] = 255

    editor.layers_controller.open_document(gray, rgba)
    assert editor.panel.smart_mask_panel.scope_label.text() == (
        "Masking: Layer 1 · Subject")
    editor.layers_controller.delete_layer(0)
    editor._refresh_active_layer_mask_scope()

    assert editor.panel.smart_mask_panel.scope_label.text() == (
        "Masking: No active layer")
