import numpy as np
import time


def test_layers_panel_exposes_accessible_brush_and_selection_controls(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers([("a", "Layer 1", True, 100, "normal")], 0)
    assert [panel.selection_shape_combo.itemText(i) for i in range(4)] == [
        "Rectangle", "Ellipse", "Polygon", "Freehand"]
    assert panel.brush_tip_combo.accessibleName() == "Mask brush tip"
    assert panel.brush_spacing_spin.accessibleName() == "Mask brush stamp spacing"
    seen = []
    panel.brush_settings_changed.connect(lambda *args: seen.append(args))
    panel.brush_tip_combo.setCurrentText("Diamond")
    panel.brush_spacing_spin.setValue(40)
    assert seen[-1] == ("diamond", 32, 100, 100, 40)


def test_layers_panel_mask_workspace_stays_inside_compact_dock(qapp_fixture):
    from PySide6.QtCore import Qt
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers([
        ("a", "Layer 1", True, 100, "normal", 0, 0, 800, 600, True, True)
    ], 0)
    panel.set_mask_state(True, enabled=True)
    panel.resize(320, 700)
    panel.show()
    panel.editor_tabs.setCurrentIndex(1)
    qapp_fixture.processEvents()

    assert panel.minimumWidth() == 280
    assert panel.maximumWidth() == 320
    assert panel.minimumSizeHint().width() <= 320
    assert panel.editor_tabs.count() == 2
    assert panel.editor_tabs.tabText(0) == "Layer"
    assert panel.editor_tabs.tabText(1) == "Mask"
    assert panel.mask_scroll.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    assert panel.mask_scroll.widgetResizable()
    assert panel.pointer_btn.text() == "Pointer"
    assert panel.paint_mask_btn.text() == "Paint Mask"
    assert panel.brush_mode_combo.currentText() == "Reveal"
    assert panel._row_targets["a"][1].text() == "Mask"
    panel.close()


def test_mask_pointer_and_paint_controls_publish_explicit_tool_intent(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers([
        ("a", "Layer 1", True, 100, "normal", 0, 0, 64, 64, True, True)
    ], 0)
    panel.set_mask_state(True, enabled=True)
    events = []
    panel.pointer_requested.connect(lambda: events.append(("pointer",)))
    panel.mask_paint_requested.connect(
        lambda enabled: events.append(("paint", enabled)))
    panel.brush_mode_changed.connect(
        lambda mode: events.append(("mode", mode)))

    panel.paint_mask_btn.click()
    panel.brush_mode_combo.setCurrentText("Hide")
    panel.pointer_btn.click()

    assert events == [
        ("paint", True), ("mode", "hide"), ("pointer",),
        ("paint", False),
    ]


def test_mask_paint_requires_an_enabled_mask_and_disarms_when_disabled(
    qapp_fixture,
):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers([
        ("a", "Layer 1", True, 100, "normal", 0, 0, 64, 64, False, False)
    ], 0)
    stopped = []
    panel.mask_paint_requested.connect(stopped.append)

    panel.set_mask_state(False)
    assert not panel.paint_mask_btn.isEnabled()
    assert "Create or import" in panel.paint_mask_btn.toolTip()

    panel.set_mask_state(True, enabled=False)
    assert not panel.paint_mask_btn.isEnabled()
    assert "Enable" in panel.paint_mask_btn.toolTip()

    panel.set_mask_state(True, enabled=True)
    assert panel.paint_mask_btn.isEnabled()
    panel.set_mask_paint_active(True)
    panel.set_mask_state(True, enabled=False)

    assert not panel.paint_mask_btn.isChecked()
    assert stopped == [False]


def test_selection_refinement_controls_publish_operation_and_radius(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers([("a", "Layer 1", True, 100, "normal")], 0)
    seen = []
    panel.selection_refine_requested.connect(lambda *args: seen.append(args))
    panel.selection_radius_spin.setValue(7)
    panel.selection_feather_btn.click()
    assert seen == [("feather", 7)]


def test_controller_path_and_refinement_are_temporary(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.layers import SelectionOperation
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((8, 8, 4), np.uint8)
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((8, 8), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    controller.initialize_source_layer()
    document = controller.document
    assert controller.update_path_selection(
        "polygon", [(1, 1), (6, 1), (3, 6)], SelectionOperation.REPLACE)
    assert controller.document is document
    assert controller.refine_temporary_selection("feather", 2)
    assert controller.document is document
    assert controller.refine_temporary_selection("all", 2)
    assert np.all(controller.temporary_selection.pixels == 255)


def test_viewport_accepts_path_selection_tools_and_transform_can_clear_them(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.set_selection_tool("polygon")
    assert view._selection_tool == "polygon"
    view.set_selection_tool("freehand")
    assert view._selection_tool == "freehand"
    view.set_selection_tool(None)
    assert view._selection_tool is None
    assert view._selection_points == []


def test_color_range_panel_publishes_pick_preview_confirm_and_cancel(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers([("a", "Layer 1", True, 100, "normal")], 0)
    events = []
    panel.color_range_pick_requested.connect(lambda: events.append(("pick",)))
    panel.color_range_changed.connect(
        lambda tolerance, softness: events.append(("change", tolerance, softness)))
    panel.color_range_confirmed.connect(lambda: events.append(("confirm",)))
    panel.color_range_cancelled.connect(lambda: events.append(("cancel",)))

    panel.color_range_btn.click()
    panel.color_range_tolerance_spin.setValue(30)
    panel.color_range_softness_spin.setValue(12)
    panel.color_range_confirm_btn.click()
    panel.color_range_cancel_btn.click()

    assert events == [
        ("pick",), ("change", 30, 16), ("change", 30, 12),
        ("confirm",), ("cancel",),
    ]
    assert panel.color_range_btn.accessibleName() == "Pick selection color from canvas"


def test_controller_color_range_preview_is_reversible_and_operation_aware(
    qapp_fixture,
):
    from ditherzam.dithering import registry
    from ditherzam.layers import SelectionOperation
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.array([[[10, 20, 30, 255], [50, 60, 70, 128]]], np.uint8)
    overlays = []
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((1, 2), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
        selection_overlay_sink=lambda pixels: overlays.append(pixels.copy()),
    )
    controller.initialize_source_layer()
    controller.refine_temporary_selection("all")
    original = controller.temporary_selection

    assert controller.begin_color_range_selection(
        1.5, 0.5, SelectionOperation.SUBTRACT, tolerance=0, softness=0)
    assert controller.temporary_selection is original
    assert overlays[-1].tolist() == [[255, 127]]
    assert controller.update_color_range_selection(70, 10)
    assert controller.cancel_color_range_selection()
    assert controller.temporary_selection is original
    assert overlays[-1].tolist() == [[255, 255]]

    assert controller.begin_color_range_selection(
        0.5, 0.5, SelectionOperation.REPLACE, tolerance=0, softness=0)
    assert controller.confirm_color_range_selection()
    assert controller.temporary_selection.pixels.tolist() == [[255, 0]]


def test_color_range_rejects_outside_click_and_document_replacement_cancels(
    qapp_fixture,
):
    from ditherzam.dithering import registry
    from ditherzam.layers import SelectionOperation
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.full((2, 2, 4), 255, np.uint8)
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((2, 2), np.float32), rgba, None),
        cap_provider=lambda: 480, frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    controller.initialize_source_layer()
    assert not controller.begin_color_range_selection(
        -1, -1, SelectionOperation.REPLACE, tolerance=0, softness=0)
    assert controller.begin_color_range_selection(
        .5, .5, SelectionOperation.REPLACE, tolerance=0, softness=0)
    controller.open_document(np.zeros((2, 2), np.float32), rgba)
    assert controller.temporary_selection is None
    assert not controller.update_color_range_selection(20, 20)


def test_viewport_color_range_click_and_escape_are_explicit(qapp_fixture):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.resize(200, 200)
    view.show()
    picked = []
    cancelled = []
    view.color_range_picked.connect(lambda x, y: picked.append((x, y)))
    view.selection_cancel_requested.connect(lambda: cancelled.append(True))
    view.set_selection_tool("color_range")
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                     pos=view.viewport().rect().center())
    assert len(picked) == 1
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert cancelled == [True]


def test_large_selection_refinement_is_nonblocking_latest_wins_and_stale_safe(
    qapp_fixture, monkeypatch,
):
    from ditherzam.dithering import registry
    from ditherzam.layers import TemporarySelection
    from ditherzam.ui import layers_controller as module
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.full((800, 800, 4), 255, np.uint8)
    overlays = []
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (
            np.zeros((800, 800), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
        selection_overlay_sink=lambda pixels: overlays.append(pixels.copy()),
    )
    controller.initialize_source_layer()
    controller._selection = TemporarySelection(
        np.zeros((800, 800), dtype=np.uint8))

    def delayed_grow(_selection, radius):
        time.sleep(0.08 if radius == 1 else 0.01)
        return TemporarySelection(
            np.full((800, 800), radius, dtype=np.uint8))

    monkeypatch.setattr(module, "grow_selection", delayed_grow)
    started = time.monotonic()
    assert controller.refine_temporary_selection("grow", 1)
    assert controller.refine_temporary_selection("grow", 2)
    assert time.monotonic() - started < 0.05

    deadline = time.monotonic() + 2.0
    while controller.selection_work_pending and time.monotonic() < deadline:
        qapp_fixture.processEvents()
        time.sleep(0.005)
    qapp_fixture.processEvents()
    assert not controller.selection_work_pending
    assert np.all(controller.temporary_selection.pixels == 2)
    assert overlays and all(not np.all(item == 1) for item in overlays)

    assert controller.refine_temporary_selection("grow", 1)
    controller.open_document(np.zeros((800, 800), np.float32), rgba)
    deadline = time.monotonic() + 2.0
    while controller.selection_work_pending and time.monotonic() < deadline:
        qapp_fixture.processEvents()
        time.sleep(0.005)
    qapp_fixture.processEvents()
    assert controller.temporary_selection is None

    def delayed_color(_rgba, _target, *, tolerance, softness):
        del softness
        time.sleep(0.08 if tolerance == 1 else 0.01)
        return TemporarySelection(
            np.full((800, 800), tolerance, dtype=np.uint8))

    from ditherzam.layers import SelectionOperation
    monkeypatch.setattr(module, "select_color_range", delayed_color)
    started = time.monotonic()
    assert controller.begin_color_range_selection(
        0.5, 0.5, SelectionOperation.REPLACE,
        tolerance=1, softness=0)
    assert controller.update_color_range_selection(2, 0)
    assert controller.confirm_color_range_selection()
    assert time.monotonic() - started < 0.05
    deadline = time.monotonic() + 2.0
    while controller.selection_work_pending and time.monotonic() < deadline:
        qapp_fixture.processEvents()
        time.sleep(0.005)
    qapp_fixture.processEvents()
    assert controller.temporary_selection is not None
    assert np.all(controller.temporary_selection.pixels == 2)
    assert controller._color_range_session is None
    controller.shutdown()
    controller._pool.waitForDone(5_000)
    qapp_fixture.processEvents()
