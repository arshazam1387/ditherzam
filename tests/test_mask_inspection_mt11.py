import numpy as np


def _proxy(*, alpha, x=0, y=0, visible=True, opacity=100, canvas_shape=None):
    from ditherzam.layers import LayerRenderProxy
    from ditherzam.layers.render import _nearest_source_index_map

    alpha = np.asarray(alpha, dtype=np.uint8)
    h, w = alpha.shape
    ch, cw = canvas_shape or (h, w)
    composite = np.zeros((ch, cw, 4), np.uint8)
    composite[...] = (40, 80, 120, 200)
    selected = np.zeros((h, w, 4), np.uint8)
    selected[..., :3] = 99
    selected[..., 3] = alpha
    for value in (composite, selected):
        value.flags.writeable = False
    return LayerRenderProxy(
        "layer", composite, selected, x, y, "multiply", opacity, visible,
        composite, selected, (),
        _nearest_source_index_map(w, w), _nearest_source_index_map(h, h))


def test_mask_only_exact_endpoints_soft_density_alpha_visibility_and_inputs():
    from ditherzam.layers import InspectionMode, RasterLayerMask, inspect_mask

    alpha = np.array([[255, 128, 64], [255, 255, 255]], np.uint8)
    pixels = np.array([[255, 128, 0], [10, 200, 30]], np.uint8)
    proxy = _proxy(alpha=alpha)
    composite = np.array(proxy.background_rgba, copy=True)
    composite.flags.writeable = False
    before_pixels = pixels.copy()
    mask = RasterLayerMask(pixels, density=50)

    actual = inspect_mask(composite, proxy, mask, InspectionMode.MASK_ONLY)
    effective = 255 - ((50 * (255 - pixels.astype(np.uint16)) + 50) // 100)
    expected = ((alpha.astype(np.uint16) * effective + 127) // 255).astype(np.uint8)
    assert np.array_equal(actual[..., 0], expected)
    assert np.array_equal(actual[..., 1], expected)
    assert np.array_equal(actual[..., 2], expected)
    assert np.all(actual[..., 3] == 255)
    assert np.array_equal(pixels, before_pixels)
    assert not composite.flags.writeable

    hidden = inspect_mask(
        composite, _proxy(alpha=alpha, visible=False), mask,
        InspectionMode.MASK_ONLY)
    assert np.all(hidden[..., :3] == 0)


def test_inspection_geometry_maps_clipping_and_cap_are_exact():
    from PIL import Image
    from ditherzam.layers import (
        InspectionMode, LayerRenderProxy, RasterLayerMask, inspect_mask)
    from ditherzam.layers.render import _nearest_source_index_map

    source = np.array([[0, 64, 128], [192, 224, 255]], np.uint8)
    selected = np.zeros((4, 6, 4), np.uint8)
    selected[..., 3] = 255
    canvas = np.zeros((3, 5, 4), np.uint8)
    proxy = LayerRenderProxy(
        "layer", canvas, selected, -2, 1, "screen", 100, True,
        canvas, selected, (),
        _nearest_source_index_map(3, 6),
        _nearest_source_index_map(2, 4))
    actual = inspect_mask(
        canvas, proxy, RasterLayerMask(source),
        InspectionMode.MASK_ONLY)
    resized = np.asarray(Image.fromarray(source).resize(
        (6, 4), resample=Image.Resampling.NEAREST))
    expected = np.zeros((3, 5), np.uint8)
    expected[1:3, 0:4] = resized[0:2, 2:6]
    assert np.array_equal(actual[..., 0], expected)


def test_normal_is_identity_and_red_overlay_marks_hidden_preserving_alpha():
    from ditherzam.layers import (
        INSPECTION_RED, INSPECTION_RED_OPACITY, InspectionMode,
        RasterLayerMask, inspect_mask)

    proxy = _proxy(alpha=np.array([[255, 128]], np.uint8))
    composite = np.array([[[20, 40, 60, 7], [100, 80, 60, 231]]], np.uint8)
    mask = RasterLayerMask(np.array([[0, 255]], np.uint8))
    normal = inspect_mask(composite, proxy, mask, InspectionMode.NORMAL)
    assert normal is composite

    red = inspect_mask(composite, proxy, mask, InspectionMode.RED_OVERLAY)
    amount = np.array([[INSPECTION_RED_OPACITY, 0]], np.uint16)
    expected_rgb = (
        composite[..., :3].astype(np.uint16) * (255 - amount[..., None])
        + np.asarray(INSPECTION_RED, np.uint16) * amount[..., None] + 127
    ) // 255
    assert np.array_equal(red[..., :3], expected_rgb.astype(np.uint8))
    assert np.array_equal(red[..., 3], composite[..., 3])


def test_disabled_and_zero_density_are_full_reveal_silhouettes():
    from ditherzam.layers import InspectionMode, RasterLayerMask, inspect_mask

    alpha = np.array([[0, 64, 255]], np.uint8)
    proxy = _proxy(alpha=alpha)
    composite = proxy.background_rgba
    for mask in (
        RasterLayerMask(np.zeros_like(alpha), enabled=False),
        RasterLayerMask(np.zeros_like(alpha), density=0),
    ):
        actual = inspect_mask(
            composite, proxy, mask, InspectionMode.MASK_ONLY)
        assert np.array_equal(actual[..., 0], alpha)


def test_inspection_coverage_applies_layer_opacity_half_up():
    from ditherzam.layers import InspectionMode, RasterLayerMask, inspect_mask

    alpha = np.array([[255, 128, 1]], np.uint8)
    mask = RasterLayerMask(np.full_like(alpha, 255))
    for opacity, expected in (
        (0, np.array([[0, 0, 0]], np.uint8)),
        (50, np.array([[128, 64, 1]], np.uint8)),
        (100, alpha),
    ):
        proxy = _proxy(alpha=alpha, opacity=opacity)
        result = inspect_mask(
            proxy.background_rgba, proxy, mask, InspectionMode.MASK_ONLY)
        assert np.array_equal(result[..., 0], expected)


def _ready_controller(qapp_fixture, monkeypatch):
    from ditherzam.dithering import registry
    from ditherzam.layers import render_layer_document_with_proxy
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    gray = np.arange(64, dtype=np.float32).reshape(8, 8)
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., :3] = 80
    rgba[..., 3] = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    frames = []
    controller = LayersController(
        panel, registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (gray, rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda frame: frames.append(np.array(frame, copy=True)),
        apply_preset=lambda _preset: None)
    requests = []
    monkeypatch.setattr(
        controller, "request_preview", lambda: requests.append(True))
    controller.initialize_source_layer()
    controller.reveal_all_raster_mask(replace_existing=False)
    controller.set_edit_target(controller.document.layers[0].id, "mask")
    requests.clear()
    rendered = render_layer_document_with_proxy(
        controller.document, registry,
        layer_id=controller.document.layers[0].id,
        target_max_side=480, look_cache=controller._look_cache)
    controller._latest_proxy = rendered.proxy
    accepted = np.array(rendered.composite_rgba, copy=True)
    accepted.flags.writeable = False
    controller._accepted_composite = accepted
    return panel, controller, frames, requests


def test_controller_toggle_is_synchronous_ui_only_and_brush_locked(
    qapp_fixture, monkeypatch
):
    from ditherzam.layers import BrushMode, BrushSettings, InspectionMode

    panel, controller, frames, requests = _ready_controller(
        qapp_fixture, monkeypatch)
    before = (
        controller.document, controller.document.revision,
        controller._history.entry_count, len(controller._thumbnail_cache),
        controller._generation, dict(controller._look_cache.metrics))
    assert controller.set_inspection_mode(InspectionMode.MASK_ONLY)
    assert len(frames) == 1
    assert requests == []
    assert (
        controller.document, controller.document.revision,
        controller._history.entry_count, len(controller._thumbnail_cache),
        controller._generation, dict(controller._look_cache.metrics)
    ) == before
    assert np.all(frames[-1][..., 3] == 255)
    assert not controller.begin_mask_brush_stroke(
        1, 1, BrushSettings(3, 100, 100, BrushMode.HIDE))
    assert controller.toggle_red_inspection()
    assert controller.inspection_mode is InspectionMode.RED_OVERLAY
    assert panel.inspection_combo.currentText() == "Red Overlay"


def test_panel_inspection_accessibility_absent_disabled_and_transform(
    qapp_fixture
):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    panel.set_layers(
        [("a", "Layer", True, 100, "normal", 0, 0, 8, 8)], 0)
    panel.set_mask_state(False)
    assert not panel.inspection_combo.isEnabled()
    assert panel.inspection_combo.currentText() == "Normal"
    assert panel.inspection_combo.accessibleName() == (
        "Raster mask inspection view")
    assert "black and white" in panel.inspection_combo.accessibleDescription()
    panel.set_mask_state(True, enabled=False, density=42)
    assert panel.inspection_combo.isEnabled()
    assert "full-reveal" in panel.inspection_combo.toolTip()
    panel.set_transform_mode(True)
    assert not panel.inspection_combo.isEnabled()


def test_current_mode_applies_to_new_terminal_and_stale_terminal_is_ignored(
    qapp_fixture, monkeypatch
):
    from ditherzam.layers import InspectionMode

    _panel, controller, frames, _requests = _ready_controller(
        qapp_fixture, monkeypatch)
    proxy = controller._latest_proxy
    composite = controller._accepted_composite
    controller._accepted_composite = None
    controller._latest_proxy = None
    assert controller.set_inspection_mode(InspectionMode.MASK_ONLY)
    assert frames == []
    generation = controller._generation
    controller._active_generation = generation
    controller._busy = True
    controller._preview_finished((composite, (), proxy), generation)
    assert np.all(frames[-1][..., 3] == 255)
    count = len(frames)
    controller._preview_finished((np.zeros_like(composite), (), proxy), generation - 1)
    assert len(frames) == count


def test_transform_forces_normal_and_restores_mode(qapp_fixture, monkeypatch):
    from ditherzam.layers import InspectionMode

    panel, controller, _frames, _requests = _ready_controller(
        qapp_fixture, monkeypatch)
    assert controller.set_inspection_mode(InspectionMode.RED_OVERLAY)
    assert controller.begin_transform_transaction()
    assert controller.inspection_mode is InspectionMode.NORMAL
    assert panel.inspection_combo.currentText() == "Normal"
    assert controller.confirm_transform_transaction() is False
    assert controller.inspection_mode is InspectionMode.RED_OVERLAY


def test_toggles_preserve_thumbnail_and_all_export_bytes(
    qapp_fixture, monkeypatch, tmp_path
):
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QPixmap
    from ditherzam.layers import InspectionMode
    from ditherzam.ui.convert import numpy_to_qimage
    from ditherzam.ui.layers_controller import _thumbnail_key

    panel, controller, _frames, requests = _ready_controller(
        qapp_fixture, monkeypatch)
    layer = controller.document.layers[0]
    pixmap = QPixmap.fromImage(numpy_to_qimage(
        np.full((3, 3, 4), 117, np.uint8))).copy()
    key = _thumbnail_key(layer)
    controller._thumbnail_cache[key] = pixmap
    panel.set_layer_thumbnail(layer.id, pixmap)

    def pixmap_bytes(value):
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        assert value.save(buffer, "PNG")
        return bytes(data)

    thumb_key = pixmap.cacheKey()
    thumb_bytes = pixmap_bytes(pixmap)
    before_doc = controller.document
    before_history = controller._history.entry_count
    before_generation = controller._generation
    png0, jpg0, mask0 = (
        tmp_path / "before.png", tmp_path / "before.jpg",
        tmp_path / "before-mask.png")
    assert controller.export_current(png0)
    assert controller.export_current(jpg0)
    assert controller.export_active_raster_mask(mask0)
    baseline = (png0.read_bytes(), jpg0.read_bytes(), mask0.read_bytes())
    before_metrics = dict(controller._look_cache.metrics)

    assert controller.set_inspection_mode(InspectionMode.MASK_ONLY)
    assert controller.set_inspection_mode(InspectionMode.RED_OVERLAY)
    assert controller.set_inspection_mode(InspectionMode.NORMAL)
    png1, jpg1, mask1 = (
        tmp_path / "after.png", tmp_path / "after.jpg",
        tmp_path / "after-mask.png")
    assert controller.export_current(png1)
    assert controller.export_current(jpg1)
    assert controller.export_active_raster_mask(mask1)
    assert baseline == (png1.read_bytes(), jpg1.read_bytes(), mask1.read_bytes())
    assert controller.document is before_doc
    assert controller._history.entry_count == before_history
    assert controller._generation == before_generation
    assert dict(controller._look_cache.metrics) == before_metrics
    assert requests == []
    assert controller._thumbnail_cache[key] is pixmap
    assert pixmap.cacheKey() == thumb_key
    assert pixmap_bytes(pixmap) == thumb_bytes


def test_backslash_guard_and_mask_target_scope(qapp_fixture, monkeypatch):
    from PySide6.QtWidgets import QLineEdit, QSpinBox
    from ditherzam.layers import InspectionMode
    from ditherzam.ui.main_window import ImageEditor

    _panel, controller, _frames, _requests = _ready_controller(
        qapp_fixture, monkeypatch)
    fake = type("Editor", (), {"layers_controller": controller})()
    for widget in (QLineEdit(), QSpinBox()):
        widget.show()
        widget.setFocus()
        qapp_fixture.processEvents()
        ImageEditor._toggle_mask_inspection(fake)
        assert controller.inspection_mode is InspectionMode.NORMAL
        widget.close()
    qapp_fixture.processEvents()
    controller.set_edit_target(controller.document.layers[0].id, "layer")
    ImageEditor._toggle_mask_inspection(fake)
    assert controller.inspection_mode is InspectionMode.NORMAL
    controller.set_edit_target(controller.document.layers[0].id, "mask")
    ImageEditor._toggle_mask_inspection(fake)
    assert controller.inspection_mode is InspectionMode.RED_OVERLAY


def test_viewport_mapping_is_unchanged_across_inspection_modes(
    qapp_fixture, monkeypatch
):
    from PySide6.QtGui import QPixmap
    from ditherzam.layers import InspectionMode
    from ditherzam.ui.convert import numpy_to_qimage
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.resize(240, 180)
    view.show()

    def sink(frame):
        view.set_pixmap(
            QPixmap.fromImage(numpy_to_qimage(frame)),
            logical_size=(8, 8), refit=False)

    _panel, controller, _frames, _requests = _ready_controller(
        qapp_fixture, monkeypatch)
    controller._frame_sink = sink
    controller._publish_accepted_inspection()
    view.scale(2.25, 2.25)
    view.horizontalScrollBar().setValue(3)
    view.verticalScrollBar().setValue(4)
    before = (
        view.transform(), view.horizontalScrollBar().value(),
        view.verticalScrollBar().value(), view.sceneRect())
    for mode in (
        InspectionMode.RED_OVERLAY, InspectionMode.MASK_ONLY,
        InspectionMode.NORMAL):
        controller.set_inspection_mode(mode)
        assert (
            view.transform(), view.horizontalScrollBar().value(),
            view.verticalScrollBar().value(), view.sceneRect()) == before
    view.close()


def test_raster_inspection_suppresses_smart_red_overlay(qapp_fixture):
    from ditherzam.layers import InspectionMode
    from ditherzam.ui.main_window import ImageEditor

    calls = []

    class Stopper:
        def stop(self):
            pass

    class Scheduler:
        def invalidate(self):
            pass

    class Viewport:
        def set_pixmap(self, *_args, **_kwargs):
            pass

    fake = type("Editor", (), {})()
    fake._debounce = fake._settle = fake._zoom_debounce = Stopper()
    fake._scheduler = Scheduler()
    fake.viewport = Viewport()
    fake.layers_controller = type("Controller", (), {
        "inspection_mode": InspectionMode.NORMAL})()
    fake._apply_active_layer_mask_overlay = lambda frame: (
        calls.append(True) or frame)
    fake._layer_reference_size = lambda: (2, 2)
    fake._sync_layer_drag_target = lambda: None
    frame = np.zeros((2, 2, 4), np.uint8)
    ImageEditor._show_layer_frame(fake, frame)
    assert calls == [True]
    fake.layers_controller.inspection_mode = InspectionMode.RED_OVERLAY
    ImageEditor._show_layer_frame(fake, frame)
    assert calls == [True]
