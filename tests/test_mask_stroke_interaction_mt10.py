import numpy as np


def _preset():
    return {
        "adjustments": {"contrast": 50},
        "dither": {"style": "None", "scale": 1, "params": {}},
    }


def _controller(panel, **overrides):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController

    gray = np.zeros((8, 8), np.float32)
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., 3] = 255
    values = dict(
        preset_provider=_preset,
        source_provider=lambda: (gray, rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    values.update(overrides)
    return LayersController(panel, registry, **values)


def _ready_controller(qapp_fixture, monkeypatch, **overrides):
    from ditherzam.ui.layers_panel import LayersPanel

    panel = LayersPanel()
    controller = _controller(panel, **overrides)
    requests = []
    monkeypatch.setattr(
        controller, "request_preview",
        lambda: requests.append(controller.document))
    controller.initialize_source_layer()
    controller.reveal_all_raster_mask(replace_existing=False)
    layer = controller.document.layers[0]
    assert controller.set_edit_target(layer.id, "mask")
    requests.clear()
    return panel, controller, requests


def test_stroke_is_isolated_until_one_release_commit(
    qapp_fixture, monkeypatch
):
    from ditherzam.layers import BrushMode, BrushSettings

    dirty = []
    clears = []
    _panel, controller, requests = _ready_controller(
        qapp_fixture, monkeypatch,
        mask_stroke_clear=lambda: clears.append(True),
    )
    controller._mask_stroke_sink = lambda authority, pixels, rect: (
        dirty.append((authority, pixels.copy(), rect)))
    before_document = controller.document
    before_mask = before_document.layers[0].raster_mask
    settings = BrushSettings(4, 100, 100, BrushMode.HIDE)

    assert controller.begin_mask_brush_stroke(2, 2, settings)
    assert controller.continue_mask_brush_stroke(5, 2)
    assert controller.document is before_document
    assert np.all(before_mask.pixels == 255)
    assert requests == []
    assert len(dirty) == 2

    assert controller.finish_mask_brush_stroke()
    after = controller.document.layers[0].raster_mask
    assert after.revision == before_mask.revision + 1
    assert np.any(after.pixels != 255)
    assert controller.undo_label == "Brush Layer Mask"
    assert len(requests) == 1
    # The dirty overlay remains until the scheduled exact frame is accepted.
    assert clears == []


def test_cancel_noop_stale_and_layer_switch_never_publish(
    qapp_fixture, monkeypatch
):
    from ditherzam.layers import BrushMode, BrushSettings

    _panel, controller, requests = _ready_controller(qapp_fixture, monkeypatch)
    original = controller.document
    hide = BrushSettings(3, 100, 100, BrushMode.HIDE)
    assert controller.begin_mask_brush_stroke(2, 2, hide)
    assert controller.cancel_mask_brush_stroke()
    assert controller.document is original
    assert requests == []

    reveal = BrushSettings(3, 100, 100, BrushMode.REVEAL)
    assert controller.begin_mask_brush_stroke(2, 2, reveal)
    assert controller.finish_mask_brush_stroke() is False
    assert controller.document is original
    assert requests == []

    assert controller.begin_mask_brush_stroke(2, 2, hide)
    controller.add_layer()
    # Unrelated mutation is blocked while the stroke remains authoritative.
    assert len(controller.document.layers) == 1
    controller.activate_layer(0)
    assert controller.mask_stroke_active is False


def test_middle_stack_proxy_replays_blends_in_original_order():
    from ditherzam.layers import (
        DirtyRect, LayerRenderProxy, blend_layer, composite_mask_stroke_roi)
    from ditherzam.layers.render import _nearest_source_index_map

    prefix = np.zeros((4, 4, 4), np.uint8)
    prefix[...] = (80, 120, 160, 255)
    selected = np.zeros((4, 4, 4), np.uint8)
    selected[...] = (220, 40, 70, 255)
    above = np.zeros((4, 4, 4), np.uint8)
    above[...] = (30, 240, 100, 180)
    for array in (prefix, selected, above):
        array.flags.writeable = False
    proxy = LayerRenderProxy(
        "middle", np.zeros_like(prefix), selected, 0, 0, "multiply", 70,
        True, prefix, selected, ((above, 0, 0, "screen", 65),),
        _nearest_source_index_map(4, 4), _nearest_source_index_map(4, 4))
    mask = np.arange(16, dtype=np.uint8).reshape(4, 4) * 17

    rect, actual = composite_mask_stroke_roi(
        proxy, mask, 80, DirtyRect(0, 0, 4, 4))
    effective = 255 - ((80 * (255 - mask.astype(np.uint16)) + 50) // 100)
    masked = selected.copy()
    masked[..., 3] = (
        (masked[..., 3].astype(np.uint16) * effective + 127) // 255
    ).astype(np.uint8)
    expected = blend_layer(prefix, masked, "multiply", 70)
    expected = blend_layer(expected, above, "screen", 65)

    assert rect == (0, 0, 4, 4)
    assert np.array_equal(actual, expected)


def test_dirty_proxy_never_resizes_the_full_mask(monkeypatch):
    import ditherzam.layers.render as render_module
    from ditherzam.layers import (
        DirtyRect, LayerRenderProxy, composite_mask_stroke_roi)

    prefix = np.zeros((360, 640, 4), np.uint8)
    selected = np.zeros_like(prefix)
    selected[..., 3] = 255
    proxy = LayerRenderProxy(
        "layer", prefix, selected, 0, 0, "normal", 100, True,
        prefix, selected, (),
        render_module._nearest_source_index_map(3840, 640),
        render_module._nearest_source_index_map(2160, 360))
    source_mask = np.full((2160, 3840), 255, np.uint8)
    source_mask[100:1124, 200:1224] = 0
    expected_mask = render_module._resize_channel_nearest(
        source_mask, (360, 640))
    monkeypatch.setattr(
        render_module, "_resize_channel_nearest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full mask resize is forbidden during a stamp")))

    rect, rgba = composite_mask_stroke_roi(
        proxy, source_mask, 100, DirtyRect(200, 100, 1224, 1124))
    assert rect[2] > rect[0] and rect[3] > rect[1]
    assert rgba.nbytes < 2 * 1024 * 1024
    x0, y0, x1, y1 = rect
    assert np.array_equal(rgba[..., 3], expected_mask[y0:y1, x0:x1])


def test_overlay_reuses_one_capped_item_for_1000_updates(qapp_fixture):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPixmap
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    patch = QPixmap(32, 32)
    patch.fill()
    before = len(view.scene().items())
    for step in range(1000):
        view.install_mask_stroke_roi(
            patch, QRectF(step % 608, step % 328, 32, 32),
            (3840, 2160), (3840, 2160))
    assert len(view.scene().items()) == before + 1
    assert view._mask_stroke_surface.width() <= 720
    assert view._mask_stroke_surface.sizeInBytes() <= 2 * 1024 * 1024
    view.clear_mask_stroke_overlay()
    assert len(view.scene().items()) == before


def test_proxy_index_maps_match_full_pillow_for_adversarial_clipped_cases():
    from ditherzam.layers import (
        DirtyRect, LayerRenderProxy, composite_mask_stroke_roi)
    from ditherzam.layers.render import (
        _nearest_source_index_map, _resize_channel_nearest)

    rng = np.random.default_rng(20260726)
    cases = [
        (240, 1, 45, 1, -7, 0, DirtyRect(0, 0, 240, 1)),
        (192, 37, 72, 13, -19, -4, DirtyRect(31, 2, 181, 36)),
        (3840, 2160, 720, 405, -113, 27, DirtyRect(777, 333, 1801, 1357)),
        (11, 7, 8, 5, 3, -2, DirtyRect(2, 1, 10, 7)),
    ]
    for _ in range(20):
        sw = int(rng.integers(2, 260))
        sh = int(rng.integers(2, 180))
        tw = int(rng.integers(1, 150))
        th = int(rng.integers(1, 110))
        x = int(rng.integers(-tw, tw))
        y = int(rng.integers(-th, th))
        x0 = int(rng.integers(0, sw))
        y0 = int(rng.integers(0, sh))
        x1 = int(rng.integers(x0 + 1, sw + 1))
        y1 = int(rng.integers(y0 + 1, sh + 1))
        cases.append((sw, sh, tw, th, x, y, DirtyRect(x0, y0, x1, y1)))

    for sw, sh, tw, th, x, y, dirty in cases:
        mask = rng.integers(0, 256, (sh, sw), dtype=np.uint8)
        canvas = np.zeros((max(3, th + 9), max(3, tw + 11), 4), np.uint8)
        selected = np.zeros((th, tw, 4), np.uint8)
        selected[..., :3] = 91
        selected[..., 3] = 255
        proxy = LayerRenderProxy(
            "layer", canvas, selected, x, y, "normal", 100, True,
            canvas, selected, (),
            _nearest_source_index_map(sw, tw),
            _nearest_source_index_map(sh, th))
        rect, actual = composite_mask_stroke_roi(proxy, mask, 100, dirty)
        dx0, dy0, dx1, dy1 = rect
        if actual.size == 0:
            continue
        full = _resize_channel_nearest(mask, (th, tw))
        expected = full[dy0 - y:dy1 - y, dx0 - x:dx1 - x]
        assert np.array_equal(actual[..., 3], expected), (
            sw, sh, tw, th, x, y, dirty, rect)
