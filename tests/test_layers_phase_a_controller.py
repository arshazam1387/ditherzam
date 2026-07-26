import numpy as np


def _preset(contrast=50):
    return {
        "adjustments": {"contrast": contrast},
        "dither": {"style": "None", "scale": 1, "params": {}},
    }


def _pixels(value, shape=(3, 4), alpha=255):
    gray = np.full(shape, value, np.float32)
    rgba = np.full((*shape, 4), value, np.uint8)
    rgba[..., 3] = alpha
    return gray, rgba


def _controller(panel, **overrides):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController

    values = dict(
        preset_provider=lambda: _preset(),
        source_provider=lambda: None,
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
        apply_source=lambda _gray, _rgba, _probability: None,
    )
    values.update(overrides)
    return LayersController(panel, registry, **values)


def test_open_then_place_preserves_first_layer_pixels(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    gray_a, rgba_a = _pixels(10)
    gray_b, rgba_b = _pixels(200)
    controller.open_document(gray_a, rgba_a)
    first = controller.document.layers[0]
    first_bytes = first.source.rgba.tobytes()
    controller.place_source(gray_b, rgba_b)
    assert controller.document.canvas.width == 4
    assert controller.document.canvas.height == 3
    assert len(controller.document.layers) == 2
    assert controller.document.layers[0] is first
    assert controller.document.layers[0].source.rgba.tobytes() == first_bytes
    assert np.all(controller.document.layers[1].source.rgba[..., :3] == 200)


def test_place_centers_a_differently_sized_native_source(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    gray_a, rgba_a = _pixels(10, shape=(6, 8))
    gray_b, rgba_b = _pixels(200, shape=(2, 4))
    controller.open_document(gray_a, rgba_a)
    controller.place_source(gray_b, rgba_b)
    placed = controller.document.layers[-1]
    assert placed.source.rgba.shape == (2, 4, 4)
    assert placed.transform.x == 2.0
    assert placed.transform.y == 2.0


def test_blank_duplicate_and_delete_to_zero_preserve_document(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    gray, rgba = _pixels(40)
    controller.open_document(gray, rgba)
    controller.new_blank_layer()
    blank = controller.document.layers[-1]
    assert np.count_nonzero(blank.source.rgba) == 0
    controller.duplicate_layer(0)
    assert controller.document.layers[0].source is controller.document.layers[1].source
    while controller.document.layers:
        controller.delete_layer(0)
    assert controller.document.canvas.width == 4
    assert controller.document.canvas.height == 3
    assert controller.active_index is None


def test_activate_layer_publishes_its_pixels_probability_and_look(qapp_fixture):
    from ditherzam.ui.layers_panel import LayersPanel

    applied_sources = []
    applied_presets = []
    controller = _controller(
        LayersPanel(),
        apply_source=lambda gray, rgba, probability:
            applied_sources.append((gray, rgba, probability)),
        apply_preset=applied_presets.append,
    )
    gray_a, rgba_a = _pixels(10)
    gray_b, rgba_b = _pixels(90)
    controller.open_document(gray_a, rgba_a)
    controller.place_source(gray_b, rgba_b)
    controller.activate_layer(0)
    assert np.array_equal(applied_sources[-1][0], gray_a)
    assert np.array_equal(applied_sources[-1][1], rgba_a)
    assert applied_sources[-1][2] is None
    assert applied_presets[-1] == _preset()


def test_stale_frame_and_thumbnail_payload_are_rejected(qapp_fixture, monkeypatch):
    from ditherzam.ui.layers_panel import LayersPanel

    frames = []
    panel = LayersPanel()
    controller = _controller(panel, frame_sink=frames.append)
    gray, rgba = _pixels(20)
    controller.open_document(gray, rgba)
    controller._generation = 7
    stale_rgba = np.full((2, 2, 4), 255, np.uint8)
    stale_key = ("missing", "old-source", ("old-look",), 0, object())
    controller._preview_finished((stale_rgba, ((stale_key, stale_rgba),)), 6)
    assert frames == []
    assert controller._thumbnail_cache == {}


def test_export_uses_current_document_graph(qapp_fixture, monkeypatch, tmp_path):
    from ditherzam.ui.layers_panel import LayersPanel
    import ditherzam.ui.layers_controller as module

    controller = _controller(LayersPanel())
    gray, rgba = _pixels(70)
    controller.open_document(gray, rgba)
    seen = []
    monkeypatch.setattr(
        module, "render_layer_document",
        lambda document, registry, **_kwargs: seen.append(document)
        or np.zeros((3, 4, 4), np.uint8),
    )
    monkeypatch.setattr(
        module, "export_frame",
        lambda image, path: tmp_path / "out.png",
    )
    controller.export_current(tmp_path / "out.png")
    assert seen == [controller.document]
