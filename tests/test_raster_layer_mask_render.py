from dataclasses import replace

import numpy as np
import pytest

from ditherzam.composition import Look
from ditherzam.layers import (
    CanvasSpec,
    Layer,
    LayerDocument,
    LayerSource,
    LayerTransform,
    RasterLayerMask,
    render_layer_document,
    render_layer_document_with_proxy,
)
from ditherzam.layers.render import (
    _apply_raster_mask,
    _normalize_layer_alpha,
)
from ditherzam.render import RenderCancelled


def _look(name="Layer"):
    return Look(name, {"dither": {"style": "None"}})


def _source(alpha, *, shape=None):
    alpha = np.asarray(alpha, dtype=np.uint8)
    if shape is not None:
        alpha = np.full(shape, alpha, dtype=np.uint8)
    gray = np.full(alpha.shape, 90, dtype=np.float32)
    rgba = np.full((*alpha.shape, 4), 90, dtype=np.uint8)
    rgba[..., 3] = alpha
    return LayerSource(gray, rgba)


def _mask(pixels, *, enabled=True, density=100):
    return RasterLayerMask(
        np.asarray(pixels, dtype=np.uint8),
        enabled=enabled,
        density=density,
    )


def test_rgb_completed_look_attaches_source_alpha_directly_at_target_geometry():
    rendered = np.full((2, 3, 3), 17, dtype=np.uint8)
    source = np.zeros((3, 2, 4), dtype=np.uint8)
    source[..., 3] = np.array([[0, 20], [80, 100], [240, 255]], np.uint8)

    result = _normalize_layer_alpha(rendered, source, (2, 3))

    expected = np.array([[0, 20, 20], [240, 255, 255]], dtype=np.uint8)
    assert np.array_equal(result[..., :3], rendered)
    assert np.array_equal(result[..., 3], expected)


def test_rgba_smart_alpha_is_preserved_without_double_applying_source_alpha():
    rendered = np.full((2, 2, 4), 33, dtype=np.uint8)
    rendered[..., 3] = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    source = np.full((2, 2, 4), 200, dtype=np.uint8)
    source[..., 3] = 128

    result = _normalize_layer_alpha(rendered, source, (2, 2))

    assert result is rendered
    assert np.array_equal(result[..., 3], [[0, 64], [128, 255]])


def test_raster_mask_exact_half_up_density_and_alpha_formulas():
    rendered = np.zeros((1, 4, 4), dtype=np.uint8)
    rendered[..., 3] = np.array([[1, 127, 128, 255]], dtype=np.uint8)
    mask = _mask([[0, 127, 128, 254]], density=50)

    result = _apply_raster_mask(rendered, mask, (1, 4))

    effective = 255 - ((50 * (255 - mask.pixels.astype(np.uint32)) + 50) // 100)
    expected = (
        rendered[..., 3].astype(np.uint32) * effective + 127
    ) // 255
    assert np.array_equal(result[..., 3], expected.astype(np.uint8))
    assert np.array_equal(result[..., :3], rendered[..., :3])


def test_mask_bypass_and_full_coverage_are_zero_allocation_identity_paths():
    rendered = np.full((2, 3, 4), 77, dtype=np.uint8)
    full = _mask(np.full((2, 3), 255, np.uint8))
    disabled = _mask(np.zeros((2, 3), np.uint8), enabled=False)
    density_zero = _mask(np.zeros((2, 3), np.uint8), density=0)

    assert _apply_raster_mask(rendered, None, (2, 3)) is rendered
    assert _apply_raster_mask(rendered, full, (2, 3)) is rendered
    assert _apply_raster_mask(rendered, disabled, (2, 3)) is rendered
    assert _apply_raster_mask(rendered, density_zero, (2, 3)) is rendered


def test_document_no_mask_disabled_and_density_zero_are_byte_identical(
    monkeypatch,
):
    class StubLookRenderer:
        def __init__(self, registry, gray, rgba, *, probability=None):
            self.rgba = rgba

        def render(self, look, **kwargs):
            return self.rgba[..., :3]

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", StubLookRenderer)
    source = _source([[0, 64], [128, 255]])
    base = Layer("a", "Layer", _look(), source=source)
    documents = [
        LayerDocument(CanvasSpec(2, 2), (base,), ("a",)),
        LayerDocument(
            CanvasSpec(2, 2),
            (replace(base, raster_mask=_mask(
                np.zeros((2, 2), np.uint8), enabled=False)),),
            ("a",),
        ),
        LayerDocument(
            CanvasSpec(2, 2),
            (replace(base, raster_mask=_mask(
                np.zeros((2, 2), np.uint8), density=0)),),
            ("a",),
        ),
    ]

    outputs = [render_layer_document(value, object()) for value in documents]

    assert np.array_equal(outputs[0], outputs[1])
    assert np.array_equal(outputs[0], outputs[2])


def test_empty_mask_fast_path_zeros_alpha_without_resize_or_general_allocations(
    monkeypatch,
):
    rendered = np.full((2, 2, 4), 91, dtype=np.uint8)
    original = rendered.copy()
    monkeypatch.setattr(
        "ditherzam.layers.render._resize_channel_nearest",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("empty fast path must not resize")),
    )

    result = _apply_raster_mask(
        rendered, _mask(np.zeros((2, 2), np.uint8)), (2, 2))

    assert result is not rendered
    assert np.array_equal(rendered, original)
    assert np.array_equal(result[..., :3], original[..., :3])
    assert not result[..., 3].any()


def test_mask_pixels_stay_source_independent_and_use_direct_nearest_geometry():
    rendered = np.full((2, 4, 4), 255, dtype=np.uint8)
    source_mask = np.array([[0, 255], [255, 0], [0, 255]], dtype=np.uint8)
    mask = _mask(source_mask)
    before = mask.pixels.copy()

    result = _apply_raster_mask(rendered, mask, (2, 4))

    assert np.array_equal(
        result[..., 3],
        [[0, 0, 255, 255], [0, 0, 255, 255]],
    )
    assert np.array_equal(mask.pixels, before)


def test_active_mask_checks_cancellation_around_resize_and_math():
    rendered = np.full((2, 4, 4), 255, dtype=np.uint8)
    mask = _mask([[0, 64], [128, 255], [32, 192]])
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(RenderCancelled):
        _apply_raster_mask(rendered, mask, (2, 4), is_cancelled=cancelled)


def test_document_render_cancels_after_raster_mask_resize(monkeypatch):
    class StubLookRenderer:
        def __init__(self, registry, gray, rgba, *, probability=None):
            self.rgba = rgba

        def render(self, look, **kwargs):
            return self.rgba

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", StubLookRenderer)
    import ditherzam.layers.render as render_module

    resized_mask = False
    real_resize = render_module._resize_channel_nearest

    def observed_resize(channel, shape):
        nonlocal resized_mask
        resized_mask = True
        return real_resize(channel, shape)

    monkeypatch.setattr(
        render_module, "_resize_channel_nearest", observed_resize)
    layer = Layer(
        "a", "Layer", _look(), source=_source(255, shape=(3, 2)),
        transform=LayerTransform(scale_x=2.0, scale_y=1.0),
        raster_mask=_mask([[0, 64], [128, 192], [32, 224]]),
    )
    document = LayerDocument(CanvasSpec(4, 3), (layer,), ("a",))

    with pytest.raises(RenderCancelled):
        render_layer_document(
            document, object(), is_cancelled=lambda: resized_mask)


def test_completed_smart_rgba_then_raster_mask_in_real_document_path(monkeypatch):
    calls = []
    smart_rgba = np.full((2, 2, 4), 40, dtype=np.uint8)
    smart_rgba[..., 3] = np.array([[0, 64], [128, 255]], dtype=np.uint8)

    class StubLookRenderer:
        def __init__(self, registry, gray, rgba, *, probability=None):
            self.source = rgba

        def render(self, look, **kwargs):
            calls.append(("complete-look", kwargs["smart_mask"]))
            return smart_rgba

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", StubLookRenderer)
    source = _source(128, shape=(2, 2))
    layer = Layer(
        "a", "Layer", _look(), source=source,
        raster_mask=_mask([[255, 0], [128, 255]]),
    )
    document = LayerDocument(CanvasSpec(2, 2), (layer,), ("a",))

    result = render_layer_document(document, object())

    assert len(calls) == 1
    assert np.array_equal(result[..., 3], [[0, 0], [64, 255]])


def test_nonuniform_scale_cap_clipping_and_proxy_share_masked_geometry(monkeypatch):
    class StubLookRenderer:
        def __init__(self, registry, gray, rgba, *, probability=None):
            self.rgba = rgba

        def render(self, look, **kwargs):
            return self.rgba[..., :3]

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", StubLookRenderer)
    source = _source([[255, 128], [64, 32]])
    layer = Layer(
        "a", "Layer", _look(), source=source,
        transform=LayerTransform(x=-1, y=1, scale_x=2.0, scale_y=1.5),
        raster_mask=_mask([[255, 0], [128, 255]]),
    )
    document = LayerDocument(CanvasSpec(6, 6), (layer,), ("a",))

    exact = render_layer_document(document, object())
    preview = render_layer_document(document, object(), target_max_side=3)
    proxied = render_layer_document_with_proxy(
        document, object(), layer_id="a", target_max_side=3)

    assert exact.shape == (6, 6, 4)
    assert preview.shape == (3, 3, 4)
    assert proxied.proxy.layer_rgba.shape == (2, 2, 4)
    assert np.array_equal(proxied.composite_rgba, preview)
    assert np.array_equal(
        proxied.proxy.layer_rgba[..., 3],
        [[255, 0], [32, 32]],
    )


def test_document_masked_frame_png_alpha_exact_and_jpeg_flattens_once(
    tmp_path, monkeypatch,
):
    from PIL import Image

    import ditherzam.composition.export as export_module
    from ditherzam.composition.export import export_frame

    class StubLookRenderer:
        def __init__(self, registry, gray, rgba, *, probability=None):
            self.rgba = rgba

        def render(self, look, **kwargs):
            return self.rgba[..., :3]

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", StubLookRenderer)
    layer = Layer(
        "a", "Layer", _look(), source=_source([[255, 128], [64, 32]]),
        raster_mask=_mask([[255, 128], [0, 255]]),
    )
    frame = render_layer_document(
        LayerDocument(CanvasSpec(2, 2), (layer,), ("a",)), object())
    calls = 0
    real_flatten = export_module.flatten_rgba_white

    def counted(value):
        nonlocal calls
        calls += 1
        return real_flatten(value)

    monkeypatch.setattr(export_module, "flatten_rgba_white", counted)
    png = tmp_path / "masked.png"
    jpg = tmp_path / "masked.jpg"

    export_frame(frame, png)
    assert calls == 0
    assert np.array_equal(np.asarray(Image.open(png).convert("RGBA")), frame)
    export_frame(frame, jpg)
    assert calls == 1
    assert Image.open(jpg).mode == "RGB"
