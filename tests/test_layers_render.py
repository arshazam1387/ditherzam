import numpy as np
import pytest

from ditherzam.composition import Look
from ditherzam.layers import (
    CanvasSpec,
    Layer,
    LayerDocument,
    LayerSource,
    LayerStack,
    LayerTransform,
    render_layer_document_with_proxy,
    render_layer_stack,
)
from ditherzam.render import RenderCancelled


def _look(name):
    return Look(name, {"dither": {"style": "None"}})


class StubRenderer:
    def __init__(self, images):
        self.source_rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        self.images = images
        self.calls = []

    def render(self, look, *, target_max_side=None, is_cancelled=None):
        self.calls.append((look.name, target_max_side, is_cancelled))
        return self.images[look.name]


def test_render_stack_skips_hidden_and_zero_opacity_and_is_bottom_to_top():
    red = np.full((2, 3, 4), [255, 0, 0, 255], dtype=np.uint8)
    blue = np.full((2, 3, 4), [0, 0, 255, 255], dtype=np.uint8)
    renderer = StubRenderer({"bottom": red, "top": blue})
    stack = LayerStack((
        Layer("a", "A", _look("bottom")),
        Layer("hidden", "Hidden", _look("unused"), visible=False),
        Layer("zero", "Zero", _look("unused"), opacity=0),
        Layer("b", "B", _look("top"), opacity=50),
    ))
    result = render_layer_stack(stack, renderer, target_max_side=64)
    assert [call[:2] for call in renderer.calls] == [
        ("bottom", 64), ("top", 64)
    ]
    assert np.array_equal(result[0, 0], [127, 0, 128, 255])


def test_empty_stack_returns_transparent_preview_sized_canvas():
    renderer = StubRenderer({})
    result = render_layer_stack(LayerStack(), renderer, target_max_side=2)
    assert result.shape == (1, 2, 4)
    assert not result.any()


def test_render_stack_checks_cancellation_between_layers():
    image = np.full((2, 3, 3), 100, dtype=np.uint8)
    renderer = StubRenderer({"a": image, "b": image})
    stack = LayerStack((
        Layer("a", "A", _look("a")),
        Layer("b", "B", _look("b")),
    ))
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(RenderCancelled):
        render_layer_stack(stack, renderer, is_cancelled=cancelled)
    assert len(renderer.calls) == 1


def _document_source(rgb, *, alpha=255, shape=(2, 2)):
    rgba = np.full((*shape, 4), (*rgb, alpha), dtype=np.uint8)
    gray = np.full(shape, 100, dtype=np.float32)
    return LayerSource(gray, rgba)


def test_document_proxy_captures_rendered_layer_without_rerendering(monkeypatch):
    calls = []

    class ProxyRenderer:
        def __init__(self, registry, gray, rgba, *, probability=None):
            self.rgba = rgba

        def render(self, look, **kwargs):
            calls.append(look.name)
            return self.rgba

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", ProxyRenderer)
    layers = (
        Layer("lower", "Lower", _look("lower"),
              source=_document_source((255, 0, 0))),
        Layer(
            "moving", "Moving", _look("moving"),
            source=_document_source((0, 255, 0)),
            transform=LayerTransform(x=1, y=0),
        ),
        Layer("upper", "Upper", _look("upper"),
              source=_document_source((0, 0, 255), alpha=128)),
    )
    document = LayerDocument(CanvasSpec(4, 2), layers, ("moving",))

    result = render_layer_document_with_proxy(
        document, object(), layer_id="moving")

    assert calls == ["lower", "moving", "upper"]
    assert result.proxy.layer_id == "moving"
    assert (result.proxy.x, result.proxy.y) == (1, 0)
    assert result.proxy.blend_mode == "normal"
    assert result.proxy.opacity == 100
    assert result.proxy.visible is True
    assert np.array_equal(
        result.proxy.layer_rgba,
        np.full((2, 2, 4), [0, 255, 0, 255], dtype=np.uint8),
    )
    # The background includes layers on both sides of the moving layer, but no
    # pixels from its old placement.
    assert np.all(result.proxy.background_rgba[..., 1] == 0)
    assert np.all(result.proxy.background_rgba[:, :2, 3] == 255)
    assert np.all(result.proxy.background_rgba[:, 2:, 3] == 0)
    assert np.all(result.composite_rgba[:, 1:3, 1] > 0)


def test_document_proxy_parts_are_immutable_and_preview_scaled(monkeypatch):
    class ProxyRenderer:
        def __init__(self, registry, gray, rgba, *, probability=None):
            self.rgba = rgba

        def render(self, look, **kwargs):
            return self.rgba

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", ProxyRenderer)
    layer = Layer(
        "moving", "Moving", _look("moving"),
        source=_document_source((10, 20, 30), shape=(4, 4)),
        transform=LayerTransform(x=2, y=2),
    )
    document = LayerDocument(CanvasSpec(8, 8), (layer,), ("moving",))

    result = render_layer_document_with_proxy(
        document, object(), layer_id="moving", target_max_side=4)

    assert result.composite_rgba.shape == (4, 4, 4)
    assert result.proxy.background_rgba.shape == (4, 4, 4)
    assert result.proxy.layer_rgba.shape == (2, 2, 4)
    assert (result.proxy.x, result.proxy.y) == (1, 1)
    assert not result.proxy.background_rgba.flags.writeable
    assert not result.proxy.layer_rgba.flags.writeable
    with pytest.raises(ValueError):
        result.proxy.layer_rgba[0, 0] = 0


def test_document_proxy_rejects_unknown_layer_id():
    document = LayerDocument(CanvasSpec(2, 2))
    with pytest.raises(ValueError, match="live document layer"):
        render_layer_document_with_proxy(
            document, object(), layer_id="missing")
