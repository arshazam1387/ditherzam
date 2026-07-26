from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from ditherzam.composition import Look
from ditherzam.layers import (
    CanvasSpec,
    Layer,
    LayerDocument,
    LayerLookRenderCache,
    LayerSource,
    RasterLayerMask,
    render_layer_document,
)
from ditherzam.render import RenderCancelled


def _look(contrast=50):
    return Look("Layer", {
        "adjustments": {"contrast": contrast},
        "dither": {"style": "None"},
    })


def _source(value=80, shape=(2, 3)):
    gray = np.full(shape, value, np.float32)
    rgba = np.full((*shape, 4), value, np.uint8)
    rgba[..., 3] = 255
    return LayerSource(gray, rgba)


def _document(layer):
    h, w = layer.source.gray.shape
    return LayerDocument(CanvasSpec(w, h), (layer,), (layer.id,))


def test_mask_only_changes_reuse_completed_look(monkeypatch):
    calls = 0

    class Stub:
        def __init__(self, registry, gray, rgba, **kwargs):
            self.rgba = rgba

        def render(self, look, **kwargs):
            nonlocal calls
            calls += 1
            return self.rgba[..., :3]

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", Stub)
    source = _source()
    pixels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    base = Layer("a", "A", _look(), source=source)
    variants = (
        base,
        replace(base, raster_mask=RasterLayerMask(pixels)),
        replace(base, raster_mask=RasterLayerMask(
            pixels, enabled=False, density=37, revision=9)),
    )
    cache = LayerLookRenderCache(1024 * 1024)

    for layer in variants:
        render_layer_document(_document(layer), object(), look_cache=cache)

    assert calls == 1


def test_source_look_and_actual_target_shape_miss(monkeypatch):
    calls = 0

    class Stub:
        def __init__(self, registry, gray, rgba, **kwargs):
            self.rgba = rgba

        def render(self, look, *, target_max_side=None, **kwargs):
            nonlocal calls
            calls += 1
            if target_max_side is None:
                return self.rgba[..., :3]
            return self.rgba[:1, :2, :3]

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", Stub)
    cache = LayerLookRenderCache(1024 * 1024)
    first = Layer("a", "A", _look(), source=_source(80))
    changed_look = replace(first, look=_look(60))
    changed_source = replace(first, source=_source(81))

    render_layer_document(_document(first), object(), look_cache=cache)
    render_layer_document(_document(changed_look), object(), look_cache=cache)
    render_layer_document(_document(changed_source), object(), look_cache=cache)
    render_layer_document(
        _document(first), object(), target_max_side=2, look_cache=cache)

    assert calls == 4


def test_cache_single_flight_immutable_and_no_failed_or_cancelled_admission():
    cache = LayerLookRenderCache(1024)
    calls = 0
    barrier = threading.Barrier(2)
    outputs = []

    def render():
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        return np.ones((2, 2, 3), np.uint8)

    def run():
        barrier.wait()
        outputs.append(cache.get_or_render("same", render))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert calls == 1
    assert outputs[0] is outputs[1]
    assert not outputs[0].flags.writeable

    with pytest.raises(RuntimeError):
        cache.get_or_render(
            "failed", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cache.metrics["entry_count"] == 1
    with pytest.raises(RenderCancelled):
        cache.get_or_render(
            "cancelled", lambda: np.zeros((1, 1, 3), np.uint8),
            is_cancelled=lambda: True,
        )
    assert cache.metrics["entry_count"] == 1


def test_cache_is_byte_bounded_and_evicts_lru():
    cache = LayerLookRenderCache(24)
    for key in ("a", "b", "c"):
        cache.get_or_render(
            key, lambda: np.zeros((2, 2, 3), np.uint8))
    assert cache.metrics["retained_bytes"] <= 24
    assert cache.metrics["entry_count"] == 2
    assert cache.metrics["eviction_count"] == 1


def test_clear_during_inflight_render_prevents_late_admission():
    cache = LayerLookRenderCache(1024)
    started = threading.Event()
    release = threading.Event()

    def render():
        started.set()
        assert release.wait(1)
        return np.ones((2, 2, 3), np.uint8)

    thread = threading.Thread(
        target=lambda: cache.get_or_render("old", render))
    thread.start()
    assert started.wait(1)
    cache.clear()
    release.set()
    thread.join(1)

    assert not thread.is_alive()
    assert cache.metrics["entry_count"] == 0


def test_frame_and_thumbnail_same_shape_share_completed_look(monkeypatch):
    calls = 0

    class Stub:
        def __init__(self, registry, gray, rgba, **kwargs):
            self.rgba = rgba

        def render(self, look, **kwargs):
            nonlocal calls
            calls += 1
            return self.rgba[..., :3]

    monkeypatch.setattr("ditherzam.layers.render.LookRenderer", Stub)
    cache = LayerLookRenderCache(1024 * 1024)
    layer = Layer("a", "A", _look(), source=_source(shape=(56, 56)))
    document = _document(layer)

    render_layer_document(
        document, object(), target_max_side=480, look_cache=cache)
    render_layer_document(
        document, object(), target_max_side=56, look_cache=cache)

    assert calls == 1
