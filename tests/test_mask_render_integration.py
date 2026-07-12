import numpy as np
import pytest

from ditherzam.masking.cache import MaskCaches
from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity
from ditherzam.masking.render import render_with_mask
from ditherzam.render import RenderCancelled
from ditherzam.masking.settings import OutsideMode, SmartMaskSettings
from ditherzam.ui.render_request import MaskContext


def _context(probability, *, outside=OutsideMode.ORIGINAL):
    probability = np.asarray(probability, np.float32)
    rgba = np.zeros((*probability.shape, 4), np.uint8)
    rgba[..., :3] = (10, 20, 30); rgba[..., 3] = 255; rgba.flags.writeable = False
    source = source_identity(rgba)
    identity = InferenceIdentity(source, ModelIdentity("m", "1", "a" * 64), "p", "primary")
    prob = ProbabilityMap(identity, np.asarray(probability, np.float32))
    return MaskContext(source, rgba, prob,
                       SmartMaskSettings(enabled=True, feather_px=0, outside=outside))


def test_disabled_is_literal_direct_historical_call(monkeypatch):
    expected = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    calls = []
    monkeypatch.setattr("ditherzam.masking.render.derive_master_mask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("mask work")))
    assert render_with_mask(lambda: calls.append(1) or expected, None) is expected
    assert calls == [1]


def test_enabled_renders_complete_branch_once_then_composites():
    context = _context([[1, 0], [0, 1]], outside=OutsideMode.BLACK)
    calls = []
    rendered = np.full((2, 2, 3), 200, np.uint8)
    result = render_with_mask(lambda: calls.append(1) or rendered, context)
    assert calls == [1]
    assert np.array_equal(result[..., 0], [[200, 0], [0, 200]])


def test_capped_composite_resizes_source_and_mask_to_render_geometry():
    context = _context(np.ones((4, 6), np.float32))
    out = render_with_mask(lambda: np.full((2, 3, 3), 77, np.uint8), context)
    assert out.shape == (2, 3, 3)
    assert np.all(out == 77)


def test_derived_and_composite_products_reuse_bounded_mask_cache(monkeypatch):
    context = _context([[1, 0], [0, 1]], outside=OutsideMode.BLACK)
    caches = MaskCaches(1024 * 1024)
    calls = {"derive": 0, "composite": 0}
    from ditherzam.masking import render as module
    real_derive, real_composite = module.derive_master_mask, module.composite_masked

    def derive(*args, **kwargs):
        calls["derive"] += 1
        return real_derive(*args, **kwargs)

    def composite(*args, **kwargs):
        calls["composite"] += 1
        return real_composite(*args, **kwargs)

    monkeypatch.setattr(module, "derive_master_mask", derive)
    monkeypatch.setattr(module, "composite_masked", composite)
    renderer = lambda: np.full((2, 2, 3), 200, np.uint8)
    first = render_with_mask(renderer, context, caches=caches, rendered_identity="branch")
    second = render_with_mask(renderer, context, caches=caches, rendered_identity="branch")
    assert np.array_equal(first, second)
    assert calls == {"derive": 1, "composite": 1}
    assert caches.metrics["derived_entries"] == 1
    assert caches.metrics["composite_entries"] == 1


def test_cancellation_during_derivation_publishes_no_partial_cache(monkeypatch):
    context = _context([[1, 0], [0, 1]], outside=OutsideMode.BLACK)
    caches = MaskCaches(1024 * 1024)
    cancelled = {"value": False}
    from ditherzam.masking import render as module
    real_derive = module.derive_master_mask

    def derive(*args, **kwargs):
        result = real_derive(*args, **kwargs)
        cancelled["value"] = True
        return result

    monkeypatch.setattr(module, "derive_master_mask", derive)
    with pytest.raises(RenderCancelled):
        render_with_mask(
            lambda: np.full((2, 2, 3), 200, np.uint8), context,
            caches=caches, rendered_identity="branch",
            is_cancelled=lambda: cancelled["value"],
        )
    assert caches.entry_count == 0


def test_full_worker_uses_request_context_with_shared_bounded_stage_cache(
        qapp_fixture, monkeypatch):
    from ditherzam.dithering import registry
    from ditherzam.render import RenderPipeline, RenderSettings
    from ditherzam.ui.main_window import _RenderWorker
    from ditherzam.ui.render_request import RenderKind, RenderRequest

    live = RenderPipeline(registry, color_engine=object(), effect_stack=object(),
                          cache_budget_bytes=4096)
    request_engine, request_effect = object(), object()
    request = RenderRequest(
        1, RenderKind.FULL, RenderSettings(style="None"), 7, 2, (2, 2),
        request_engine, request_effect,
        source_gray=np.zeros((2, 2), np.float32),
    )
    observed = {}

    def render_cached(self, source, settings, **kwargs):
        observed.update(engine=self.color_engine, effect=self.effect_stack,
                        cache=self._cache, lock=self._cache_lock, source=source)
        return np.zeros((2, 2, 3), np.uint8)

    monkeypatch.setattr(RenderPipeline, "render_cached", render_cached)
    worker = _RenderWorker(live, np.ones((2, 2), np.float32), request)
    # Simulate GUI context reassignment after request capture.
    live.color_engine, live.effect_stack = object(), object()
    worker.run()
    assert observed["engine"] is request_engine
    assert observed["effect"] is request_effect
    assert observed["cache"] is live._cache
    assert observed["lock"] is live._cache_lock
    assert observed["source"] is request.source_gray
