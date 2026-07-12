import numpy as np
import pytest

from ditherzam.masking.cache import MaskCaches
from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap, source_identity
from ditherzam.masking.render import render_with_mask
from ditherzam.render import RenderCancelled
from ditherzam.masking.settings import OutsideMode, SmartMaskSettings
from ditherzam.ui.render_request import MaskContext


def _context(probability, *, outside=OutsideMode.ORIGINAL, **setting_changes):
    probability = np.asarray(probability, np.float32)
    rgba = np.zeros((*probability.shape, 4), np.uint8)
    rgba[..., :3] = (10, 20, 30); rgba[..., 3] = 255; rgba.flags.writeable = False
    source = source_identity(rgba)
    identity = InferenceIdentity(source, ModelIdentity("m", "1", "a" * 64), "p", "primary")
    prob = ProbabilityMap(identity, np.asarray(probability, np.float32))
    return MaskContext(source, rgba, prob,
                       SmartMaskSettings(enabled=True, feather_px=0, outside=outside,
                                         **setting_changes))


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

    class Engine:
        context = type("Context", (), {"key": ("palette",)})()
        source_rgb = None

    class Effects:
        items = ()

    live = RenderPipeline(registry, color_engine=Engine(), effect_stack=Effects(),
                          cache_budget_bytes=4096)
    request_engine, request_effect = Engine(), Effects()
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
    live.color_engine, live.effect_stack = Engine(), Effects()
    worker.run()
    assert observed["engine"] is request_engine
    assert observed["effect"] is request_effect
    assert observed["cache"] is live._cache
    assert observed["lock"] is live._cache_lock
    assert observed["source"] is request.source_gray


def test_proxy_complete_branch_runs_once_across_mask_only_edits(monkeypatch):
    import ditherzam.render as render_module
    from ditherzam.dithering import registry
    from ditherzam.render import RenderPipeline, RenderSettings
    from ditherzam.ui.preview import render_preview

    base = np.arange(16 * 24, dtype=np.float32).reshape(16, 24) % 256
    pipeline = RenderPipeline(registry, cache_budget_bytes=1024 * 1024)
    caches = MaskCaches(1024 * 1024)
    real = render_module.apply_contrast
    calls = {"branch": 0}

    def contrast(*args, **kwargs):
        calls["branch"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(render_module, "apply_contrast", contrast)
    probability = np.linspace(0, 1, base.size, dtype=np.float32).reshape(base.shape)
    contexts = [
        _context(probability),
        _context(probability, sensitivity=60),
        _context(probability, expansion_px=2),
        _context(probability, invert=True),
        _context(probability, outside=OutsideMode.BLACK),
    ]
    for context in contexts:
        render_preview(
            pipeline, base, RenderSettings(style="None", scale=1), 12,
            mask_context=context, mask_caches=caches,
            rendered_identity=(context.source, "creative-a"),
        )
    assert calls["branch"] == 1


def test_disabled_proxy_full_render_now_and_export_never_build_mask_identity(
        qapp_fixture, monkeypatch):
    from ditherzam.dithering import registry
    from ditherzam.render import RenderPipeline, RenderSettings
    from ditherzam.ui.main_window import ImageEditor, _RenderWorker
    from ditherzam.ui.render_request import RenderKind, RenderRequest

    def forbidden(_self):
        raise AssertionError("disabled render evaluated masked identity")

    monkeypatch.setattr(RenderRequest, "rendered_identity", property(forbidden))
    base = np.zeros((4, 4), np.float32)
    for kind in (RenderKind.DRAG, RenderKind.FULL):
        request = RenderRequest(1, kind, RenderSettings(style="None", scale=1),
                                1, 4, (4, 4), source_gray=base)
        _RenderWorker(RenderPipeline(registry), base, request).run()

    editor = ImageEditor(registry=registry)
    editor.load_array(base)
    editor.render_now()
    editor._rendered_rgb()


def test_exact_reconstructed_value_context_hits_and_creative_change_misses(
        qapp_fixture, monkeypatch):
    from ditherzam.dithering import registry
    from ditherzam.render import RenderPipeline
    from ditherzam.ui.main_window import ImageEditor

    base = np.full((2, 2), 120, np.float32)
    context = _context(np.ones((2, 2), np.float32), outside=OutsideMode.BLACK)
    editor = ImageEditor(registry=registry)
    editor.load_array(base)
    editor._configure_editor_caches(True)
    monkeypatch.setattr(editor, "_current_mask_context", lambda: context)
    real = RenderPipeline.render
    calls = {"render": 0}

    def render(self, *args, **kwargs):
        calls["render"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(RenderPipeline, "render", render)
    editor._rendered_rgb()
    editor._rendered_rgb()  # independently reconstructed but value-equal context
    assert calls["render"] == 1
    editor.panel.state["contrast"] = 60
    editor._rendered_rgb()
    assert calls["render"] == 2


def test_jpeg_transparency_notice_is_nonblocking_and_shown_once(qapp_fixture, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    from ditherzam.dithering import registry
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor(registry=registry)
    editor.load_array(np.zeros((2, 2), np.float32))
    rgba = np.zeros((2, 2, 4), np.uint8)
    paths = iter((str(tmp_path / "one.jpg"), str(tmp_path / "two.jpg")))
    messages = []
    monkeypatch.setattr(editor, "_rendered_rgb", lambda: rgba)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *args: (next(paths), ""))
    )
    monkeypatch.setattr(
        editor.statusBar(), "showMessage", lambda message, timeout=0: messages.append((message, timeout))
    )

    editor._on_export_raster("JPEG Files (*.jpg)", ".jpg")
    editor._on_export_raster("JPEG Files (*.jpg)", ".jpg")

    assert messages == [
        ("JPEG does not support transparency; transparent pixels are flattened onto white.", 8000)
    ]


def test_raster_notice_follows_selected_filename_not_action_filter(
        qapp_fixture, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    from ditherzam.dithering import registry
    from ditherzam.ui.main_window import ImageEditor

    editor = ImageEditor(registry=registry)
    editor.load_array(np.zeros((2, 2), np.float32))
    rgba = np.zeros((2, 2, 4), np.uint8)
    selections = iter((
        (str(tmp_path / "png-action.JPG"), ""),
        (str(tmp_path / "jpeg-action.png"), ""),
        ("", ""),
    ))
    messages = []
    monkeypatch.setattr(editor, "_rendered_rgb", lambda: rgba)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *args: next(selections))
    )
    monkeypatch.setattr(
        editor.statusBar(), "showMessage", lambda message, timeout=0: messages.append(message)
    )

    editor._on_export_raster("PNG Files (*.png)", ".png")
    editor._jpeg_flatten_notice_shown = False
    editor._on_export_raster("JPEG Files (*.jpg)", ".jpg")
    editor._on_export_raster("JPEG Files (*.jpg)", ".jpg")

    assert messages == [
        "JPEG does not support transparency; transparent pixels are flattened onto white."
    ]
