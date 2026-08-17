import numpy as np
from threading import Event
import time

from PySide6.QtCore import QRunnable, QTimer

from ditherzam.masking.adapter import InferenceResult
from ditherzam.masking.contracts import (
    InferenceIdentity,
    ModelIdentity,
    ProbabilityMap,
    source_identity,
)
from ditherzam.masking.inference_request import (
    InferenceOutcome,
    InferenceRequest,
    InferenceTerminal,
)
from ditherzam.masking.settings import MaskTarget, SmartMaskSettings
from ditherzam.masking.cache import MaskCaches
from ditherzam.render_cache import MIB
from ditherzam.ui.main_window import ImageEditor
from ditherzam.ui.render_request import RenderKind
from ditherzam.ui.smart_mask_panel import MaskPanelStatus


MODEL = ModelIdentity("test", "1", "a" * 64)


class Adapter:
    def infer(self, rgba, *, should_cancel=None):
        raise AssertionError("tests capture launches")


def source(value=0):
    rgba = np.full((3, 4, 4), value, np.uint8); rgba[..., 3] = 255
    gray = np.full((3, 4), value, np.float32)
    return gray, rgba[..., :3].copy(), rgba


def editor():
    value = ImageEditor(mask_adapter=Adapter(), mask_model=MODEL)
    launched = []
    value._launch_mask_worker = launched.append
    return value, launched


def success(request):
    identity = InferenceIdentity(request.source, request.model,
                                 request.preprocessing_version, "primary")
    result = InferenceResult("primary", ProbabilityMap(identity,
        np.full((request.source.height, request.source.width), .75, np.float32)))
    return InferenceOutcome(request, InferenceTerminal.SUCCESS, result=result)


def test_add_mask_target_opens_mask_tools_without_arming_invalid_brush(
    qapp_fixture,
):
    window, _launched = editor()
    window.load_array(*source())
    layer = window.layers_controller.document.layers[0]

    window.layers_panel._row_targets[layer.id][1].click()
    qapp_fixture.processEvents()

    assert window.layers_panel.editor_tabs.currentIndex() == 1
    assert window.layers_controller.edit_target.kind == "mask"
    assert not window.viewport._mask_brush_mode
    assert not window.layers_panel.paint_mask_btn.isEnabled()

    assert window.layers_controller.reveal_all_raster_mask(
        replace_existing=False)
    qapp_fixture.processEvents()
    assert window.layers_panel.paint_mask_btn.isEnabled()

    window.layers_panel._row_targets[layer.id][1].click()
    qapp_fixture.processEvents()
    assert window.viewport._mask_brush_mode


def test_disabled_load_and_whole_image_do_not_infer(qapp_fixture, monkeypatch):
    window, launched = editor()
    calls = []
    from ditherzam.masking.contracts import source_identity as real_source_identity
    monkeypatch.setattr("ditherzam.ui.main_window.source_identity",
                        lambda rgba: calls.append(rgba) or real_source_identity(rgba))
    window.load_array(*source())
    assert launched == []
    assert calls == []
    window.panel.smart_mask_panel.set_settings(SmartMaskSettings(enabled=True,
        target=MaskTarget.WHOLE_IMAGE))
    window._on_mask_settings_changed(window.panel.smart_mask_panel.settings)
    assert launched == []
    assert calls == []


def test_layer_from_whole_image_smart_needs_no_model_or_probability(
    qapp_fixture, monkeypatch
):
    window, _launched = editor()
    window.load_array(*source())
    settings = SmartMaskSettings(
        enabled=True, target=MaskTarget.WHOLE_IMAGE)
    window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings)
    window.layers_controller.update_active_from_editor()
    monkeypatch.setattr(
        window, "_mask_dependencies_available", lambda: False)
    window._mask_probability = None
    window._mask_source = None

    assert window.layers_controller.raster_mask_from_smart()
    mask = window.layers_controller.document.layers[0].raster_mask
    assert np.all(mask.pixels == 255)

    window.layers_controller.delete_active_raster_mask()
    disabled = SmartMaskSettings(target=MaskTarget.WHOLE_IMAGE)
    window.panel.smart_mask_panel.set_settings(disabled)
    window._on_mask_settings_changed(disabled)
    window.layers_controller.update_active_from_editor()
    assert window.layers_controller.raster_mask_from_smart() is False
    assert window.layers_controller.document.layers[0].raster_mask is None


def test_enable_infers_once_edits_reuse_and_request_freezes_context(qapp_fixture):
    window, launched = editor(); window.load_array(*source())
    settings = SmartMaskSettings(enabled=True)
    window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings)
    assert len(launched) == 1
    window._on_mask_terminal(success(launched[0]))
    edited = SmartMaskSettings(enabled=True, sensitivity=70)
    window.panel.smart_mask_panel.set_settings(edited)
    window._on_mask_settings_changed(edited)
    assert len(launched) == 1
    request = window._build_request(RenderKind.DRAG)
    assert request.mask_context is not None
    assert request.mask_context.settings is edited


def test_redetect_failure_and_cancel_retain_last_valid(qapp_fixture):
    window, launched = editor(); window.load_array(*source())
    settings = SmartMaskSettings(enabled=True); window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings); window._on_mask_terminal(success(launched[0]))
    prior = window._mask_probability
    window._request_mask_detection(); active = launched[-1]
    window._cancel_mask_detection()
    assert window._mask_probability is prior
    window._on_mask_terminal(InferenceOutcome(active, InferenceTerminal.CANCELLED))
    assert window._mask_probability is prior


def test_source_replacement_invalidates_stale_terminal(qapp_fixture):
    window, launched = editor(); window.load_array(*source())
    settings = SmartMaskSettings(enabled=True); window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings); stale = launched[0]
    window.load_array(*source(20))
    assert len(launched) == 1
    window._on_mask_terminal(success(stale))
    assert len(launched) == 2
    assert window._mask_probability is None
    assert window._current_mask_context() is None


def test_terminal_promotes_at_most_one_trailing(qapp_fixture):
    window, launched = editor(); window.load_array(*source())
    settings = SmartMaskSettings(enabled=True); window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings); first = launched[0]
    window._request_mask_detection(); window._request_mask_detection()
    assert len(launched) == 1
    window._on_mask_terminal(InferenceOutcome(first, InferenceTerminal.CANCELLED))
    assert len(launched) == 2
    window._on_mask_terminal(InferenceOutcome(first, InferenceTerminal.CANCELLED))
    assert len(launched) == 2


def test_actual_editor_cache_budgets_obey_global_ceiling(qapp_fixture):
    window, _ = editor()
    assert window.pipeline.cache_metrics["budget_bytes"] == 160 * MIB
    assert window._mask_caches.budget_bytes == 0
    assert window.layers_controller._look_cache.metrics["budget_bytes"] == 32 * MIB
    window.load_array(*source())
    settings = SmartMaskSettings(enabled=True); window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings)
    assert window.pipeline.cache_metrics["budget_bytes"] == 96 * MIB
    assert window._mask_caches.budget_bytes == 64 * MIB
    assert (window.pipeline.cache_metrics["budget_bytes"] + window._mask_caches.budget_bytes
            + window.layers_controller._look_cache.metrics["budget_bytes"]
            <= 192 * MIB)
    assert (window.pipeline.cache_metrics["retained_bytes"] + window._mask_caches.retained_bytes
            + window.layers_controller._look_cache.metrics["retained_bytes"]
            <= 192 * MIB)
    window._on_mask_terminal(success(window._mask_scheduler._active))
    assert window._mask_probability is not None
    assert window._mask_caches.get_inference(window._mask_probability.identity) is window._mask_probability
    assert window._mask_probability.values.nbytes <= window._mask_caches.retained_bytes
    disabled = SmartMaskSettings(enabled=False)
    window.panel.smart_mask_panel.set_settings(disabled)
    window._on_mask_settings_changed(disabled)
    assert window._mask_probability is None and window._mask_source is None
    assert window._mask_caches.retained_bytes == 0
    assert window.pipeline.cache_metrics["budget_bytes"] == 160 * MIB


def test_only_current_progress_is_published_and_terminal_clears_it(qapp_fixture):
    window, launched = editor(); window.load_array(*source())
    settings = SmartMaskSettings(enabled=True); window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings); current = launched[0]
    window._on_mask_progress(current, 10)
    assert window.panel.smart_mask_panel.progress_label.text() == "Detecting 10%"
    window.load_array(*source(2))
    window._on_mask_progress(current, 90)
    assert window.panel.smart_mask_panel.progress_label.text() != "Detecting 90%"
    window._on_mask_terminal(InferenceOutcome(current, InferenceTerminal.CANCELLED))
    assert window.panel.smart_mask_panel.progress_label.text() == ""


def test_inference_uses_editor_owned_serial_pool(qapp_fixture):
    window, _ = editor()
    assert window._mask_pool is not window._pool
    assert window._mask_pool.maxThreadCount() == 1


def test_real_mask_worker_is_retained_until_queued_terminal(qapp_fixture, monkeypatch):
    window = ImageEditor(mask_adapter=Adapter(), mask_model=MODEL)
    captured = []
    monkeypatch.setattr(window._mask_pool, "start", captured.append)
    rgba = source()[2]
    request = window._mask_scheduler.request(
        InferenceRequest(
            source_identity(rgba),
            MODEL,
            window._mask_preprocessing_version,
            rgba,
        )
    )

    window._launch_mask_worker(request)

    assert captured == [next(iter(window._mask_workers))]
    worker = captured[0]
    worker.signals.cancelled.emit(
        InferenceOutcome(request, InferenceTerminal.CANCELLED)
    )
    assert worker not in window._mask_workers


def test_cache_rejection_never_publishes_direct_probability(qapp_fixture):
    window, launched = editor(); window.load_array(*source())
    settings = SmartMaskSettings(enabled=True); window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings)
    window._mask_caches = MaskCaches(8)
    window._on_mask_terminal(success(launched[0]))
    assert window._mask_probability is None
    assert window._current_mask_context() is None
    assert window._mask_caches.retained_bytes == 0
    assert window.panel.smart_mask_panel.status is MaskPanelStatus.ERROR


class _BlockingRunnable(QRunnable):
    def __init__(self, started, release, finished):
        super().__init__(); self.started = started; self.release = release; self.finished = finished

    def run(self):
        self.started.set(); self.release.wait(5); self.finished.set()


def test_blocking_inference_pool_does_not_block_render_pool_and_close_retires_it(qapp_fixture):
    window, _ = editor()
    window.show()
    started, release, finished = Event(), Event(), Event()
    window._mask_pool.start(_BlockingRunnable(started, release, finished))
    assert started.wait(1)
    render_done = Event()

    class Quick(QRunnable):
        def run(self): render_done.set()

    window._pool.start(Quick())
    assert render_done.wait(1), "global render pool was blocked by inference"
    heartbeat_seen = Event()
    heartbeat = QTimer(); heartbeat.setInterval(5)
    heartbeat.timeout.connect(heartbeat_seen.set)
    heartbeat.start()
    try:
        assert window.close() is False
        assert not finished.is_set(), "close must not wait for active inference"
        assert window.isVisible(), "first close must be ignored while inference is active"
        assert window._mask_closing and not window._mask_close_finalizing
        deadline = time.monotonic() + 2
        while not heartbeat_seen.is_set() and time.monotonic() < deadline:
            qapp_fixture.processEvents()
        assert heartbeat_seen.is_set(), "close must leave the Qt event loop responsive"
        release.set()
        deadline = time.monotonic() + 2
        while window.isVisible() and time.monotonic() < deadline:
            qapp_fixture.processEvents()
        assert finished.is_set()
        assert not window.isVisible()
        assert window._mask_pool.activeThreadCount() == 0
        assert window._mask_closing is True
        assert window._mask_close_finalizing is True
    finally:
        heartbeat.stop()
        release.set()
