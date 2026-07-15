import numpy as np
from threading import Event
import time

from PySide6.QtCore import QRunnable, QTimer

from ditherzam.masking.adapter import InferenceResult
from ditherzam.masking.contracts import InferenceIdentity, ModelIdentity, ProbabilityMap
from ditherzam.masking.inference_request import InferenceOutcome, InferenceTerminal
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
    assert window.pipeline.cache_metrics["budget_bytes"] == 192 * MIB
    assert window._mask_caches.budget_bytes == 0
    window.load_array(*source())
    settings = SmartMaskSettings(enabled=True); window.panel.smart_mask_panel.set_settings(settings)
    window._on_mask_settings_changed(settings)
    assert window.pipeline.cache_metrics["budget_bytes"] == 128 * MIB
    assert window._mask_caches.budget_bytes == 64 * MIB
    assert (window.pipeline.cache_metrics["budget_bytes"] + window._mask_caches.budget_bytes
            <= 192 * MIB)
    assert (window.pipeline.cache_metrics["retained_bytes"] + window._mask_caches.retained_bytes
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
    assert window.pipeline.cache_metrics["budget_bytes"] == 192 * MIB


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
    heartbeats = []
    heartbeat = QTimer(); heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: heartbeats.append(time.monotonic()))
    heartbeat.start()
    assert window.close() is False
    deadline = time.monotonic() + 1
    while len(heartbeats) < 2 and time.monotonic() < deadline:
        qapp_fixture.processEvents(); time.sleep(.005)
    assert len(heartbeats) >= 2, "GUI event loop stalled while inference was active"
    assert window.isVisible(), "first close must be ignored while inference is active"
    assert window._mask_closing and not window._mask_close_finalizing
    release.set()
    deadline = time.monotonic() + 2
    while window.isVisible() and time.monotonic() < deadline:
        qapp_fixture.processEvents(); time.sleep(.005)
    heartbeat.stop()
    assert finished.is_set()
    assert not window.isVisible()
    assert window._mask_pool.activeThreadCount() == 0
    assert window._mask_closing is True
    assert window._mask_close_finalizing is True
