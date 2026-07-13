import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import types
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from ditherzam.animation.temporal import PATTERNS
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.animation.timeline import Timeline
from ditherzam.ui.render_request import RenderKind
from ditherzam.ui.timeline_panel import TimelinePanel, AnimationController, _AnimRequest

_app = QApplication.instance() or QApplication([])


def test_panel_lists_none_plus_nine_patterns():
    p = TimelinePanel(length=10)
    items = [p.pattern_combo.itemText(i) for i in range(p.pattern_combo.count())]
    assert items[0] == "none"
    assert tuple(items[1:]) == PATTERNS
    assert len(items) == 10


def test_panel_signals_exist():
    p = TimelinePanel()
    for sig in ("keyframe_requested", "frame_changed", "export_requested", "play_toggled"):
        assert hasattr(p, sig)


def test_length_updates_frame_slider_range():
    p = TimelinePanel(length=10)
    p.length_spin.setValue(20)
    assert p.frame_slider.maximum() == 19


def test_play_toggle_drives_timer():
    p = TimelinePanel(length=5)
    p.play_btn.setChecked(True)
    assert p._timer.isActive()
    p.play_btn.setChecked(False)
    assert not p._timer.isActive()


def _sync_controller(base, pipeline=None, cap_provider=None, settings=None, length=5,
                     pattern="static", amp=60):
    """An AnimationController whose thread pool runs workers inline (no real
    threading/timing dependency), mirroring the pattern in
    test_render_cancellation.py / test_preview_lifecycle.py."""
    p = TimelinePanel(length=length)
    p.pattern_combo.setCurrentText(pattern)
    p.amp_slider.setValue(amp)
    pipeline = pipeline or RenderPipeline(registry)
    tl = Timeline(length=length)
    settings = settings or RenderSettings(style="Bayer-Matrix 4x4", scale=1)
    ctrl = AnimationController(
        p, pipeline, lambda: (base, settings), tl, seed=0, cap_provider=cap_provider)
    # Run workers inline via a disposable stand-in pool; never mutate the shared
    # QThreadPool.globalInstance() singleton (that leaks into other tests).
    ctrl._pool = types.SimpleNamespace(start=lambda worker: worker.run())
    return ctrl


def _make_anim_request(ctrl, frame_index, target_max_side=8):
    gray, settings = ctrl.provide_base()
    ctrl._last_base_gray = gray
    return _AnimRequest(generation=0, kind=RenderKind.DRAG, frame_index=frame_index,
                        settings=settings, temporal_field=None,
                        target_max_side=target_max_side)


def test_controller_renders_frame_to_sink():
    base = np.full((16, 16), 128.0, np.float32)
    ctrl = _sync_controller(base)
    captured = {}
    ctrl.on_frame = lambda img: captured.setdefault("img", img)
    result = ctrl.render_frame(2)

    assert result is None                          # async: fire-and-forget
    assert captured["img"].shape == (16, 16, 3)
    assert captured["img"].dtype == np.uint8


def test_render_frame_screen_render_uses_cap_provider():
    # A large source capped tightly must land on the capped shape -- the
    # animation screen render goes through the same render_preview() path as
    # the still-image capped preview.
    base = np.random.default_rng(0).uniform(0, 255, (1080, 1920)).astype(np.float32)
    ctrl = _sync_controller(
        base, cap_provider=lambda: 640,
        settings=RenderSettings(style="Bayer-Matrix 4x4", scale=5),
        pattern="none", amp=0)
    captured = {}
    ctrl.on_frame = lambda img: captured.setdefault("img", img)

    ctrl.render_frame(0)

    assert captured["img"].shape == (360, 640, 3)   # capped, not full 1080x1920


def test_render_frame_defaults_cap_when_no_provider_given():
    # No cap_provider passed -> a sane default still produces a valid render
    # for a small (already-fits) source, matching the pre-4.1 synchronous path.
    base = np.full((16, 16), 128.0, np.float32)
    ctrl = _sync_controller(base, cap_provider=None)
    captured = {}
    ctrl.on_frame = lambda img: captured.setdefault("img", img)

    ctrl.render_frame(1)

    assert captured["img"].shape == (16, 16, 3)


def test_temporal_field_shape_consistent_across_scale_cap_combos():
    # Hazard #1: feed the capped async path through several scale/cap combos
    # that force real proxy downscaling; must not crash, must produce a
    # correctly capped-size output.
    base = np.random.default_rng(1).uniform(0, 255, (720, 1280)).astype(np.float32)
    for scale, cap in ((1, 480), (3, 480), (5, 720), (2, 1280)):
        ctrl = _sync_controller(
            base, cap_provider=lambda cap=cap: cap,
            settings=RenderSettings(style="Floyd-Steinberg", scale=scale),
            pattern="vhs-jitter", amp=60)
        captured = {}
        ctrl.on_frame = lambda img: captured.setdefault("img", img)

        ctrl.render_frame(3)

        assert captured["img"].dtype == np.uint8
        assert max(captured["img"].shape[:2]) <= cap


# ---- latest-wins / async, non-blocking (hazard #3) --------------------------

def test_playback_is_latest_wins_stale_frames_dropped():
    # While one frame render is "in flight" (not yet finished), two more
    # scrub/playback ticks arrive. Only the LAST one's image reaches the sink;
    # the coalesced middle request is dropped entirely, never even launched.
    base = np.full((8, 8), 128.0, np.float32)
    ctrl = _sync_controller(base, pattern="none", amp=0)

    # Freeze scheduling by hand: request three frames back-to-back before any
    # worker is allowed to run, exactly like main_window's DRAG-burst tests.
    ctrl._scheduler._busy = True   # simulate a render already in flight
    for idx in (1, 2, 3):
        stamped = ctrl._scheduler.request(_make_anim_request(ctrl, frame_index=idx))
        assert stamped is None            # coalesced, nothing launched yet

    assert ctrl._scheduler._pending.frame_index == 3   # only the freshest survives

    captured = []
    ctrl.on_frame = captured.append
    ctrl._advance_scheduler()   # the in-flight render "finishes" -> launches trailing

    assert len(captured) == 1   # frames 1 and 2 were coalesced away, never rendered
    assert captured[0].shape == (8, 8, 3)


def test_stale_result_delivered_after_newer_request_is_not_painted():
    # A worker whose request belongs to an OLD generation reports its
    # terminal signal after fresher renders have since completed (the
    # scheduler's current generation has moved on): is_current() must be
    # False, so the GUI thread must not paint it even though the signal fired.
    base = np.full((8, 8), 128.0, np.float32)
    ctrl = _sync_controller(base, pattern="none", amp=0)
    captured = []
    ctrl.on_frame = captured.append

    stale = replace(_make_anim_request(ctrl, frame_index=1), generation=1)
    ctrl._scheduler._gen = 5   # several fresher renders have since completed

    ctrl._on_frame_rendered(np.zeros((8, 8, 3), np.uint8), stale)

    assert captured == []   # stale (superseded) frame never reaches the sink


# ---- worker lifetime: queued signals must survive worker GC -----------------

def test_frame_delivery_survives_worker_gc():
    """Regression: nothing kept the launched _AnimRenderWorker alive, so its
    signals QObject died when run() returned and the queued finished emission
    was dropped -- the frame never reached the sink AND the scheduler stayed
    busy forever, so play/scrub/amplitude went completely dead. Must use the
    REAL thread pool with no test-held worker reference (the inline stand-in
    pool used elsewhere keeps the worker alive and masks the bug)."""
    import gc
    import time

    base = np.full((16, 16), 128.0, np.float32)
    p = TimelinePanel(length=5)
    p.pattern_combo.setCurrentText("static")
    p.amp_slider.setValue(60)
    settings = RenderSettings(style="Bayer-Matrix 4x4", scale=1)
    ctrl = AnimationController(
        p, RenderPipeline(registry), lambda: (base, settings),
        Timeline(length=5), seed=0)
    captured = []
    ctrl.on_frame = captured.append

    ctrl.render_frame(2)

    deadline = time.time() + 10.0
    while time.time() < deadline and not captured:
        gc.collect()
        _app.processEvents()
        time.sleep(0.01)

    assert captured, "frame result signal was dropped; animation preview dead"
    assert captured[0].shape == (16, 16, 3)
    # the scheduler must be released too, or every later frame wedges
    assert not ctrl._scheduler._busy


# ---- export stays exact and cap-independent (hazard #2) ---------------------

def test_export_never_reads_cap_provider():
    base = np.full((16, 16), 128.0, np.float32)

    def poisoned_cap():
        raise AssertionError("export must never read the preview cap")

    ctrl = _sync_controller(base, cap_provider=poisoned_cap)
    calls = []

    def fake_export_animation(pipeline, gray, settings, timeline, pattern, amp,
                              out_path, fps=24, seed=0):
        calls.append((gray, settings, pattern, amp, out_path, fps, seed))
        return out_path

    import ditherzam.animation as animation_mod
    orig = animation_mod.export_animation
    animation_mod.export_animation = fake_export_animation
    try:
        result = ctrl.export("out.mp4", fps=24)
    finally:
        animation_mod.export_animation = orig

    assert result == "out.mp4"
    assert len(calls) == 1


def test_export_output_identical_regardless_of_selected_cap():
    base = np.random.default_rng(7).uniform(0, 255, (32, 32)).astype(np.float32)
    settings = RenderSettings(style="Bayer-Matrix 4x4", scale=2)

    def _export_frames(cap_value):
        ctrl = _sync_controller(
            base, cap_provider=lambda: cap_value, settings=settings,
            pattern="static", amp=40, length=3)
        from ditherzam.animation import render_animation
        return list(render_animation(
            ctrl.pipeline, base, settings, ctrl.timeline,
            ctrl.panel.pattern(), ctrl.panel.amplitude(), seed=ctrl.seed))

    frames_small_cap = _export_frames(64)
    frames_large_cap = _export_frames(4000)

    assert len(frames_small_cap) == len(frames_large_cap)
    for a, b in zip(frames_small_cap, frames_large_cap):
        np.testing.assert_array_equal(a, b)


def test_export_renders_from_export_pipeline_snapshot(monkeypatch):
    """When an export-pipeline provider is set, export() renders the animation
    from that snapshot, not the live preview pipeline (Task 4.2)."""
    import ditherzam.animation as anim_mod
    base = np.full((16, 16), 128.0, np.float32)
    ctrl = _sync_controller(base)
    snapshot = object()
    ctrl.export_pipeline_provider = lambda: snapshot

    captured = {}

    def fake_export_animation(pipeline, *a, **k):
        captured["pipeline"] = pipeline
        return "out.mp4"

    monkeypatch.setattr(anim_mod, "export_animation", fake_export_animation)

    result = ctrl.export("out.mp4", fps=24)

    assert captured["pipeline"] is snapshot
    assert captured["pipeline"] is not ctrl.pipeline
    assert result == "out.mp4"
