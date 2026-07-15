"""Contract tests for cooperative, stage-boundary preview cancellation.

Cancellation is deliberately an interactive-render concern.  The core accepts
an optional predicate, checks it only between complete stages, and reports a
distinct terminal outcome.  Callers such as export that omit the predicate are
never made cancellable by UI scheduler state.
"""
from __future__ import annotations

import numpy as np
import pytest

import ditherzam.render as render_mod
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.ui.render_request import RenderKind, RenderRequest
from ditherzam.ui.render_scheduler import RenderScheduler


BASE = np.arange(64, dtype=np.float32).reshape(8, 8) * 4
SETTINGS = RenderSettings(style="None", scale=1)
# Keep the red suite collectable before production introduces the public
# outcome type; once present, every test automatically exercises the real one.
RenderCancelled = getattr(render_mod, "RenderCancelled", type("RenderCancelled", (Exception,), {}))


def _request(kind=RenderKind.DRAG, marker=0):
    return RenderRequest(
        generation=0,
        kind=kind,
        settings=RenderSettings(contrast=marker),
        source_id=1,
        target_max_side=64,
        logical_size=(64, 64),
    )


def test_scheduler_marks_obsolete_inflight_request_cancelled():
    scheduler = RenderScheduler()
    first = scheduler.request(_request(marker=1))
    assert first is not None
    assert not scheduler.should_cancel(first)

    scheduler.request(_request(marker=2))

    assert scheduler.should_cancel(first)


def test_cancelled_render_promotes_only_newest_trailing_request():
    scheduler = RenderScheduler()
    first = scheduler.request(_request(marker=1))
    scheduler.request(_request(marker=2))
    scheduler.request(_request(marker=3))

    assert scheduler.should_cancel(first)
    trailing = scheduler.on_finished()  # cancelled is still a terminal outcome

    assert trailing is not None
    assert trailing.settings.contrast == 3
    assert scheduler.is_current(trailing)
    assert not scheduler.should_cancel(trailing)


def test_render_waits_for_running_stage_then_cancels_at_boundary(monkeypatch):
    calls = []
    cancel = {"requested": False}

    def contrast(image, _value):
        calls.append("contrast-start")
        cancel["requested"] = True  # cancellation arrives during this stage
        calls.append("contrast-end")
        return image

    def midtones(image, _value):
        calls.append("midtones")
        return image

    monkeypatch.setattr(render_mod, "apply_contrast", contrast)
    monkeypatch.setattr(render_mod, "apply_midtones", midtones)

    with pytest.raises(RenderCancelled):
        RenderPipeline(registry).render(
            BASE, SETTINGS, is_cancelled=lambda: cancel["requested"]
        )

    assert calls == ["contrast-start", "contrast-end"]


def test_cancelled_cached_render_does_not_publish_partial_cache(monkeypatch):
    pipeline = RenderPipeline(registry)
    # Legitimate prior entry under the real RenderCache (id/.get/.put/.keys),
    # not the plain-dict shape ``render_cache.py`` replaced.
    sentinel = np.array([[17]], dtype=np.float32)
    keep_key = "keep"
    pipeline._cache.put(keep_key, {"marker": sentinel})
    before_keys = pipeline._cache.keys()
    cancel = {"requested": False}

    def contrast(image, _value):
        cancel["requested"] = True
        return image + 1

    monkeypatch.setattr(render_mod, "apply_contrast", contrast)

    with pytest.raises(RenderCancelled):
        pipeline.render_cached(
            BASE, SETTINGS, is_cancelled=lambda: cancel["requested"]
        )

    assert pipeline._cache.keys() == before_keys
    assert pipeline._cache.get(keep_key)["marker"] is sentinel
    assert pipeline._cache.get(id(BASE)) is None


def test_export_style_render_without_predicate_is_never_ui_cancelled():
    """Exact/export callers omit the UI predicate and remain independent."""
    scheduler = RenderScheduler()
    scheduler.request(_request(marker=1))
    scheduler.request(_request(marker=2))  # makes the UI request obsolete

    result = RenderPipeline(registry).render(BASE, SETTINGS)

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8


def test_worker_cancel_is_distinct_and_emits_exactly_one_terminal(qapp_fixture, monkeypatch):
    pytest.importorskip("PySide6")
    import ditherzam.ui.main_window as mw

    request = _request(RenderKind.SETTLE)

    def cancelled(*_args, **_kwargs):
        raise RenderCancelled

    monkeypatch.setattr(mw, "render_preview", cancelled)
    worker = mw._RenderWorker(RenderPipeline(registry), BASE, request)
    got = {"finished": 0, "failed": 0, "cancelled": 0}
    worker.signals.finished.connect(
        lambda *_args: got.__setitem__("finished", got["finished"] + 1)
    )
    worker.signals.failed.connect(
        lambda *_args: got.__setitem__("failed", got["failed"] + 1)
    )
    worker.signals.cancelled.connect(
        lambda *_args: got.__setitem__("cancelled", got["cancelled"] + 1)
    )

    worker.run()

    assert got == {"finished": 0, "failed": 0, "cancelled": 1}


def test_cancelled_worker_does_not_paint_fail_or_wedge_and_launches_trailing(
    qapp_fixture, monkeypatch
):
    pytest.importorskip("PySide6")
    import ditherzam.ui.main_window as mw

    editor = mw.ImageEditor()
    editor._base_gray = BASE
    first = editor._scheduler.request(editor._build_request(RenderKind.DRAG))
    editor._scheduler.request(editor._build_request(RenderKind.SETTLE))
    launched = []
    monkeypatch.setattr(editor, "_launch_worker", launched.append)
    old_image = editor.last_qimage

    editor._on_render_cancelled(first)

    assert editor.last_qimage is old_image
    assert len(launched) == 1
    assert launched[0].kind is RenderKind.SETTLE
    assert editor._scheduler._busy is True
