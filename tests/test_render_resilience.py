"""A render worker that raises must not permanently wedge the scheduler.

Regression for the "app gets stuck the more you use it" bug: an unhandled
exception in the background render thread left RenderScheduler._busy stuck
True, so every subsequent render request coalesced to None and the preview
froze forever. See systematic-debugging session 2026-07-08.
"""
import numpy as np
import pytest

pytest.importorskip("PySide6")


def _throw(*_a, **_k):
    raise RuntimeError("simulated render failure")


def test_worker_emits_failed_not_finished_on_exception(qapp_fixture, monkeypatch):
    import ditherzam.ui.main_window as mw
    from ditherzam.render import RenderPipeline, RenderSettings
    from ditherzam.dithering import registry
    from ditherzam.ui.render_request import RenderKind, RenderRequest

    # FULL is the only kind whose mode is "full" (render_cached); SETTLE/DRAG/ZOOM
    # now render through the capped render_preview path (task 2.3).
    monkeypatch.setattr(RenderPipeline, "render_cached", _throw)
    request = RenderRequest(
        generation=7, kind=RenderKind.FULL, settings=RenderSettings(),
        source_id=1, target_max_side=16, logical_size=(16, 16),
    )
    worker = mw._RenderWorker(RenderPipeline(registry), np.zeros((16, 16), np.float32),
                              request)
    got = {"finished": [], "failed": []}
    worker.signals.finished.connect(lambda _img, r: got["finished"].append(r))
    worker.signals.failed.connect(lambda r: got["failed"].append(r))

    worker.run()   # must NOT raise out of run()

    assert got["failed"] == [request]
    assert got["finished"] == []


def test_render_failure_releases_scheduler(qapp_fixture, monkeypatch):
    import ditherzam.ui.main_window as mw

    win = mw.ImageEditor()
    g = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.float32)
    win.load_array(g, np.stack([g] * 3, -1).astype(np.uint8))
    monkeypatch.setattr(mw.RenderPipeline, "render_cached", _throw)
    monkeypatch.setattr(mw, "render_preview", _throw)

    request = win._scheduler.request(win._build_request(mw.RenderKind.SETTLE))
    assert request is not None and win._scheduler._busy is True

    worker = mw._RenderWorker(win.pipeline, win._base_gray, request)
    worker.signals.failed.connect(win._on_render_failed)
    worker.run()

    # Despite the failure, the scheduler must be released so future renders run.
    assert win._scheduler._busy is False
