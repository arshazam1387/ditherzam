"""A render worker that raises must not permanently wedge the coalescer.

Regression for the "app gets stuck the more you use it" bug: an unhandled
exception in the background render thread left RenderCoalescer._busy stuck True,
so every subsequent render request coalesced to None and the preview froze
forever. See systematic-debugging session 2026-07-08.
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

    monkeypatch.setattr(RenderPipeline, "render_cached", _throw)
    worker = mw._RenderWorker(RenderPipeline(registry), np.zeros((16, 16), np.float32),
                              RenderSettings(), token=7, mode="full")
    got = {"finished": [], "failed": []}
    worker.signals.finished.connect(lambda _img, t: got["finished"].append(t))
    worker.signals.failed.connect(lambda t: got["failed"].append(t))

    worker.run()   # must NOT raise out of run()

    assert got["failed"] == [7]
    assert got["finished"] == []


def test_render_failure_releases_coalescer(qapp_fixture, monkeypatch):
    import ditherzam.ui.main_window as mw

    win = mw.ImageEditor()
    g = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.float32)
    win.load_array(g, np.stack([g] * 3, -1).astype(np.uint8))
    monkeypatch.setattr(mw.RenderPipeline, "render_cached", _throw)
    monkeypatch.setattr(mw, "render_preview", _throw)

    win._render_mode = "full"
    token = win._coalescer.request()
    assert token is not None and win._coalescer._busy is True

    worker = mw._RenderWorker(win.pipeline, win._base_gray,
                              mw.settings_from_controls(win.panel.state),
                              token, mode="full")
    worker.signals.failed.connect(win._on_render_failed)
    worker.run()

    # Despite the failure, the coalescer must be released so future renders run.
    assert win._coalescer._busy is False
