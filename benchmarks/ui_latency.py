"""Offscreen UI drag-to-pixmap latency probe.

Simulates a rapid slider drag (many `panel.state` changes -> schedule_render) and
measures how many full renders actually executed and how many delivered results
were immediately superseded (wasted paints). Demonstrates the cancel-superseded win.

Run:
    QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m benchmarks.ui_latency
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, QEventLoop

from ditherzam.ui.main_window import ImageEditor
from .common import make_gray


def _pump(ms: float) -> None:
    app = QCoreApplication.instance()
    end = time.perf_counter() + ms / 1000.0
    while time.perf_counter() < end:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 5)
        time.sleep(0.001)


def probe(n_changes: int = 20, interval_ms: float = 40.0, size=(1080, 1920)) -> dict:
    import ditherzam.ui.main_window as MW
    from ditherzam.render import RenderPipeline

    app = QApplication.instance() or QApplication([])
    ed = ImageEditor(debounce_ms=20)
    ed._pool.setMaxThreadCount(2)  # realistic: renders are CPU-bound
    ed.load_array(make_gray(*size))

    counters = {"proxy": 0, "full": 0}
    marks = {"first_feedback": None}
    drag_start = {"t": None}

    # Count proxy vs full renders by wrapping the two pipeline entry points the
    # worker calls, and stamp the time of the first delivered feedback frame.
    _orig_preview = MW.render_preview
    _orig_cached = RenderPipeline.render_cached

    def _counted_preview(pipe, base, settings, max_side):
        counters["proxy"] += 1
        out = _orig_preview(pipe, base, settings, max_side)
        if marks["first_feedback"] is None and drag_start["t"] is not None:
            # stamp when the first proxy result is ready (~ first on-screen feedback)
            marks["first_feedback"] = (time.perf_counter() - drag_start["t"]) * 1000.0
        return out

    def _counted_cached(self, base, settings, temporal_field=None):
        counters["full"] += 1
        return _orig_cached(self, base, settings, temporal_field)

    MW.render_preview = _counted_preview
    RenderPipeline.render_cached = _counted_cached

    ed.render_now()  # initial paint + JIT warm
    counters["full"] = 0

    t0 = time.perf_counter()
    drag_start["t"] = t0
    for i in range(n_changes):
        ed.panel.state["luminance_threshold"] = 10 + (i * 7) % 80
        ed.schedule_render()
        _pump(interval_ms)

    # drain: run the debounce + settle + all queued workers to completion, safely
    _pump(400.0)
    ed._pool.waitForDone(10000)
    _pump(200.0)
    total = (time.perf_counter() - t0) * 1000.0
    MW.render_preview = _orig_preview
    RenderPipeline.render_cached = _orig_cached
    ff = marks["first_feedback"]
    return {
        "n_changes": n_changes,
        "proxy_renders": counters["proxy"],
        "full_renders": counters["full"],
        "time_to_first_feedback_ms": round(ff, 1) if ff is not None else None,
        "drag+drain_ms": round(total, 1),
    }


def main() -> None:
    r = probe()
    print("== UI drag latency probe (1080p, 20-step luminance drag) ==")
    for k, v in r.items():
        print(f"  {k:>26}: {v}")


if __name__ == "__main__":
    main()
