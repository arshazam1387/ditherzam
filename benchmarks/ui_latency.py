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


def probe(n_changes: int = 20, interval_ms: float = 40.0, size=(720, 1280)) -> dict:
    import ditherzam.ui.main_window as MW

    app = QApplication.instance() or QApplication([])
    ed = ImageEditor(debounce_ms=20)
    ed._pool.setMaxThreadCount(2)  # realistic: renders are CPU-bound
    ed.load_array(make_gray(*size))

    counters = {"do_render": 0, "delivered": 0}

    # Non-invasive counting: an extra slot on the debounce timer (renders started),
    # and a wrapper on the module-level numpy_to_qimage (one call per render that
    # actually completed in a worker and produced a pixmap).
    ed._debounce.timeout.connect(lambda: counters.__setitem__("do_render", counters["do_render"] + 1))
    _orig_q2i = MW.numpy_to_qimage

    def _counted_q2i(arr):
        counters["delivered"] += 1
        return _orig_q2i(arr)
    MW.numpy_to_qimage = _counted_q2i

    ed.render_now()  # initial paint + JIT warm
    counters["delivered"] = 0  # discount the warm render

    t0 = time.perf_counter()
    for i in range(n_changes):
        ed.panel.state["luminance_threshold"] = 10 + (i * 7) % 80
        ed.schedule_render()
        _pump(interval_ms)

    # drain: run the debounce + all queued workers to completion, safely
    _pump(200.0)
    ed._pool.waitForDone(10000)
    _pump(200.0)
    total = (time.perf_counter() - t0) * 1000.0
    MW.numpy_to_qimage = _orig_q2i
    return {
        "n_changes": n_changes,
        "do_render_calls": counters["do_render"],
        "delivered_pixmaps": counters["delivered"],
        "wasted_paints": max(0, counters["delivered"] - 1),
        "drag+drain_ms": round(total, 1),
    }


def main() -> None:
    r = probe()
    print("== UI drag latency probe (720p) ==")
    for k, v in r.items():
        print(f"  {k:>18}: {v}")


if __name__ == "__main__":
    main()
