"""Bounded threading policy — pure budget math + best-effort numba masking."""
import os
import types
from contextlib import contextmanager

import pytest

from ditherzam import threading_policy as tp


def test_snap_to_supported_picks_largest_supported_le_n():
    assert tp.snap_to_supported(0) == 1
    assert tp.snap_to_supported(1) == 1
    assert tp.snap_to_supported(3) == 2
    assert tp.snap_to_supported(4) == 4
    assert tp.snap_to_supported(7) == 4
    assert tp.snap_to_supported(8) == 8
    assert tp.snap_to_supported(16) == 8   # capped at the max supported count


def test_interactive_budget_caps_at_four():
    # 4.3 baseline: parallel color kernels plateau by ~4 threads.
    assert tp.interactive_budget(1) == 1
    assert tp.interactive_budget(2) == 2
    assert tp.interactive_budget(4) == 4
    assert tp.interactive_budget(8) == 4
    assert tp.interactive_budget(16) == 4


def test_export_budget_reserves_interactive_headroom():
    # Export claims cpu minus the interactive reserve, snapped to supported.
    assert tp.export_budget(1) == 1
    assert tp.export_budget(2) == 1
    assert tp.export_budget(4) == 1        # 4 - 4 interactive -> min 1
    assert tp.export_budget(8) == 4        # 8 - 4 interactive -> 4
    assert tp.export_budget(16) == 8       # 16 - 4 interactive -> 12 -> snap 8
    for cpu in (1, 2, 4, 8, 16):
        assert tp.export_budget(cpu) >= 1


def test_budgets_default_to_detected_cpu(monkeypatch):
    monkeypatch.setattr(tp, "cpu_threads", lambda: 8)
    assert tp.interactive_budget() == 4
    assert tp.export_budget() == 4


def test_numba_threads_sets_and_restores(monkeypatch):
    state = {"n": 3}
    fake = types.SimpleNamespace(
        get_num_threads=lambda: state["n"],
        set_num_threads=lambda v: state.__setitem__("n", v))
    monkeypatch.setattr(tp, "_numba", lambda: fake)
    with tp.numba_threads(2):
        assert state["n"] == 2
    assert state["n"] == 3                  # restored even without an error


def test_numba_threads_restores_on_exception(monkeypatch):
    state = {"n": 5}
    fake = types.SimpleNamespace(
        get_num_threads=lambda: state["n"],
        set_num_threads=lambda v: state.__setitem__("n", v))
    monkeypatch.setattr(tp, "_numba", lambda: fake)
    try:
        with tp.numba_threads(1):
            assert state["n"] == 1
            raise ValueError("boom")
    except ValueError:
        pass
    assert state["n"] == 5


def test_numba_threads_best_effort_when_numba_unavailable(monkeypatch):
    monkeypatch.setattr(tp, "_numba", lambda: None)
    with tp.numba_threads(2):
        pass                                # no crash, no-op


def test_install_interactive_budget_sets_process_default(monkeypatch):
    state = {"n": 0}
    fake = types.SimpleNamespace(
        get_num_threads=lambda: state["n"],
        set_num_threads=lambda v: state.__setitem__("n", v))
    monkeypatch.setattr(tp, "_numba", lambda: fake)
    monkeypatch.setattr(tp, "cpu_threads", lambda: 8)
    assert tp.install_interactive_budget() == 4
    assert state["n"] == 4


def test_video_dither_worker_runs_under_export_budget(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from ditherzam.video import workers

    entered = []

    @contextmanager
    def spy_budget(n):
        entered.append(n)
        yield

    monkeypatch.setattr(workers, "numba_threads", spy_budget)
    monkeypatch.setattr(workers, "export_budget", lambda: 4)
    # Only assert the frames call happens inside the budget window.
    monkeypatch.setattr(
        workers, "dither_frames",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no budget"))
        if not entered else ["f0"])

    worker = workers.VideoDitherWorker("in", "out", pipeline=None, settings=None)
    finished = []
    worker.signals.finished.connect(finished.append)
    worker.run()

    assert entered == [4]              # export budget applied
    assert finished == [["f0"]]        # dither ran inside the budget window
