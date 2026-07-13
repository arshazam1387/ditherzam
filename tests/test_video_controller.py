"""Task 4.1: capped, display-only video frame previews.

Video playback shows already-dithered PNG frames read from disk
(``dithered_frames/``) at full resolution today. This caps the in-memory
raster BEFORE QImage conversion so the GUI never paints/scales an oversized
pixmap, while the on-disk dithered frames (which ARE the export) and the
ffmpeg assemble/dither workers stay completely untouched and cap-independent.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtWidgets import QApplication, QWidget

import ditherzam.ui.video_controller as vc_mod
from ditherzam.ui.video_controller import VideoController

_app = QApplication.instance() or QApplication([])


class _FakeViewport:
    def __init__(self) -> None:
        self.calls = []   # (pixmap, logical_size, refit)

    def set_pixmap(self, pixmap, logical_size=None, refit=True) -> None:
        self.calls.append((pixmap, logical_size, refit))


def _window():
    win = QWidget()
    win.viewport = _FakeViewport()
    return win


def _controller(cap=1440, pipeline=None, settings=None):
    win = _window()
    ctrl = VideoController(
        win, pipeline, settings_provider=lambda: settings,
        expert_provider=lambda: False, cap_provider=lambda: cap)
    return ctrl


# ---- (a)/(e): display frame capped before QImage conversion -----------------

def test_show_frame_caps_before_qimage_conversion():
    ctrl = _controller(cap=64)
    rgb = np.random.default_rng(0).integers(0, 255, (256, 512, 3), dtype=np.uint8)

    ctrl._show_frame(rgb)

    assert len(ctrl.win.viewport.calls) == 1
    pixmap, logical_size, _refit = ctrl.win.viewport.calls[0]
    assert pixmap.width() == 64 and pixmap.height() == 32   # aspect-preserving cap
    assert logical_size == (512, 256)   # source-logical geometry preserved (hazard #4)


def test_show_frame_no_downscale_when_already_within_cap():
    ctrl = _controller(cap=1440)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    ctrl._show_frame(rgb)

    pixmap, logical_size, _refit = ctrl.win.viewport.calls[0]
    assert pixmap.width() == 64 and pixmap.height() == 64
    assert logical_size == (64, 64)


def test_show_frame_uses_current_cap_provider_value_each_call():
    caps = iter([64, 1440])
    win = _window()
    ctrl = VideoController(win, None, lambda: None, lambda: False,
                           cap_provider=lambda: next(caps))
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    ctrl._show_frame(rgb)
    ctrl._show_frame(rgb)

    first, second = ctrl.win.viewport.calls
    assert first[0].width() == 64
    assert second[0].width() == 256   # already fits under 1440, unchanged


def test_show_frame_does_not_mutate_source_file_on_disk(tmp_path):
    from PIL import Image
    p = tmp_path / "frame.png"
    Image.fromarray(np.full((300, 300, 3), 200, dtype=np.uint8)).save(p)
    before = p.read_bytes()
    ctrl = _controller(cap=64)
    arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)

    ctrl._show_frame(arr)

    assert p.read_bytes() == before   # display-only; the on-disk frame is untouched


# ---- imported preview frame also goes through the capped path ---------------

def test_imported_preview_frame_is_capped(tmp_path, monkeypatch):
    from PIL import Image

    ctrl = _controller(cap=64)
    ctrl.build_menu()   # populates _export_action, referenced by _on_imported
    ctrl.temp_dir = tmp_path
    frames_dir = tmp_path / "original_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((200, 400, 3), 180, dtype=np.uint8)).save(frames_dir / "frame000000.png")

    monkeypatch.setattr(vc_mod, "detect_preview_frame",
                        lambda d: str(frames_dir / "frame000000.png"))

    class _Dlg:
        def close(self):
            pass

    ctrl._on_imported(_Dlg())

    assert len(ctrl.win.viewport.calls) == 1
    pixmap, logical_size, _refit = ctrl.win.viewport.calls[0]
    assert max(pixmap.width(), pixmap.height()) == 64
    assert logical_size == (400, 200)


# ---- default cap_provider is Qt-free and safe with no image loaded ---------

def test_default_cap_provider_is_sane_when_none_given():
    win = _window()
    ctrl = VideoController(win, None, lambda: None, lambda: False)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    ctrl._show_frame(rgb)   # must not raise

    pixmap, _logical, _refit = ctrl.win.viewport.calls[0]
    assert pixmap.width() == 64 and pixmap.height() == 64


# ---- hazard #2: export is exact and cap-independent --------------------------

class _FakeSignals(QObject):
    progress = Signal(int, int)
    error = Signal(str)
    finished = Signal(object)


class _FakeDitherWorker:
    def __init__(self, in_dir, out_dir, pipeline, settings) -> None:
        self.args = (in_dir, out_dir, pipeline, settings)
        self.signals = _FakeSignals()

    def cancel(self) -> None:
        pass

    def run(self) -> None:
        pass


class _FakeAssembleWorker:
    def __init__(self, frames_dir, fps, orig_video, out) -> None:
        self.args = (frames_dir, fps, orig_video, out)
        self.signals = _FakeSignals()

    def run(self) -> None:
        pass


def _prep_export(monkeypatch, tmp_path, cap_provider, pipeline, settings):
    win = _window()
    ctrl = VideoController(win, pipeline, lambda: settings, lambda: False,
                           cap_provider=cap_provider)
    ctrl.temp_dir = tmp_path
    (tmp_path / "original_frames").mkdir(parents=True, exist_ok=True)
    (tmp_path / "dithered_frames").mkdir(parents=True, exist_ok=True)
    ctrl.input_file = "in.mp4"
    ctrl.framerate = 24.0

    monkeypatch.setattr(vc_mod, "VideoDitherWorker", _FakeDitherWorker)
    monkeypatch.setattr(vc_mod, "VideoAssembleWorker", _FakeAssembleWorker)
    monkeypatch.setattr(vc_mod.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "out.mp4"), ""))
    started = []
    monkeypatch.setattr(ctrl.pool, "start", started.append)
    return ctrl, started


def test_export_video_never_reads_cap_provider(tmp_path, monkeypatch):
    def poisoned_cap():
        raise AssertionError("export must never read the preview cap")

    pipeline, settings = object(), object()
    ctrl, started = _prep_export(monkeypatch, tmp_path, poisoned_cap, pipeline, settings)

    ctrl.export_video()   # must not raise -- cap_provider is never invoked

    assert len(started) == 1
    in_dir, out_dir, used_pipeline, used_settings = started[0].args
    assert in_dir == str(tmp_path / "original_frames")
    assert out_dir == str(tmp_path / "dithered_frames")
    assert used_pipeline is pipeline
    assert used_settings is settings


def test_export_video_worker_args_identical_regardless_of_cap(tmp_path, monkeypatch):
    pipeline, settings = object(), object()
    results = []
    for cap_value in (64, 4000):
        ctrl, started = _prep_export(
            monkeypatch, tmp_path, lambda cap_value=cap_value: cap_value, pipeline, settings)
        ctrl.export_video()
        results.append(started[0].args)

    assert results[0] == results[1]


# ---- worker lifetime: queued terminal signals must survive worker GC --------

class _ThreadedDitherWorker(QRunnable):
    """Real QRunnable that emits finished from a pool thread, like the real
    worker. The pool deletes it (autoDelete) the instant run() returns, so the
    queued finished emission is only delivered if the controller kept the
    Python wrapper (and its signals QObject) alive."""

    def __init__(self, in_dir, out_dir, pipeline, settings) -> None:
        super().__init__()
        self.signals = _FakeSignals()

    def cancel(self) -> None:
        pass

    def run(self) -> None:
        self.signals.finished.emit(1)


_assemble_instances: list = []


class _RecordingAssembleWorker(QRunnable):
    def __init__(self, frames_dir, fps, orig_video, out) -> None:
        super().__init__()
        _assemble_instances.append(self)
        self.signals = _FakeSignals()

    def run(self) -> None:
        self.signals.finished.emit("out")


def test_dither_finished_survives_worker_gc(tmp_path, monkeypatch):
    """Regression: nothing kept the started worker alive, so the WorkerSignals
    QObject was destroyed when run() returned and the queued finished/error
    emission was silently dropped -- export stalled after the dither phase with
    no output and no error dialog."""
    import gc
    import time

    _assemble_instances.clear()
    win = _window()
    ctrl = VideoController(win, object(), lambda: object(), lambda: False,
                           cap_provider=lambda: 1440)
    ctrl.temp_dir = tmp_path
    (tmp_path / "original_frames").mkdir(parents=True, exist_ok=True)
    (tmp_path / "dithered_frames").mkdir(parents=True, exist_ok=True)
    ctrl.input_file = "in.mp4"
    ctrl.framerate = 24.0
    monkeypatch.setattr(vc_mod, "VideoDitherWorker", _ThreadedDitherWorker)
    monkeypatch.setattr(vc_mod, "VideoAssembleWorker", _RecordingAssembleWorker)
    monkeypatch.setattr(vc_mod.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "out.mp4"), ""))

    class _SilentMessageBox:   # the modal export-complete box would block the loop
        critical = staticmethod(lambda *a, **k: None)
        warning = staticmethod(lambda *a, **k: None)
        information = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(vc_mod, "QMessageBox", _SilentMessageBox)

    ctrl.export_video()   # real QThreadPool; no test-held reference to the worker

    deadline = time.time() + 10.0
    while time.time() < deadline and not _assemble_instances:
        gc.collect()
        _app.processEvents()
        time.sleep(0.01)

    assert len(_assemble_instances) == 1, \
        "dither finished signal was dropped; assemble phase never started"

    # and the assemble worker's own finished must arrive too (export-complete UI)
    deadline = time.time() + 10.0
    while time.time() < deadline and ctrl._active_workers:
        gc.collect()
        _app.processEvents()
        time.sleep(0.01)
    assert not ctrl._active_workers, "terminal signals must release kept workers"


def test_export_video_uses_export_pipeline_snapshot(tmp_path, monkeypatch):
    """The async dither worker must receive the export-pipeline snapshot, not the
    live shared pipeline that later UI edits mutate (Task 4.2)."""
    live_pipeline, settings, snapshot = object(), object(), object()
    win = _window()
    ctrl = VideoController(win, live_pipeline, lambda: settings, lambda: False,
                           cap_provider=lambda: 1440,
                           export_pipeline_provider=lambda: snapshot)
    ctrl.temp_dir = tmp_path
    (tmp_path / "original_frames").mkdir(parents=True, exist_ok=True)
    (tmp_path / "dithered_frames").mkdir(parents=True, exist_ok=True)
    ctrl.input_file = "in.mp4"
    ctrl.framerate = 24.0
    monkeypatch.setattr(vc_mod, "VideoDitherWorker", _FakeDitherWorker)
    monkeypatch.setattr(vc_mod, "VideoAssembleWorker", _FakeAssembleWorker)
    monkeypatch.setattr(vc_mod.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "out.mp4"), ""))
    started = []
    monkeypatch.setattr(ctrl.pool, "start", started.append)

    ctrl.export_video()

    _, _, used_pipeline, used_settings = started[0].args
    assert used_pipeline is snapshot
    assert used_pipeline is not live_pipeline
    assert used_settings is settings
