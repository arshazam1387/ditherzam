import pytest

pytest.importorskip("PySide6")  # skip on headless boxes without Qt

from PySide6.QtWidgets import QApplication
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.video.workers import (
    WorkerSignals, VideoImportWorker, VideoDitherWorker, VideoAssembleWorker,
)

_app = QApplication.instance() or QApplication([])


def test_signals_have_expected_channels():
    s = WorkerSignals()
    assert hasattr(s, "finished") and hasattr(s, "error") and hasattr(s, "progress")


def test_import_worker_constructs_with_fake_runner():
    w = VideoImportWorker("in.mp4", "/frames", runner=lambda cmd: "")
    assert hasattr(w.signals, "finished")


def test_dither_worker_is_cancellable():
    pipe = RenderPipeline(registry)
    w = VideoDitherWorker("/in", "/out", pipe, RenderSettings(style="Floyd-Steinberg"))
    assert w._is_canceled is False
    w.cancel()
    assert w._is_canceled is True


def test_assemble_worker_constructs():
    w = VideoAssembleWorker("/frames", 30, None, "out.mp4", runner=lambda cmd: "")
    assert hasattr(w.signals, "progress")


def test_import_worker_run_emits_finished(tmp_path):
    frames = tmp_path / "frames"
    got = {}
    w = VideoImportWorker("in.mp4", str(frames), runner=lambda cmd: "")
    w.signals.finished.connect(lambda payload: got.setdefault("done", payload))
    w.run()
    assert got.get("done") == str(frames)


def test_import_worker_run_emits_error_on_failure():
    def boom(cmd):
        raise RuntimeError("ffmpeg exploded")
    w = VideoImportWorker("in.mp4", "/frames", runner=boom)
    errs = []
    w.signals.error.connect(lambda m: errs.append(m))
    w.run()
    assert errs and "exploded" in errs[0]
