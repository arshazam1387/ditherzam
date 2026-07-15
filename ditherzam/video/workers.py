"""Qt QRunnable workers for the video pipeline.

This is the ONLY module under ditherzam/video/ permitted to import PySide6 (see
roadmap "Core is Qt-free"). Each worker delegates to the pure, headless functions
in ffmpeg.py / frames.py and reports via Signals.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .ffmpeg import (
    assemble_video, cmd_extract_frames, run_command,
)
from .frames import dither_frames
from ..threading_policy import export_budget, numba_threads


class WorkerSignals(QObject):
    finished = Signal(object)   # payload varies per worker
    error = Signal(str)
    progress = Signal(int, int)  # (done, total)


class VideoImportWorker(QRunnable):
    """Extract original frames from a video into `frames_dir` (spec §12.3)."""

    def __init__(self, video, frames_dir, runner=run_command) -> None:
        super().__init__()
        self.video = video
        self.frames_dir = str(frames_dir)
        self.runner = runner
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            os.makedirs(self.frames_dir, exist_ok=True)
            self.runner(cmd_extract_frames(self.video, self.frames_dir))
            self.signals.finished.emit(self.frames_dir)
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI thread
            self.signals.error.emit(str(e))


class VideoDitherWorker(QRunnable):
    """Dither every extracted frame; cancellable mid-run (spec §12.4)."""

    def __init__(self, in_dir, out_dir, pipeline, settings) -> None:
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.pipeline = pipeline
        self.settings = settings
        self._is_canceled = False
        self.signals = WorkerSignals()

    def cancel(self) -> None:
        self._is_canceled = True

    @Slot()
    def run(self) -> None:
        try:
            # This runs async on a pool thread while the UI stays live; drop to the
            # export budget so a concurrent interactive render keeps its reserve.
            with numba_threads(export_budget()):
                written = dither_frames(
                    self.in_dir, self.out_dir, self.pipeline, self.settings,
                    progress=lambda done, total: self.signals.progress.emit(done, total),
                    is_cancelled=lambda: self._is_canceled,
                )
            self.signals.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(str(e))


class VideoAssembleWorker(QRunnable):
    """Encode dithered frames + mux audio into the final MP4 (spec §12.5)."""

    def __init__(self, frames_dir, fps, orig_video, out, runner=run_command) -> None:
        super().__init__()
        self.frames_dir = frames_dir
        self.fps = fps
        self.orig_video = orig_video
        self.out = out
        self.runner = runner
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            assemble_video(
                self.frames_dir, self.fps, self.orig_video, self.out,
                runner=self.runner,
            )
            self.signals.finished.emit(self.out)
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(str(e))
