"""UI glue for the video pipeline: menu actions, worker orchestration, playback.

Qt-only (allowed under ditherzam/ui/). All heavy lifting is delegated to the
headless functions in ditherzam.video.* and to the QRunnable workers.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox, QProgressDialog

from ditherzam.video.ffmpeg import check_video_limits, probe_duration, probe_fps
from ditherzam.video.frames import detect_preview_frame
from ditherzam.video.workers import (
    VideoAssembleWorker, VideoDitherWorker, VideoImportWorker,
)

from .convert import numpy_to_qimage

_VIDEO_FILTER = "Video Files (*.mp4 *.avi *.mov *.mkv)"
_MP4_FILTER = "MP4 Files (*.mp4)"


def _new_temp_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "ditherzam" / uuid.uuid4().hex
    (d / "original_frames").mkdir(parents=True, exist_ok=True)
    (d / "dithered_frames").mkdir(parents=True, exist_ok=True)
    return d


class FramePlayer:
    """QTimer-driven playback over a directory of dithered frames."""

    def __init__(self, on_frame, fps: float = 24.0) -> None:
        self._on_frame = on_frame
        self._frames: list[Path] = []
        self._idx = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self.step)
        self.set_fps(fps)

    def set_fps(self, fps: float) -> None:
        interval = int(1000.0 / fps) if fps and fps > 0 else 42
        self._timer.setInterval(max(1, interval))

    def load(self, frames_dir) -> None:
        self._frames = sorted(Path(frames_dir).glob("frame*.png"))
        self._idx = 0

    def start(self) -> None:
        if self._frames:
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def step(self) -> None:
        if not self._frames:
            return
        p = self._frames[self._idx % len(self._frames)]
        self._on_frame(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
        self._idx += 1


class VideoController:
    """Owns video state and wires the Video menu to workers."""

    def __init__(self, main_window, pipeline, settings_provider, expert_provider) -> None:
        self.win = main_window
        self.pipeline = pipeline
        self._settings_provider = settings_provider   # () -> RenderSettings
        self._expert_provider = expert_provider        # () -> bool
        self.pool = QThreadPool.globalInstance()
        self.temp_dir: Path | None = None
        self.input_file: str | None = None
        self.framerate: float = 24.0
        self.player = FramePlayer(self._show_frame)

    # --- menu construction ---
    def build_menu(self) -> QMenu:
        menu = QMenu("Video", self.win)
        menu.addAction("Import Video", self.import_video)
        self._export_action = menu.addAction("Export Video", self.export_video)
        self._export_action.setEnabled(False)
        return menu

    def _show_frame(self, rgb_u8: np.ndarray) -> None:
        # Convert the HxWx3 uint8 array to a pixmap for the graphics viewport.
        self.win.viewport.set_pixmap(QPixmap.fromImage(numpy_to_qimage(rgb_u8)))

    def _error(self, msg: str) -> None:
        QMessageBox.critical(self.win, "Video", msg)

    # --- import ---
    def import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.win, "Import Video", "", _VIDEO_FILTER)
        if not path:
            return
        expert = bool(self._expert_provider())
        try:
            fps = probe_fps(path)
            duration = probe_duration(path)
        except Exception as e:  # noqa: BLE001
            self._error(str(e))
            return
        limit_msg = check_video_limits(fps, duration, expert=expert)
        if limit_msg is not None:
            self._error(limit_msg)
            return

        self.temp_dir = _new_temp_dir()
        self.input_file = path
        self.framerate = fps or 24.0
        self.player.set_fps(self.framerate)

        dlg = QProgressDialog("Extracting frames...", "", 0, 0, self.win)
        dlg.setCancelButton(None)
        dlg.show()

        worker = VideoImportWorker(path, str(self.temp_dir / "original_frames"))
        worker.signals.error.connect(lambda m: (dlg.close(), self._error(m)))
        worker.signals.finished.connect(lambda _fd: self._on_imported(dlg))
        self.pool.start(worker)

    def _on_imported(self, dlg) -> None:
        dlg.close()
        assert self.temp_dir is not None
        preview = detect_preview_frame(self.temp_dir / "original_frames")
        if preview is None:
            self._error("Could not detect a non-black frame for preview.")
        else:
            self._show_frame(np.asarray(Image.open(preview).convert("RGB"), np.uint8))
        self._export_action.setEnabled(True)

    # --- export ---
    def export_video(self) -> None:
        if self.temp_dir is None:
            return
        out, _ = QFileDialog.getSaveFileName(self.win, "Export Video", "", _MP4_FILTER)
        if not out:
            return
        in_dir = self.temp_dir / "original_frames"
        out_dir = self.temp_dir / "dithered_frames"

        prog = QProgressDialog("Processing frames...", "Cancel", 0, 100, self.win)
        dither = VideoDitherWorker(
            str(in_dir), str(out_dir), self.pipeline, self._settings_provider()
        )
        prog.canceled.connect(dither.cancel)

        def on_dither_progress(done: int, total: int) -> None:
            prog.setMaximum(max(total, 1))
            prog.setValue(done)

        def on_dither_done(_written: int) -> None:
            prog.close()
            self.player.load(out_dir)
            self.player.start()
            reasm = QProgressDialog("Reassembling video...", "", 0, 0, self.win)
            reasm.setCancelButton(None)
            reasm.show()
            assemble = VideoAssembleWorker(
                str(out_dir), self.framerate, self.input_file, out
            )
            assemble.signals.error.connect(lambda m: (reasm.close(), self._error(m)))
            assemble.signals.finished.connect(
                lambda _o: (reasm.close(),
                            QMessageBox.information(self.win, "Video",
                                                    "Video export complete!"))
            )
            self.pool.start(assemble)

        dither.signals.progress.connect(on_dither_progress)
        dither.signals.error.connect(lambda m: (prog.close(), self._error(m)))
        dither.signals.finished.connect(on_dither_done)
        self.pool.start(dither)
