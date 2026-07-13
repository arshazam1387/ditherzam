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
from .preview import preview_target_size
from ditherzam.masking.scope import mask_allows_media, unsupported_mask_message

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

    def __init__(self, main_window, pipeline, settings_provider, expert_provider,
                 cap_provider=None, export_pipeline_provider=None,
                 mask_settings_provider=None) -> None:
        self.win = main_window
        self.pipeline = pipeline
        self._settings_provider = settings_provider   # () -> RenderSettings
        self._expert_provider = expert_provider        # () -> bool
        self.cap_provider = cap_provider or (lambda: 1440)   # () -> int, Qt-free
        # () -> RenderPipeline snapshot for EXACT export, isolated from live edits;
        # falls back to the live pipeline when unset (tests/back-compat).
        self.export_pipeline_provider = export_pipeline_provider
        self.mask_settings_provider = mask_settings_provider
        self.pool = QThreadPool.globalInstance()
        self._active_workers: set = set()
        self.temp_dir: Path | None = None
        self.input_file: str | None = None
        self.framerate: float = 24.0
        self.player = FramePlayer(self._show_frame)

    def _start_worker(self, worker) -> None:
        """Start a QRunnable worker, holding the Python wrapper until a terminal
        signal is delivered. QThreadPool owns only the C++ runnable and deletes
        it the moment run() returns; without this reference the WorkerSignals
        QObject is destroyed with the wrapper, and the queued finished/error
        emission (posted to the GUI thread but not yet delivered) is dropped --
        the export chain then stalls silently after the worker completes."""
        self._active_workers.add(worker)

        def release(*_args) -> None:
            self._active_workers.discard(worker)

        worker.signals.finished.connect(release)
        worker.signals.error.connect(release)
        self.pool.start(worker)

    # --- menu construction ---
    def build_menu(self) -> QMenu:
        menu = QMenu("Video", self.win)
        self._import_action = menu.addAction("Import Video", self.import_video)
        self._export_action = menu.addAction("Export Video", self.export_video)
        self._export_action.setEnabled(False)
        return menu

    def _show_frame(self, rgb_u8: np.ndarray) -> None:
        # Display-only: cap BEFORE QImage conversion (Task 4.1). The on-disk
        # dithered frames (the export) are never touched -- only this
        # in-memory copy shown in the viewport is downscaled.
        h, w = rgb_u8.shape[:2]
        cap = max(1, int(self.cap_provider()))
        target_h, target_w = preview_target_size(h, w, cap)
        display = rgb_u8
        if (target_h, target_w) != (h, w):
            display = np.asarray(
                Image.fromarray(rgb_u8).resize((target_w, target_h), Image.NEAREST),
                dtype=np.uint8)
        qimg = numpy_to_qimage(display)
        # Source-logical size preserved so the capped pixmap fills the
        # viewport the same way the still-image capped preview does.
        self.win.viewport.set_pixmap(QPixmap.fromImage(qimg), logical_size=(w, h))

    def _error(self, msg: str) -> None:
        QMessageBox.critical(self.win, "Video", msg)

    def _mask_allows_video(self) -> bool:
        if (self.mask_settings_provider is None
                or mask_allows_media("video", self.mask_settings_provider())):
            return True
        QMessageBox.warning(self.win, "Smart Mask", unsupported_mask_message("video"))
        return False

    def refresh_mask_scope(self) -> None:
        """Keep import/export availability truthful without weakening prerequisites."""
        allowed = (self.mask_settings_provider is None
                   or mask_allows_media("video", self.mask_settings_provider()))
        message = "" if allowed else unsupported_mask_message("video")
        import_action = getattr(self, "_import_action", None)
        if import_action is not None:
            import_action.setEnabled(allowed)
            import_action.setToolTip(message)
            import_action.setStatusTip(message)
        export_action = getattr(self, "_export_action", None)
        if export_action is not None:
            export_action.setEnabled(self.temp_dir is not None and allowed)
            export_action.setToolTip(message)
            export_action.setStatusTip(message)

    # --- import ---
    def import_video(self) -> None:
        if not self._mask_allows_video():
            return
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
        self._start_worker(worker)

    def _on_imported(self, dlg) -> None:
        dlg.close()
        assert self.temp_dir is not None
        preview = detect_preview_frame(self.temp_dir / "original_frames")
        if preview is None:
            self._error("Could not detect a non-black frame for preview.")
        else:
            self._show_frame(np.asarray(Image.open(preview).convert("RGB"), np.uint8))
        self._export_action.setEnabled(True)
        self.refresh_mask_scope()

    # --- export ---
    def export_video(self) -> None:
        if self.temp_dir is None:
            return
        if not self._mask_allows_video():
            return
        out, _ = QFileDialog.getSaveFileName(self.win, "Export Video", "", _MP4_FILTER)
        if not out:
            return
        in_dir = self.temp_dir / "original_frames"
        out_dir = self.temp_dir / "dithered_frames"

        # Snapshot the render context at launch so a later UI edit cannot change
        # this in-flight (async) export.
        export_pipeline = (self.export_pipeline_provider()
                           if self.export_pipeline_provider else self.pipeline)
        prog = QProgressDialog("Processing frames...", "Cancel", 0, 100, self.win)
        dither = VideoDitherWorker(
            str(in_dir), str(out_dir), export_pipeline, self._settings_provider()
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
            self._start_worker(assemble)

        dither.signals.progress.connect(on_dither_progress)
        dither.signals.error.connect(lambda m: (prog.close(), self._error(m)))
        dither.signals.finished.connect(on_dither_done)
        self._start_worker(dither)
