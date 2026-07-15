from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..animation.temporal import PATTERNS, temporal_noise
from ..render import RenderCancelled
from .preview import render_preview
from .render_request import RenderKind
from .render_scheduler import RenderScheduler


class TimelinePanel(QWidget):
    """Temporal pattern picker, keyframe editor trigger, scrubber and playback."""

    keyframe_requested = Signal(int)   # current frame index
    frame_changed = Signal(int)        # scrub / playback frame index
    export_requested = Signal()
    play_toggled = Signal(bool)

    def __init__(self, length: int = 30, parent=None) -> None:
        super().__init__(parent)

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItem("none")
        self.pattern_combo.addItems(list(PATTERNS))

        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(20)

        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 3600)
        self.length_spin.setValue(length)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, length - 1))

        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.key_btn = QPushButton("Add Keyframe")
        self.export_btn = QPushButton("Export Animation (MP4)")

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // 24)   # ~24 fps preview

        self._build_layout()
        self._connect()

    def _build_layout(self) -> None:
        top = QHBoxLayout()
        top.addWidget(QLabel("Pattern"))
        top.addWidget(self.pattern_combo)
        top.addWidget(QLabel("Amplitude"))
        top.addWidget(self.amp_slider)
        top.addWidget(QLabel("Frames"))
        top.addWidget(self.length_spin)

        mid = QHBoxLayout()
        mid.addWidget(self.play_btn)
        mid.addWidget(self.frame_slider)

        bottom = QHBoxLayout()
        bottom.addWidget(self.key_btn)
        bottom.addWidget(self.export_btn)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid)
        root.addLayout(bottom)

    def _connect(self) -> None:
        self.length_spin.valueChanged.connect(self._on_length)
        self.frame_slider.valueChanged.connect(self.frame_changed.emit)
        self.play_btn.toggled.connect(self._on_play)
        self.key_btn.clicked.connect(
            lambda: self.keyframe_requested.emit(self.frame_slider.value()))
        self.export_btn.clicked.connect(self.export_requested.emit)
        self._timer.timeout.connect(self._advance)

    def _on_length(self, n: int) -> None:
        self.frame_slider.setRange(0, max(0, int(n) - 1))

    def _on_play(self, on: bool) -> None:
        self.play_btn.setText("Pause" if on else "Play")
        if on:
            self._timer.start()
        else:
            self._timer.stop()
        self.play_toggled.emit(on)

    def _advance(self) -> None:
        n = max(1, self.length_spin.value())
        self.frame_slider.setValue((self.frame_slider.value() + 1) % n)

    # --- read-only accessors for controllers ---
    def pattern(self) -> str:
        return self.pattern_combo.currentText()

    def amplitude(self) -> float:
        return float(self.amp_slider.value())

    def length(self) -> int:
        return int(self.length_spin.value())


@dataclass(frozen=True, eq=False)
class _AnimRequest:
    """One immutable animation screen-render ask (mirrors ``RenderRequest`` in
    ``render_request.py`` but scoped to what a capped animation frame needs).
    ``eq=False``: ``temporal_field`` may be a numpy array."""

    generation: int
    kind: RenderKind
    frame_index: int
    settings: object          # RenderSettings
    temporal_field: object    # np.ndarray | None
    target_max_side: int


class _AnimRenderSignals(QObject):
    finished = Signal(object, object)   # (rgb_u8, _AnimRequest)
    failed = Signal(object)             # _AnimRequest
    cancelled = Signal(object)          # _AnimRequest


class _AnimRenderWorker(QRunnable):
    """Runs one capped animation frame render off the GUI thread (mirrors
    ``_RenderWorker`` in main_window.py: exactly one
    terminal signal per run, even on exception)."""

    def __init__(self, pipeline, base_gray, request: _AnimRequest, is_cancelled=None):
        super().__init__()
        self._pipeline = pipeline
        self._base_gray = base_gray
        self._request = request
        self._is_cancelled = is_cancelled
        self.signals = _AnimRenderSignals()

    def run(self) -> None:
        try:
            rgb = render_preview(
                self._pipeline, self._base_gray, self._request.settings,
                self._request.target_max_side,
                is_cancelled=self._is_cancelled,
                temporal_field=self._request.temporal_field)
        except RenderCancelled:
            self.signals.cancelled.emit(self._request)
            return
        except Exception:
            import traceback
            traceback.print_exc()
            self.signals.failed.emit(self._request)
            return
        self.signals.finished.emit(rgb, self._request)


class AnimationController:
    """Bridges a TimelinePanel to a RenderPipeline and an image sink.

    Screen frames (scrub/playback) render asynchronously off the GUI thread,
    capped at ``cap_provider()`` and latest-wins (mirrors the still-image
    ``_RenderWorker``/``RenderScheduler`` pattern in main_window.py) -- a slow
    render never blocks the GUI thread and stale frames are dropped rather
    than painted out of order. ``export()`` stays synchronous, exact, and
    fully independent of the screen cap.
    """

    _INACTIVE = {"", "none", "None", "off"}

    def __init__(self, panel: TimelinePanel, pipeline, provide_base, timeline,
                 seed: int = 0, cap_provider=None,
                 export_pipeline_provider=None) -> None:
        self.panel = panel
        self.pipeline = pipeline
        self.provide_base = provide_base    # () -> (gray_f32, RenderSettings) | None
        self.timeline = timeline
        self.seed = int(seed)
        self.cap_provider = cap_provider or (lambda: 1440)   # () -> int, Qt-free
        # () -> RenderPipeline snapshot for EXACT export, isolated from live edits;
        # falls back to the live pipeline when unset (tests/back-compat).
        self.export_pipeline_provider = export_pipeline_provider
        self.on_frame = None                # callable(np.uint8 HxWx3) | None
        self._pool = QThreadPool.globalInstance()
        self._active_workers: set = set()
        self._scheduler = RenderScheduler()
        self._last_base_gray = None
        panel.frame_changed.connect(self.render_frame)
        panel.export_requested.connect(self._on_export)

    def render_frame(self, frame_index: int) -> None:
        """Schedule a capped, async screen render of ``frame_index``. Never
        blocks; a render already in flight coalesces this into the latest
        pending trailing request (latest-wins)."""
        base = self.provide_base()
        if base is None:
            return
        gray, settings = base
        settings = self.timeline.settings_at(settings, int(frame_index))
        h, w = gray.shape[:2]
        factor = max(1, int(settings.scale))
        small_shape = (max(1, h // factor), max(1, w // factor))
        pattern = self.panel.pattern()
        amp = self.panel.amplitude()
        field = None
        if pattern not in self._INACTIVE and amp > 0.0:
            field = temporal_noise(int(frame_index), small_shape, pattern, amp, self.seed)
        req = _AnimRequest(
            generation=0, kind=RenderKind.DRAG, frame_index=int(frame_index),
            settings=settings, temporal_field=field,
            target_max_side=max(1, int(self.cap_provider())))
        self._last_base_gray = gray
        stamped = self._scheduler.request(req)
        if stamped is not None:
            self._launch_worker(stamped)

    def _launch_worker(self, request: _AnimRequest) -> None:
        worker = _AnimRenderWorker(
            self.pipeline, self._last_base_gray, request,
            is_cancelled=lambda: self._scheduler.should_cancel(request))
        worker.signals.finished.connect(self._on_frame_rendered)
        worker.signals.failed.connect(self._on_frame_terminal)
        worker.signals.cancelled.connect(self._on_frame_terminal)
        # Hold the wrapper until a terminal signal is delivered: QThreadPool
        # deletes the C++ runnable when run() returns, and without this ref the
        # signals QObject dies with it, dropping the queued emission -- the
        # frame is lost and the scheduler wedges busy forever (same bug as
        # VideoController._start_worker; AnimationController is not a QObject,
        # so these bound-method slots don't anchor delivery either).
        self._active_workers.add(worker)

        def release(*_args) -> None:
            self._active_workers.discard(worker)

        worker.signals.finished.connect(release)
        worker.signals.failed.connect(release)
        worker.signals.cancelled.connect(release)
        self._pool.start(worker)

    def _on_frame_rendered(self, rgb, request: _AnimRequest) -> None:
        # Drop stale/out-of-order results; only the most-recently-started
        # render is delivered to the sink.
        if self._scheduler.is_current(request) and self.on_frame is not None:
            self.on_frame(rgb)
        self._advance_scheduler()

    def _on_frame_terminal(self, request: _AnimRequest) -> None:
        # Failed or cancelled: don't paint, but always release the scheduler
        # so a raising/obsolete render can never wedge future frames.
        self._advance_scheduler()

    def _advance_scheduler(self) -> None:
        nxt = self._scheduler.on_finished()
        if nxt is not None:
            self._launch_worker(nxt)

    def export(self, out_path: str, fps: int = 24) -> "str | None":
        from ..animation import export_animation
        base = self.provide_base()
        if base is None:
            return None
        gray, settings = base
        pipeline = (self.export_pipeline_provider() if self.export_pipeline_provider
                    else self.pipeline)
        return export_animation(
            pipeline, gray, settings, self.timeline,
            self.panel.pattern(), self.panel.amplitude(),
            out_path, fps=fps, seed=self.seed)

    def _on_export(self) -> None:
        # Actual file dialog + worker is wired by the main window; this default is a
        # no-op hook so the panel's export button is always connected to a live slot.
        pass
