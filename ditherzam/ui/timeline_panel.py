from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..animation.temporal import PATTERNS, temporal_noise


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


class AnimationController:
    """Bridges a TimelinePanel to a RenderPipeline and an image sink."""

    _INACTIVE = {"", "none", "None", "off"}

    def __init__(self, panel: TimelinePanel, pipeline, provide_base, timeline,
                 seed: int = 0) -> None:
        self.panel = panel
        self.pipeline = pipeline
        self.provide_base = provide_base    # () -> (gray_f32, RenderSettings) | None
        self.timeline = timeline
        self.seed = int(seed)
        self.on_frame = None                # callable(np.uint8 HxWx3) | None
        panel.frame_changed.connect(self.render_frame)
        panel.export_requested.connect(self._on_export)

    def render_frame(self, frame_index: int) -> "np.ndarray | None":
        base = self.provide_base()
        if base is None:
            return None
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
        img = self.pipeline.render(gray, settings, temporal_field=field)
        if self.on_frame is not None:
            self.on_frame(img)
        return img

    def export(self, out_path: str, fps: int = 24) -> "str | None":
        from ..animation import export_animation
        base = self.provide_base()
        if base is None:
            return None
        gray, settings = base
        return export_animation(
            self.pipeline, gray, settings, self.timeline,
            self.panel.pattern(), self.panel.amplitude(),
            out_path, fps=fps, seed=self.seed)

    def _on_export(self) -> None:
        # Actual file dialog + worker is wired by the main window; this default is a
        # no-op hook so the panel's export button is always connected to a live slot.
        pass
