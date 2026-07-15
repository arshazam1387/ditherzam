# ditherzam/ui/glow_panel.py
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..effects.glow_params import GLOW_DEFAULTS
from .widgets import InvisibleSpinBox, ResettableGlowSlider

# (state_key, label, min, max, spin_display_max)
_GLOW_SLIDERS = [
    ("glow_threshold", "Threshold", 0, 255, 255),
    ("glow_smoothing", "Smoothing", 0, 128, 128),
    ("glow_radius", "Radius", 0, 200, 200),
    ("glow_intensity", "Intensity", 0, 100, 100),
    ("glow_epsilon", "Epsilon", 0, 100, 100),
    ("glow_falloff", "Falloff", 0, 100, 100),
    ("glow_distance", "Distance Scale", 1, 100, 100),
    ("glow_aspect", "Aspect", 0, 100, 100),
]


class GlowPanel(QWidget):
    """Dedicated, fully-customizable Epsilon Glow controls (its own tab)."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glow_panel")
        self.state: dict = dict(GLOW_DEFAULTS)
        self._sliders: dict[str, ResettableGlowSlider] = {}
        self._spins: dict[str, InvisibleSpinBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Epsilon Glow")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        self.enable_toggle = QCheckBox("Enable Glow")
        self.enable_toggle.toggled.connect(self._on_enable)
        layout.addWidget(self.enable_toggle)

        for key, label, lo, hi, disp in _GLOW_SLIDERS:
            default = int(GLOW_DEFAULTS[key])
            slider = ResettableGlowSlider(default=default, glow_color="#5e89ed")
            slider.setRange(lo, hi)
            slider.setValue(default)
            spin = InvisibleSpinBox(max_display=disp)
            spin.setValue(int(round(default / max(hi, 1) * 100)))
            slider.valueChanged.connect(self._make_handler(key))
            slider.valueChanged.connect(
                lambda v, s=spin, h=hi: s.setValue(int(round(v / max(h, 1) * 100))))
            self._sliders[key] = slider
            self._spins[key] = spin
            layout.addWidget(_row(label, slider, spin))

        self.reset_btn = QPushButton("Reset all")
        self.reset_btn.clicked.connect(self.reset_all)
        layout.addWidget(self.reset_btn)
        layout.addStretch(1)

    def _make_handler(self, key: str):
        def handler(value: int) -> None:
            self.state[key] = int(value)
            self.changed.emit()
        return handler

    def _on_enable(self, checked: bool) -> None:
        self.state["glow_enabled"] = bool(checked)
        self.changed.emit()

    def reset_all(self) -> None:
        for key, slider in self._sliders.items():
            slider.setValue(int(GLOW_DEFAULTS[key]))
        self.enable_toggle.setChecked(bool(GLOW_DEFAULTS["glow_enabled"]))


def _row(text: str, widget: QWidget, spin: QWidget) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel(text))
    row.addWidget(widget, 1)
    row.addWidget(spin)
    return container
