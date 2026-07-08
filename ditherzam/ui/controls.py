from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .delegates import populate_dither_combo
from .widgets import (
    InvisibleSpinBox,
    NoScrollComboBox,
    ResettableGlowSlider,
)

# Adjustment sliders: (state_key, label, spin_display_max, neutral_default).
# Range 0..100. Neutral is 50 for tonal sliders (50 == identity), but Blur's
# identity is 0 — value=50 is a 25px Gaussian blur, so it must start at 0.
_ADJUSTMENTS = [
    ("contrast", "Contrast", 250, 50),
    ("midtones", "Midtones", 10, 50),
    ("highlights", "Highlights", 50, 50),
    ("luminance_threshold", "Luminance Threshold", 100, 50),
    ("blur", "Blur", 100, 0),
]

_PALETTES = ["grayscale", "gameboy", "cga", "pico8", "sepia"]
_COLOR_MODES = ["off", "nearest", "ordered", "diffused"]
_EFFECTS = ["Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch", "Epsilon Glow"]


class ControlPanel(QWidget):
    """The right-hand control panel. Every edit mutates ``self.state`` then emits
    ``changed`` — the window turns that into a debounced re-render."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("control_panel")
        self.state: dict = {
            "contrast": 50, "midtones": 50, "highlights": 50,
            "luminance_threshold": 50, "blur": 0, "saturation": 50,
            "invert": False, "preview_disabled": False,
            "style": "None", "scale": 5, "params": {},
            "palette": "grayscale", "color_mode": "off", "effects": [],
        }
        self._sliders: dict[str, ResettableGlowSlider] = {}
        self._spins: dict[str, InvisibleSpinBox] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._build_dither_section(layout)
        self._build_adjustments_section(layout)
        self._build_color_section(layout)
        self._build_effects_section(layout)
        layout.addStretch(1)

    # ---- section builders ---------------------------------------------------
    def _build_dither_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Dither Controls"))

        self.preview_toggle = QCheckBox("Disable Preview")
        self.preview_toggle.toggled.connect(self._on_preview_toggle)
        layout.addWidget(self.preview_toggle)

        self.dither_combo = NoScrollComboBox()
        populate_dither_combo(self.dither_combo, {"Default": ["None"]})
        self.dither_combo.currentIndexChanged.connect(self._on_style_changed)
        layout.addWidget(_labeled("Style", self.dither_combo))

        self.scale_slider = ResettableGlowSlider(default=5, glow_color="#5e89ed")
        self.scale_slider.setRange(1, 20)
        self.scale_spin = InvisibleSpinBox(max_display=20)
        self.scale_spin.setValue(round(5 / 20 * 100))
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        # Scale slider is 1..20; the 0..100 spin displays that value directly.
        self.scale_slider.valueChanged.connect(
            lambda v: self.scale_spin.setValue(round(v / 20 * 100)))
        self._spins["scale"] = self.scale_spin
        layout.addWidget(_labeled("Scale", self.scale_slider, self.scale_spin))

    def _build_adjustments_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Adjustments"))

        self.invert_toggle = QCheckBox("Invert Output")
        self.invert_toggle.toggled.connect(self._on_invert_toggle)
        layout.addWidget(self.invert_toggle)

        for key, label, disp_max, default in _ADJUSTMENTS:
            slider = ResettableGlowSlider(default=default, glow_color="#5e89ed")
            slider.setRange(0, 100)
            spin = InvisibleSpinBox(max_display=disp_max)
            spin.setValue(default)
            slider.valueChanged.connect(self._make_slider_handler(key))
            slider.valueChanged.connect(spin.setValue)   # keep the number in sync
            self._sliders[key] = slider
            self._spins[key] = spin
            layout.addWidget(_labeled(label, slider, spin))

        # expose the Contrast slider by name for the tests / window
        self.contrast_slider = self._sliders["contrast"]

    def _build_color_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Color"))

        self.palette_combo = NoScrollComboBox()
        self.palette_combo.addItems(_PALETTES)
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        layout.addWidget(_labeled("Palette", self.palette_combo))

        self.mode_combo = NoScrollComboBox()
        self.mode_combo.addItems(_COLOR_MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(_labeled("Mode", self.mode_combo))

        self.saturation_slider = ResettableGlowSlider(default=50, glow_color="#5e89ed")
        self.saturation_slider.setRange(0, 100)
        self.saturation_spin = InvisibleSpinBox(max_display=100)
        self.saturation_spin.setValue(50)
        self.saturation_slider.valueChanged.connect(self._make_slider_handler("saturation"))
        self.saturation_slider.valueChanged.connect(self.saturation_spin.setValue)
        self._spins["saturation"] = self.saturation_spin
        layout.addWidget(_labeled("Saturation", self.saturation_slider, self.saturation_spin))

    def _build_effects_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Effects"))

        self.effects_list = QListWidget()
        layout.addWidget(self.effects_list)

        row = QHBoxLayout()
        self.effect_combo = NoScrollComboBox()
        self.effect_combo.addItems(_EFFECTS)
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        add_btn.clicked.connect(self._on_add_effect)
        remove_btn.clicked.connect(self._on_remove_effect)
        row.addWidget(self.effect_combo)
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        container = QWidget()
        container.setLayout(row)
        layout.addWidget(container)

    # ---- public API ---------------------------------------------------------
    def set_registry_categories(self, by_category) -> None:
        current = self.state["style"]
        self.dither_combo.blockSignals(True)
        populate_dither_combo(self.dither_combo, by_category)
        self.dither_combo.blockSignals(False)
        self.set_style(current)

    def set_style(self, name: str) -> None:
        model = self.dither_combo.model()
        for row in range(self.dither_combo.count()):
            idx = model.index(row, 0)
            if idx.data(Qt.ItemDataRole.UserRole) == name:
                self.dither_combo.setCurrentIndex(row)
                break
        self.state["style"] = name
        self.changed.emit()

    # ---- signal handlers ----------------------------------------------------
    def _on_style_changed(self, _index: int) -> None:
        data = self.dither_combo.currentData(Qt.ItemDataRole.UserRole)
        if data is not None:
            self.state["style"] = data
            self.changed.emit()

    def _on_scale_changed(self, value: int) -> None:
        self.state["scale"] = int(value)
        self.changed.emit()

    def _on_preview_toggle(self, checked: bool) -> None:
        self.state["preview_disabled"] = bool(checked)
        self.changed.emit()

    def _on_invert_toggle(self, checked: bool) -> None:
        self.state["invert"] = bool(checked)
        self.changed.emit()

    def _on_palette_changed(self, text: str) -> None:
        self.state["palette"] = text
        self.changed.emit()

    def _on_mode_changed(self, text: str) -> None:
        self.state["color_mode"] = text
        self.changed.emit()

    def _make_slider_handler(self, key: str):
        def handler(value: int) -> None:
            self.state[key] = int(value)
            self.changed.emit()
        return handler

    def _on_add_effect(self) -> None:
        name = self.effect_combo.currentText()
        self.effects_list.addItem(name)
        self.state["effects"] = self._effects_from_list()
        self.changed.emit()

    def _on_remove_effect(self) -> None:
        row = self.effects_list.currentRow()
        if row >= 0:
            self.effects_list.takeItem(row)
            self.state["effects"] = self._effects_from_list()
            self.changed.emit()

    def _effects_from_list(self) -> list[str]:
        return [self.effects_list.item(i).text() for i in range(self.effects_list.count())]


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
    return lbl


def _labeled(text: str, widget: QWidget, spin: QWidget | None = None) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel(text))
    row.addWidget(widget, 1)
    if spin is not None:
        row.addWidget(spin)
    return container
