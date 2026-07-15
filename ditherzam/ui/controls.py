from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..color.palette_store import PaletteStore
from ..color.ramp import RAMP_MODES
from ..dithering.parameters import parameter_specs
from .delegates import populate_dither_combo
from .palette_editor import SwatchStrip
from .palette_picker import PalettePicker
from .smart_mask_panel import SmartMaskPanel
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

_COLOR_MODES = ["off", "source", "nearest", "ordered", "diffused", "ramp"]
_EFFECTS = ["Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch"]


class ControlPanel(QWidget):
    """The right-hand control panel. Every edit mutates ``self.state`` then emits
    ``changed`` — the window turns that into a debounced re-render."""

    changed = Signal()
    from_image_requested = Signal()
    palette_preview = Signal(object)

    def __init__(self, parent=None, store: PaletteStore | None = None):
        super().__init__(parent)
        self.setObjectName("control_panel")
        self.store = store if store is not None else PaletteStore()
        self.state: dict = {
            "contrast": 50, "midtones": 50, "highlights": 50,
            "luminance_threshold": 50, "blur": 0, "saturation": 50,
            "invert": False, "preview_disabled": False,
            "style": "None", "scale": 5, "params": {},
            "palette": "grayscale", "color_mode": "off", "effects": [],
            "source_dither": 100, "source_dither_brighten": False,
            "depth": 2, "color_mapping": "match",
            "palette_autosave": False, "extract_unit": "k",
            "palette_preview": True, "palette_wheel_cycle": False,
        }
        self._sliders: dict[str, ResettableGlowSlider] = {}
        self._spins: dict[str, InvisibleSpinBox] = {}
        self._registry = None
        self._style_params: dict[str, dict] = {"None": {}}
        self.param_sliders: dict[str, ResettableGlowSlider] = {}
        self.param_value_labels: dict[str, QLabel] = {}
        self.working_palette = self.store.get(self.state["palette"])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._build_dither_section(layout)
        self.smart_mask_panel = SmartMaskPanel()
        layout.addWidget(self.smart_mask_panel)
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
        self._dither_categories = {"Default": ["None"]}
        self.dither_search = QLineEdit()
        self.dither_search.setPlaceholderText("Search dither styles…")
        self.dither_search.setClearButtonEnabled(True)
        self.dither_search.setVisible(False)
        self.dither_search.textChanged.connect(self._filter_dither_styles)
        self.dither_search_btn = QPushButton("Search")
        self.dither_search_btn.setCheckable(True)
        self.dither_search_btn.toggled.connect(self._toggle_dither_search)

        style_row = QHBoxLayout()
        style_row.setContentsMargins(0, 0, 0, 0)
        style_row.addWidget(self.dither_combo, 1)
        style_row.addWidget(self.dither_search_btn)
        style_container = QWidget()
        style_container.setLayout(style_row)
        layout.addWidget(_labeled("Style", style_container))
        layout.addWidget(self.dither_search)

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

        self.param_controls = QWidget()
        self.param_controls_layout = QVBoxLayout(self.param_controls)
        self.param_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.param_controls_layout.setSpacing(6)
        layout.addWidget(self.param_controls)

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

        self.palette_picker = PalettePicker()
        self.palette_picker.populate(self.store)
        self.palette_picker.select(self.state["palette"])
        self.palette_picker.selected.connect(self._on_palette_changed)
        self.palette_picker.preview.connect(self.palette_preview.emit)
        layout.addWidget(_labeled("Palette", self.palette_picker))

        self.swatch_strip = SwatchStrip()
        self.swatch_strip.set_palette(self.working_palette)
        self.swatch_strip.edited.connect(self._on_palette_edited)
        layout.addWidget(self.swatch_strip)

        pal_btns = QHBoxLayout()
        self.shuffle_btn = QPushButton("Shuffle")
        self.save_palette_btn = QPushButton("Save palette")
        self.reset_palette_btn = QPushButton("Reset to builtin")
        self.from_image_btn = QPushButton("From Image")
        self.shuffle_btn.clicked.connect(lambda: self.swatch_strip.shuffle())
        self.save_palette_btn.clicked.connect(self._on_save_palette)
        self.reset_palette_btn.clicked.connect(self._on_reset_palette)
        self.from_image_btn.clicked.connect(lambda: self.from_image_requested.emit())
        for b in (self.shuffle_btn, self.save_palette_btn,
                  self.reset_palette_btn, self.from_image_btn):
            pal_btns.addWidget(b)
        pal_container = QWidget()
        pal_container.setLayout(pal_btns)
        layout.addWidget(pal_container)

        self.category_combo = NoScrollComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.addItems(sorted(self.store.list_by_category().keys()))
        layout.addWidget(_labeled("Category", self.category_combo))

        self.extract_slider = ResettableGlowSlider(default=8, glow_color="#5e89ed")
        self.extract_slider.setRange(2, 64)
        layout.addWidget(_labeled("From-Image Colors", self.extract_slider))

        self.extract_unit_combo = NoScrollComboBox()
        self.extract_unit_combo.addItems(["k", "%"])
        self.extract_unit_combo.currentTextChanged.connect(self._on_extract_unit_changed)
        layout.addWidget(_labeled("From-Image Unit", self.extract_unit_combo))

        self.autosave_toggle = QCheckBox("Autosave palette")
        self.autosave_toggle.toggled.connect(self._on_autosave_toggled)
        layout.addWidget(self.autosave_toggle)

        self.palette_preview_toggle = QCheckBox("Preview on hover")
        self.palette_preview_toggle.setChecked(True)
        self.palette_preview_toggle.toggled.connect(self._on_palette_preview_toggled)
        layout.addWidget(self.palette_preview_toggle)

        self.wheel_cycle_toggle = QCheckBox("Wheel cycles palettes")
        self.wheel_cycle_toggle.toggled.connect(self._on_wheel_cycle_toggled)
        layout.addWidget(self.wheel_cycle_toggle)

        self._update_reset_enabled()

        self.mode_combo = NoScrollComboBox()
        self.mode_combo.addItems(_COLOR_MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(_labeled("Mode", self.mode_combo))

        self.source_dither_slider = ResettableGlowSlider(default=100, glow_color="#5e89ed")
        self.source_dither_slider.setRange(0, 100)
        self.source_dither_spin = InvisibleSpinBox(max_display=100)
        self.source_dither_spin.setValue(100)
        self.source_dither_slider.valueChanged.connect(
            self._make_slider_handler("source_dither"))
        self.source_dither_slider.valueChanged.connect(self.source_dither_spin.setValue)
        self._sliders["source_dither"] = self.source_dither_slider
        self._spins["source_dither"] = self.source_dither_spin
        self.source_dither_row = _labeled("Colored Dither", self.source_dither_slider,
                                          self.source_dither_spin)
        self.source_dither_row.setVisible(False)
        layout.addWidget(self.source_dither_row)

        self.source_dither_brighten_check = QCheckBox("Brighten marks")
        self.source_dither_brighten_check.setToolTip(
            "Colored Dither marks lift the image toward white instead of darkening it")
        self.source_dither_brighten_check.toggled.connect(
            self._on_source_dither_brighten_toggle)
        self.source_dither_brighten_check.setVisible(False)
        layout.addWidget(self.source_dither_brighten_check)

        self.mapping_combo = NoScrollComboBox()
        self.mapping_combo.addItems(list(RAMP_MODES))
        self.mapping_combo.currentTextChanged.connect(self._on_mapping_changed)
        layout.addWidget(_labeled("Mapping", self.mapping_combo))

        self.depth_slider = ResettableGlowSlider(default=2, glow_color="#5e89ed")
        self.depth_slider.setRange(1, 64)
        self.depth_spin = InvisibleSpinBox(max_display=64)
        self.depth_spin.setValue(round(2 / 64 * 100))
        self.depth_slider.valueChanged.connect(self._make_slider_handler("depth"))
        # Depth slider is 1..64; the 0..100 spin displays that value directly.
        self.depth_slider.valueChanged.connect(
            lambda v: self.depth_spin.setValue(round(v / 64 * 100)))
        self._sliders["depth"] = self.depth_slider
        self._spins["depth"] = self.depth_spin
        layout.addWidget(_labeled("Depth", self.depth_slider, self.depth_spin))

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
        self._dither_categories = {
            str(category): list(names) for category, names in by_category.items()
        }
        self._filter_dither_styles(self.dither_search.text())

    def set_registry(self, registry) -> None:
        """Attach the registry that supplies style-specific control metadata."""
        self._registry = registry
        self.set_registry_categories(registry.by_category())
        self._rebuild_parameter_controls()

    def refresh_parameter_controls(self) -> None:
        """Re-read current preset/state values into the visible style controls."""
        self._rebuild_parameter_controls()

    def _rebuild_parameter_controls(self) -> None:
        while self.param_controls_layout.count():
            item = self.param_controls_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.param_sliders.clear()
        self.param_value_labels.clear()

        entry = (self._registry.get_entry(self.state["style"])
                 if self._registry is not None else None)
        specs = parameter_specs(entry) if entry is not None else ()
        params = self.state.setdefault("params", {})
        for spec in specs:
            value = max(spec.minimum, min(spec.maximum,
                        int(params.get(spec.key, spec.default))))
            params.setdefault(spec.key, value)
            slider = ResettableGlowSlider(default=spec.default, glow_color="#5e89ed")
            slider.setRange(spec.minimum, spec.maximum)
            slider.setValue(value)
            number = QLabel(str(value))
            number.setMinimumWidth(24)
            number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            slider.valueChanged.connect(
                lambda v, key=spec.key: self._on_parameter_changed(key, v))
            slider.valueChanged.connect(lambda v, label=number: label.setText(str(v)))
            self.param_sliders[spec.key] = slider
            self.param_value_labels[spec.key] = number
            self.param_controls_layout.addWidget(_labeled(spec.label, slider, number))
        self.param_controls.setVisible(bool(specs))

    def _on_parameter_changed(self, key: str, value: int) -> None:
        self.state.setdefault("params", {})[key] = int(value)
        self._style_params[self.state["style"]] = dict(self.state["params"])
        self.changed.emit()

    def _toggle_dither_search(self, visible: bool) -> None:
        self.dither_search.setVisible(visible)
        if visible:
            self.dither_search.setFocus()
        else:
            self.dither_search.clear()

    def _filter_dither_styles(self, query: str) -> None:
        needle = query.strip().casefold()
        if needle:
            filtered = {
                category: [name for name in names if needle in name.casefold()]
                for category, names in self._dither_categories.items()
            }
            filtered = {category: names for category, names in filtered.items() if names}
        else:
            filtered = self._dither_categories

        current = self.state["style"]
        self.dither_combo.blockSignals(True)
        populate_dither_combo(self.dither_combo, filtered)
        model = self.dither_combo.model()
        for row in range(self.dither_combo.count()):
            idx = model.index(row, 0)
            if idx.data(Qt.ItemDataRole.UserRole) == current:
                self.dither_combo.setCurrentIndex(row)
                break
        self.dither_combo.blockSignals(False)

    def set_style(self, name: str, params: dict | None = None) -> None:
        previous = self.state["style"]
        self._style_params[previous] = dict(self.state.get("params", {}))
        model = self.dither_combo.model()
        for row in range(self.dither_combo.count()):
            idx = model.index(row, 0)
            if idx.data(Qt.ItemDataRole.UserRole) == name:
                self.dither_combo.setCurrentIndex(row)
                break
        self.state["style"] = name
        if params is not None:
            self._style_params[name] = dict(params)
        self.state["params"] = dict(self._style_params.get(name, {}))
        self._rebuild_parameter_controls()
        self.changed.emit()

    # ---- signal handlers ----------------------------------------------------
    def _on_style_changed(self, _index: int) -> None:
        data = self.dither_combo.currentData(Qt.ItemDataRole.UserRole)
        if data is not None:
            previous = self.state["style"]
            self._style_params[previous] = dict(self.state.get("params", {}))
            self.state["style"] = data
            self.state["params"] = dict(self._style_params.get(data, {}))
            self._rebuild_parameter_controls()
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

    def _on_source_dither_brighten_toggle(self, checked: bool) -> None:
        self.state["source_dither_brighten"] = bool(checked)
        self.changed.emit()

    def _on_palette_changed(self, text: str) -> None:
        self.state["palette"] = text
        self.working_palette = self.store.get(text)
        self.swatch_strip.set_palette(self.working_palette)
        self.palette_preview.emit(None)                # clear any hover preview
        self._sync_category_combo(self.working_palette.category)
        self._update_reset_enabled()
        self.changed.emit()

    def _on_palette_edited(self, palette) -> None:
        self.working_palette = palette
        if self.state.get("palette_autosave"):
            self.store.save(palette)
            self._update_reset_enabled()
        self.changed.emit()

    def set_working_palette(self, palette) -> None:
        self.working_palette = palette
        self.state["palette"] = palette.name
        self.swatch_strip.set_palette(palette)
        self.palette_picker.select(palette.name)
        self._sync_category_combo(getattr(palette, "category", ""))
        self.changed.emit()

    def _sync_category_combo(self, category: str) -> None:
        self.category_combo.blockSignals(True)
        self.category_combo.setCurrentText(category or "")
        self.category_combo.blockSignals(False)

    def _on_autosave_toggled(self, checked: bool) -> None:
        self.state["palette_autosave"] = bool(checked)

    def _on_palette_preview_toggled(self, checked: bool) -> None:
        self.state["palette_preview"] = bool(checked)
        self.palette_picker.set_preview_enabled(bool(checked))

    def _on_wheel_cycle_toggled(self, checked: bool) -> None:
        self.state["palette_wheel_cycle"] = bool(checked)
        self.palette_picker.set_wheel_cycle(bool(checked))

    def _on_extract_unit_changed(self, text: str) -> None:
        unit = "pct" if text == "%" else "k"
        self.state["extract_unit"] = unit
        if unit == "pct":
            self.extract_slider.setRange(0, 100)
            self.extract_slider.setValue(50)
        else:
            self.extract_slider.setRange(2, 64)
            self.extract_slider.setValue(8)

    def _on_save_palette(self) -> None:
        self.working_palette.category = self.category_combo.currentText().strip()
        self.store.save(self.working_palette)
        self._refresh_palette_picker(self.working_palette.name)
        self._update_reset_enabled()

    def _on_reset_palette(self) -> None:
        name = self.working_palette.name
        if self.store.is_user(name) and self.store.is_builtin(name):
            self.working_palette = self.store.reset_to_builtin(name)
            self.swatch_strip.set_palette(self.working_palette)
        self._refresh_palette_picker(name if self.store.is_builtin(name) else None)
        self._update_reset_enabled()
        self.changed.emit()

    def _refresh_palette_picker(self, select: str | None) -> None:
        self.palette_picker.populate(self.store)
        if select is not None:
            self.palette_picker.select(select)

    def _update_reset_enabled(self) -> None:
        name = self.working_palette.name
        self.reset_palette_btn.setEnabled(
            self.store.is_user(name) and self.store.is_builtin(name))

    def _on_mode_changed(self, text: str) -> None:
        self.state["color_mode"] = text
        self.source_dither_row.setVisible(text == "source")
        self.source_dither_brighten_check.setVisible(text == "source")
        self.changed.emit()

    def _on_mapping_changed(self, text: str) -> None:
        self.state["color_mapping"] = text
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
