"""Compact, state-only Smart Mask controls.

The panel deliberately does not perform inference.  It publishes immutable
settings and user intents for the controller introduced by SM-12.
"""
from __future__ import annotations

from enum import Enum

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt

from ditherzam.masking.settings import (
    EXPANSION_MAX_PX, EXPANSION_MIN_PX, SENSITIVITY_MAX, SENSITIVITY_MIN,
    MaskTarget, OutsideMode, SmartMaskSettings,
)


class MaskPanelStatus(Enum):
    DISABLED = "Disabled"
    NEEDS_DETECTION = "Needs detection"
    DETECTING = "Detecting"
    READY = "Ready"
    NO_CLEAR_SUBJECT = "No clear subject"
    CANCELLED = "Cancelled"
    MODEL_UNAVAILABLE = "Model unavailable"
    ERROR = "Error"


class SmartMaskPanel(QGroupBox):
    """Smart Mask settings editor and inference-lifecycle display."""

    settings_changed = Signal(object)
    redetect_requested = Signal()
    cancel_requested = Signal()
    overlay_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Smart Mask", parent)
        self._settings = SmartMaskSettings()
        self._status = MaskPanelStatus.DISABLED
        self._source_available = False
        self._model_available = False
        self._has_valid_mask = False
        self._build_ui()
        self._sync_controls_from_settings()
        self._set_expanded(True)
        self._refresh_enabled_state()

    @property
    def settings(self) -> SmartMaskSettings:
        return self._settings

    @property
    def status(self) -> MaskPanelStatus:
        return self._status

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)
        self.disclosure_button = QToolButton()
        self.disclosure_button.setText("Controls")
        self.disclosure_button.setCheckable(True)
        self.disclosure_button.setChecked(True)
        self.disclosure_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.disclosure_button.setArrowType(Qt.ArrowType.DownArrow)
        self.disclosure_button.setAccessibleName("Show Smart Mask controls")
        outer.addWidget(self.disclosure_button)
        self.controls_widget = QWidget()
        outer.addWidget(self.controls_widget)
        layout = QVBoxLayout(self.controls_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.enabled_check = QCheckBox("Enabled")
        self.target_combo = QComboBox()
        self.target_combo.addItem("Subject", MaskTarget.SUBJECT)
        self.target_combo.addItem("Background", MaskTarget.BACKGROUND)
        self.target_combo.addItem("Whole Image", MaskTarget.WHOLE_IMAGE)
        self.candidate_combo = QComboBox()
        self.candidate_combo.addItem("Primary (1 of 1)")
        self.candidate_combo.setEnabled(False)
        self.candidate_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout.addWidget(self.enabled_check)
        layout.addWidget(self._labeled("Target", self.target_combo))
        layout.addWidget(self._labeled("Detected subject", self.candidate_combo))

        self.sensitivity_slider, self.sensitivity_spin = self._slider_row(
            layout, "Sensitivity", SENSITIVITY_MIN, SENSITIVITY_MAX, 50)
        self.feather_slider, self.feather_spin = self._slider_row(
            layout, "Edge feather", 0, 256, 8, " px")
        self.expansion_slider, self.expansion_spin = self._slider_row(
            layout, "Expand / contract", EXPANSION_MIN_PX, EXPANSION_MAX_PX, 0, " px")

        self.invert_check = QCheckBox("Invert mask")
        self.outside_combo = QComboBox()
        for label, value in (("Original", OutsideMode.ORIGINAL),
                             ("Transparent", OutsideMode.TRANSPARENT),
                             ("White", OutsideMode.WHITE), ("Black", OutsideMode.BLACK)):
            self.outside_combo.addItem(label, value)
        self.bake_check = QCheckBox("Bake fill into dither")
        self.bake_check.setToolTip(
            "Paint the White/Black outside fill into the image before dithering "
            "so dither and effects render across the background too.")
        self.overlay_check = QCheckBox("Show mask overlay")
        layout.addWidget(self.invert_check)
        layout.addWidget(self._labeled("Outside region", self.outside_combo))
        layout.addWidget(self.bake_check)
        layout.addWidget(self.overlay_check)

        buttons = QHBoxLayout()
        self.redetect_button = QPushButton("Re-detect")
        self.cancel_button = QPushButton("Cancel")
        self.progress_label = QLabel("")
        buttons.addWidget(self.redetect_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.progress_label, 1)
        layout.addLayout(buttons)
        self.status_label = QLabel("Disabled")
        self.status_label.setAccessibleName("Smart Mask status")
        layout.addWidget(self.status_label)

        self.disclosure_button.toggled.connect(self._set_expanded)

        self.enabled_check.toggled.connect(self._settings_edited)
        self.target_combo.currentIndexChanged.connect(self._settings_edited)
        self.sensitivity_slider.valueChanged.connect(self.sensitivity_spin.setValue)
        self.feather_slider.valueChanged.connect(self.feather_spin.setValue)
        self.expansion_slider.valueChanged.connect(self.expansion_spin.setValue)
        for slider in (self.sensitivity_slider, self.feather_slider, self.expansion_slider):
            slider.valueChanged.connect(self._settings_edited)
        self.invert_check.toggled.connect(self._settings_edited)
        self.outside_combo.currentIndexChanged.connect(self._settings_edited)
        self.bake_check.toggled.connect(self._settings_edited)
        self.overlay_check.toggled.connect(self.overlay_changed)
        self.redetect_button.clicked.connect(self.redetect_requested)
        self.cancel_button.clicked.connect(self.cancel_requested)
        for spin in (self.sensitivity_spin, self.feather_spin, self.expansion_spin):
            spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._set_tab_order()

    @staticmethod
    def _labeled(text: str, widget: QWidget) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel(text)
        label.setBuddy(widget)
        widget.setAccessibleName(text)
        row.addWidget(label)
        row.addWidget(widget, 1)
        return box

    def _slider_row(self, layout: QVBoxLayout, label: str, minimum: int,
                    maximum: int, value: int, suffix: str = "") -> tuple[QSlider, QSpinBox]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setValue(value)
        spin.setReadOnly(True)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        text_label = QLabel(label)
        text_label.setBuddy(slider)
        slider.setAccessibleName(label)
        spin.setAccessibleName(f"{label} value")
        row_layout.addWidget(text_label)
        row_layout.addWidget(slider, 1)
        row_layout.addWidget(spin)
        layout.addWidget(row)
        return slider, spin

    def _set_expanded(self, expanded: bool) -> None:
        self.controls_widget.setVisible(bool(expanded))
        self.disclosure_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.disclosure_button.setAccessibleName(
            "Hide Smart Mask controls" if expanded else "Show Smart Mask controls")

    def _set_tab_order(self) -> None:
        chain = (self.disclosure_button, self.enabled_check, self.target_combo,
                 self.sensitivity_slider, self.feather_slider,
                 self.expansion_slider, self.invert_check, self.outside_combo,
                 self.bake_check, self.overlay_check, self.redetect_button,
                 self.cancel_button)
        for first, second in zip(chain, chain[1:]):
            QWidget.setTabOrder(first, second)

    def _settings_edited(self, *_args: object) -> None:
        self._settings = SmartMaskSettings(
            enabled=self.enabled_check.isChecked(),
            target=self.target_combo.currentData(),
            sensitivity=self.sensitivity_slider.value(),
            feather_px=self.feather_slider.value(),
            expansion_px=self.expansion_slider.value(),
            invert=self.invert_check.isChecked(),
            outside=self.outside_combo.currentData(),
            bake_fill=self.bake_check.isChecked(),
        )
        if not self._settings.enabled:
            self._status = MaskPanelStatus.DISABLED
            self.status_label.setText(self._status.value)
        elif self._status is MaskPanelStatus.DISABLED:
            self._status = (MaskPanelStatus.READY if self._has_valid_mask
                            else MaskPanelStatus.NEEDS_DETECTION)
            self.status_label.setText(self._status.value)
        self._refresh_enabled_state()
        self.settings_changed.emit(self._settings)

    def set_settings(self, settings: SmartMaskSettings) -> None:
        if not isinstance(settings, SmartMaskSettings):
            raise TypeError("settings must be SmartMaskSettings")
        self._settings = settings
        self._sync_controls_from_settings()
        self._status = (
            MaskPanelStatus.DISABLED if not settings.enabled
            else MaskPanelStatus.READY if self._has_valid_mask
            else MaskPanelStatus.NEEDS_DETECTION
        )
        self.status_label.setText(self._status.value)
        self.progress_label.clear()
        self._refresh_enabled_state()

    def _sync_controls_from_settings(self) -> None:
        controls = (self.enabled_check, self.target_combo, self.sensitivity_slider,
                    self.feather_slider, self.expansion_slider, self.invert_check,
                    self.outside_combo, self.bake_check)
        for control in controls:
            control.blockSignals(True)
        self.enabled_check.setChecked(self._settings.enabled)
        self.target_combo.setCurrentIndex(self.target_combo.findData(self._settings.target))
        self.sensitivity_slider.setValue(self._settings.sensitivity)
        self.feather_slider.setValue(self._settings.feather_px)
        self.expansion_slider.setValue(self._settings.expansion_px)
        self.invert_check.setChecked(self._settings.invert)
        self.outside_combo.setCurrentIndex(self.outside_combo.findData(self._settings.outside))
        self.bake_check.setChecked(self._settings.bake_fill)
        for control in controls:
            control.blockSignals(False)
        self.sensitivity_spin.setValue(self._settings.sensitivity)
        self.feather_spin.setValue(self._settings.feather_px)
        self.expansion_spin.setValue(self._settings.expansion_px)

    def set_availability(self, *, source: bool, model: bool) -> None:
        self._source_available = bool(source)
        self._model_available = bool(model)
        self._refresh_enabled_state()

    def set_valid_mask_available(self, available: bool) -> None:
        """Record whether the current source/model has a publishable mask."""
        self._has_valid_mask = bool(available)
        if self._settings.enabled and self._status is not MaskPanelStatus.DETECTING:
            self._status = (MaskPanelStatus.READY if self._has_valid_mask
                            else MaskPanelStatus.NEEDS_DETECTION)
            self.status_label.setText(self._status.value)
        self._refresh_enabled_state()

    def set_status(self, status: MaskPanelStatus | str, progress: int | None = None) -> None:
        self._status = status if isinstance(status, MaskPanelStatus) else MaskPanelStatus(status)
        if self._status is MaskPanelStatus.READY:
            self._has_valid_mask = True
        self.status_label.setText(self._status.value)
        self.progress_label.setText(
            f"Detecting {max(0, min(100, int(progress)))}%"
            if self._status is MaskPanelStatus.DETECTING and progress is not None else "")
        self._refresh_enabled_state()

    def reset_for_source(self) -> None:
        """Clear session-only display state when a new source is loaded."""
        self._has_valid_mask = False
        self.overlay_check.setChecked(False)
        self.set_status(MaskPanelStatus.NEEDS_DETECTION if self._settings.enabled
                        else MaskPanelStatus.DISABLED)

    def _refresh_enabled_state(self) -> None:
        """Apply the complete state-to-enablement matrix in one place."""
        enabled = self._settings.enabled
        whole = self._settings.target is MaskTarget.WHOLE_IMAGE
        boundary = enabled and not whole
        detecting = self._status is MaskPanelStatus.DETECTING
        self.target_combo.setEnabled(enabled)
        self.candidate_combo.setEnabled(False)  # v1 has exactly one real candidate
        for control in (self.sensitivity_slider, self.sensitivity_spin,
                        self.feather_slider, self.feather_spin,
                        self.expansion_slider, self.expansion_spin,
                        self.invert_check, self.overlay_check):
            control.setEnabled(boundary)
        self.outside_combo.setEnabled(enabled and not whole)
        self.bake_check.setEnabled(
            boundary and self._settings.outside in (OutsideMode.WHITE, OutsideMode.BLACK))
        self.redetect_button.setEnabled(
            boundary and self._source_available and self._model_available and not detecting)
        self.cancel_button.setEnabled(boundary and detecting)
