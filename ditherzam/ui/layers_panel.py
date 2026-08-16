"""Widget-only controls for the spatial layer stack."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QSize, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QApplication,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class _TargetButton(QToolButton):
    """Thumbnail button with explicit Enter as well as native Space activation."""

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
            event.accept()
            return
        super().keyPressEvent(event)


class LayersPanel(QWidget):
    """Present layer state and publish index-based editing intents.

    Core stacks are bottom-to-top while the list is intentionally displayed
    top-to-bottom. Every list item therefore carries its core index in
    ``Qt.UserRole``; visual row numbers are never used as model indices.
    """

    new_blank_requested = Signal()
    place_image_requested = Signal()
    # Compatibility signal for integrations which still listen for the old name.
    add_requested = Signal()
    duplicate_requested = Signal(int)
    delete_requested = Signal(int)
    move_requested = Signal(int, int)
    visibility_changed = Signal(int, bool)
    name_changed = Signal(int, str)
    blend_changed = Signal(int, str)
    opacity_changed = Signal(int, int)
    transform_changed = Signal(int, int, int, int, int)
    center_requested = Signal(int)
    fit_requested = Signal(int)
    transform_mode_requested = Signal(int)
    transform_confirmed = Signal()
    transform_cancelled = Signal()
    selection_changed = Signal(int)
    export_requested = Signal()
    reveal_mask_requested = Signal(int, bool)
    hide_mask_requested = Signal(int, bool)
    transparency_mask_requested = Signal(int, bool)
    mask_enabled_changed = Signal(int, bool)
    mask_density_changed = Signal(int, int)
    smart_mask_requested = Signal()
    invert_mask_requested = Signal()
    fill_white_mask_requested = Signal()
    fill_black_mask_requested = Signal()
    reset_mask_requested = Signal()
    delete_mask_requested = Signal()
    import_mask_requested = Signal()
    export_mask_requested = Signal()
    edit_target_requested = Signal(str, str)
    mask_replacement_confirmed = Signal(str)
    mask_replacement_cancelled = Signal(str)
    inspection_mode_changed = Signal(str)
    smart_refinement_started = Signal()
    smart_refinement_changed = Signal(int, str, int, int, bool)
    smart_refinement_confirmed = Signal(str)
    smart_refinement_cancelled = Signal(str)
    luminance_mask_requested = Signal(str)
    gradient_mask_requested = Signal(str, float, float, float, float)
    gradient_tool_requested = Signal(str)
    pattern_mask_requested = Signal(str, int, int, int, int, int, int, bool)
    selection_tool_requested = Signal(str)
    selection_clear_requested = Signal()
    selection_from_mask_requested = Signal()
    selection_refine_requested = Signal(str, int)
    color_range_pick_requested = Signal()
    color_range_changed = Signal(int, int)
    color_range_confirmed = Signal()
    color_range_cancelled = Signal()
    brush_settings_changed = Signal(str, int, int, int, int)
    brush_mode_changed = Signal(str)
    mask_paint_requested = Signal(bool)
    pointer_requested = Signal()

    _EMPTY_TEXT = "No layers. Place an image or create a blank layer."
    _BLENDS = (
        ("Normal", "normal"),
        ("Multiply", "multiply"),
        ("Screen", "screen"),
        ("Overlay", "overlay"),
        ("Difference", "difference"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(320)
        self._layers: list[tuple] = []
        self._thumbnails: dict[str, QPixmap] = {}
        self._placeholder = self._make_placeholder()
        self._inspection_mode = "Normal"

        self.empty_label = QLabel(self._EMPTY_TEXT)
        self.empty_label.setWordWrap(True)
        self.empty_label.setAccessibleName("Layers empty state")

        self.layer_list = QListWidget()
        self.layer_list.setIconSize(QSize(56, 56))
        self.layer_list.setUniformItemSizes(True)
        self.layer_list.setAccessibleName(
            "Layers, topmost first, with visibility, preview, and name")
        self.inspection_combo = QComboBox()
        self.inspection_combo.addItems(("Normal", "Red Overlay", "Mask Only"))
        self.inspection_combo.setAccessibleName("Raster mask inspection view")
        self.inspection_combo.setAccessibleDescription(
            "Shows the normal composite, hidden mask areas in red, or mask "
            "coverage as black and white.")

        self.new_blank_btn = QPushButton("New Blank")
        self.new_blank_btn.setAccessibleName("Create new blank layer")
        self.place_image_btn = QPushButton("Place Image")
        self.place_image_btn.setAccessibleName("Place image as a new layer")
        # Attribute compatibility for callers which have not migrated yet.
        self.add_btn = self.new_blank_btn
        self.duplicate_btn = QPushButton("Duplicate")
        self.delete_btn = QPushButton("Delete")
        self.up_btn = QPushButton("Up")
        self.down_btn = QPushButton("Down")

        self.visibility_check = QCheckBox("Visible")
        self.name_edit = QLineEdit()
        self.name_edit.setAccessibleName("Selected layer name")

        self.blend_combo = QComboBox()
        self.blend_combo.setAccessibleName("Selected layer blend mode")
        for label, value in self._BLENDS:
            self.blend_combo.addItem(label, value)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setAccessibleName("Selected layer opacity")
        self.opacity_label = QLabel("100%")
        self.opacity_label.setAccessibleName("Selected layer opacity value")

        self.mask_state_label = QLabel("No raster mask")
        self.mask_state_label.setWordWrap(True)
        self.mask_state_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.mask_state_label.setAccessibleName("Selected layer raster mask state")
        self.mask_combination_combo = QComboBox()
        self.mask_combination_combo.addItems(
            ("Replace", "Add", "Subtract", "Intersect"))
        self.mask_combination_combo.setAccessibleName(
            "Mask candidate combination mode")
        self.reveal_mask_btn = QPushButton("Reveal All")
        self.reveal_mask_btn.setAccessibleName("Create fully revealed raster mask")
        self.hide_mask_btn = QPushButton("Hide All")
        self.hide_mask_btn.setAccessibleName("Create fully hidden raster mask")
        self.transparency_mask_btn = QPushButton("From Transparency")
        self.transparency_mask_btn.setAccessibleName(
            "Create raster mask from source transparency")
        # Compact visible vocabulary; legacy buttons above remain callable
        # compatibility attributes for existing integrations.
        self.create_mask_btn = QToolButton()
        self.create_mask_btn.setText("Create")
        self.create_mask_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self.create_mask_btn.setAccessibleName("Create or replace raster mask")
        self.create_mask_menu = QMenu(self.create_mask_btn)
        self.create_mask_btn.setMenu(self.create_mask_menu)
        self.reveal_mask_action = self.create_mask_menu.addAction("Reveal All")
        self.hide_mask_action = self.create_mask_menu.addAction("Hide All")
        self.transparency_mask_action = self.create_mask_menu.addAction(
            "From Transparency")
        self.smart_mask_action = self.create_mask_menu.addAction("From Smart Mask")
        self.shadows_mask_action = self.create_mask_menu.addAction("From Shadows")
        self.midtones_mask_action = self.create_mask_menu.addAction("From Midtones")
        self.highlights_mask_action = self.create_mask_menu.addAction(
            "From Highlights")
        self.linear_gradient_action = self.create_mask_menu.addAction(
            "Linear Gradient…")
        self.radial_gradient_action = self.create_mask_menu.addAction(
            "Radial Gradient…")
        self.pattern_mask_action = self.create_mask_menu.addAction("Pattern...")
        self.freeze_smart_action = self.create_mask_menu.addAction(
            "Freeze Smart…")

        self.edit_mask_btn = QToolButton()
        self.edit_mask_btn.setText("Edit")
        self.edit_mask_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self.edit_mask_btn.setAccessibleName("Edit selected raster mask")
        self.edit_mask_menu = QMenu(self.edit_mask_btn)
        self.edit_mask_btn.setMenu(self.edit_mask_menu)
        self.invert_mask_action = self.edit_mask_menu.addAction("Invert")
        self.fill_white_mask_action = self.edit_mask_menu.addAction("Fill White")
        self.fill_black_mask_action = self.edit_mask_menu.addAction("Fill Black")
        self.reset_mask_action = self.edit_mask_menu.addAction("Reset / Replace")
        self.dither_mask_action = self.edit_mask_menu.addAction(
            "Dither the Mask...")
        self.delete_mask_action = self.edit_mask_menu.addAction("Delete")

        self.import_mask_btn = QPushButton("Import")
        self.import_mask_btn.setAccessibleName("Import raster mask PNG")
        self.export_mask_btn = QPushButton("Export")
        self.export_mask_btn.setAccessibleName("Export raster mask PNG")
        self.mask_enabled_check = QCheckBox("Enabled")
        self.mask_enabled_check.setAccessibleName(
            "Enable selected layer raster mask")
        self.mask_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_density_slider.setRange(0, 100)
        self.mask_density_slider.setValue(100)
        self.mask_density_slider.setAccessibleName(
            "Selected layer raster mask density")
        self.mask_density_label = QLabel("100%")
        self.mask_density_label.setAccessibleName(
            "Selected layer raster mask density value")
        self.selection_shape_combo = QComboBox()
        self.selection_shape_combo.addItems(("Rectangle", "Ellipse", "Polygon", "Freehand"))
        self.selection_shape_combo.setAccessibleName("Temporary selection shape")
        self.selection_operation_combo = QComboBox()
        self.selection_operation_combo.addItems(("Replace", "Add", "Subtract"))
        self.selection_operation_combo.setAccessibleName(
            "Temporary selection operation")
        self.selection_draw_btn = QPushButton("Draw")
        self.selection_draw_btn.setText("Select on Canvas")
        self.selection_draw_btn.setAccessibleName(
            "Draw temporary selection on canvas")
        self.selection_clear_btn = QPushButton("Clear")
        self.selection_clear_btn.setAccessibleName("Clear temporary selection")
        self.from_selection_btn = QPushButton("From Selection")
        self.from_selection_btn.setAccessibleName(
            "Create raster mask from temporary selection")
        self.selection_radius_spin = QSpinBox()
        self.selection_radius_spin.setRange(1, 64)
        self.selection_radius_spin.setValue(4)
        self.selection_radius_spin.setSuffix(" px")
        self.selection_radius_spin.setAccessibleName("Selection refinement radius")
        self.selection_all_btn = QPushButton("All")
        self.selection_invert_btn = QPushButton("Invert")
        self.selection_grow_btn = QPushButton("Grow")
        self.selection_shrink_btn = QPushButton("Shrink")
        self.selection_feather_btn = QPushButton("Feather")
        for widget, name in (
            (self.selection_all_btn, "Select entire canvas"),
            (self.selection_invert_btn, "Invert temporary selection"),
            (self.selection_grow_btn, "Grow temporary selection"),
            (self.selection_shrink_btn, "Shrink temporary selection"),
            (self.selection_feather_btn, "Feather temporary selection"),
        ):
            widget.setAccessibleName(name)

        self.color_range_btn = QPushButton("Color Range")
        self.color_range_btn.setAccessibleName("Pick selection color from canvas")
        self.color_range_tolerance_spin = QSpinBox()
        self.color_range_tolerance_spin.setRange(0, 441)
        self.color_range_tolerance_spin.setValue(24)
        self.color_range_tolerance_spin.setAccessibleName("Color range tolerance")
        self.color_range_softness_spin = QSpinBox()
        self.color_range_softness_spin.setRange(0, 441)
        self.color_range_softness_spin.setValue(16)
        self.color_range_softness_spin.setAccessibleName("Color range softness")
        self.color_range_confirm_btn = QPushButton("Confirm")
        self.color_range_confirm_btn.setAccessibleName("Confirm color range selection")
        self.color_range_cancel_btn = QPushButton("Cancel")
        self.color_range_cancel_btn.setAccessibleName("Cancel color range selection")

        self.pointer_btn = QPushButton("Pointer")
        self.pointer_btn.setAccessibleName("Return to the normal pointer")
        self.pointer_btn.setToolTip("Leave canvas tools and restore the normal pointer")
        self.paint_mask_btn = QPushButton("Paint Mask")
        self.paint_mask_btn.setCheckable(True)
        self.paint_mask_btn.setAccessibleName("Paint the selected layer mask")
        self.paint_mask_btn.setToolTip(
            "Arm the mask brush. Pointer / Done exits painting.")
        self.brush_mode_combo = QComboBox()
        self.brush_mode_combo.addItems(("Reveal", "Hide"))
        self.brush_mode_combo.setAccessibleName("Mask brush paint mode")
        self.brush_mode_combo.setToolTip(
            "Reveal makes painted areas visible; Hide conceals them.")

        self.brush_tip_combo = QComboBox()
        self.brush_tip_combo.addItems(("Round", "Square", "Diamond", "Texture"))
        self.brush_tip_combo.setAccessibleName("Mask brush tip")
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 2048)
        self.brush_size_spin.setValue(32)
        self.brush_size_spin.setSuffix(" px")
        self.brush_size_spin.setAccessibleName("Mask brush size")
        self.brush_hardness_spin = QSpinBox()
        self.brush_hardness_spin.setRange(0, 100)
        self.brush_hardness_spin.setValue(100)
        self.brush_hardness_spin.setSuffix("%")
        self.brush_hardness_spin.setAccessibleName("Mask brush hardness")
        self.brush_strength_spin = QSpinBox()
        self.brush_strength_spin.setRange(0, 100)
        self.brush_strength_spin.setValue(100)
        self.brush_strength_spin.setSuffix("%")
        self.brush_strength_spin.setAccessibleName("Mask brush strength")
        self.brush_spacing_spin = QSpinBox()
        self.brush_spacing_spin.setRange(1, 100)
        self.brush_spacing_spin.setValue(25)
        self.brush_spacing_spin.setSuffix("%")
        self.brush_spacing_spin.setAccessibleName("Mask brush stamp spacing")
        self._mask_present = False
        self._selection_pending = False
        self._edit_target: tuple[str, str] | None = None
        self._row_targets: dict[str, tuple[QToolButton, QToolButton]] = {}

        self.mask_confirmation = QWidget()
        confirmation_layout = QVBoxLayout(self.mask_confirmation)
        confirmation_layout.setContentsMargins(0, 0, 0, 0)
        self.mask_confirmation_label = QLabel("Replace the existing raster mask?")
        self.mask_confirmation_label.setWordWrap(True)
        self.mask_confirmation_label.setAccessibleName(
            "Raster mask replacement confirmation")
        self.replace_mask_btn = QPushButton("Replace")
        self.replace_mask_btn.setAccessibleName("Replace existing raster mask")
        self.cancel_replace_mask_btn = QPushButton("Cancel")
        self.cancel_replace_mask_btn.setAccessibleName(
            "Cancel raster mask replacement")
        confirmation_layout.addWidget(self.mask_confirmation_label)
        confirmation_actions = QHBoxLayout()
        confirmation_actions.addWidget(self.replace_mask_btn)
        confirmation_actions.addWidget(self.cancel_replace_mask_btn)
        confirmation_layout.addLayout(confirmation_actions)
        self._replacement_token: str | None = None

        self.smart_refinement = QWidget()
        refinement = QVBoxLayout(self.smart_refinement)
        refinement.setContentsMargins(0, 0, 0, 0)
        refinement.setSpacing(4)
        refinement_form = QFormLayout()
        refinement_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.smart_threshold = QSpinBox()
        self.smart_threshold.setRange(0, 100)
        self.smart_threshold.setAccessibleName("Smart mask threshold")
        self.smart_refine_kind = QComboBox()
        self.smart_refine_kind.addItems(("None", "Grow", "Shrink"))
        self.smart_refine_kind.setAccessibleName("Smart mask morphology")
        self.smart_refine_radius = QSpinBox()
        self.smart_refine_radius.setRange(0, 64)
        self.smart_refine_radius.setAccessibleName(
            "Smart mask morphology radius in source pixels")
        self.smart_feather_radius = QSpinBox()
        self.smart_feather_radius.setRange(0, 64)
        self.smart_feather_radius.setAccessibleName(
            "Smart mask feather radius in source pixels")
        refinement_form.addRow("Threshold", self.smart_threshold)
        refinement_form.addRow("Morphology", self.smart_refine_kind)
        refinement_form.addRow("Radius", self.smart_refine_radius)
        refinement_form.addRow("Feather", self.smart_feather_radius)
        refinement.addLayout(refinement_form)
        options = QHBoxLayout()
        self.smart_refine_invert = QCheckBox("Invert")
        self.smart_refine_invert.setAccessibleName(
            "Invert frozen Smart mask")
        self.smart_refine_note = QLabel("Preview only · exact on Confirm")
        self.smart_refine_note.setWordWrap(True)
        self.smart_refine_note.setAccessibleName(
            "Smart refinement preview accuracy notice")
        options.addWidget(self.smart_refine_invert)
        options.addWidget(self.smart_refine_note, 1)
        refinement.addLayout(options)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.confirm_smart_refine_btn = QPushButton("Confirm")
        self.confirm_smart_refine_btn.setAccessibleName(
            "Confirm exact Smart mask refinement")
        self.cancel_smart_refine_btn = QPushButton("Cancel")
        self.cancel_smart_refine_btn.setAccessibleName(
            "Cancel Smart mask refinement")
        actions.addWidget(self.confirm_smart_refine_btn)
        actions.addWidget(self.cancel_smart_refine_btn)
        refinement.addLayout(actions)
        self._smart_refinement_token: str | None = None
        self._smart_refinement_focus_widgets = (
            self.smart_threshold, self.smart_refine_kind,
            self.smart_refine_radius, self.smart_feather_radius,
            self.smart_refine_invert,
            self.confirm_smart_refine_btn, self.cancel_smart_refine_btn)
        for widget in self._smart_refinement_focus_widgets:
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.gradient_editor = QWidget()
        gradient_layout = QVBoxLayout(self.gradient_editor)
        gradient_layout.setContentsMargins(0, 0, 0, 0)
        self.gradient_note = QLabel(
            "Drag on canvas or enter normalized geometry.")
        self.gradient_note.setWordWrap(True)
        gradient_layout.addWidget(self.gradient_note)
        geometry = QGridLayout()
        geometry.setHorizontalSpacing(4)
        geometry.setVerticalSpacing(4)
        self.gradient_spins = []
        for index, (label, value) in enumerate(
            (("X1", 0), ("Y1", 0), ("X2", 100), ("Y2", 0))
        ):
            row, pair = divmod(index, 2)
            geometry.addWidget(QLabel(label), row, pair * 2)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix("%")
            spin.setValue(value)
            spin.setAccessibleName(f"Gradient {label} source coordinate")
            geometry.addWidget(spin, row, pair * 2 + 1)
            self.gradient_spins.append(spin)
        gradient_layout.addLayout(geometry)
        gradient_actions = QHBoxLayout()
        gradient_actions.addStretch(1)
        self.confirm_gradient_btn = QPushButton("Create")
        self.cancel_gradient_btn = QPushButton("Cancel")
        gradient_actions.addWidget(self.confirm_gradient_btn)
        gradient_actions.addWidget(self.cancel_gradient_btn)
        gradient_layout.addLayout(gradient_actions)
        self._gradient_kind: str | None = None
        self.gradient_editor.setVisible(False)

        self.pattern_editor = QWidget()
        pattern_layout = QVBoxLayout(self.pattern_editor)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.setSpacing(4)
        pattern_form = QFormLayout()
        pattern_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.pattern_family = QComboBox()
        for label, value in (
            ("Bayer", "bayer"), ("Lines", "lines"), ("Noise", "noise")
        ):
            self.pattern_family.addItem(label, value)
        self.pattern_family.setAccessibleName("Mask pattern family")
        self.pattern_scale = QSpinBox()
        self.pattern_scale.setRange(1, 4096)
        self.pattern_scale.setValue(1)
        self.pattern_scale.setAccessibleName("Mask pattern scale in source pixels")
        self.pattern_orientation = QComboBox()
        self.pattern_orientation.setAccessibleName("Mask pattern orientation")
        pattern_form.addRow("Family", self.pattern_family)
        pattern_form.addRow("Scale", self.pattern_scale)
        pattern_form.addRow("Direction", self.pattern_orientation)
        pattern_layout.addLayout(pattern_form)
        offsets = QFormLayout()
        offsets.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.pattern_offset_x = QSpinBox()
        self.pattern_offset_y = QSpinBox()
        for axis, spin in (
            ("X", self.pattern_offset_x), ("Y", self.pattern_offset_y)
        ):
            spin.setRange(-1_000_000, 1_000_000)
            spin.setAccessibleName(f"Mask pattern {axis} offset")
            offsets.addRow(f"{axis} offset", spin)
        self.pattern_seed = QSpinBox()
        self.pattern_seed.setRange(-(2 ** 31), 2 ** 31 - 1)
        self.pattern_seed.setAccessibleName("Seeded noise pattern seed")
        offsets.addRow("Seed", self.pattern_seed)
        pattern_layout.addLayout(offsets)
        mix_row = QHBoxLayout()
        self.pattern_mix_label = QLabel("Dither mix")
        self.pattern_mix = QSpinBox()
        self.pattern_mix.setRange(0, 100)
        self.pattern_mix.setValue(100)
        self.pattern_mix.setSuffix("%")
        self.pattern_mix.setAccessibleName("Dither mask mix")
        mix_row.addWidget(self.pattern_mix_label)
        mix_row.addWidget(self.pattern_mix)
        mix_row.addStretch(1)
        pattern_layout.addLayout(mix_row)
        pattern_actions = QHBoxLayout()
        pattern_actions.addStretch(1)
        self.confirm_pattern_btn = QPushButton("Create")
        self.confirm_pattern_btn.setAccessibleName("Create raster mask pattern")
        self.cancel_pattern_btn = QPushButton("Cancel")
        self.cancel_pattern_btn.setAccessibleName("Cancel mask pattern")
        pattern_actions.addWidget(self.confirm_pattern_btn)
        pattern_actions.addWidget(self.cancel_pattern_btn)
        pattern_layout.addLayout(pattern_actions)
        self._pattern_dither = False
        self._pattern_editor_open = False
        self._refresh_pattern_controls()
        self.pattern_editor.setVisible(False)

        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        for label, spin in (("Layer X position", self.x_spin),
                            ("Layer Y position", self.y_spin)):
            spin.setRange(-100000, 100000)
            spin.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setAccessibleName(label)
        self.width_spin = QSpinBox()
        self.height_spin = QSpinBox()
        for label, spin in (("Layer width", self.width_spin),
                            ("Layer height", self.height_spin)):
            spin.setRange(1, 100000)
            spin.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setAccessibleName(label)
        self.lock_aspect_check = QCheckBox("Lock aspect")
        self.lock_aspect_check.setChecked(True)
        self.center_btn = QPushButton("Center")
        self.fit_btn = QPushButton("Fit")
        self.transform_btn = QPushButton("Transform")
        self.transform_btn.setAccessibleName(
            "Move or resize selected layer on canvas")
        self.confirm_transform_btn = QPushButton("Confirm")
        self.confirm_transform_btn.setAccessibleName(
            "Confirm layer position and size")
        self.cancel_transform_btn = QPushButton("Cancel")
        self.cancel_transform_btn.setAccessibleName(
            "Cancel layer position and size changes")
        self._transform_mode = False
        self._transform_syncing = False
        self._aspect_ratio = 1.0

        self.export_btn = QPushButton("Export Layers")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Layers status")
        self.status_label.setProperty("error", False)
        self.transform_guidance_label = QLabel(
            "Confirm or cancel the active layer transform first.")
        self.transform_guidance_label.setWordWrap(True)
        self.transform_guidance_label.setAccessibleName(
            "Active transform guidance")

        self._build_layout()
        self.setFocusProxy(self.layer_list)
        self._connect()
        self._document_available = False
        self._refresh_state()

    def _build_layout(self) -> None:
        actions = QGridLayout()
        actions.setHorizontalSpacing(4)
        actions.setVerticalSpacing(4)
        actions.addWidget(self.new_blank_btn, 0, 0)
        actions.addWidget(self.place_image_btn, 0, 1)
        actions.addWidget(self.duplicate_btn, 1, 0)
        actions.addWidget(self.delete_btn, 1, 1)

        ordering = QHBoxLayout()
        ordering.addWidget(self.up_btn)
        ordering.addWidget(self.down_btn)
        ordering.addStretch(1)

        layer_form = QFormLayout()
        layer_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layer_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layer_form.addRow(self.visibility_check)
        name_label = QLabel("Name")
        name_label.setBuddy(self.name_edit)
        layer_form.addRow(name_label, self.name_edit)
        blend_label = QLabel("Blend")
        blend_label.setBuddy(self.blend_combo)
        layer_form.addRow(blend_label, self.blend_combo)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_label)
        opacity_label = QLabel("Opacity")
        opacity_label.setBuddy(self.opacity_slider)
        layer_form.addRow(opacity_label, opacity_row)

        layer_form.addRow("Position X", self.x_spin)
        layer_form.addRow("Position Y", self.y_spin)
        layer_form.addRow("Width", self.width_spin)
        layer_form.addRow("Height", self.height_spin)

        transform_actions = QHBoxLayout()
        transform_actions.addWidget(self.center_btn)
        transform_actions.addWidget(self.fit_btn)
        layer_form.addRow(self.lock_aspect_check)
        layer_form.addRow(transform_actions)

        canvas_transform_actions = QHBoxLayout()
        canvas_transform_actions.addWidget(self.transform_btn)
        canvas_transform_actions.addWidget(self.confirm_transform_btn)
        canvas_transform_actions.addWidget(self.cancel_transform_btn)
        layer_form.addRow("Canvas", canvas_transform_actions)

        layer_page = QWidget()
        self.layer_page = layer_page
        layer_page_layout = QVBoxLayout(layer_page)
        layer_page_layout.setContentsMargins(6, 6, 6, 6)
        layer_page_layout.setSpacing(6)
        layer_page_layout.addLayout(layer_form)
        layer_page_layout.addWidget(self.export_btn)
        layer_page_layout.addStretch(1)

        mask_content = QWidget()
        self.mask_content = mask_content
        mask_content.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        mask_root = QVBoxLayout(mask_content)
        mask_root.setContentsMargins(6, 6, 6, 6)
        mask_root.setSpacing(8)

        tool_row = QHBoxLayout()
        tool_row.addWidget(self.pointer_btn)
        tool_row.addWidget(self.paint_mask_btn)
        mask_root.addLayout(tool_row)
        mode_row = QHBoxLayout()
        mode_label = QLabel("Brush action")
        mode_label.setBuddy(self.brush_mode_combo)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.brush_mode_combo, 1)
        mask_root.addLayout(mode_row)

        mask_actions = QHBoxLayout()
        mask_actions.addWidget(self.create_mask_btn)
        mask_actions.addWidget(self.edit_mask_btn)
        mask_io_actions = QHBoxLayout()
        mask_io_actions.addWidget(self.import_mask_btn)
        mask_io_actions.addWidget(self.export_mask_btn)
        mask_setup = QGroupBox("Mask")
        mask_setup_layout = QVBoxLayout(mask_setup)
        mask_setup_layout.setContentsMargins(6, 8, 6, 6)
        mask_setup_layout.setSpacing(4)
        mask_setup_layout.addWidget(self.mask_state_label)
        mask_setup_layout.addLayout(mask_actions)
        mask_setup_layout.addLayout(mask_io_actions)
        combine_row = QHBoxLayout()
        combine_row.addWidget(QLabel("New masks"))
        combine_row.addWidget(self.mask_combination_combo, 1)
        mask_setup_layout.addLayout(combine_row)
        mask_density_row = QHBoxLayout()
        mask_density_row.addWidget(self.mask_enabled_check)
        mask_density_row.addWidget(self.mask_density_slider, 1)
        mask_density_row.addWidget(self.mask_density_label)
        mask_setup_layout.addLayout(mask_density_row)
        inspection_label = QLabel("Inspect")
        inspection_label.setBuddy(self.inspection_combo)
        inspection_row = QHBoxLayout()
        inspection_row.addWidget(inspection_label)
        inspection_row.addWidget(self.inspection_combo, 1)
        mask_setup_layout.addLayout(inspection_row)
        mask_root.addWidget(mask_setup)

        brush_group = QGroupBox("Brush")
        brush_form = QFormLayout(brush_group)
        brush_form.setContentsMargins(6, 8, 6, 6)
        brush_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        brush_form.addRow("Tip", self.brush_tip_combo)
        brush_form.addRow("Size", self.brush_size_spin)
        brush_form.addRow("Hardness", self.brush_hardness_spin)
        brush_form.addRow("Strength", self.brush_strength_spin)
        brush_form.addRow("Spacing", self.brush_spacing_spin)
        mask_root.addWidget(brush_group)

        selection_group = QGroupBox("Selection")
        selection_layout = QVBoxLayout(selection_group)
        selection_layout.setContentsMargins(6, 8, 6, 6)
        selection_layout.setSpacing(4)
        selection_form = QFormLayout()
        selection_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        selection_form.addRow("Shape", self.selection_shape_combo)
        selection_form.addRow("Mode", self.selection_operation_combo)
        selection_layout.addLayout(selection_form)
        selection_primary = QVBoxLayout()
        selection_primary.setSpacing(4)
        selection_primary.addWidget(self.selection_draw_btn)
        selection_primary.addWidget(self.selection_clear_btn)
        selection_layout.addLayout(selection_primary)
        refine_grid = QGridLayout()
        refine_grid.setHorizontalSpacing(4)
        refine_grid.setVerticalSpacing(4)
        refine_grid.addWidget(QLabel("Radius"), 0, 0)
        refine_grid.addWidget(self.selection_radius_spin, 0, 1)
        refine_grid.addWidget(self.selection_all_btn, 1, 0)
        refine_grid.addWidget(self.selection_invert_btn, 1, 1)
        refine_grid.addWidget(self.selection_grow_btn, 2, 0)
        refine_grid.addWidget(self.selection_shrink_btn, 2, 1)
        refine_grid.addWidget(self.selection_feather_btn, 3, 0, 1, 2)
        selection_layout.addLayout(refine_grid)
        selection_layout.addWidget(self.color_range_btn)
        range_form = QFormLayout()
        range_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        range_form.addRow("Tolerance", self.color_range_tolerance_spin)
        range_form.addRow("Softness", self.color_range_softness_spin)
        selection_layout.addLayout(range_form)
        range_actions = QHBoxLayout()
        range_actions.addWidget(self.color_range_confirm_btn)
        range_actions.addWidget(self.color_range_cancel_btn)
        selection_layout.addLayout(range_actions)
        selection_layout.addWidget(self.from_selection_btn)
        mask_root.addWidget(selection_group)
        mask_root.addWidget(self.mask_confirmation)
        mask_root.addWidget(self.smart_refinement)
        mask_root.addWidget(self.gradient_editor)
        mask_root.addWidget(self.pattern_editor)
        mask_root.addStretch(1)

        self.mask_scroll = QScrollArea()
        self.mask_scroll.setWidgetResizable(True)
        self.mask_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.mask_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.mask_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mask_scroll.setWidget(mask_content)

        self.editor_tabs = QTabWidget()
        self.editor_tabs.setAccessibleName("Layer and mask editing tools")
        self.editor_tabs.addTab(layer_page, "Layer")
        self.editor_tabs.addTab(self.mask_scroll, "Mask")
        self.editor_tabs.setMinimumHeight(250)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self.empty_label)
        root.addLayout(actions)
        root.addWidget(self.layer_list, 1)
        root.addLayout(ordering)
        root.addWidget(self.editor_tabs, 2)
        root.addWidget(self.transform_guidance_label)
        root.addWidget(self.status_label)

        QWidget.setTabOrder(self.new_blank_btn, self.place_image_btn)
        QWidget.setTabOrder(self.place_image_btn, self.duplicate_btn)
        QWidget.setTabOrder(self.duplicate_btn, self.delete_btn)
        QWidget.setTabOrder(self.delete_btn, self.layer_list)
        QWidget.setTabOrder(self.layer_list, self.up_btn)
        QWidget.setTabOrder(self.up_btn, self.down_btn)
        QWidget.setTabOrder(self.down_btn, self.visibility_check)
        QWidget.setTabOrder(self.visibility_check, self.name_edit)
        QWidget.setTabOrder(self.name_edit, self.blend_combo)
        QWidget.setTabOrder(self.blend_combo, self.opacity_slider)
        QWidget.setTabOrder(self.opacity_slider, self.create_mask_btn)
        QWidget.setTabOrder(self.create_mask_btn, self.edit_mask_btn)
        QWidget.setTabOrder(self.edit_mask_btn, self.import_mask_btn)
        QWidget.setTabOrder(self.import_mask_btn, self.export_mask_btn)
        QWidget.setTabOrder(self.export_mask_btn, self.mask_enabled_check)
        QWidget.setTabOrder(
            self.mask_enabled_check, self.mask_density_slider)
        QWidget.setTabOrder(self.mask_density_slider, self.inspection_combo)
        QWidget.setTabOrder(self.inspection_combo, self.x_spin)
        QWidget.setTabOrder(self.smart_threshold, self.smart_refine_kind)
        QWidget.setTabOrder(self.smart_refine_kind, self.smart_refine_radius)
        QWidget.setTabOrder(
            self.smart_refine_radius, self.smart_feather_radius)
        QWidget.setTabOrder(
            self.smart_feather_radius, self.smart_refine_invert)
        QWidget.setTabOrder(
            self.smart_refine_invert, self.confirm_smart_refine_btn)
        QWidget.setTabOrder(
            self.confirm_smart_refine_btn, self.cancel_smart_refine_btn)
        QWidget.setTabOrder(self.cancel_smart_refine_btn, self.x_spin)
        QWidget.setTabOrder(self.x_spin, self.y_spin)
        QWidget.setTabOrder(self.y_spin, self.width_spin)
        QWidget.setTabOrder(self.width_spin, self.height_spin)
        QWidget.setTabOrder(self.height_spin, self.lock_aspect_check)
        QWidget.setTabOrder(self.lock_aspect_check, self.center_btn)
        QWidget.setTabOrder(self.center_btn, self.fit_btn)
        QWidget.setTabOrder(self.fit_btn, self.transform_btn)
        QWidget.setTabOrder(self.transform_btn, self.confirm_transform_btn)
        QWidget.setTabOrder(
            self.confirm_transform_btn, self.cancel_transform_btn)
        QWidget.setTabOrder(self.cancel_transform_btn, self.export_btn)

    def _connect(self) -> None:
        self.new_blank_btn.clicked.connect(self.new_blank_requested.emit)
        self.new_blank_btn.clicked.connect(self.add_requested.emit)
        self.place_image_btn.clicked.connect(self.place_image_requested.emit)
        self.duplicate_btn.clicked.connect(self._request_duplicate)
        self.delete_btn.clicked.connect(self._request_delete)
        self.up_btn.clicked.connect(self._request_up)
        self.down_btn.clicked.connect(self._request_down)
        self.layer_list.currentItemChanged.connect(self._selection_changed)
        self.layer_list.itemChanged.connect(self._row_visibility_edited)
        self.visibility_check.toggled.connect(self._visibility_edited)
        self.name_edit.editingFinished.connect(self._name_edited)
        self.blend_combo.currentIndexChanged.connect(self._blend_edited)
        self.opacity_slider.valueChanged.connect(self._opacity_edited)
        self.reveal_mask_btn.clicked.connect(
            lambda: self._request_mask_generator(self.reveal_mask_requested))
        self.hide_mask_btn.clicked.connect(
            lambda: self._request_mask_generator(self.hide_mask_requested))
        self.transparency_mask_btn.clicked.connect(
            lambda: self._request_mask_generator(
                self.transparency_mask_requested))
        self.reveal_mask_action.triggered.connect(
            lambda: self._request_mask_generator(self.reveal_mask_requested))
        self.hide_mask_action.triggered.connect(
            lambda: self._request_mask_generator(self.hide_mask_requested))
        self.transparency_mask_action.triggered.connect(
            lambda: self._request_mask_generator(
                self.transparency_mask_requested))
        self.smart_mask_action.triggered.connect(self.smart_mask_requested.emit)
        self.freeze_smart_action.triggered.connect(
            self.smart_refinement_started.emit)
        self.shadows_mask_action.triggered.connect(
            lambda: self.luminance_mask_requested.emit("shadows"))
        self.midtones_mask_action.triggered.connect(
            lambda: self.luminance_mask_requested.emit("midtones"))
        self.highlights_mask_action.triggered.connect(
            lambda: self.luminance_mask_requested.emit("highlights"))
        self.linear_gradient_action.triggered.connect(
            lambda: self.show_gradient_editor("linear"))
        self.radial_gradient_action.triggered.connect(
            lambda: self.show_gradient_editor("radial"))
        self.pattern_mask_action.triggered.connect(
            lambda: self.show_pattern_editor(False))
        self.dither_mask_action.triggered.connect(
            lambda: self.show_pattern_editor(True))
        self.confirm_gradient_btn.clicked.connect(self._confirm_gradient)
        self.cancel_gradient_btn.clicked.connect(
            lambda: self.show_gradient_editor(None))
        self.pattern_family.currentIndexChanged.connect(
            self._refresh_pattern_controls)
        self.confirm_pattern_btn.clicked.connect(self._confirm_pattern)
        self.cancel_pattern_btn.clicked.connect(
            lambda: self.show_pattern_editor(None))
        self.invert_mask_action.triggered.connect(self.invert_mask_requested.emit)
        self.fill_white_mask_action.triggered.connect(
            self.fill_white_mask_requested.emit)
        self.fill_black_mask_action.triggered.connect(
            self.fill_black_mask_requested.emit)
        self.reset_mask_action.triggered.connect(self.reset_mask_requested.emit)
        self.delete_mask_action.triggered.connect(self.delete_mask_requested.emit)
        self.import_mask_btn.clicked.connect(self.import_mask_requested.emit)
        self.export_mask_btn.clicked.connect(self.export_mask_requested.emit)
        self.mask_enabled_check.toggled.connect(self._mask_enabled_edited)
        self.mask_density_slider.valueChanged.connect(
            self._mask_density_edited)
        self.inspection_combo.currentTextChanged.connect(
            self.inspection_mode_changed.emit)
        self.selection_draw_btn.clicked.connect(
            lambda: self.selection_tool_requested.emit(
                self.selection_shape_combo.currentText().lower()))
        self.selection_clear_btn.clicked.connect(
            self.selection_clear_requested.emit)
        self.from_selection_btn.clicked.connect(
            self.selection_from_mask_requested.emit)
        self.color_range_btn.clicked.connect(self.color_range_pick_requested.emit)
        self.color_range_tolerance_spin.valueChanged.connect(
            lambda value: self.color_range_changed.emit(
                value, self.color_range_softness_spin.value()))
        self.color_range_softness_spin.valueChanged.connect(
            lambda value: self.color_range_changed.emit(
                self.color_range_tolerance_spin.value(), value))
        self.color_range_confirm_btn.clicked.connect(
            self.color_range_confirmed.emit)
        self.color_range_cancel_btn.clicked.connect(
            self.color_range_cancelled.emit)
        self.pointer_btn.clicked.connect(self._request_pointer)
        self.paint_mask_btn.toggled.connect(self.mask_paint_requested.emit)
        self.brush_mode_combo.currentTextChanged.connect(
            lambda text: self.brush_mode_changed.emit(text.lower()))
        for button, operation in (
            (self.selection_all_btn, "all"),
            (self.selection_invert_btn, "invert"),
            (self.selection_grow_btn, "grow"),
            (self.selection_shrink_btn, "shrink"),
            (self.selection_feather_btn, "feather"),
        ):
            button.clicked.connect(
                lambda _checked=False, op=operation:
                self.selection_refine_requested.emit(
                    op, self.selection_radius_spin.value()))
        self.brush_tip_combo.currentTextChanged.connect(self._emit_brush_settings)
        for control in (
            self.brush_size_spin, self.brush_hardness_spin,
            self.brush_strength_spin, self.brush_spacing_spin,
        ):
            control.valueChanged.connect(self._emit_brush_settings)
        self.replace_mask_btn.clicked.connect(self._confirm_mask_replacement)
        self.cancel_replace_mask_btn.clicked.connect(
            self._cancel_mask_replacement)
        self.smart_threshold.valueChanged.connect(
            self._emit_smart_refinement)
        self.smart_refine_kind.currentTextChanged.connect(
            self._emit_smart_refinement)
        self.smart_refine_radius.valueChanged.connect(
            self._emit_smart_refinement)
        self.smart_feather_radius.valueChanged.connect(
            self._emit_smart_refinement)
        self.smart_refine_invert.toggled.connect(
            self._emit_smart_refinement)
        self.confirm_smart_refine_btn.clicked.connect(
            self._confirm_smart_refinement)
        self.cancel_smart_refine_btn.clicked.connect(
            self._cancel_smart_refinement)
        for spin in (self.x_spin, self.y_spin, self.width_spin, self.height_spin):
            spin.valueChanged.connect(self._transform_edited)
        self.center_btn.clicked.connect(self._request_center)
        self.fit_btn.clicked.connect(self._request_fit)
        self.transform_btn.clicked.connect(self._request_transform_mode)
        self.confirm_transform_btn.clicked.connect(
            self.transform_confirmed.emit)
        self.cancel_transform_btn.clicked.connect(
            self.transform_cancelled.emit)
        self.export_btn.clicked.connect(self.export_requested.emit)

    @staticmethod
    def _make_placeholder() -> QPixmap:
        pixmap = QPixmap(56, 56)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        colors = (QColor(196, 196, 196, 150), QColor(230, 230, 230, 150))
        square = 7
        for y in range(0, 56, square):
            for x in range(0, 56, square):
                painter.fillRect(x, y, square, square,
                                 colors[(x // square + y // square) % 2])
        painter.end()
        return pixmap

    def show_gradient_editor(self, kind: str | None) -> None:
        if kind is not None and kind not in {"linear", "radial"}:
            raise ValueError("gradient kind must be linear or radial")
        self._gradient_kind = kind
        self.gradient_editor.setVisible(kind is not None)
        self.gradient_tool_requested.emit("" if kind is None else kind)
        if kind is not None:
            self.gradient_note.setText(
                f"{kind.title()} gradient · drag on canvas or enter geometry.")
            self.gradient_spins[0].setFocus(Qt.FocusReason.OtherFocusReason)

    def selection_operation_text(self) -> str:
        return self.selection_operation_combo.currentText()

    def _emit_brush_settings(self, *_args) -> None:
        self.brush_settings_changed.emit(
            self.brush_tip_combo.currentText().lower(),
            self.brush_size_spin.value(),
            self.brush_hardness_spin.value(),
            self.brush_strength_spin.value(),
            self.brush_spacing_spin.value(),
        )

    def _request_pointer(self) -> None:
        self.pointer_requested.emit()
        if self.paint_mask_btn.isChecked():
            self.paint_mask_btn.setChecked(False)

    def set_mask_paint_active(self, active: bool) -> None:
        """Mirror the viewport tool without feeding the request signal back."""
        with QSignalBlocker(self.paint_mask_btn):
            self.paint_mask_btn.setChecked(bool(active))

    def set_gradient_geometry(
        self, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> None:
        """Mirror canvas drag geometry into the keyboard-editable fields."""
        for spin, value in zip(
            self.gradient_spins, (start_x, start_y, end_x, end_y)
        ):
            spin.setValue(round(max(0.0, min(1.0, float(value))) * 100.0))

    def _confirm_gradient(self) -> None:
        if self._gradient_kind is None:
            return
        values = [spin.value() / 100.0 for spin in self.gradient_spins]
        self.gradient_mask_requested.emit(self._gradient_kind, *values)
        self.show_gradient_editor(None)

    def show_pattern_editor(self, dither: bool | None) -> None:
        self._pattern_editor_open = dither is not None
        self.pattern_editor.setVisible(dither is not None)
        if dither is None:
            return
        self._pattern_dither = bool(dither)
        self.confirm_pattern_btn.setText("Dither" if dither else "Create")
        self.confirm_pattern_btn.setAccessibleName(
            "Dither the active raster mask" if dither
            else "Create raster mask pattern")
        self.pattern_mix_label.setVisible(bool(dither))
        self.pattern_mix.setVisible(bool(dither))
        self._refresh_pattern_controls()
        self.pattern_family.setFocus(Qt.FocusReason.OtherFocusReason)

    def _refresh_pattern_controls(self) -> None:
        kind = self.pattern_family.currentData()
        self.pattern_seed.setEnabled(kind == "noise")
        self.pattern_orientation.setEnabled(kind != "noise")
        angles = {
            "bayer": (0, 90, 180, 270),
            "lines": (0, 45, 90, 135),
            "noise": (0,),
        }[kind]
        current = self.pattern_orientation.currentData()
        blocker = QSignalBlocker(self.pattern_orientation)
        self.pattern_orientation.clear()
        for angle in angles:
            self.pattern_orientation.addItem(f"{angle} degrees", angle)
        index = self.pattern_orientation.findData(current)
        self.pattern_orientation.setCurrentIndex(max(0, index))
        del blocker

    def _confirm_pattern(self) -> None:
        if not self._pattern_editor_open:
            return
        self.pattern_mask_requested.emit(
            str(self.pattern_family.currentData()),
            self.pattern_scale.value(),
            int(self.pattern_orientation.currentData()),
            self.pattern_offset_x.value(),
            self.pattern_offset_y.value(),
            self.pattern_mix.value() if self._pattern_dither else 100,
            self.pattern_seed.value(),
            self._pattern_dither,
        )
        self.show_pattern_editor(None)

    def set_layers(
        self,
        layers: Iterable[tuple[str, str, bool, int, str]],
        selected: int | None,
    ) -> None:
        """Replace displayed state without emitting user-edit signals.

        Clearing the list destroys its row widgets.  Qt may synchronously move
        focus while doing that, which can route an editor refresh back into
        this method.  Ignore that nested publication: the outer publication
        already represents the controller's authoritative document, and
        allowing both rebuilds leaves the outer call holding deleted item
        wrappers.
        """
        if getattr(self, "_setting_layers", False):
            return
        self._setting_layers = True
        try:
            self._set_layers_once(layers, selected)
        finally:
            self._setting_layers = False

    def _set_layers_once(
        self,
        layers: Iterable[tuple[str, str, bool, int, str]],
        selected: int | None,
    ) -> None:
        """Replace displayed state without emitting user-edit signals."""
        focused_target: tuple[str, str] | None = None
        focused = QApplication.focusWidget()
        for layer_id, buttons in self._row_targets.items():
            if focused is buttons[0]:
                focused_target = (layer_id, "layer")
            elif focused is buttons[1]:
                focused_target = (layer_id, "mask")

        clean = []
        for raw in layers:
            values = tuple(raw)
            if len(values) == 5:
                values = (*values, 0, 0, 1, 1, False, False)
            elif len(values) == 9:
                values = (*values, False, False)
            if len(values) != 11:
                raise ValueError("layer rows must have five, nine, or eleven values")
            (
                layer_id, name, visible, opacity, blend, x, y, width, height,
                mask_present, mask_enabled,
            ) = values
            clean.append((
                str(layer_id), str(name), bool(visible), int(opacity), str(blend),
                int(x), int(y), max(1, int(width)), max(1, int(height)),
                bool(mask_present), bool(mask_enabled),
            ))
        self._layers = clean
        live_ids = {layer_id for layer_id, *_rest in clean}
        self._thumbnails = {
            layer_id: pixmap
            for layer_id, pixmap in self._thumbnails.items()
            if layer_id in live_ids
        }

        with QSignalBlocker(self.layer_list):
            self.layer_list.clear()
            self._row_targets.clear()
            selected_item: QListWidgetItem | None = None
            for core_index in range(len(clean) - 1, -1, -1):
                (
                    layer_id, name, visible, _opacity, _blend,
                    _x, _y, _width, _height, mask_present, mask_enabled,
                ) = clean[core_index]
                item = QListWidgetItem(name)
                item.setData(Qt.ItemDataRole.UserRole, core_index)
                item.setData(int(Qt.ItemDataRole.UserRole) + 1, layer_id)
                item.setCheckState(
                    Qt.CheckState.Checked if visible
                    else Qt.CheckState.Unchecked)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setSizeHint(QSize(0, 62))
                item.setIcon(QIcon(
                    self._thumbnails.get(layer_id, self._placeholder)))
                state_text = "visible" if visible else "hidden"
                accessible_text = f"{name}, {state_text}"
                item.setToolTip(accessible_text)
                item.setData(
                    Qt.ItemDataRole.AccessibleTextRole, accessible_text)
                self.layer_list.addItem(item)
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(2, 2, 2, 2)
                row_layout.setSpacing(4)
                layer_target = _TargetButton()
                layer_target.setCheckable(True)
                layer_target.setIcon(QIcon(
                    self._thumbnails.get(layer_id, self._placeholder)))
                layer_target.setIconSize(QSize(48, 48))
                layer_target.setAccessibleName(f"Edit pixels for {name}")
                layer_target.setAccessibleDescription(
                    "Layer thumbnail. Press Enter or Space to edit layer pixels.")
                layer_target.setToolTip(f"{name} layer pixels")
                mask_target = _TargetButton()
                mask_target.setCheckable(True)
                mask_target.setText("Mask" if mask_present else "Add Mask")
                mask_target.setToolButtonStyle(
                    Qt.ToolButtonStyle.ToolButtonTextOnly)
                mask_target.setFixedSize(64, 48)
                mask_target.setAccessibleName(f"Edit raster mask for {name}")
                mask_target.setAccessibleDescription(
                    "Raster mask thumbnail linked to the layer transform. "
                    + (
                        "The mask is disabled; enable it before editing pixels. "
                        if mask_present and not mask_enabled else
                        "No raster mask; activating selects its creation target. "
                        if not mask_present else
                        ""
                    )
                    + "Press Enter or Space to activate.")
                mask_target.setToolTip(
                    "Raster mask linked to layer transform"
                    if mask_present else
                    "No raster mask · linked creation target")
                label = QLabel(name)
                label.setToolTip(accessible_text)
                row_layout.addWidget(layer_target)
                row_layout.addWidget(mask_target)
                row_layout.addWidget(label, 1)
                layer_target.clicked.connect(
                    lambda _checked=False, lid=layer_id, it=item:
                    self._activate_row_target(it, lid, "layer"))
                mask_target.clicked.connect(
                    lambda _checked=False, lid=layer_id, it=item:
                    self._activate_row_target(it, lid, "mask"))
                self._row_targets[layer_id] = (layer_target, mask_target)
                self.layer_list.setItemWidget(item, row)
                if selected is not None and core_index == int(selected):
                    selected_item = item
            self.layer_list.setCurrentItem(selected_item)
        self._sync_edit_target()
        if focused_target is not None:
            buttons = self._row_targets.get(focused_target[0])
            if buttons is not None:
                buttons[0 if focused_target[1] == "layer" else 1].setFocus(
                    Qt.FocusReason.OtherFocusReason)
        self._refresh_state()

    def set_document_available(self, available: bool) -> None:
        """Publish document existence separately from its possibly empty layer list."""
        self._document_available = bool(available)
        self._refresh_state()

    def set_layer_thumbnail(self, layer_id: str, pixmap: QPixmap) -> None:
        """Cache a defensive thumbnail copy and update its live row in place."""
        key = str(layer_id)
        copied = QPixmap(pixmap)
        self._thumbnails[key] = copied
        id_role = int(Qt.ItemDataRole.UserRole) + 1
        for row in range(self.layer_list.count()):
            item = self.layer_list.item(row)
            if item.data(id_role) == key:
                with QSignalBlocker(self.layer_list):
                    item.setIcon(QIcon(copied))
                    targets = self._row_targets.get(key)
                    if targets is not None:
                        targets[0].setIcon(QIcon(copied))
                break

    def set_edit_target(self, layer_id: str | None, kind: str = "layer") -> None:
        if layer_id is None:
            self._edit_target = None
        else:
            if kind not in {"layer", "mask"}:
                raise ValueError("edit target kind must be layer or mask")
            self._edit_target = (str(layer_id), kind)
        self._sync_edit_target()

    def _sync_edit_target(self) -> None:
        for layer_id, buttons in self._row_targets.items():
            for kind, button in zip(("layer", "mask"), buttons):
                with QSignalBlocker(button):
                    button.setChecked(self._edit_target == (layer_id, kind))
                button.setProperty(
                    "editTarget", self._edit_target == (layer_id, kind))
                button.style().unpolish(button)
                button.style().polish(button)

    def _activate_row_target(
        self, item: QListWidgetItem, layer_id: str, kind: str
    ) -> None:
        self.layer_list.setCurrentItem(item)
        self.editor_tabs.setCurrentIndex(1 if kind == "mask" else 0)
        self.edit_target_requested.emit(layer_id, kind)
        buttons = self._row_targets.get(layer_id)
        if buttons is not None:
            buttons[0 if kind == "layer" else 1].setFocus(
                Qt.FocusReason.OtherFocusReason)

    def show_mask_replacement(
        self, token: str | None, label: str = "Replace the existing raster mask?"
    ) -> None:
        self._replacement_token = None if token is None else str(token)
        self.mask_confirmation_label.setText(str(label))
        self.mask_confirmation.setVisible(token is not None)

    def show_smart_refinement(
        self, token: str | None, threshold: int = 50, invert: bool = False
    ) -> None:
        self._smart_refinement_token = None if token is None else str(token)
        for widget in self._smart_refinement_focus_widgets:
            widget.setFocusPolicy(
                Qt.FocusPolicy.StrongFocus if token is not None
                else Qt.FocusPolicy.NoFocus)
        if token is not None:
            QWidget.setTabOrder(self.inspection_combo, self.smart_threshold)
            QWidget.setTabOrder(
                self.cancel_smart_refine_btn, self.x_spin)
        else:
            QWidget.setTabOrder(self.inspection_combo, self.x_spin)
        if token is not None:
            for widget, value in (
                (self.smart_threshold, int(threshold)),
                (self.smart_refine_radius, 0),
                (self.smart_feather_radius, 0),
            ):
                with QSignalBlocker(widget):
                    widget.setValue(value)
            with QSignalBlocker(self.smart_refine_kind):
                self.smart_refine_kind.setCurrentText("None")
            with QSignalBlocker(self.smart_refine_invert):
                self.smart_refine_invert.setChecked(bool(invert))
            self.set_smart_refinement_pending(False)
        self._refresh_state()
        if token is not None:
            self.smart_threshold.setFocus(Qt.FocusReason.OtherFocusReason)

    def _emit_smart_refinement(self, *_args) -> None:
        if self._smart_refinement_token is not None:
            self.smart_refinement_changed.emit(
                self.smart_threshold.value(),
                self.smart_refine_kind.currentText().lower(),
                self.smart_refine_radius.value(),
                self.smart_feather_radius.value(),
                self.smart_refine_invert.isChecked())

    def set_smart_refinement_pending(self, pending: bool) -> None:
        for widget in self._smart_refinement_focus_widgets:
            widget.setEnabled(
                not pending or widget is self.cancel_smart_refine_btn)
        self.smart_refine_note.setText(
            "Computing exact mask…"
            if pending else "Preview only · exact on Confirm")

    def _confirm_smart_refinement(self) -> None:
        if self._smart_refinement_token is not None:
            self.smart_refinement_confirmed.emit(
                self._smart_refinement_token)

    def _cancel_smart_refinement(self) -> None:
        if self._smart_refinement_token is not None:
            self.smart_refinement_cancelled.emit(
                self._smart_refinement_token)

    def _confirm_mask_replacement(self) -> None:
        if self._replacement_token is not None:
            self.mask_replacement_confirmed.emit(self._replacement_token)

    def _cancel_mask_replacement(self) -> None:
        if self._replacement_token is not None:
            self.mask_replacement_cancelled.emit(self._replacement_token)

    def selected_index(self) -> int | None:
        item = self.layer_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(value, int) and 0 <= value < len(self._layers):
            return value
        return None

    def set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(str(text))
        self.status_label.setProperty("error", bool(error))
        style = self.status_label.style()
        style.unpolish(self.status_label)
        style.polish(self.status_label)
        self.status_label.update()

    def set_mask_state(
        self, present: bool, enabled: bool = False, density: int = 100
    ) -> None:
        disarm_paint = (
            self.paint_mask_btn.isChecked()
            and (not bool(present) or not bool(enabled))
        )
        self._mask_present = bool(present)
        self._mask_enabled = bool(enabled)
        with QSignalBlocker(self.mask_enabled_check):
            self.mask_enabled_check.setChecked(bool(enabled))
        with QSignalBlocker(self.mask_density_slider):
            self.mask_density_slider.setValue(int(density))
        self.mask_density_label.setText(f"{int(density)}%")
        if not present:
            text = "No raster mask"
        else:
            state = "enabled" if enabled else "disabled"
            text = f"Raster mask {state} · {int(density)}%"
        self.mask_state_label.setText(text)
        self._refresh_state()
        if disarm_paint:
            self.set_mask_paint_active(False)
            self.mask_paint_requested.emit(False)

    def set_selection_pending(self, pending: bool) -> None:
        """Show immediate feedback while exact selection math runs off-thread."""
        self._selection_pending = bool(pending)
        self.color_range_confirm_btn.setText(
            "Working…" if pending else "Confirm")
        self.from_selection_btn.setEnabled(
            not pending and self.from_selection_btn.isEnabled())
        self._refresh_state()

    def set_inspection_mode(self, mode: str) -> None:
        index = self.inspection_combo.findText(str(mode))
        if index < 0:
            raise ValueError("unknown inspection mode")
        self._inspection_mode = str(mode)
        with QSignalBlocker(self.inspection_combo):
            self.inspection_combo.setCurrentIndex(index)
        self._refresh_state()

    def _selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        self._refresh_state()
        if current is not None:
            index = self.selected_index()
            if index is not None:
                self.selection_changed.emit(index)

    def _row_visibility_edited(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(value, int) or not 0 <= value < len(self._layers):
            return
        checked = item.checkState() == Qt.CheckState.Checked
        if checked == self._layers[value][2]:
            return
        if item is self.layer_list.currentItem():
            with QSignalBlocker(self.visibility_check):
                self.visibility_check.setChecked(checked)
        self.visibility_changed.emit(value, checked)

    def _refresh_state(self) -> None:
        index = self.selected_index()
        has_layers = bool(self._layers)
        has_selection = index is not None
        refining = self._smart_refinement_token is not None

        self.empty_label.setVisible(not has_layers)
        self.layer_list.setVisible(has_layers)
        self.layer_list.setEnabled(not self._transform_mode and not refining)
        self.new_blank_btn.setEnabled(not self._transform_mode and not refining)
        self.place_image_btn.setEnabled(not self._transform_mode and not refining)
        self.duplicate_btn.setEnabled(has_selection and not self._transform_mode)
        self.delete_btn.setEnabled(has_selection and not self._transform_mode)
        self.up_btn.setEnabled(
            has_selection and not self._transform_mode
            and index is not None and index < len(self._layers) - 1
        )
        self.down_btn.setEnabled(
            has_selection and not self._transform_mode
            and index is not None and index > 0
        )
        self.visibility_check.setEnabled(has_selection and not self._transform_mode)
        self.name_edit.setEnabled(has_selection and not self._transform_mode)
        self.blend_combo.setEnabled(has_selection and not self._transform_mode)
        self.opacity_slider.setEnabled(has_selection and not self._transform_mode)
        can_mutate_mask = has_selection and not self._transform_mode and not refining
        self.pointer_btn.setEnabled(not self._transform_mode)
        for widget in (
            self.paint_mask_btn, self.brush_mode_combo,
            self.brush_tip_combo, self.brush_size_spin,
            self.brush_hardness_spin, self.brush_strength_spin,
            self.brush_spacing_spin, self.selection_shape_combo,
            self.selection_operation_combo, self.selection_draw_btn,
            self.selection_clear_btn, self.from_selection_btn,
            self.selection_radius_spin, self.selection_all_btn,
            self.selection_invert_btn, self.selection_grow_btn,
            self.selection_shrink_btn, self.selection_feather_btn,
            self.color_range_btn, self.color_range_tolerance_spin,
            self.color_range_softness_spin, self.color_range_confirm_btn,
            self.color_range_cancel_btn,
        ):
            widget.setEnabled(can_mutate_mask)
        can_paint_mask = (
            can_mutate_mask and self._mask_present and self._mask_enabled)
        self.paint_mask_btn.setEnabled(can_paint_mask)
        if not self._mask_present:
            self.paint_mask_btn.setToolTip(
                "Create or import a raster mask before painting.")
        elif not self._mask_enabled:
            self.paint_mask_btn.setToolTip(
                "Enable the raster mask before painting.")
        else:
            self.paint_mask_btn.setToolTip(
                "Arm the mask brush. Pointer / Done exits painting.")
        self.from_selection_btn.setEnabled(
            can_mutate_mask and not self._selection_pending)
        for widget in (
            self.reveal_mask_btn, self.hide_mask_btn,
            self.transparency_mask_btn,
        ):
            widget.setEnabled(can_mutate_mask)
        self.create_mask_btn.setEnabled(can_mutate_mask)
        self.import_mask_btn.setEnabled(can_mutate_mask)
        self.edit_mask_btn.setEnabled(can_mutate_mask and self._mask_present)
        self.export_mask_btn.setEnabled(can_mutate_mask and self._mask_present)
        self.mask_enabled_check.setEnabled(
            can_mutate_mask and self._mask_present)
        self.mask_density_slider.setEnabled(
            can_mutate_mask and self._mask_present)
        self.inspection_combo.setEnabled(
            can_mutate_mask and self._mask_present)
        if not self._mask_present:
            self.inspection_combo.setToolTip(
                "Create or import a raster mask to inspect it.")
        elif not self._mask_enabled:
            self.inspection_combo.setToolTip(
                "The mask is disabled; inspection shows the full-reveal "
                "source-alpha silhouette.")
        else:
            self.inspection_combo.setToolTip(
                "Inspect raster-mask coverage without changing the document.")
        for widget in (
            self.x_spin, self.y_spin, self.width_spin, self.height_spin,
            self.lock_aspect_check, self.center_btn, self.fit_btn,
        ):
            widget.setEnabled(has_selection)
        self.transform_btn.setEnabled(has_selection)
        self.transform_btn.setVisible(not self._transform_mode)
        self.confirm_transform_btn.setVisible(self._transform_mode)
        self.cancel_transform_btn.setVisible(self._transform_mode)
        self.export_btn.setEnabled(
            (self._document_available or has_layers) and not self._transform_mode)
        self.transform_guidance_label.setVisible(self._transform_mode)
        self.mask_confirmation.setVisible(
            self._replacement_token is not None and not self._transform_mode)
        self.smart_refinement.setVisible(refining and not self._transform_mode)
        if refining:
            for widget in (
                self.duplicate_btn, self.delete_btn, self.up_btn, self.down_btn,
                self.visibility_check, self.name_edit, self.blend_combo,
                self.opacity_slider, self.import_mask_btn, self.export_mask_btn,
                self.mask_enabled_check, self.mask_density_slider,
                self.inspection_combo, self.x_spin, self.y_spin,
                self.width_spin, self.height_spin, self.lock_aspect_check,
                self.center_btn, self.fit_btn, self.transform_btn,
                self.export_btn,
            ):
                widget.setEnabled(False)

        if index is None:
            with QSignalBlocker(self.visibility_check):
                self.visibility_check.setChecked(False)
            with QSignalBlocker(self.name_edit):
                self.name_edit.clear()
            with QSignalBlocker(self.blend_combo):
                self.blend_combo.setCurrentIndex(0)
            with QSignalBlocker(self.opacity_slider):
                self.opacity_slider.setValue(100)
            self.opacity_label.setText("100%")
            self._set_transform_controls(0, 0, 1, 1)
            return

        (
            _layer_id, name, visible, opacity, blend,
            x, y, width, height, _mask_present, _mask_enabled,
        ) = self._layers[index]

        with QSignalBlocker(self.visibility_check):
            self.visibility_check.setChecked(visible)
        with QSignalBlocker(self.name_edit):
            self.name_edit.setText(name)
        blend_index = self.blend_combo.findData(blend)
        with QSignalBlocker(self.blend_combo):
            self.blend_combo.setCurrentIndex(max(0, blend_index))
        with QSignalBlocker(self.opacity_slider):
            self.opacity_slider.setValue(opacity)
        self.opacity_label.setText(f"{opacity}%")
        self._set_transform_controls(x, y, width, height)

    def keyPressEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Escape
            and self._smart_refinement_token is not None
        ):
            self._cancel_smart_refinement()
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_transform_controls(
        self, x: int, y: int, width: int, height: int
    ) -> None:
        self._transform_syncing = True
        try:
            for spin, value in (
                (self.x_spin, x),
                (self.y_spin, y),
                (self.width_spin, width),
                (self.height_spin, height),
            ):
                with QSignalBlocker(spin):
                    spin.setValue(int(value))
            self._aspect_ratio = width / float(max(1, height))
        finally:
            self._transform_syncing = False

    def _request_duplicate(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.duplicate_requested.emit(index)

    def _request_delete(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.delete_requested.emit(index)

    def _request_up(self) -> None:
        index = self.selected_index()
        if index is not None and index < len(self._layers) - 1:
            self.move_requested.emit(index, index + 1)

    def _request_down(self) -> None:
        index = self.selected_index()
        if index is not None and index > 0:
            self.move_requested.emit(index, index - 1)

    def _visibility_edited(self, visible: bool) -> None:
        index = self.selected_index()
        if index is not None:
            self.visibility_changed.emit(index, bool(visible))

    def _name_edited(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.name_changed.emit(index, self.name_edit.text())

    def _blend_edited(self, _combo_index: int) -> None:
        index = self.selected_index()
        if index is not None:
            value = self.blend_combo.currentData()
            if isinstance(value, str):
                self.blend_changed.emit(index, value)

    def _opacity_edited(self, opacity: int) -> None:
        self.opacity_label.setText(f"{int(opacity)}%")
        index = self.selected_index()
        if index is not None:
            self.opacity_changed.emit(index, int(opacity))

    def _request_mask_generator(self, signal) -> None:
        index = self.selected_index()
        if index is not None:
            # Kept in the legacy signal payload for binary compatibility. The
            # controller deliberately routes this through its proposal boundary.
            signal.emit(index, True)

    def _mask_enabled_edited(self, enabled: bool) -> None:
        index = self.selected_index()
        if index is not None and self._mask_present:
            self.mask_enabled_changed.emit(index, bool(enabled))

    def _mask_density_edited(self, density: int) -> None:
        self.mask_density_label.setText(f"{int(density)}%")
        index = self.selected_index()
        if index is not None and self._mask_present:
            self.mask_density_changed.emit(index, int(density))

    def _transform_edited(self, _value: int) -> None:
        if self._transform_syncing:
            return
        self._transform_syncing = True
        try:
            sender = self.sender()
            if self.lock_aspect_check.isChecked():
                if sender is self.width_spin:
                    height = max(
                        1, int(round(self.width_spin.value() / self._aspect_ratio)))
                    with QSignalBlocker(self.height_spin):
                        self.height_spin.setValue(height)
                elif sender is self.height_spin:
                    width = max(
                        1, int(round(self.height_spin.value() * self._aspect_ratio)))
                    with QSignalBlocker(self.width_spin):
                        self.width_spin.setValue(width)
            else:
                self._aspect_ratio = (
                    self.width_spin.value() / float(self.height_spin.value()))
        finally:
            self._transform_syncing = False
        index = self.selected_index()
        if index is not None:
            self.transform_changed.emit(
                index,
                self.x_spin.value(),
                self.y_spin.value(),
                self.width_spin.value(),
                self.height_spin.value(),
            )

    def _request_center(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.center_requested.emit(index)

    def _request_fit(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.fit_requested.emit(index)

    def _request_transform_mode(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.transform_mode_requested.emit(index)

    def set_transform_mode(self, active: bool) -> None:
        self._transform_mode = bool(active)
        self._refresh_state()
