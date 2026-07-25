"""Widget-only controls for the spatial layer stack."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QSize, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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
        self._layers: list[tuple[str, str, bool, int, str, int, int, int, int]] = []
        self._thumbnails: dict[str, QPixmap] = {}
        self._placeholder = self._make_placeholder()

        self.empty_label = QLabel(self._EMPTY_TEXT)
        self.empty_label.setWordWrap(True)
        self.empty_label.setAccessibleName("Layers empty state")

        self.layer_list = QListWidget()
        self.layer_list.setIconSize(QSize(56, 56))
        self.layer_list.setUniformItemSizes(True)
        self.layer_list.setAccessibleName(
            "Layers, topmost first, with visibility, preview, and name")

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

        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        for label, spin in (("Layer X position", self.x_spin),
                            ("Layer Y position", self.y_spin)):
            spin.setRange(-100000, 100000)
            spin.setAccessibleName(label)
        self.width_spin = QSpinBox()
        self.height_spin = QSpinBox()
        for label, spin in (("Layer width", self.width_spin),
                            ("Layer height", self.height_spin)):
            spin.setRange(1, 100000)
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
        actions = QHBoxLayout()
        actions.addWidget(self.new_blank_btn)
        actions.addWidget(self.place_image_btn)
        actions.addWidget(self.duplicate_btn)
        actions.addWidget(self.delete_btn)

        ordering = QHBoxLayout()
        ordering.addWidget(self.up_btn)
        ordering.addWidget(self.down_btn)
        ordering.addStretch(1)

        form = QFormLayout()
        form.addRow(self.visibility_check)
        name_label = QLabel("Name")
        name_label.setBuddy(self.name_edit)
        form.addRow(name_label, self.name_edit)
        blend_label = QLabel("Blend")
        blend_label.setBuddy(self.blend_combo)
        form.addRow(blend_label, self.blend_combo)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_label)
        opacity_label = QLabel("Opacity")
        opacity_label.setBuddy(self.opacity_slider)
        form.addRow(opacity_label, opacity_row)

        position_row = QHBoxLayout()
        position_row.addWidget(QLabel("X"))
        position_row.addWidget(self.x_spin)
        position_row.addWidget(QLabel("Y"))
        position_row.addWidget(self.y_spin)
        form.addRow("Position", position_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("W"))
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("H"))
        size_row.addWidget(self.height_spin)
        form.addRow("Size", size_row)

        transform_actions = QHBoxLayout()
        transform_actions.addWidget(self.lock_aspect_check)
        transform_actions.addStretch(1)
        transform_actions.addWidget(self.center_btn)
        transform_actions.addWidget(self.fit_btn)
        form.addRow(transform_actions)

        canvas_transform_actions = QHBoxLayout()
        canvas_transform_actions.addWidget(self.transform_btn)
        canvas_transform_actions.addWidget(self.confirm_transform_btn)
        canvas_transform_actions.addWidget(self.cancel_transform_btn)
        form.addRow("Canvas", canvas_transform_actions)

        finishing = QHBoxLayout()
        finishing.addWidget(self.export_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self.empty_label)
        root.addLayout(actions)
        root.addWidget(self.layer_list, 1)
        root.addLayout(ordering)
        root.addLayout(form)
        root.addLayout(finishing)
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
        QWidget.setTabOrder(self.opacity_slider, self.x_spin)
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

    def set_layers(
        self,
        layers: Iterable[tuple[str, str, bool, int, str]],
        selected: int | None,
    ) -> None:
        """Replace displayed state without emitting user-edit signals."""
        clean = []
        for raw in layers:
            values = tuple(raw)
            if len(values) == 5:
                values = (*values, 0, 0, 1, 1)
            if len(values) != 9:
                raise ValueError("layer rows must have five or nine values")
            layer_id, name, visible, opacity, blend, x, y, width, height = values
            clean.append((
                str(layer_id), str(name), bool(visible), int(opacity), str(blend),
                int(x), int(y), max(1, int(width)), max(1, int(height)),
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
            selected_item: QListWidgetItem | None = None
            for core_index in range(len(clean) - 1, -1, -1):
                (
                    layer_id, name, visible, _opacity, _blend,
                    _x, _y, _width, _height,
                ) = clean[core_index]
                item = QListWidgetItem(name)
                item.setData(Qt.ItemDataRole.UserRole, core_index)
                item.setData(int(Qt.ItemDataRole.UserRole) + 1, layer_id)
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if visible
                    else Qt.CheckState.Unchecked)
                item.setIcon(QIcon(
                    self._thumbnails.get(layer_id, self._placeholder)))
                state_text = "visible" if visible else "hidden"
                accessible_text = f"{name}, {state_text}"
                item.setToolTip(accessible_text)
                item.setData(
                    Qt.ItemDataRole.AccessibleTextRole, accessible_text)
                self.layer_list.addItem(item)
                if selected is not None and core_index == int(selected):
                    selected_item = item
            self.layer_list.setCurrentItem(selected_item)
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
                break

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

        self.empty_label.setVisible(not has_layers)
        self.layer_list.setVisible(has_layers)
        self.layer_list.setEnabled(not self._transform_mode)
        self.new_blank_btn.setEnabled(not self._transform_mode)
        self.place_image_btn.setEnabled(not self._transform_mode)
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
            x, y, width, height,
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
