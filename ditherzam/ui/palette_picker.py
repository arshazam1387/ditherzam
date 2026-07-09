from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

_NAME_ROLE = Qt.ItemDataRole.UserRole


def _swatch_icon(colors: np.ndarray, w: int = 112, h: int = 22) -> QIcon:
    pix = QPixmap(w, h)
    pix.fill(QColor(0, 0, 0, 0))
    n = max(1, int(colors.shape[0]))
    painter = QPainter(pix)
    cell = w / n
    for i in range(n):
        r, g, b = (int(round(c)) for c in colors[i])
        painter.fillRect(int(i * cell), 0, int(cell) + 1, h, QColor(r, g, b))
    painter.end()
    return QIcon(pix)


class PalettePicker(QTreeWidget):
    """Category-grouped palette tree. Hover/scroll preview, click to commit."""

    selected = Signal(str)      # committed palette name
    preview = Signal(object)    # a Palette to preview, or None to revert

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setMouseTracking(True)
        # Bigger swatches + a taller list so more palettes are visible at once.
        self.setIconSize(QSize(112, 22))
        self.setMinimumHeight(260)
        self._preview_enabled = True
        self._wheel_cycle = False
        self._store = None
        self.itemClicked.connect(self._on_item_clicked)
        self.itemActivated.connect(self._on_item_clicked)   # Enter / dbl-click
        self.currentItemChanged.connect(self._on_current_changed)
        self.itemEntered.connect(self._on_item_entered)

    # -- config ---------------------------------------------------------------
    def set_preview_enabled(self, on: bool) -> None:
        self._preview_enabled = bool(on)

    def set_wheel_cycle(self, on: bool) -> None:
        self._wheel_cycle = bool(on)

    # -- population -----------------------------------------------------------
    def populate(self, store) -> None:
        self._store = store
        self.blockSignals(True)
        self.clear()
        for category, names in store.list_by_category().items():
            header = QTreeWidgetItem([category])
            header.setFlags(Qt.ItemFlag.ItemIsEnabled)     # not selectable
            header.setData(0, _NAME_ROLE, None)
            self.addTopLevelItem(header)
            for name in names:
                child = QTreeWidgetItem([name])
                child.setData(0, _NAME_ROLE, name)
                child.setIcon(0, _swatch_icon(store.get(name).colors))
                header.addChild(child)
            header.setExpanded(True)
        self.blockSignals(False)

    def select(self, name: str) -> None:
        item = self._find_item(name)
        if item is not None:
            self.blockSignals(True)
            self.setCurrentItem(item)
            self.blockSignals(False)

    # -- helpers --------------------------------------------------------------
    def _palette_items(self) -> list[QTreeWidgetItem]:
        out = []
        for i in range(self.topLevelItemCount()):
            top = self.topLevelItem(i)
            for j in range(top.childCount()):
                out.append(top.child(j))
        return out

    def _find_item(self, name: str):
        for it in self._palette_items():
            if it.data(0, _NAME_ROLE) == name:
                return it
        return None

    @staticmethod
    def _name_of(item):
        return None if item is None else item.data(0, _NAME_ROLE)

    def _emit_preview(self, name) -> None:
        if not self._preview_enabled or self._store is None:
            return
        self.preview.emit(self._store.get(name) if name is not None else None)

    # -- signal handlers ------------------------------------------------------
    def _on_item_clicked(self, item, _col) -> None:
        name = self._name_of(item)
        if name is not None:
            self.selected.emit(name)

    def _on_current_changed(self, current, _prev) -> None:
        self._emit_preview(self._name_of(current))

    def _on_item_entered(self, item, _col) -> None:
        self._emit_preview(self._name_of(item))

    # -- events ---------------------------------------------------------------
    def leaveEvent(self, event):
        if self._preview_enabled:
            self.preview.emit(None)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if not self._wheel_cycle:
            super().wheelEvent(event)
            return
        self._cycle(-1 if event.angleDelta().y() > 0 else 1)
        event.accept()

    def _cycle(self, step: int) -> None:
        items = self._palette_items()
        if not items:
            return
        cur = self.currentItem()
        idx = items.index(cur) if cur in items else -1
        idx = max(0, min(len(items) - 1, idx + step))
        self.setCurrentItem(items[idx])   # fires _on_current_changed -> preview
