from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal, QMimeData
from PySide6.QtGui import QColor, QDrag
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from ..color.palette import Palette

_MIN_SWATCHES = 1


class _SwatchButton(QPushButton):
    """Swatch button that starts a drag (carrying its index) on press-drag,
    while still behaving as a normal click target for short press/release."""

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self._press_pos: QPoint | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._press_pos is not None
            and bool(event.buttons() & Qt.MouseButton.LeftButton)
            and (event.position().toPoint() - self._press_pos).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._press_pos = None
            self.setDown(False)
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(str(self.index))
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.MoveAction)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        super().mouseReleaseEvent(event)


class SwatchStrip(QWidget):
    """Editable row of palette swatches over an in-memory working Palette."""

    edited = Signal(object)   # emits the working Palette

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette = Palette.from_list("empty", [[0, 0, 0]])
        self._locked: set[int] = set()
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(2)
        self._buttons: list[QPushButton] = []
        self._rebuild()
        self.setAcceptDrops(True)

    # -- public API -----------------------------------------------------------
    def set_palette(self, palette: Palette) -> None:
        self._palette = Palette(
            name=palette.name, colors=palette.colors.copy(),
            category=getattr(palette, "category", ""),
        )
        self._locked = set()
        self._rebuild()

    def palette(self) -> Palette:
        return self._palette

    def locked(self) -> set[int]:
        return set(self._locked)

    def set_swatch_color(self, i: int, rgb) -> None:
        self._palette.colors[i] = np.asarray(rgb, dtype=np.float32)
        self._rebuild()
        self.edited.emit(self._palette)

    def add_swatch(self) -> None:
        last = self._palette.colors[-1:].copy()
        self._palette = Palette(
            name=self._palette.name,
            colors=np.vstack([self._palette.colors, last]).astype(np.float32),
        )
        self._rebuild()
        self.edited.emit(self._palette)

    def remove_swatch(self, i: int) -> None:
        if self._palette.colors.shape[0] <= _MIN_SWATCHES:
            return
        self._palette = Palette(
            name=self._palette.name,
            colors=np.delete(self._palette.colors, i, axis=0).astype(np.float32),
        )
        self._locked = {j - 1 if j > i else j for j in self._locked if j != i}
        self._rebuild()
        self.edited.emit(self._palette)

    def move_swatch(self, src_i: int, dst_i: int) -> None:
        n = self._palette.colors.shape[0]
        if not (0 <= src_i < n and 0 <= dst_i < n) or src_i == dst_i:
            return
        order = list(range(n))
        order.insert(dst_i, order.pop(src_i))
        new_colors = self._palette.colors[order].astype(np.float32)
        remap = {old: new for new, old in enumerate(order)}
        self._locked = {remap[i] for i in self._locked}
        self._palette = Palette(
            name=self._palette.name, colors=new_colors,
            category=self._palette.category,
        )
        self._rebuild()
        self.edited.emit(self._palette)

    def toggle_lock(self, i: int) -> None:
        if i in self._locked:
            self._locked.discard(i)
        else:
            self._locked.add(i)
        self._rebuild()

    def shuffle(self, rng=None) -> None:
        rng = rng if rng is not None else np.random.default_rng()
        self._palette = self._palette.shuffle(self._locked, rng)
        self._rebuild()
        self.edited.emit(self._palette)

    # -- rendering ------------------------------------------------------------
    def _rebuild(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._buttons = []
        for i in range(self._palette.colors.shape[0]):
            r, g, b = (int(round(c)) for c in self._palette.colors[i])
            btn = _SwatchButton(i)
            btn.setFixedSize(22, 22)
            border = "2px solid #f0d000" if i in self._locked else "1px solid #333"
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: {border};")
            btn.clicked.connect(lambda _=False, idx=i: self._pick_color(idx))
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda _pos, idx=i: self.remove_swatch(idx))
            self._row.addWidget(btn)
            self._buttons.append(btn)
        add = QPushButton("+")
        add.setFixedSize(22, 22)
        add.clicked.connect(lambda _=False: self.add_swatch())
        self._row.addWidget(add)

    def _pick_color(self, i: int) -> None:
        c = self._palette.colors[i]
        initial = QColor(int(c[0]), int(c[1]), int(c[2]))
        chosen = QColorDialog.getColor(initial, self, "Pick swatch color")
        if chosen.isValid():
            self.set_swatch_color(i, (chosen.red(), chosen.green(), chosen.blue()))

    # -- drag-drop reordering --------------------------------------------------
    def _target_index(self, x: float) -> int:
        for i, b in enumerate(self._buttons):
            if x < b.x() + b.width() / 2:
                return i
        return len(self._buttons) - 1

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            src = int(event.mimeData().text())
        except (TypeError, ValueError):
            return
        dst = self._target_index(event.position().x())
        self.move_swatch(src, dst)
        event.acceptProposedAction()
