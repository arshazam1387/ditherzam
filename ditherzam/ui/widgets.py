from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsDropShadowEffect,
    QLabel,
    QSlider,
    QSpinBox,
)


class GlowSlider(QSlider):
    """Horizontal slider with a colored glow and no accidental scroll-wheel edits."""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None,
                 glow_color="#5e89ed"):
        super().__init__(orientation, parent)
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setBlurRadius(15)
        self._glow.setOffset(0, 0)
        self._glow.setColor(QColor(glow_color))
        self.setGraphicsEffect(self._glow)

    def set_glow_color(self, color: str) -> None:
        self._glow.setColor(QColor(color))

    def wheelEvent(self, event):  # swallow — never change value on scroll
        event.accept()


class ResettableGlowSlider(GlowSlider):
    """GlowSlider that remembers a default and restores it on double-click."""

    def __init__(self, default: int = 0, orientation=Qt.Orientation.Horizontal,
                 parent=None, glow_color="#5e89ed"):
        super().__init__(orientation, parent, glow_color)
        self._default = int(default)
        self.setValue(self._default)

    @property
    def default(self) -> int:
        return self._default

    def reset(self) -> None:
        self.setValue(self._default)

    def mouseDoubleClickEvent(self, event):
        self.reset()
        event.accept()


class InvisibleSpinBox(QSpinBox):
    """0..100 spinbox with hidden arrows that displays a friendly scaled number."""

    def __init__(self, max_display: int = 100, parent=None):
        super().__init__(parent)
        self._max_display = int(max_display)
        self.setRange(0, 100)
        self.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

    def textFromValue(self, value: int) -> str:
        return str(round(value / 100.0 * self._max_display))

    def wheelEvent(self, event):  # swallow
        event.accept()


class ClickableLabel(QLabel):
    """Bold label that emits ``clicked`` on left press (used for the shuffle icon)."""

    clicked = Signal()

    def __init__(self, text: str = "", font_size: int = 11, parent=None):
        super().__init__(text, parent)
        f = self.font()
        f.setPointSize(int(font_size))
        self.setFont(f)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ResettableLabel(ClickableLabel):
    """ClickableLabel that also emits ``double_clicked`` (reset-to-default affordance)."""

    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class NoScrollComboBox(QComboBox):
    """Combo box that ignores the scroll wheel so hovering never changes selection."""

    def wheelEvent(self, event):  # swallow
        event.accept()
