from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QComboBox, QStyle, QStyledItemDelegate

from .settings_map import build_dither_rows

HEADER_ROLE = Qt.ItemDataRole.UserRole + 1


def populate_dither_combo(combo: QComboBox, by_category) -> None:
    """Fill a combo with grouped rows; category headers are bold + non-selectable."""
    model = QStandardItemModel(combo)
    for is_header, label, style in build_dither_rows(by_category):
        item = QStandardItem(label)
        if is_header:
            item.setData(True, HEADER_ROLE)
            item.setData(None, Qt.ItemDataRole.UserRole)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)          # enabled, not selectable
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        else:
            item.setData(False, HEADER_ROLE)
            item.setData(style, Qt.ItemDataRole.UserRole)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        model.appendRow(item)
    combo.setModel(model)
    combo.setItemDelegate(DitherStyleDelegate(combo))


class DitherStyleDelegate(QStyledItemDelegate):
    """Renders category header rows in bold and without the selection highlight."""

    def paint(self, painter, option, index):
        if index.data(HEADER_ROLE):
            option.font.setBold(True)
            option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return size
