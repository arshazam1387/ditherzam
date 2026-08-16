from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QPlainTextEdit, QVBoxLayout,
)

from ditherzam.diagnostics import build_diagnostic_report, clear_logs


class DiagnosticsDialog(QDialog):
    def __init__(self, log_path: str | Path, parent=None):
        super().__init__(parent)
        self.log_path = Path(log_path)
        self.setWindowTitle("Program Notes")
        self.resize(820, 560)
        label = QLabel(
            "Local crash and load notes. Attach this report when describing a freeze."
        )
        label.setWordWrap(True)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        copy = QPushButton("Copy Report")
        copy.clicked.connect(self.copy_report)
        folder = QPushButton("Open Log Folder")
        folder.clicked.connect(self.open_folder)
        clear = QPushButton("Clear Logs")
        clear.clicked.connect(self.clear)
        buttons = QHBoxLayout()
        for button in (refresh, copy, folder, clear):
            buttons.addWidget(button)
        buttons.addStretch(1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.report, 1)
        layout.addLayout(buttons)
        layout.addWidget(close)
        self.refresh()

    def refresh(self) -> None:
        self.report.setPlainText(build_diagnostic_report(self.log_path))
        cursor = self.report.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.report.setTextCursor(cursor)

    def copy_report(self) -> None:
        QGuiApplication.clipboard().setText(self.report.toPlainText())

    def open_folder(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_path.parent)))

    def clear(self) -> None:
        answer = QMessageBox.question(
            self, "Clear Program Notes",
            "Delete the current and rotated diagnostic logs?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            clear_logs(self.log_path)
            self.refresh()
