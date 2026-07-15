"""Preset/export menu builder. This is the only PySide6 import in Phase 6.

The pure functions in ``ditherzam.presets``, ``ditherzam.export`` and
``ditherzam.batch`` do all the real work; this module only wires QActions.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QAction, QKeySequence

# (key, menu label, keyboard shortcut or None)
MENU_SPEC: list[tuple[str, str, str | None]] = [
    ("save_preset",   "Save Preset...",   None),
    ("load_preset",   "Load Preset...",   None),
    ("import_preset", "Import Preset(s)...", None),
    ("export_preset", "Export Preset...", None),
    ("export_png",    "Export PNG...",    "Ctrl+Shift+S"),
    ("export_jpg",    "Export JPG (Transparency on White)...", None),
    ("export_svg",    "Export as Vector (SVG)...", None),
    ("batch_folder",  "Batch — Select Folder...", None),
]


def create_export_menu(menubar, handlers: dict[str, Callable]):
    """Add an "&Export" menu to ``menubar`` and connect each action to a handler.

    Actions without a handler are created disabled so the menu still lists them.
    Returns ``(menu, {key: QAction})``.
    """
    menu = menubar.addMenu("&Export")
    actions: dict[str, QAction] = {}
    for key, label, shortcut in MENU_SPEC:
        action = QAction(label, menu)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        handler = handlers.get(key)
        if handler is None:
            action.setEnabled(False)
        else:
            action.triggered.connect(lambda _checked=False, h=handler: h())
        menu.addAction(action)
        actions[key] = action
    return menu, actions
