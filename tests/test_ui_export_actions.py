import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMainWindow
from ditherzam.ui.export_actions import create_export_menu, MENU_SPEC

_app = QApplication.instance() or QApplication([])


def test_menu_spec_has_all_actions():
    keys = [k for k, _label, _sc in MENU_SPEC]
    for expected in ("save_preset", "load_preset", "import_preset", "export_preset",
                     "export_png", "export_jpg", "export_svg", "batch_folder"):
        assert expected in keys


def test_create_menu_builds_actions_with_labels():
    win = QMainWindow()
    calls = []
    handlers = {k: (lambda key=k: calls.append(key)) for k, _l, _s in MENU_SPEC}
    menu, actions = create_export_menu(win.menuBar(), handlers)
    assert menu.title() == "&Export"
    assert set(actions.keys()) == {k for k, _l, _s in MENU_SPEC}
    for key, label, _sc in MENU_SPEC:
        assert actions[key].text() == label


def test_triggering_action_calls_handler():
    win = QMainWindow()
    calls = []
    handlers = {k: (lambda key=k: calls.append(key)) for k, _l, _s in MENU_SPEC}
    _menu, actions = create_export_menu(win.menuBar(), handlers)
    actions["export_svg"].trigger()
    actions["save_preset"].trigger()
    assert calls == ["export_svg", "save_preset"]


def test_missing_handler_action_is_disabled():
    win = QMainWindow()
    _menu, actions = create_export_menu(win.menuBar(), {"export_png": lambda: None})
    assert actions["export_png"].isEnabled() is True
    assert actions["export_svg"].isEnabled() is False
