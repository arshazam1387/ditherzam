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
                     "export_png", "export_jpg", "export_svg", "export_preview",
                     "batch_folder"):
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


def test_export_preview_saves_displayed_qimage_bytes(qapp_fixture, tmp_path, monkeypatch):
    # Export Preview must write the preview raster exactly as displayed —
    # never re-render through the export pipeline.
    import numpy as np
    from PySide6.QtWidgets import QFileDialog
    from ditherzam.ui.main_window import ImageEditor

    window = ImageEditor()
    gray = np.linspace(0, 255, 12, dtype=np.float32).reshape(3, 4)
    rgba = np.zeros((3, 4, 4), np.uint8)
    rgba[..., 3] = 255
    window.load_array(gray, rgba[..., :3].copy(), rgba)
    window.render_now()
    assert window.last_qimage is not None

    out = tmp_path / "preview.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "PNG Files (*.png)")))
    exploded = lambda *a, **k: (_ for _ in ()).throw(AssertionError("export pipeline used"))
    monkeypatch.setattr(window, "_rendered_rgb", exploded)
    monkeypatch.setattr(window, "_export_pipeline", exploded)
    window._on_export_preview()
    assert out.exists()

    from PySide6.QtGui import QImage
    saved = QImage(str(out))
    shown = window.last_qimage
    assert (saved.width(), saved.height()) == (shown.width(), shown.height())
    probe = [(0, 0), (shown.width() - 1, shown.height() - 1)]
    for x, y in probe:
        assert saved.pixel(x, y) == shown.pixel(x, y)


def test_export_preview_without_image_is_a_safe_noop(qapp_fixture, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from ditherzam.ui.main_window import ImageEditor

    window = ImageEditor()
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("dialog opened"))))
    window._on_export_preview()  # no image loaded: must not open a dialog or raise
