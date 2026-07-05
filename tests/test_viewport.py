import numpy as np
import pytest

pytest.importorskip("PySide6")


def _pixmap(w, h):
    from PySide6.QtGui import QPixmap
    from ditherzam.ui.convert import numpy_to_qimage
    arr = np.random.RandomState(0).randint(0, 256, (h, w, 3), np.uint8)
    return QPixmap.fromImage(numpy_to_qimage(arr))


def test_set_pixmap_and_zoom(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView
    v = CustomGraphicsView()
    v.resize(200, 200)
    v.set_pixmap(_pixmap(64, 64))
    before = v.transform().m11()
    v.zoom_in()
    assert v.transform().m11() > before          # zoomed in


def test_zoom_percent_signal(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView
    v = CustomGraphicsView()
    v.resize(200, 200)
    seen = []
    v.zoom_changed.connect(seen.append)
    v.set_pixmap(_pixmap(64, 64))
    assert seen and isinstance(seen[-1], int)


def test_zoom_hard_cap_no_crash(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView
    v = CustomGraphicsView()
    v.resize(200, 200)
    v.set_pixmap(_pixmap(64, 64))
    for _ in range(200):                          # spam past the 100x cap
        v.zoom_in()
    assert v.transform().m11() <= 100.0
