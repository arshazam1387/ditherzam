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


def test_capped_pixmap_uses_source_logical_scene_bounds(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView

    v = CustomGraphicsView()
    v.resize(800, 600)
    v.set_pixmap(_pixmap(400, 225), logical_size=(3840, 2160))

    bounds = v._pix_item.sceneBoundingRect()
    assert bounds.width() == pytest.approx(3840.0)
    assert bounds.height() == pytest.approx(2160.0)
    assert v.sceneRect().width() == pytest.approx(3840.0)
    assert v.sceneRect().height() == pytest.approx(2160.0)


def test_ordinary_pixmap_replacement_preserves_view_transform(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView

    v = CustomGraphicsView()
    v.resize(800, 600)
    v.set_pixmap(_pixmap(400, 225), logical_size=(3840, 2160))
    v.zoom_in()
    before = v.transform()

    v.set_pixmap(_pixmap(640, 360), logical_size=(3840, 2160), refit=False)

    assert v.transform() == before
    assert v._pix_item.sceneBoundingRect().width() == pytest.approx(3840.0)


def test_source_replacement_can_refit(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView

    v = CustomGraphicsView()
    v.resize(800, 600)
    v.set_pixmap(_pixmap(400, 225), logical_size=(3840, 2160))
    v.zoom_in()
    zoomed = v.transform().m11()

    v.set_pixmap(_pixmap(300, 400), logical_size=(1200, 1600), refit=True)

    assert v.transform().m11() != pytest.approx(zoomed)
    assert v._pix_item.sceneBoundingRect().size().width() == pytest.approx(1200.0)
    assert v._pix_item.sceneBoundingRect().size().height() == pytest.approx(1600.0)


def test_viewport_device_demand_reports_drawable_pixels(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView

    v = CustomGraphicsView()
    v.resize(640, 360)
    logical_w = v.viewport().width()
    logical_h = v.viewport().height()
    dpr = v.viewport().devicePixelRatioF()

    assert v.viewport_device_demand() == (
        int(__import__("math").ceil(logical_w * dpr)),
        int(__import__("math").ceil(logical_h * dpr)),
    )
from PySide6.QtGui import QPainter, QPixmap


def test_pixmap_filtering_is_smooth_only_while_downscaled(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.resize(200, 200)
    view.show()
    qapp_fixture.processEvents()

    view.set_pixmap(QPixmap(1000, 1000), logical_size=(1000, 1000), refit=True)
    assert view.renderHints() & QPainter.RenderHint.SmoothPixmapTransform

    view.resetTransform()
    view._update_pixmap_filtering()
    assert not (view.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)
