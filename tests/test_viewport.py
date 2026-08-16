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


def test_left_drag_inside_active_layer_emits_document_delta(qapp_fixture):
    from PySide6.QtCore import QPoint, QRectF, Qt
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(240, 240)
    view.show()
    view.set_pixmap(_pixmap(100, 100), logical_size=(100, 100))
    view.set_layer_drag_target(QRectF(10, 10, 30, 20))
    qapp_fixture.processEvents()
    start = view.mapFromScene(20, 20)
    end = view.mapFromScene(27, 25)
    deltas = []
    view.layer_dragged.connect(lambda x, y: deltas.append((x, y)))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert deltas
    assert deltas[-1][0] == pytest.approx(7.0, abs=0.75)
    assert deltas[-1][1] == pytest.approx(5.0, abs=0.75)


def test_middle_drag_pans_when_active_layer_covers_canvas(qapp_fixture):
    from PySide6.QtCore import QPoint, QRectF, Qt
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(180, 180)
    view.show()
    view.set_pixmap(_pixmap(400, 400), logical_size=(400, 400))
    view.set_layer_drag_target(QRectF(0, 0, 400, 400))
    qapp_fixture.processEvents()
    view.horizontalScrollBar().setValue(100)
    start = view.viewport().rect().center()
    end = start + QPoint(20, 0)
    QTest.mousePress(view.viewport(), Qt.MouseButton.MiddleButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.MiddleButton, pos=end)
    assert view.horizontalScrollBar().value() < 100


def test_corner_drag_emits_document_resize_delta(qapp_fixture):
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(240, 240)
    view.show()
    view.set_pixmap(_pixmap(100, 100), logical_size=(100, 100))
    view.set_layer_drag_target(QRectF(20, 20, 40, 30))
    qapp_fixture.processEvents()
    start = view.mapFromScene(60, 50)
    end = view.mapFromScene(68, 56)
    resized = []
    view.layer_resized.connect(
        lambda corner, dx, dy: resized.append((corner, dx, dy)))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert resized
    assert resized[-1][0] == "se"
    assert resized[-1][1] == pytest.approx(8.0, abs=0.75)
    assert resized[-1][2] == pytest.approx(6.0, abs=0.75)


def _transform_view(qapp_fixture):
    from PySide6.QtCore import QRectF
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(320, 260)
    view.show()
    view.set_pixmap(_pixmap(100, 100), logical_size=(100, 100))
    view.set_layer_drag_target(QRectF(20, 25, 60, 50))
    qapp_fixture.processEvents()
    return view


def test_transform_exposes_eight_device_stable_handle_targets(qapp_fixture):
    view = _transform_view(qapp_fixture)

    handles = view._layer_handle_rects()

    assert set(handles) == {"nw", "n", "ne", "e", "se", "s", "sw", "w"}
    dpr = view.viewport().devicePixelRatioF()
    for handle in handles.values():
        screen_rect = view.mapFromScene(handle).boundingRect()
        assert screen_rect.width() * dpr >= 16
        assert screen_rect.height() * dpr >= 16

    view.scale(3.0, 3.0)
    zoomed = view._layer_handle_rects()
    for handle in zoomed.values():
        screen_rect = view.mapFromScene(handle).boundingRect()
        assert screen_rect.width() * dpr >= 16
        assert screen_rect.height() * dpr >= 16


def test_visible_transform_handles_scale_with_selected_image(qapp_fixture):
    from PySide6.QtCore import QRectF

    view = _transform_view(qapp_fixture)
    view.set_layer_drag_target(QRectF(20, 20, 20, 20))
    small = view._layer_visible_handle_rects()["nw"].width()

    view.set_layer_drag_target(QRectF(10, 10, 80, 80))
    large = view._layer_visible_handle_rects()["nw"].width()

    assert large > small
    assert large == pytest.approx(small * 4.0)


def test_visible_transform_handle_is_fully_inside_click_target_when_zoomed(
    qapp_fixture,
):
    from PySide6.QtCore import QRectF

    view = _transform_view(qapp_fixture)
    view.set_layer_drag_target(QRectF(10, 10, 80, 80))
    view.scale(3.0, 3.0)

    visible = view._layer_visible_handle_rects()
    targets = view._layer_handle_rects()

    for name, visible_rect in visible.items():
        assert targets[name].contains(visible_rect)


@pytest.mark.parametrize(
    ("handle", "cursor"),
    [
        ("nw", "SizeFDiagCursor"),
        ("n", "SizeVerCursor"),
        ("ne", "SizeBDiagCursor"),
        ("e", "SizeHorCursor"),
        ("se", "SizeFDiagCursor"),
        ("s", "SizeVerCursor"),
        ("sw", "SizeBDiagCursor"),
        ("w", "SizeHorCursor"),
    ],
)
def test_transform_handle_cursor_changes_before_press(
    qapp_fixture, handle, cursor
):
    from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    view = _transform_view(qapp_fixture)
    try:
        point = view.mapFromScene(view._layer_handle_rects()[handle].center())
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(point),
            QPointF(view.viewport().mapToGlobal(point)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        assert QCoreApplication.sendEvent(view.viewport(), event)
        qapp_fixture.processEvents()

        assert view.cursor().shape() == getattr(Qt.CursorShape, cursor)
    finally:
        view.close()
        view.deleteLater()
        qapp_fixture.processEvents()


def test_transform_selection_uses_contrast_halo_pens(qapp_fixture):
    view = _transform_view(qapp_fixture)

    pens = view._selection_outline_pens()

    assert [pen.color().name() for pen in pens] == [
        "#000000",
        "#ffffff",
        "#2f80ff",
    ]
    assert [pen.width() for pen in pens] == [6, 4, 2]
    assert all(pen.isCosmetic() for pen in pens)


def test_canvas_tools_are_mutually_exclusive_and_cursor_tracks_enter_leave(
    qapp_fixture,
):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QEnterEvent
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.show()
    view.set_mask_brush_mode(True)
    assert view._mask_brush_mode
    assert view._selection_tool is None
    assert view._gradient_tool is None
    assert view.cursor().shape() == Qt.CursorShape.BlankCursor

    qapp_fixture.sendEvent(view.viewport(), QEvent(QEvent.Type.Leave))
    assert view.cursor().shape() == Qt.CursorShape.ArrowCursor
    qapp_fixture.sendEvent(
        view.viewport(), QEnterEvent(QPointF(), QPointF(), QPointF()))
    assert view.cursor().shape() == Qt.CursorShape.BlankCursor

    view.set_selection_tool("rectangle")
    assert not view._mask_brush_mode
    assert view._gradient_tool is None
    assert view.cursor().shape() == Qt.CursorShape.CrossCursor

    view.set_gradient_tool("linear")
    assert view._selection_tool is None
    assert not view._mask_brush_mode
    assert view._gradient_tool == "linear"
    view.close()


def test_rectangle_selection_disarms_before_publishing_completion(qapp_fixture):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(200, 200)
    view.set_pixmap(QPixmap(100, 100), logical_size=(100, 100))
    view.show()
    seen = []
    view.selection_dragged.connect(
        lambda *_args: seen.append(
            (view._selection_tool, view.cursor().shape())))
    view.set_selection_tool("rectangle")

    QTest.mousePress(
        view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    QTest.mouseRelease(
        view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(120, 120))

    assert seen == [(None, Qt.CursorShape.ArrowCursor)]
    view.close()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("Key_Left", (-1, 0)),
        ("Key_Right", (1, 0)),
        ("Key_Up", (0, -1)),
        ("Key_Down", (0, 1)),
    ],
)
def test_transform_arrow_keys_request_one_pixel_nudge(
    qapp_fixture, key, expected
):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    view = _transform_view(qapp_fixture)
    nudges = []
    view.layer_nudge_requested.connect(lambda dx, dy: nudges.append((dx, dy)))

    QTest.keyClick(view.viewport(), getattr(Qt.Key, key))

    assert nudges == [expected]


def test_transform_shift_arrow_requests_ten_pixel_nudge(qapp_fixture):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    view = _transform_view(qapp_fixture)
    nudges = []
    view.layer_nudge_requested.connect(lambda dx, dy: nudges.append((dx, dy)))

    QTest.keyClick(
        view.viewport(),
        Qt.Key.Key_Right,
        Qt.KeyboardModifier.ShiftModifier,
    )

    assert nudges == [(10, 0)]


@pytest.mark.parametrize("key", ["Key_Return", "Key_Enter"])
def test_transform_enter_keys_request_confirm(qapp_fixture, key):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    view = _transform_view(qapp_fixture)
    confirmations = []
    view.layer_transform_confirm_requested.connect(
        lambda: confirmations.append(True)
    )

    QTest.keyClick(view.viewport(), getattr(Qt.Key, key))

    assert confirmations == [True]


def test_transform_escape_requests_cancel(qapp_fixture):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    view = _transform_view(qapp_fixture)
    cancellations = []
    view.layer_transform_cancel_requested.connect(
        lambda: cancellations.append(True)
    )

    QTest.keyClick(view.viewport(), Qt.Key.Key_Escape)

    assert cancellations == [True]


def test_edge_handle_drag_emits_direction_and_document_delta(qapp_fixture):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    view = _transform_view(qapp_fixture)
    start = view.mapFromScene(view._layer_handle_rects()["e"].center())
    end = view.mapFromScene(
        view._layer_handle_rects()["e"].center().x() + 7,
        view._layer_handle_rects()["e"].center().y(),
    )
    resized = []
    view.layer_resized.connect(
        lambda handle, dx, dy: resized.append((handle, dx, dy))
    )

    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert resized
    assert resized[-1][0] == "e"
    assert resized[-1][1] == pytest.approx(7.0, abs=0.75)
    assert resized[-1][2] == pytest.approx(0.0, abs=0.75)


def test_transform_shortcuts_are_inactive_without_target(qapp_fixture):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(200, 200)
    view.show()
    nudges = []
    confirmations = []
    view.layer_nudge_requested.connect(lambda dx, dy: nudges.append((dx, dy)))
    view.layer_transform_confirm_requested.connect(
        lambda: confirmations.append(True)
    )

    QTest.keyClick(view.viewport(), Qt.Key.Key_Left)
    QTest.keyClick(view.viewport(), Qt.Key.Key_Return)

    assert nudges == []
    assert confirmations == []


def test_transform_shortcuts_do_not_hijack_active_numeric_editor(qapp_fixture):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QSpinBox

    view = _transform_view(qapp_fixture)
    editor = QSpinBox()
    editor.show()
    editor.setFocus()
    qapp_fixture.processEvents()
    nudges = []
    view.layer_nudge_requested.connect(lambda dx, dy: nudges.append((dx, dy)))

    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Right,
        Qt.KeyboardModifier.NoModifier,
    )
    view.keyPressEvent(event)

    assert nudges == []


def test_transform_proxy_installs_without_replacing_authoritative_pixmap(
    qapp_fixture,
):
    from PySide6.QtCore import QRectF
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    authoritative = _pixmap(100, 80)
    background = _pixmap(50, 40)
    layer = _pixmap(20, 10)
    view.set_pixmap(authoritative, logical_size=(100, 80))

    view.install_transform_proxy(
        background,
        layer,
        QRectF(10, 15, 40, 20),
    )

    assert view._pix_item.pixmap().cacheKey() == authoritative.cacheKey()
    assert view._transform_proxy_root is not None
    assert view._transform_proxy_background.sceneBoundingRect() == QRectF(
        0, 0, 100, 80
    )
    assert view._transform_proxy_layer.sceneBoundingRect() == QRectF(
        10, 15, 40, 20
    )
    assert view.sceneRect() == QRectF(0, 0, 100, 80)


def test_transform_proxy_geometry_updates_synchronously_and_tracks_handles(
    qapp_fixture,
):
    from PySide6.QtCore import QRectF
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.set_pixmap(_pixmap(100, 80), logical_size=(100, 80))
    view.set_layer_drag_target(QRectF(10, 15, 40, 20))
    view.install_transform_proxy(
        _pixmap(100, 80),
        _pixmap(20, 10),
        QRectF(10, 15, 40, 20),
    )

    updated = QRectF(-10, 25, 60, 30)
    view.update_transform_proxy_geometry(updated)

    assert view._transform_proxy_layer.sceneBoundingRect() == updated
    assert view._layer_drag_rect == updated
    assert view.sceneRect() == QRectF(0, 0, 100, 80)


def test_transform_proxy_is_clipped_to_document_canvas(qapp_fixture):
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QGraphicsItem
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.set_pixmap(_pixmap(100, 80), logical_size=(100, 80))
    view.install_transform_proxy(
        _pixmap(100, 80),
        _pixmap(20, 10),
        QRectF(-10, -5, 40, 20),
    )

    assert (
        view._transform_proxy_root.flags()
        & QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape
    )
    assert view._transform_proxy_root.rect() == QRectF(0, 0, 100, 80)


def test_clear_transform_proxy_restores_base_and_preserves_view_state(
    qapp_fixture,
):
    from PySide6.QtCore import QRectF
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.resize(300, 240)
    view.set_pixmap(_pixmap(100, 80), logical_size=(100, 80))
    view.zoom_in()
    before_transform = view.transform()
    before_scene = view.sceneRect()
    view.install_transform_proxy(
        _pixmap(100, 80),
        _pixmap(20, 10),
        QRectF(10, 15, 40, 20),
    )

    view.clear_transform_proxy()

    assert view._transform_proxy_root is None
    assert view._transform_proxy_background is None
    assert view._transform_proxy_layer is None
    assert view._pix_item.isVisible()
    assert view.transform() == before_transform
    assert view.sceneRect() == before_scene
