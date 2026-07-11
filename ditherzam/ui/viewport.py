from __future__ import annotations

import time

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .viewport_math import (
    clamp_velocity,
    inertia_step,
    next_zoom,
    viewport_device_size,
    zoom_percent,
)


class CustomGraphicsView(QGraphicsView):
    """Image viewport: Shift+wheel zoom, drag-pan with fling inertia, drag-drop import."""

    image_dropped = Signal(str)
    zoom_changed = Signal(int)

    def __init__(self, bg_color="#1f1f1f", friction=0.95, enable_inertia=True,
                 velocity_scale=0.5, max_velocity=2000.0, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self.setBackgroundBrush(QColor(bg_color))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAcceptDrops(True)

        self._friction = friction
        self._enable_inertia = enable_inertia
        self._velocity_scale = velocity_scale
        self._max_velocity = max_velocity
        self._velocity = QPointF(0.0, 0.0)
        self._last_pos: QPointF | None = None
        self._last_time: float | None = None
        self._panning = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_inertia_tick)

    # ---- image / zoom -------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap, logical_size=None, refit=True) -> None:
        """Display *pixmap* in source-logical scene coordinates.

        ``logical_size`` is the source ``(width, height)`` represented by a
        potentially capped raster.  Set ``refit=False`` for an ordinary
        preview-quality replacement so the user's view transform is retained.
        """
        self._pix_item.setPixmap(pixmap)
        bounds = self._pix_item.boundingRect()
        if logical_size is None:
            logical_width, logical_height = bounds.width(), bounds.height()
        elif hasattr(logical_size, "width"):
            logical_width = logical_size.width()
            logical_height = logical_size.height()
        else:
            logical_width, logical_height = logical_size
        if (not pixmap.isNull()
                and (logical_width <= 0 or logical_height <= 0)):
            raise ValueError("logical source dimensions must be positive")

        if pixmap.isNull():
            self._pix_item.setTransform(QTransform())
        else:
            self._pix_item.setTransform(QTransform.fromScale(
                logical_width / bounds.width(),
                logical_height / bounds.height(),
            ))
        self._scene.setSceneRect(self._pix_item.sceneBoundingRect())
        if refit:
            self.fit_image_to_viewport()
        else:
            self._update_pixmap_filtering()

    def _update_pixmap_filtering(self) -> None:
        """Avoid screen-space moire when a fine dither is being reduced.

        Nearest-neighbour display is desirable at 1:1 and above, but while a
        pixmap is smaller on screen than its raster dimensions it drops whole
        rows/columns.  Repeating dither patterns then appear as large blank
        rectangles or density bands.  Smooth only that reduction; zoomed-in
        pixels remain crisp.
        """
        pixmap = self._pix_item.pixmap()
        if pixmap.isNull():
            smooth = False
        else:
            item_scale = abs(self._pix_item.transform().m11())
            view_scale = abs(self.transform().m11())
            device_scale = item_scale * view_scale * self.devicePixelRatioF()
            smooth = device_scale < 1.0 - 1e-9
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)

    def viewport_device_demand(self) -> tuple[int, int]:
        """Drawable viewport dimensions in physical display pixels."""
        viewport = self.viewport()
        return viewport_device_size(
            viewport.width(), viewport.height(), viewport.devicePixelRatioF(),
        )

    def fit_image_to_viewport(self) -> None:
        if not self._pix_item.pixmap().isNull():
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_pixmap_filtering()
        self.zoom_changed.emit(self.current_zoom_percent())

    def current_zoom_percent(self) -> int:
        return zoom_percent(self.transform().m11())

    def is_image_zoomed(self) -> bool:
        rect = self._pix_item.sceneBoundingRect()
        vp = self.viewport().rect()
        return (rect.width() * self.transform().m11() > vp.width()
                or rect.height() * self.transform().m22() > vp.height())

    def _apply_zoom(self, direction: int) -> None:
        if next_zoom(self.transform().m11(), direction) is None:
            return
        factor = 1.2 if direction > 0 else 0.8
        self.scale(factor, factor)
        self._update_pixmap_filtering()
        self.zoom_changed.emit(self.current_zoom_percent())

    def zoom_in(self) -> None:
        self._apply_zoom(1)

    def zoom_out(self) -> None:
        self._apply_zoom(-1)

    def reset_zoom(self) -> None:
        self.fit_image_to_viewport()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._apply_zoom(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
        else:
            super().wheelEvent(event)

    # ---- pan + inertia ------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_pos = event.position()
            self._last_time = time.monotonic()
            self._velocity = QPointF(0.0, 0.0)
            self._timer.stop()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_pos is not None:
            delta = event.position() - self._last_pos
            self._last_pos = event.position()
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - int(delta.x()))
            vbar.setValue(vbar.value() - int(delta.y()))
            now = time.monotonic()
            dt = max(now - (self._last_time or now), 1e-3)
            self._velocity = QPointF(delta.x() / dt, delta.y() / dt)
            self._last_time = now
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._enable_inertia:
                vx = clamp_velocity(self._velocity.x(), self._velocity_scale, self._max_velocity)
                vy = clamp_velocity(self._velocity.y(), self._velocity_scale, self._max_velocity)
                self._velocity = QPointF(vx, vy)
                if abs(vx) >= 10 or abs(vy) >= 10:
                    self._timer.start(16)
        super().mouseReleaseEvent(event)

    def _on_inertia_tick(self):
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        nx, vx = inertia_step(hbar.value(), self._velocity.x(), hbar.maximum(), self._friction)
        ny, vy = inertia_step(vbar.value(), self._velocity.y(), vbar.maximum(), self._friction)
        hbar.setValue(int(nx))
        vbar.setValue(int(ny))
        self._velocity = QPointF(vx, vy)
        if abs(vx) < 10 and abs(vy) < 10:
            self._timer.stop()

    # ---- drag-and-drop import ----------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.image_dropped.emit(path)
                break
        event.acceptProposedAction()
