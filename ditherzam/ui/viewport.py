from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QLineEdit,
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
    layer_drag_started = Signal()
    layer_dragged = Signal(float, float)
    layer_drag_finished = Signal()
    layer_resize_started = Signal(str)
    layer_resized = Signal(str, float, float)
    layer_resize_finished = Signal()
    layer_nudge_requested = Signal(int, int)
    layer_transform_confirm_requested = Signal()
    layer_transform_cancel_requested = Signal()
    mask_brush_started = Signal(float, float)
    mask_brush_moved = Signal(float, float)
    mask_brush_finished = Signal()
    mask_brush_cancel_requested = Signal()
    mask_brush_size_delta_requested = Signal(int)
    mask_brush_mode_swap_requested = Signal()
    selection_dragged = Signal(str, float, float, float, float)
    gradient_dragged = Signal(str, float, float, float, float)

    def __init__(self, bg_color="#1f1f1f", friction=0.95, enable_inertia=True,
                 velocity_scale=0.5, max_velocity=2000.0, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self._transform_proxy_root: QGraphicsRectItem | None = None
        self._transform_proxy_background: QGraphicsPixmapItem | None = None
        self._transform_proxy_layer: QGraphicsPixmapItem | None = None
        self._mask_stroke_item: QGraphicsPixmapItem | None = None
        self._mask_stroke_surface: QImage | None = None
        self._mask_stroke_raster_size: tuple[int, int] | None = None
        self.setBackgroundBrush(QColor(bg_color))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self._friction = friction
        self._enable_inertia = enable_inertia
        self._velocity_scale = velocity_scale
        self._max_velocity = max_velocity
        self._velocity = QPointF(0.0, 0.0)
        self._last_pos: QPointF | None = None
        self._last_time: float | None = None
        self._panning = False
        self._layer_drag_rect: QRectF | None = None
        self._layer_dragging = False
        self._layer_drag_start_scene: QPointF | None = None
        self._layer_resize_corner: str | None = None
        self._mask_brush_mode = False
        self._mask_brush_size = 32.0
        self._mask_brush_down = False
        self._space_pan = False
        self._brush_scene_pos: QPointF | None = None
        self._selection_tool: str | None = None
        self._selection_start: QPointF | None = None
        self._selection_end: QPointF | None = None
        self._selection_item: QGraphicsPixmapItem | None = None
        self._gradient_tool: str | None = None
        self._gradient_start: QPointF | None = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_inertia_tick)

    def set_mask_brush_mode(self, enabled: bool, size: float | None = None) -> None:
        self._mask_brush_mode = bool(enabled)
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
                raise ValueError("brush size must be positive")
            self._mask_brush_size = float(size)
        if not enabled:
            self._mask_brush_down = False
            self._brush_scene_pos = None
            self._space_pan = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(Qt.CursorShape.BlankCursor)
        self.viewport().update()

    def set_mask_brush_size(self, size: float) -> None:
        self.set_mask_brush_mode(self._mask_brush_mode, size)

    def set_selection_tool(self, shape: str | None) -> None:
        if shape not in {None, "rectangle", "ellipse"}:
            raise ValueError("selection tool must be rectangle, ellipse, or None")
        self._selection_tool = shape
        self._selection_start = None
        self._selection_end = None
        if shape is not None:
            self.set_mask_brush_mode(False)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def install_selection_overlay(self, coverage) -> None:
        import numpy as np
        from .convert import numpy_to_qimage

        value = np.asarray(coverage)
        if value.dtype != np.uint8 or value.ndim != 2 or not value.size:
            raise ValueError("selection coverage must be non-empty 2-D uint8")
        rgba = np.zeros((*value.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 47
        rgba[..., 1] = 128
        rgba[..., 2] = 255
        rgba[..., 3] = (value.astype(np.uint16) * 96 // 255).astype(np.uint8)
        if self._selection_item is None:
            self._selection_item = QGraphicsPixmapItem()
            self._selection_item.setZValue(4.0)
            self._scene.addItem(self._selection_item)
        self._selection_item.setPixmap(
            QPixmap.fromImage(numpy_to_qimage(rgba)))
        self._selection_item.setVisible(True)

    def clear_selection_overlay(self) -> None:
        if self._selection_item is not None:
            self._selection_item.setVisible(False)

    def set_gradient_tool(self, kind: str | None) -> None:
        if kind not in {None, "linear", "radial"}:
            raise ValueError("gradient tool must be linear, radial, or None")
        self._gradient_tool = kind
        self._gradient_start = None
        if kind is not None:
            self.set_selection_tool(None)
            self.setCursor(Qt.CursorShape.CrossCursor)

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

    def set_layer_drag_target(self, rect: QRectF | None) -> None:
        """Set the selected layer's bounds in document/scene coordinates."""
        if rect is not None and not isinstance(rect, QRectF):
            raise ValueError("layer drag target must be a QRectF or None")
        self._layer_drag_rect = None if rect is None else QRectF(rect)
        if rect is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def install_transform_proxy(
        self,
        background_canvas: QPixmap,
        selected_layer: QPixmap,
        layer_rect: QRectF,
        *,
        document_rect: QRectF | None = None,
        opacity: float = 1.0,
    ) -> None:
        """Install a synchronous, viewport-only layer transform preview.

        ``background_canvas`` is the document composite with the selected
        layer omitted. ``selected_layer`` is that layer's alpha-bearing raster.
        Both are mapped into document/scene coordinates; no displayed pixmap is
        copied back into the authoritative ``_pix_item``.
        """
        if background_canvas.isNull() or selected_layer.isNull():
            raise ValueError("transform proxy pixmaps must not be null")
        if not isinstance(layer_rect, QRectF):
            raise ValueError("layer_rect must be a QRectF")
        if layer_rect.width() <= 0 or layer_rect.height() <= 0:
            raise ValueError("layer_rect dimensions must be positive")
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between 0 and 1")
        if document_rect is None:
            document_rect = self._pix_item.sceneBoundingRect()
        elif not isinstance(document_rect, QRectF):
            raise ValueError("document_rect must be a QRectF or None")
        else:
            document_rect = QRectF(document_rect)
        if document_rect.width() <= 0 or document_rect.height() <= 0:
            raise ValueError("document_rect dimensions must be positive")

        self.clear_transform_proxy()
        root = QGraphicsRectItem(document_rect)
        root.setPen(QPen(Qt.PenStyle.NoPen))
        root.setBrush(Qt.BrushStyle.NoBrush)
        root.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True
        )
        root.setZValue(1.0)
        background = QGraphicsPixmapItem(background_canvas, root)
        layer = QGraphicsPixmapItem(selected_layer, root)
        background.setZValue(0.0)
        layer.setZValue(1.0)
        layer.setOpacity(opacity)
        self._scene.addItem(root)
        self._transform_proxy_root = root
        self._transform_proxy_background = background
        self._transform_proxy_layer = layer
        self._set_proxy_item_geometry(background, document_rect)
        self._set_proxy_item_geometry(layer, layer_rect)
        self._pix_item.setVisible(False)
        self.set_layer_drag_target(layer_rect)
        self.viewport().update()

    @staticmethod
    def _set_proxy_item_geometry(
        item: QGraphicsPixmapItem, rect: QRectF
    ) -> None:
        bounds = item.boundingRect()
        item.setPos(rect.topLeft())
        item.setTransform(QTransform.fromScale(
            rect.width() / bounds.width(),
            rect.height() / bounds.height(),
        ))

    def update_transform_proxy_geometry(self, layer_rect: QRectF) -> None:
        """Move/scale the active proxy immediately in scene coordinates."""
        if self._transform_proxy_layer is None:
            raise RuntimeError("no transform proxy is installed")
        if not isinstance(layer_rect, QRectF):
            raise ValueError("layer_rect must be a QRectF")
        if layer_rect.width() <= 0 or layer_rect.height() <= 0:
            raise ValueError("layer_rect dimensions must be positive")
        rect = QRectF(layer_rect)
        self._set_proxy_item_geometry(self._transform_proxy_layer, rect)
        self.set_layer_drag_target(rect)
        self.viewport().update()

    def clear_transform_proxy(self) -> None:
        """Remove the viewport proxy without changing pan, zoom, or base pixels."""
        if self._transform_proxy_root is not None:
            self._scene.removeItem(self._transform_proxy_root)
        self._transform_proxy_root = None
        self._transform_proxy_background = None
        self._transform_proxy_layer = None
        self._pix_item.setVisible(True)
        self.viewport().update()

    def install_mask_stroke_roi(
        self, pixmap: QPixmap, rect: QRectF, raster_size: tuple[int, int],
        logical_size: tuple[int, int], max_side: int = 720,
    ) -> None:
        """Merge an ROI into one bounded reusable stroke overlay item."""
        if pixmap.isNull() or rect.width() <= 0 or rect.height() <= 0:
            return
        rw, rh = raster_size
        lw, lh = logical_size
        if min(rw, rh, lw, lh, max_side) <= 0:
            raise ValueError("stroke overlay dimensions must be positive")
        scale = min(1.0, max_side / max(rw, rh))
        sw, sh = max(1, round(rw * scale)), max(1, round(rh * scale))
        if (
            self._mask_stroke_surface is None
            or self._mask_stroke_raster_size != (rw, rh)
            or self._mask_stroke_surface.size().width() != sw
            or self._mask_stroke_surface.size().height() != sh
        ):
            self.clear_mask_stroke_overlay()
            self._mask_stroke_surface = QImage(
                sw, sh, QImage.Format.Format_RGBA8888)
            self._mask_stroke_surface.fill(Qt.GlobalColor.transparent)
            self._mask_stroke_raster_size = (rw, rh)
            self._mask_stroke_item = QGraphicsPixmapItem()
            self._mask_stroke_item.setZValue(3.0)
            self._scene.addItem(self._mask_stroke_item)
        assert self._mask_stroke_surface is not None
        assert self._mask_stroke_item is not None
        target = QRectF(
            rect.x() * sw / rw, rect.y() * sh / rh,
            rect.width() * sw / rw, rect.height() * sh / rh)
        painter = QPainter(self._mask_stroke_surface)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Source)
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
        painter.end()
        self._mask_stroke_item.setPixmap(
            QPixmap.fromImage(self._mask_stroke_surface))
        self._set_proxy_item_geometry(
            self._mask_stroke_item, QRectF(0, 0, lw, lh))
        self.viewport().update()

    def clear_mask_stroke_overlay(self) -> None:
        if self._mask_stroke_item is not None:
            self._scene.removeItem(self._mask_stroke_item)
        self._mask_stroke_item = None
        self._mask_stroke_surface = None
        self._mask_stroke_raster_size = None
        self.viewport().update()

    def _device_pixels_to_scene(self, pixels: float) -> float:
        view_scale = max(abs(self.transform().m11()), 1e-6)
        dpr = max(self.viewport().devicePixelRatioF(), 1e-6)
        return pixels / (view_scale * dpr)

    def _layer_handle_rects(self) -> dict[str, QRectF]:
        """Eight resize hit targets, each at least 16 physical display pixels."""
        return self._layer_handle_rects_for_device_size(16.0)

    def _layer_visible_handle_rects(self) -> dict[str, QRectF]:
        """Compact visible handles independent of zoom and display scaling."""
        return self._layer_handle_rects_for_device_size(10.0)

    def _layer_handle_rects_for_device_size(
        self, device_size: float
    ) -> dict[str, QRectF]:
        rect = self._layer_drag_rect
        if rect is None:
            return {}
        size = self._device_pixels_to_scene(device_size)
        half = size / 2.0
        center_x = rect.center().x()
        center_y = rect.center().y()
        return {
            "nw": QRectF(rect.left() - half, rect.top() - half, size, size),
            "n": QRectF(center_x - half, rect.top() - half, size, size),
            "ne": QRectF(rect.right() - half, rect.top() - half, size, size),
            "e": QRectF(rect.right() - half, center_y - half, size, size),
            "se": QRectF(rect.right() - half, rect.bottom() - half, size, size),
            "s": QRectF(center_x - half, rect.bottom() - half, size, size),
            "sw": QRectF(rect.left() - half, rect.bottom() - half, size, size),
            "w": QRectF(rect.left() - half, center_y - half, size, size),
        }

    @staticmethod
    def _selection_outline_pens() -> tuple[QPen, QPen, QPen]:
        pens = []
        for color, width in (
            (QColor("#000000"), 6),
            (QColor("#ffffff"), 4),
            (QColor("#2f80ff"), 2),
        ):
            pen = QPen(color)
            pen.setCosmetic(True)
            pen.setWidth(width)
            pens.append(pen)
        return tuple(pens)

    @staticmethod
    def _resize_cursor(handle: str) -> Qt.CursorShape:
        if handle in {"nw", "se"}:
            return Qt.CursorShape.SizeFDiagCursor
        if handle in {"ne", "sw"}:
            return Qt.CursorShape.SizeBDiagCursor
        if handle in {"n", "s"}:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeHorCursor

    def _handle_at(self, scene_pos: QPointF) -> str | None:
        for handle, hit_rect in self._layer_handle_rects().items():
            if hit_rect.contains(scene_pos):
                return handle
        return None

    def _update_transform_hover_cursor(self, scene_pos: QPointF) -> None:
        handle = self._handle_at(scene_pos)
        if handle is not None:
            self.setCursor(self._resize_cursor(handle))
        elif (
            self._layer_drag_rect is not None
            and self._layer_drag_rect.contains(scene_pos)
        ):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if (
            self._mask_brush_mode and not self._space_pan
            and self._brush_scene_pos is not None
        ):
            radius = self._mask_brush_size / 2.0
            pen = QPen(QColor("#ffffff"))
            pen.setCosmetic(True)
            pen.setWidth(1)
            painter.save()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawEllipse(self._brush_scene_pos, radius, radius)
            painter.restore()
        if (
            self._selection_tool is not None
            and self._selection_start is not None
            and self._selection_end is not None
        ):
            selection_rect = QRectF(
                self._selection_start, self._selection_end).normalized()
            painter.save()
            pen = QPen(QColor("#ffffff"))
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(47, 128, 255, 32))
            if self._selection_tool == "ellipse":
                painter.drawEllipse(selection_rect)
            else:
                painter.drawRect(selection_rect)
            painter.restore()
        target = self._layer_drag_rect
        if target is None:
            return
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        visible_handles = self._layer_visible_handle_rects().values()
        for pen in self._selection_outline_pens():
            painter.setPen(pen)
            painter.drawRect(target)
            for handle in visible_handles:
                painter.drawRect(handle)
        painter.setBrush(QColor("#2f80ff"))
        painter.setPen(Qt.PenStyle.NoPen)
        for handle in visible_handles:
            painter.drawRect(handle)
        painter.restore()

    # ---- pan + inertia ------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pos = event.position()
            self._last_time = time.monotonic()
            self._velocity = QPointF(0.0, 0.0)
            self._timer.stop()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (
            self._gradient_tool is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._gradient_start = self.mapToScene(event.position().toPoint())
            event.accept()
            return
        if (
            self._selection_tool is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            self._selection_start = scene_pos
            self._selection_end = scene_pos
            self.viewport().update()
            event.accept()
            return
        if (
            self._mask_brush_mode
            and event.button() == Qt.MouseButton.LeftButton
            and not self._space_pan
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            self._brush_scene_pos = scene_pos
            self._mask_brush_down = True
            self.mask_brush_started.emit(
                float(scene_pos.x()), float(scene_pos.y()))
            self.viewport().update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            handle = self._handle_at(scene_pos)
            if handle is not None:
                self._layer_resize_corner = handle
                self._layer_drag_start_scene = scene_pos
                self._layer_dragging = False
                self._panning = False
                self._timer.stop()
                self.setCursor(self._resize_cursor(handle))
                self.layer_resize_started.emit(handle)
                event.accept()
                return
            if (
                self._layer_drag_rect is not None
                and self._layer_drag_rect.contains(scene_pos)
            ):
                self._layer_dragging = True
                self._layer_drag_start_scene = scene_pos
                self._panning = False
                self._timer.stop()
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                self.layer_drag_started.emit()
                event.accept()
                return
            self._panning = True
            self._last_pos = event.position()
            self._last_time = time.monotonic()
            self._velocity = QPointF(0.0, 0.0)
            self._timer.stop()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_tool is not None and self._selection_start is not None:
            self._selection_end = self.mapToScene(event.position().toPoint())
            self.viewport().update()
            event.accept()
            return
        if self._mask_brush_mode:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._brush_scene_pos = scene_pos
            if self._mask_brush_down and not self._space_pan:
                self.mask_brush_moved.emit(
                    float(scene_pos.x()), float(scene_pos.y()))
                event.accept()
                self.viewport().update()
                return
            self.viewport().update()
        if (
            self._layer_resize_corner is not None
            and self._layer_drag_start_scene is not None
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            delta = scene_pos - self._layer_drag_start_scene
            self.layer_resized.emit(
                self._layer_resize_corner,
                float(delta.x()),
                float(delta.y()),
            )
            event.accept()
            return
        if self._layer_dragging and self._layer_drag_start_scene is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            delta = scene_pos - self._layer_drag_start_scene
            self.layer_dragged.emit(float(delta.x()), float(delta.y()))
            event.accept()
            return
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
        elif self._layer_drag_rect is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._update_transform_hover_cursor(scene_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self._gradient_tool is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self._gradient_start is not None
        ):
            end = self.mapToScene(event.position().toPoint())
            start = self._gradient_start
            self._gradient_start = None
            self.gradient_dragged.emit(
                self._gradient_tool,
                float(start.x()), float(start.y()),
                float(end.x()), float(end.y()),
            )
            event.accept()
            return
        if (
            self._selection_tool is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self._selection_start is not None
        ):
            end = self.mapToScene(event.position().toPoint())
            start = self._selection_start
            self._selection_start = None
            self._selection_end = None
            self.selection_dragged.emit(
                self._selection_tool,
                float(start.x()), float(start.y()),
                float(end.x()), float(end.y()),
            )
            self.viewport().update()
            event.accept()
            return
        if (
            self._mask_brush_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._mask_brush_down
        ):
            self._mask_brush_down = False
            self.mask_brush_finished.emit()
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._layer_resize_corner is not None
        ):
            self._layer_resize_corner = None
            self._layer_drag_start_scene = None
            self._update_transform_hover_cursor(
                self.mapToScene(event.position().toPoint())
            )
            self.layer_resize_finished.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._layer_dragging:
            self._layer_dragging = False
            self._layer_drag_start_scene = None
            self._update_transform_hover_cursor(
                self.mapToScene(event.position().toPoint())
            )
            self.layer_drag_finished.emit()
            event.accept()
            return
        if (
            event.button() in {
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.MiddleButton,
            }
            and self._panning
        ):
            self._panning = False
            self.setCursor(
                Qt.CursorShape.BlankCursor
                if self._mask_brush_mode and not self._space_pan
                else Qt.CursorShape.ArrowCursor)
            if self._enable_inertia:
                vx = clamp_velocity(self._velocity.x(), self._velocity_scale, self._max_velocity)
                vy = clamp_velocity(self._velocity.y(), self._velocity_scale, self._max_velocity)
                self._velocity = QPointF(vx, vy)
                if abs(vx) >= 10 or abs(vy) >= 10:
                    self._timer.start(16)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        focus = QApplication.focusWidget()
        if (
            self._mask_brush_mode
            and not isinstance(focus, (QLineEdit, QAbstractSpinBox))
        ):
            key = event.key()
            if key == Qt.Key.Key_Space and not event.isAutoRepeat():
                self._space_pan = True
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                event.accept()
                return
            if key == Qt.Key.Key_Escape:
                self._mask_brush_down = False
                self.mask_brush_cancel_requested.emit()
                event.accept()
                return
            if key in {Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight}:
                self.mask_brush_size_delta_requested.emit(
                    -1 if key == Qt.Key.Key_BracketLeft else 1)
                event.accept()
                return
            if key == Qt.Key.Key_X:
                self.mask_brush_mode_swap_requested.emit()
                event.accept()
                return
        if self._layer_drag_rect is None:
            super().keyPressEvent(event)
            return
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.layer_transform_confirm_requested.emit()
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.layer_transform_cancel_requested.emit()
            event.accept()
            return
        deltas = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        if key in deltas:
            step = (
                10
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                else 1
            )
            dx, dy = deltas[key]
            self.layer_nudge_requested.emit(dx * step, dy * step)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if (
            self._mask_brush_mode
            and event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
        ):
            self._space_pan = False
            self.setCursor(Qt.CursorShape.BlankCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

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
