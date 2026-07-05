from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QWidget,
)

from ditherzam.dithering import registry as _dither_registry
from ditherzam.render import RenderPipeline

from .controls import ControlPanel
from .convert import numpy_to_qimage
from .hotkeys import get_hotkeys
from .settings_map import settings_from_controls
from .viewport import CustomGraphicsView


class _RenderSignals(QObject):
    finished = Signal(QImage)


class _RenderWorker(QRunnable):
    """Runs one render off the GUI thread and emits the finished QImage."""

    def __init__(self, pipeline: RenderPipeline, base_gray: np.ndarray, settings):
        super().__init__()
        self._pipeline = pipeline
        self._base_gray = base_gray
        self._settings = settings
        self.signals = _RenderSignals()

    def run(self) -> None:
        rgb = self._pipeline.render(self._base_gray, self._settings)
        self.signals.finished.emit(numpy_to_qimage(rgb))


class ImageEditor(QMainWindow):
    def __init__(self, registry=None, color_engine=None, effect_stack=None,
                 debounce_ms: int = 20, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ditherzam")
        self._registry = registry or _dither_registry
        self.pipeline = RenderPipeline(self._registry, color_engine, effect_stack)
        self._base_gray: np.ndarray | None = None
        self.last_qimage: QImage | None = None
        self._pool = QThreadPool.globalInstance()
        self._debounce_ms = debounce_ms

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)

        self.viewport = CustomGraphicsView()
        self.panel = ControlPanel()
        self.panel.set_registry_categories(self._registry.by_category())
        self.panel.changed.connect(self.schedule_render)
        self.viewport.image_dropped.connect(self._on_image_dropped)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self.viewport)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        # single-child layout for the central widget
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_render)

        self._install_shortcuts()

    # ---- public API ---------------------------------------------------------
    def load_array(self, gray_f32) -> None:
        self._base_gray = np.asarray(gray_f32, dtype=np.float32)

    def set_style(self, name: str) -> None:
        self.panel.set_style(name)

    def render_now(self) -> QImage:
        """Synchronous render (used by tests and the initial paint)."""
        if self._base_gray is None:
            raise RuntimeError("No image loaded")
        settings = settings_from_controls(self.panel.state)
        rgb = self.pipeline.render(self._base_gray, settings)
        qimg = numpy_to_qimage(rgb)
        self.last_qimage = qimg
        self.viewport.set_pixmap(QPixmap.fromImage(qimg))
        return qimg

    def schedule_render(self) -> None:
        self._debounce.start(self._debounce_ms)

    # ---- internals ----------------------------------------------------------
    def _do_render(self) -> None:
        if self._base_gray is None:
            return
        settings = settings_from_controls(self.panel.state)
        worker = _RenderWorker(self.pipeline, self._base_gray, settings)
        worker.signals.finished.connect(self._on_rendered)
        self._pool.start(worker)

    def _on_rendered(self, qimg: QImage) -> None:
        self.last_qimage = qimg
        self.viewport.set_pixmap(QPixmap.fromImage(qimg))

    def _on_image_dropped(self, path: str) -> None:
        from PIL import Image

        from ditherzam.imaging import to_gray_f32
        try:
            self.load_array(to_gray_f32(Image.open(path)))
        except Exception:
            return
        self.render_now()

    def _install_shortcuts(self) -> None:
        hk = get_hotkeys(sys.platform)
        self._actions: dict[str, QAction] = {}
        bindings = {
            "zoom_in": self.viewport.zoom_in,
            "zoom_out": self.viewport.zoom_out,
            "zoom_reset": self.viewport.reset_zoom,
        }
        for action_name, slot in bindings.items():
            act = QAction(self)
            act.setShortcut(QKeySequence(hk[action_name]))
            act.triggered.connect(slot)
            self.addAction(act)
            self._actions[action_name] = act
