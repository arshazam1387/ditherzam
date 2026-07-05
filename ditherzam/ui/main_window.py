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
from .export_actions import create_export_menu
from .hotkeys import get_hotkeys
from .settings_map import settings_from_controls
from .viewport import CustomGraphicsView
from ..presets import PresetManager, settings_to_preset, preset_to_settings
from ..export.raster import save_raster
from ..export.vector import raster_to_svg
from ..batch import batch_process


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

        self.expert_mode = False
        self._install_shortcuts()
        self._wire_export()
        self._wire_video()

    # ---- video (Phase 7, UI layer only) -------------------------------------
    def _wire_video(self) -> None:
        from .video_controller import VideoController
        self.video_controller = VideoController(
            self,
            self.pipeline,
            settings_provider=self._collect_settings,
            expert_provider=lambda: self.expert_mode,
        )
        self.menuBar().addMenu(self.video_controller.build_menu())

    # ---- presets & export (Phase 6, UI layer only) --------------------------
    def _wire_export(self) -> None:
        from platformdirs import user_data_dir
        from pathlib import Path
        self._preset_manager = PresetManager(Path(user_data_dir("ditherzam")) / "PRESETS")
        self._export_menu, self._export_actions = create_export_menu(
            self.menuBar(),
            {
                "save_preset":   self._on_save_preset,
                "load_preset":   self._on_load_preset,
                "import_preset": self._on_import_preset,
                "export_preset": self._on_export_preset,
                "export_png":    lambda: self._on_export_raster("PNG Files (*.png)", ".png"),
                "export_jpg":    lambda: self._on_export_raster("JPEG Files (*.jpg)", ".jpg"),
                "export_svg":    self._on_export_svg,
                "batch_folder":  self._on_batch_folder,
            },
        )

    # -- pure state accessors used by the handlers --
    def _collect_settings(self):
        return settings_from_controls(self.panel.state)

    def _color_mode(self) -> str:
        return str(self.panel.state.get("color_mode", "off"))

    def _current_palette(self):
        if self._color_mode() == "off":
            return None
        from ..color.palette import builtin_palettes
        return builtin_palettes().get(self.panel.state.get("palette"))

    def _current_effect_stack(self):
        from ..effects.stack import EffectStack
        stack = EffectStack()
        for name in self.panel.state.get("effects", []) or []:
            stack.add(name)
        return stack

    def _reference_size(self) -> tuple[int, int]:
        if self._base_gray is None:
            return (0, 0)
        h, w = self._base_gray.shape[:2]
        return (int(w), int(h))

    def _rendered_rgb(self) -> np.ndarray:
        if self._base_gray is None:
            raise RuntimeError("No image loaded")
        return self.pipeline.render(self._base_gray, self._collect_settings())

    def _apply_preset(self, settings, palette, effects) -> None:
        panel = self.panel
        for key in ("contrast", "midtones", "highlights", "luminance_threshold", "blur"):
            value = int(getattr(settings, key))
            panel.state[key] = value
            if key in panel._sliders:
                panel._sliders[key].setValue(value)
        panel.saturation_slider.setValue(int(settings.saturation))
        panel.scale_slider.setValue(int(settings.scale))
        panel.invert_toggle.setChecked(bool(settings.invert))
        panel.preview_toggle.setChecked(bool(settings.preview_disabled))
        panel.state["params"] = dict(settings.params)
        panel.effects_list.clear()
        for name, _params in effects:
            panel.effects_list.addItem(name)
        panel.state["effects"] = [name for name, _params in effects]
        if palette is not None:
            panel.state["palette"] = palette.name
        panel.set_style(settings.style)
        if self._base_gray is not None:
            self.render_now()

    # -- menu handlers --
    def _on_save_preset(self):
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")
        if not ok or not name:
            return
        preset = settings_to_preset(
            self._collect_settings(), self._current_palette(),
            self._current_effect_stack(), self._color_mode(),
        )
        self._preset_manager.save(name, preset)
        QMessageBox.information(self, "Presets", f"Preset '{name}' saved successfully!")

    def _on_load_preset(self):
        from PySide6.QtWidgets import QInputDialog
        names = self._preset_manager.list()
        if not names:
            return
        name, ok = QInputDialog.getItem(self, "Load Preset", "Preset:", names, 0, False)
        if not ok:
            return
        settings, palette, effects = preset_to_settings(self._preset_manager.load(name))
        self._apply_preset(settings, palette, effects)

    def _on_import_preset(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(self, "Import Preset(s)", "",
                                              "Preset Files (*.yaml *.yml)")
        if not path:
            return
        try:
            name = self._preset_manager.import_file(path)
        except ValueError:
            QMessageBox.warning(self, "Presets", "Not a valid preset file.")
            return
        QMessageBox.information(self, "Presets", f"Imported preset '{name}'.")

    def _on_export_preset(self):
        from PySide6.QtWidgets import QFileDialog
        import yaml
        path, _ = QFileDialog.getSaveFileName(self, "Export Preset", "",
                                              "Preset Files (*.yaml)")
        if not path:
            return
        preset = settings_to_preset(
            self._collect_settings(), self._current_palette(),
            self._current_effect_stack(), self._color_mode(),
        )
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(preset, f, sort_keys=False, allow_unicode=True)

    def _on_export_raster(self, file_filter, ext):
        from PySide6.QtWidgets import QFileDialog
        if self._base_gray is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", file_filter)
        if not path:
            return
        save_raster(self._rendered_rgb(), path)

    def _on_export_svg(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        if self._base_gray is None:
            return
        cont = QMessageBox.warning(
            self, "Export as Vector",
            "WARNING: vector export is experimental. For best results use a larger "
            "scale and fewer fine details. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if cont != QMessageBox.StandardButton.Yes:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Vector", "",
                                              "SVG Files (*.svg)")
        if not path:
            return
        settings = self._collect_settings()
        gray = self._base_gray
        threshold = int(settings.luminance_threshold / 100.0 * 255.0)
        svg = raster_to_svg(np.asarray(gray).astype("uint8"), threshold,
                            bool(settings.invert))
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)

    def _on_batch_folder(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        from pathlib import Path
        out = Path(folder) / "batch_processed"
        processed, skipped = batch_process(
            folder, out, self._collect_settings(),
            self.pipeline, self._reference_size(),
        )
        QMessageBox.information(
            self, "Batch",
            f"Processed {processed} images. Skipped {skipped} images.",
        )

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
