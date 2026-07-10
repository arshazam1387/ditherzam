from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSettings, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QImage, QKeySequence, QPixmap
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
from .preview import render_preview
from .preview_preferences import (
    PREVIEW_RESOLUTIONS,
    PreviewPreferences,
    load_preview_preferences,
    save_preview_preferences,
)
from .render_scheduler import RenderCoalescer
from .export_actions import create_export_menu
from .hotkeys import get_hotkeys
from .settings_map import settings_from_controls
from .viewport import CustomGraphicsView
from ..presets import PresetManager, settings_to_preset, preset_to_settings
from ..export.raster import save_raster
from ..export.vector import raster_to_svg
from ..batch import batch_process


# Default parameters for post-effects added from the Effects panel (which stores
# only names). Every EFFECTS function has a required strength/amount arg, so a
# param-less add would crash the render — these give each a sensible default.
_EFFECT_DEFAULTS: dict[str, dict] = {
    "Blur": {"radius": 2.0},
    "Sharpen": {"amount": 1.0},
    "Chromatic Aberration": {"shift": 2},
    "JPEG Glitch": {"quality": 15},
    "Epsilon Glow": {"threshold": 64.0, "smoothing": 32.0, "radius": 8.0,
                     "intensity": 1.0, "epsilon": 0.4, "falloff": 0.5,
                     "distance_scale": 1.0, "aspect": 1.0},
}


class _RenderSignals(QObject):
    finished = Signal(QImage, int)
    failed = Signal(int)


class _RenderWorker(QRunnable):
    """Runs one render off the GUI thread and emits (QImage, generation token)."""

    def __init__(self, pipeline: RenderPipeline, base_gray: np.ndarray, settings,
                 token: int, mode: str = "full", proxy_max_side: int = 640):
        super().__init__()
        self._pipeline = pipeline
        self._base_gray = base_gray
        self._settings = settings
        self._token = token
        self._mode = mode
        self._proxy_max_side = proxy_max_side
        self.signals = _RenderSignals()

    def run(self) -> None:
        # A render raising here would otherwise never emit `finished`, so the
        # coalescer's `_busy` flag would stick True and freeze all future renders.
        # Always report an outcome (finished OR failed) so the coalescer recovers.
        try:
            if self._mode == "proxy":
                rgb = render_preview(self._pipeline, self._base_gray, self._settings,
                                     self._proxy_max_side)
            else:
                rgb = self._pipeline.render_cached(self._base_gray, self._settings)
            qimg = numpy_to_qimage(rgb)
        except Exception:
            import traceback
            traceback.print_exc()
            self.signals.failed.emit(self._token)
            return
        self.signals.finished.emit(qimg, self._token)


class ImageEditor(QMainWindow):
    def __init__(self, registry=None, color_engine=None, effect_stack=None,
                 debounce_ms: int = 20, settle_ms: int = 160,
                 proxy_max_side: int = 640, parent=None, preference_store=None):
        super().__init__(parent)
        self.setWindowTitle("ditherzam")
        self._registry = registry or _dither_registry
        self.pipeline = RenderPipeline(self._registry, color_engine, effect_stack)
        self._base_gray: np.ndarray | None = None
        self._base_rgb: np.ndarray | None = None
        self._preview_palette = None
        self.last_qimage: QImage | None = None
        self._pool = QThreadPool.globalInstance()
        self._debounce_ms = debounce_ms
        self._settle_ms = settle_ms
        self._proxy_max_side = proxy_max_side
        self._render_mode = "full"
        self._coalescer = RenderCoalescer()
        self._preference_store = preference_store or QSettings()
        self.preview_preferences = load_preview_preferences(self._preference_store)

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)

        self.viewport = CustomGraphicsView()
        self.panel = ControlPanel()
        self.panel.set_registry_categories(self._registry.by_category())
        self.panel.changed.connect(self.schedule_render)
        self.panel.from_image_requested.connect(self._on_from_image_requested)
        self.panel.palette_preview.connect(self._on_palette_preview)
        self.viewport.image_dropped.connect(self._on_image_dropped)

        from PySide6.QtWidgets import QTabWidget
        from .glow_panel import GlowPanel

        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setWidget(self.panel)

        self.glow_panel = GlowPanel()
        self.glow_panel.changed.connect(self.schedule_render)
        glow_scroll = QScrollArea()
        glow_scroll.setWidgetResizable(True)
        glow_scroll.setWidget(self.glow_panel)

        self.tabs = QTabWidget()
        self.tabs.addTab(editor_scroll, "Editor")
        self.tabs.addTab(glow_scroll, "Glow")

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self.viewport)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        # single-child layout for the central widget
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)

        # Two-stage scheduling: a short debounce fires a fast downscaled *proxy*
        # for live feedback; a longer settle timer (restarted on every change)
        # fires the exact full-resolution render once the drag stops.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_render)

        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.timeout.connect(self._do_full_render)

        self.expert_mode = False
        self._install_shortcuts()
        self._wire_preview_preferences()
        self._wire_export()
        self._wire_video()
        self._wire_animation()

    def _wire_preview_preferences(self) -> None:
        """Build the View menu controls for application-level preview policy."""
        self.view_menu = self.menuBar().addMenu("&View")
        self.preview_resolution_menu = self.view_menu.addMenu("Preview Resolution")
        self.preview_resolution_group = QActionGroup(self)
        self.preview_resolution_group.setExclusive(True)
        self.preview_resolution_actions: dict[str, QAction] = {}

        for resolution in PREVIEW_RESOLUTIONS:
            action = QAction(resolution, self)
            action.setCheckable(True)
            action.setChecked(resolution == self.preview_preferences.resolution)
            action.triggered.connect(
                lambda _checked=False, value=resolution:
                    self._set_preview_resolution(value)
            )
            self.preview_resolution_group.addAction(action)
            self.preview_resolution_menu.addAction(action)
            self.preview_resolution_actions[resolution] = action

        self.view_menu.addSeparator()
        self.rerender_on_zoom_action = QAction(
            "Rerender Preview When Zooming In", self
        )
        self.rerender_on_zoom_action.setCheckable(True)
        self.rerender_on_zoom_action.setChecked(
            self.preview_preferences.rerender_on_zoom
        )
        self.rerender_on_zoom_action.toggled.connect(self._set_rerender_on_zoom)
        self.view_menu.addAction(self.rerender_on_zoom_action)

    def _set_preview_resolution(self, resolution: str) -> None:
        self.preview_preferences = PreviewPreferences(
            resolution, self.preview_preferences.rerender_on_zoom
        )
        self._save_preview_preferences_and_schedule()

    def _set_rerender_on_zoom(self, enabled: bool) -> None:
        self.preview_preferences = PreviewPreferences(
            self.preview_preferences.resolution, enabled
        )
        self._save_preview_preferences_and_schedule()

    def _save_preview_preferences_and_schedule(self) -> None:
        save_preview_preferences(self._preference_store, self.preview_preferences)
        # Task 2.3 owns the one-off Full action. If that state has been introduced,
        # any policy edit supersedes it without coupling these controls to its shape.
        if hasattr(self, "_full_preview_requested"):
            self._full_preview_requested = False
        self.schedule_render()

    # ---- animation (Phase 8, UI layer only) ---------------------------------
    def _wire_animation(self) -> None:
        from PySide6.QtWidgets import QDockWidget
        from ..animation.timeline import Timeline
        from .timeline_panel import TimelinePanel, AnimationController

        self.timeline = Timeline(length=30)
        self.timeline_panel = TimelinePanel(length=30, parent=self)
        dock = QDockWidget("Animation", self)
        dock.setWidget(self.timeline_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self.anim_controller = AnimationController(
            self.timeline_panel, self.pipeline,
            self._provide_animation_base, self.timeline, seed=0)
        self.anim_controller.on_frame = self._show_animation_frame
        self.timeline_panel.keyframe_requested.connect(self._add_keyframe_at)
        self.timeline_panel.export_requested.connect(self._export_animation)

    def _provide_animation_base(self):
        if self._base_gray is None:          # no image loaded yet
            return None
        return self._base_gray, self._collect_settings()

    def _show_animation_frame(self, rgb_u8) -> None:
        qimg = numpy_to_qimage(rgb_u8)
        self.last_qimage = qimg
        self.viewport.set_pixmap(QPixmap.fromImage(qimg))

    def _add_keyframe_at(self, frame_index: int) -> None:
        from ..animation.timeline import Keyframe
        s = self._collect_settings()
        self.timeline.add(
            Keyframe(int(frame_index), "luminance_threshold", s.luminance_threshold))

    def _export_animation(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        if self._base_gray is None:
            return
        out, _ = QFileDialog.getSaveFileName(self, "Export Animation", "animation.mp4",
                                             "MP4 Video (*.mp4)")
        if not out:
            return
        self.anim_controller.export(out, fps=24)

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
        if self._preview_palette is not None:
            return self._preview_palette
        return self.panel.working_palette

    def _on_palette_preview(self, palette) -> None:
        if not self.panel.state.get("palette_preview", True):
            return
        self._preview_palette = palette          # a Palette, or None to revert
        if self._base_gray is not None:
            self.schedule_render()

    def _on_from_image_requested(self) -> None:
        if self._base_rgb is None:
            return
        from ..color.palette import generate_palette
        unit = str(self.panel.state.get("extract_unit", "k"))
        value = int(self.panel.extract_slider.value())
        palette = generate_palette(self._base_rgb, unit, value)
        self.panel.set_working_palette(palette)

    def _current_color_engine(self):
        """ColorEngine reflecting the panel's Palette + Mode, or None when off."""
        if self._color_mode() == "off":
            return None
        palette = self._current_palette()
        if palette is None:
            return None
        from ..color.engine import ColorEngine
        return ColorEngine(palette, self._color_mode())

    def _current_effect_stack(self):
        from ..effects.stack import EffectStack
        from ..effects.glow_params import glow_params_from_state
        names = self.panel.state.get("effects", []) or []
        glow_on = bool(self.glow_panel.state.get("glow_enabled"))
        if not names and not glow_on:
            return None
        stack = EffectStack()
        for name in names:
            stack.add(name, **_EFFECT_DEFAULTS.get(name, {}))
        if glow_on:
            stack.add("Epsilon Glow", **glow_params_from_state(self.glow_panel.state))
        return stack

    def _sync_pipeline(self) -> None:
        """Refresh the pipeline's color engine + effect stack from panel state.

        The render pipeline reads these attributes at render time, so they must be
        rebuilt before every render or the Color/Effects controls do nothing.
        """
        self.pipeline.color_engine = self._current_color_engine()
        self.pipeline.effect_stack = self._current_effect_stack()

    def _reference_size(self) -> tuple[int, int]:
        if self._base_gray is None:
            return (0, 0)
        h, w = self._base_gray.shape[:2]
        return (int(w), int(h))

    def _rendered_rgb(self) -> np.ndarray:
        if self._base_gray is None:
            raise RuntimeError("No image loaded")
        self._sync_pipeline()
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
        from ..effects.glow_params import glow_state_from_params, GLOW_DEFAULTS
        panel.effects_list.clear()
        glow_state = None
        non_glow = []
        for name, params in effects:
            if name == "Epsilon Glow":
                glow_state = glow_state_from_params(params)
            else:
                panel.effects_list.addItem(name)
                non_glow.append(name)
        panel.state["effects"] = non_glow
        # push glow params into the Glow tab (enable + sliders), or disable if absent
        gp = self.glow_panel
        gp.enable_toggle.setChecked(bool(glow_state))
        if glow_state:
            for key, slider in gp._sliders.items():
                slider.setValue(int(glow_state.get(key, GLOW_DEFAULTS[key])))
        if palette is not None:
            panel.set_working_palette(palette)
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
    def load_array(self, gray_f32, rgb_u8=None) -> None:
        self._base_gray = np.asarray(gray_f32, dtype=np.float32)
        self._base_rgb = None if rgb_u8 is None else np.asarray(rgb_u8, dtype=np.uint8)
        self.pipeline.clear_cache()  # drop the previous image's cached intermediates

    def set_style(self, name: str) -> None:
        self.panel.set_style(name)

    def render_now(self) -> QImage:
        """Synchronous render (used by tests and the initial paint)."""
        if self._base_gray is None:
            raise RuntimeError("No image loaded")
        self._debounce.stop()
        self._settle.stop()
        self._coalescer.invalidate()  # supersede any in-flight background render
        self._sync_pipeline()
        settings = settings_from_controls(self.panel.state)
        rgb = self.pipeline.render_cached(self._base_gray, settings)
        qimg = numpy_to_qimage(rgb)
        self.last_qimage = qimg
        self.viewport.set_pixmap(QPixmap.fromImage(qimg))
        return qimg

    def schedule_render(self) -> None:
        self._debounce.start(self._debounce_ms)   # fast proxy
        self._settle.start(self._settle_ms)        # full-res once idle

    # ---- internals ----------------------------------------------------------
    def _do_render(self) -> None:
        """Debounce tick: request a fast proxy render."""
        if self._base_gray is None:
            return
        self._render_mode = "proxy"
        token = self._coalescer.request()
        if token is not None:
            self._launch_worker(token)

    def _do_full_render(self) -> None:
        """Settle tick: request the exact full-resolution render."""
        if self._base_gray is None:
            return
        self._render_mode = "full"
        token = self._coalescer.request()
        if token is not None:
            self._launch_worker(token)

    def _launch_worker(self, token: int) -> None:
        """Snapshot the current state + mode and start one background render."""
        self._sync_pipeline()
        settings = settings_from_controls(self.panel.state)
        worker = _RenderWorker(self.pipeline, self._base_gray, settings, token,
                               mode=self._render_mode,
                               proxy_max_side=self._proxy_max_side)
        worker.signals.finished.connect(self._on_rendered)
        worker.signals.failed.connect(self._on_render_failed)
        self._pool.start(worker)

    def _on_rendered(self, qimg: QImage, token: int) -> None:
        # Drop stale/out-of-order results; only the most-recently-started render
        # is painted.
        if self._coalescer.is_current(token):
            self.last_qimage = qimg
            self.viewport.set_pixmap(QPixmap.fromImage(qimg))
        # If state changed while this render was in flight, run one trailing render
        # with the freshest state.
        nxt = self._coalescer.on_finished()
        if nxt is not None:
            self._launch_worker(nxt)

    def _on_render_failed(self, token: int) -> None:
        # A background render raised (already logged in the worker). Don't paint,
        # but release the coalescer so rendering recovers instead of freezing.
        nxt = self._coalescer.on_finished()
        if nxt is not None:
            self._launch_worker(nxt)

    def _on_image_dropped(self, path: str) -> None:
        from PIL import Image

        from ditherzam.imaging import to_gray_f32
        try:
            img = Image.open(path)
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            self.load_array(to_gray_f32(img), rgb)
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
