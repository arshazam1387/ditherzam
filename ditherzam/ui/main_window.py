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
from ditherzam.color.context import ColorContextCache
from ditherzam.render import RenderCancelled, RenderPipeline
from ditherzam.masking.cache import MaskCaches, editor_cache_allocation
from ditherzam.masking.contracts import (InferenceIdentity, ModelIdentity, ProbabilityMap,
                                         source_identity)
from ditherzam.masking.inference_request import InferenceOutcome, InferenceRequest, InferenceTerminal
from ditherzam.masking.inference_scheduler import InferenceScheduler
from ditherzam.masking.ort_adapter import PREPROCESSING_VERSION
from ditherzam.masking.model_assets import ModelAssetError
from ditherzam.masking.settings import MaskTarget, SmartMaskSettings
from ditherzam.masking.render import render_with_mask
from ditherzam.masking.geometry import derive_master_mask, resize_mask_area

from .controls import ControlPanel
from .convert import numpy_to_qimage
from .preview import auto_preview_resolution, preview_cap, render_preview, zoom_preview_bucket
from .preview_preferences import (
    PREVIEW_RESOLUTIONS,
    PreviewPreferences,
    load_preview_preferences,
    save_preview_preferences,
)
from .render_request import MaskContext, RenderKind, RenderRequest
from .mask_workers import InferenceWorker
from .smart_mask_panel import MaskPanelStatus
from .mask_overlay import apply_mask_overlay
from .render_scheduler import RenderScheduler
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
    finished = Signal(QImage, object)   # (image, RenderRequest)
    failed = Signal(object)             # RenderRequest
    cancelled = Signal(object)          # RenderRequest -- distinct from failed: not an error


class _RenderWorker(QRunnable):
    """Runs one render off the GUI thread for an immutable RenderRequest."""

    def __init__(self, pipeline: RenderPipeline, base_gray: np.ndarray,
                 request: RenderRequest, is_cancelled=None, mask_caches=None):
        super().__init__()
        self._pipeline = pipeline
        self._base_gray = base_gray
        self._request = request
        self._is_cancelled = is_cancelled
        self._mask_caches = mask_caches
        self.signals = _RenderSignals()

    def run(self) -> None:
        # A render raising here would otherwise never emit `finished`, so the
        # scheduler's `_busy` flag would stick True and freeze all future renders.
        # Always report an outcome (finished, failed, OR cancelled) so the
        # scheduler recovers -- exactly one terminal signal per run.
        try:
            # Isolate the request's context from later GUI reassignment and
            # synchronous renders on the live editor pipeline.
            pipeline = self._pipeline.snapshot_context(
                self._request.color_engine, self._request.effect_stack)
            # RenderPipeline's cache is internally locked and all cache keys
            # include the request-local engine/effect signatures. Sharing this
            # one bounded owner preserves completed branches across mask-only
            # edits without exposing mutable live pipeline context.
            source = (self._request.source_gray if self._request.source_gray is not None
                      else self._base_gray)
            if self._request.mode == "proxy":
                if self._request.mask_context is None:
                    result = render_preview(
                        pipeline, source, self._request.settings,
                        self._request.target_max_side, is_cancelled=self._is_cancelled)
                else:
                    result = render_preview(
                        pipeline, source, self._request.settings,
                        self._request.target_max_side, is_cancelled=self._is_cancelled,
                        mask_context=self._request.mask_context,
                        mask_caches=self._mask_caches,
                        rendered_identity=self._request.rendered_identity)
            else:
                # A baked base is call-fresh, so it renders uncached; the
                # baked result is cached by render_with_mask instead.
                render = lambda bake=None: (
                    pipeline.render_cached(
                        source, self._request.settings, is_cancelled=self._is_cancelled)
                    if bake is None else
                    pipeline.render(bake(source), self._request.settings,
                                    is_cancelled=self._is_cancelled))
                if self._request.mask_context is None:
                    result = render()
                else:
                    result = render_with_mask(
                        render, self._request.mask_context,
                        caches=self._mask_caches,
                        rendered_identity=self._request.rendered_identity,
                        is_cancelled=self._is_cancelled,
                        target_shape=source.shape[:2])
            if self._request.show_mask_overlay and self._request.mask_context is not None:
                if self._is_cancelled is not None and self._is_cancelled():
                    raise RenderCancelled
                context = self._request.mask_context
                s = context.settings
                mask = derive_master_mask(
                    context.probability, sensitivity=s.sensitivity, target=s.target,
                    invert=s.invert, expansion_px=s.expansion_px,
                    feather_px=s.feather_px, source_shape=context.source_rgba.shape[:2])
                if self._is_cancelled is not None and self._is_cancelled():
                    raise RenderCancelled
                if mask.shape != result.shape[:2]:
                    mask = resize_mask_area(mask, result.shape[:2])
                if self._is_cancelled is not None and self._is_cancelled():
                    raise RenderCancelled
                result = apply_mask_overlay(result, mask)
            qimg = numpy_to_qimage(result)
        except RenderCancelled:
            self.signals.cancelled.emit(self._request)
            return
        except Exception:
            import traceback
            traceback.print_exc()
            self.signals.failed.emit(self._request)
            return
        self.signals.finished.emit(qimg, self._request)


class _DecodeSignals(QObject):
    finished = Signal(object, object, object)   # (gray_f32, rgb_u8, rgba_u8)


class _DecodeWorker(QRunnable):
    """Decodes an image file off the GUI thread for the initial drop/import."""

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self.signals = _DecodeSignals()

    def run(self) -> None:
        from PIL import Image
        try:
            with Image.open(self._path) as img:
                # RGBA is the canonical decoded source.  PIL's RGBA conversion
                # produces straight (not premultiplied) channels, so transparent
                # pixels retain their original RGB values for later compositing.
                rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
            rgb = rgba[..., :3].copy()
            # Preserve the existing PIL RGB->L render input exactly; deriving it
            # from the canonical RGB also keeps alpha out of source luminance.
            gray = np.array(Image.fromarray(rgb, "RGB").convert("L"), dtype=np.float32)
            rgba.setflags(write=False)
        except Exception:
            return  # swallow decode failures, same as the old synchronous path
        self.signals.finished.emit(gray, rgb, rgba)


class ImageEditor(QMainWindow):
    def __init__(self, registry=None, color_engine=None, effect_stack=None,
                 debounce_ms: int = 20, settle_ms: int = 160, zoom_debounce_ms: int = 150,
                 proxy_max_side: int = 640, parent=None, preference_store=None,
                 mask_adapter=None, mask_model: ModelIdentity | None = None,
                 mask_preprocessing_version: str = PREPROCESSING_VERSION):
        super().__init__(parent)
        self.setWindowTitle("ditherzam")
        self._registry = registry or _dither_registry
        # Engines are immutable request snapshots, while their palette-derived
        # data is safe to reuse for the lifetime of this editor.  Keeping the
        # cache editor-owned avoids global cross-document retention.
        self._color_context_cache = ColorContextCache()
        self.pipeline = RenderPipeline(self._registry, color_engine, effect_stack)
        allocation = editor_cache_allocation(False)
        self.pipeline.configure_cache_budget(allocation.render_bytes)
        self._mask_caches = MaskCaches(allocation.mask_bytes)
        self._mask_cache_enabled = False
        self._mask_adapter = mask_adapter
        self._mask_model = mask_model
        self._mask_preprocessing_version = mask_preprocessing_version
        self._mask_scheduler = InferenceScheduler()
        self._mask_closing = False
        self._mask_close_finalizing = False
        self._mask_close_timer: QTimer | None = None
        self._mask_source = None
        self._mask_probability: ProbabilityMap | None = None
        self._base_gray: np.ndarray | None = None
        self._base_rgb: np.ndarray | None = None
        self._base_rgba: np.ndarray | None = None
        self._preview_palette = None
        self._applying_preset = False
        self.last_qimage: QImage | None = None
        self._pool = QThreadPool.globalInstance()
        self._mask_pool = QThreadPool(self)
        self._mask_pool.setMaxThreadCount(1)
        self._debounce_ms = debounce_ms
        self._settle_ms = settle_ms
        self._zoom_debounce_ms = zoom_debounce_ms
        self._proxy_max_side = proxy_max_side
        self._scheduler = RenderScheduler()
        self._preference_store = preference_store or QSettings()
        self.preview_preferences = load_preview_preferences(self._preference_store)
        # Set by the Full Quality Preview action; makes the next settle tick
        # exact/uncapped, then any further edit resets it via schedule_render().
        self._full_preview_requested = False
        # True only for the paint right after a new source loads -- refits the
        # viewport once, then ordinary renders keep the user's zoom/pan.
        self._pending_refit = False
        # Longest-side bucket of the last zoom-triggered render, so we schedule
        # a bucket at most once until the user zooms back out below it.
        self._last_zoom_bucket: int | None = None

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)

        self.viewport = CustomGraphicsView()
        self.panel = ControlPanel()
        self.panel.set_registry(self._registry)
        self.panel.changed.connect(self.schedule_render)
        self.panel.from_image_requested.connect(self._on_from_image_requested)
        self.panel.palette_preview.connect(self._on_palette_preview)
        mask_panel = self.panel.smart_mask_panel
        mask_panel.settings_changed.connect(self._on_mask_settings_changed)
        mask_panel.overlay_changed.connect(lambda _enabled: self.schedule_render())
        mask_panel.redetect_requested.connect(self._request_mask_detection)
        mask_panel.cancel_requested.connect(self._cancel_mask_detection)
        mask_panel.set_availability(source=False, model=self._mask_dependencies_available())
        self.viewport.image_dropped.connect(self._on_image_dropped)
        self.viewport.zoom_changed.connect(self._on_zoom_changed)

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

        # Debounces zoom bursts (wheel/pinch spam) before considering a bucketed
        # refinement render; only fires the optional Task-2.4 zoom refinement.
        self._zoom_debounce = QTimer(self)
        self._zoom_debounce.setSingleShot(True)
        self._zoom_debounce.timeout.connect(self._on_zoom_debounced)

        self.expert_mode = False
        self._install_shortcuts()
        self._wire_preview_preferences()
        self._wire_export()
        self._wire_video()
        self._wire_animation()
        self._refresh_mask_scope_actions()

    def _mask_dependencies_available(self) -> bool:
        return (callable(getattr(self._mask_adapter, "infer", None))
                and isinstance(self._mask_model, ModelIdentity)
                and isinstance(self._mask_preprocessing_version, str)
                and bool(self._mask_preprocessing_version.strip()))

    def _configure_editor_caches(self, enabled: bool) -> None:
        if enabled == self._mask_cache_enabled:
            return
        allocation = editor_cache_allocation(enabled)
        self.pipeline.configure_cache_budget(allocation.render_bytes)
        self._mask_caches = MaskCaches(allocation.mask_bytes)
        self._mask_cache_enabled = enabled

    def _on_mask_settings_changed(self, settings: SmartMaskSettings) -> None:
        self._apply_mask_settings_lifecycle(settings)
        self._refresh_mask_scope_actions()
        self.schedule_render()

    def _apply_mask_settings_lifecycle(self, settings: SmartMaskSettings) -> None:
        """Apply mask ownership/inference effects once after an atomic state update."""
        self._configure_editor_caches(settings.enabled)
        if not settings.enabled:
            self._cancel_mask_detection()
            self._mask_probability = None
            self._mask_source = None
            self._mask_caches.clear()
            self.panel.smart_mask_panel.set_valid_mask_available(False)
        elif settings.target is MaskTarget.WHOLE_IMAGE:
            self._cancel_mask_detection()
        elif self._mask_probability is None:
            if (self._base_rgba is not None and self._mask_dependencies_available()):
                source = source_identity(self._base_rgba)
                identity = InferenceIdentity(
                    source, self._mask_model, self._mask_preprocessing_version, "primary")
                cached = self._mask_caches.get_inference(identity)
                if cached is not None:
                    self._mask_source = source
                    self._mask_probability = cached
                    self.panel.smart_mask_panel.set_valid_mask_available(True)
                    self.panel.smart_mask_panel.set_status(MaskPanelStatus.READY)
                else:
                    self._request_mask_detection()

    def _request_mask_detection(self) -> None:
        settings = self.panel.smart_mask_panel.settings
        if (not settings.enabled or settings.target is MaskTarget.WHOLE_IMAGE
                or self._base_rgba is None):
            return
        if not self._mask_dependencies_available():
            self.panel.smart_mask_panel.set_status(MaskPanelStatus.MODEL_UNAVAILABLE)
            return
        if self._mask_source is None:
            self._mask_source = source_identity(self._base_rgba)
        request = InferenceRequest(
            self._mask_source, self._mask_model, self._mask_preprocessing_version,
            self._base_rgba,
        )
        launch = self._mask_scheduler.request(request)
        self.panel.smart_mask_panel.set_status(MaskPanelStatus.DETECTING)
        if launch is not None:
            self._launch_mask_worker(launch)

    def _cancel_mask_detection(self) -> None:
        self._mask_scheduler.invalidate_source(self._mask_source)
        panel = self.panel.smart_mask_panel
        if panel.status is MaskPanelStatus.DETECTING:
            panel.set_status(MaskPanelStatus.CANCELLED)

    def _launch_mask_worker(self, request: InferenceRequest) -> None:
        worker = InferenceWorker(request, self._mask_adapter)
        for signal in (worker.signals.succeeded, worker.signals.no_subject,
                       worker.signals.cancelled, worker.signals.model_unavailable,
                       worker.signals.failed):
            signal.connect(self._on_mask_terminal)
        worker.signals.progress.connect(self._on_mask_progress)
        self._mask_pool.start(worker)

    def _on_mask_progress(self, request: InferenceRequest, progress: int) -> None:
        if (not self._mask_closing and self._mask_scheduler.is_current(request)
                and request.source == self._mask_source
                and request.model == self._mask_model
                and self.panel.smart_mask_panel.status is MaskPanelStatus.DETECTING):
            self.panel.smart_mask_panel.set_status(MaskPanelStatus.DETECTING, progress)

    def _on_mask_terminal(self, outcome: InferenceOutcome) -> None:
        if self._mask_closing:
            self._mask_scheduler.on_terminal(outcome)
            return
        current = self._mask_scheduler.is_current(outcome.request)
        if current and outcome.request.source == self._mask_source:
            panel = self.panel.smart_mask_panel
            if outcome.terminal is InferenceTerminal.SUCCESS:
                candidate = outcome.result.probability
                if self._mask_caches.put_inference(candidate):
                    self._mask_probability = candidate
                    panel.set_valid_mask_available(True)
                    panel.set_status(MaskPanelStatus.READY)
                    self.schedule_render()
                else:
                    # Cache admission is the publication boundary. Never retain
                    # an unaccounted full-resolution array in editor authority.
                    panel.set_status(MaskPanelStatus.ERROR)
            elif outcome.terminal is InferenceTerminal.NO_SUBJECT:
                panel.set_status(MaskPanelStatus.NO_CLEAR_SUBJECT)
            elif outcome.terminal is InferenceTerminal.CANCELLED:
                panel.set_status(MaskPanelStatus.CANCELLED)
            else:
                panel.set_status(MaskPanelStatus.MODEL_UNAVAILABLE
                                 if isinstance(outcome.error, ModelAssetError)
                                 else MaskPanelStatus.ERROR)
        trailing = self._mask_scheduler.on_terminal(outcome)
        if trailing is not None:
            self.panel.smart_mask_panel.set_status(MaskPanelStatus.DETECTING)
            self._launch_mask_worker(trailing)

    def closeEvent(self, event) -> None:
        """Cancel inference and retire the owned pool without blocking the GUI."""
        if self._mask_close_finalizing:
            super().closeEvent(event)
            return
        self._mask_closing = True
        self._mask_scheduler.invalidate_source(None)
        self._mask_pool.clear()
        if self._mask_pool.activeThreadCount() == 0:
            self._mask_close_finalizing = True
            super().closeEvent(event)
            return
        event.ignore()
        if self._mask_close_timer is None:
            self._mask_close_timer = QTimer(self)
            self._mask_close_timer.setInterval(10)
            self._mask_close_timer.timeout.connect(self._poll_mask_pool_close)
        self._mask_close_timer.start()

    def _poll_mask_pool_close(self) -> None:
        if self._mask_pool.activeThreadCount() != 0:
            return
        if self._mask_close_timer is not None:
            self._mask_close_timer.stop()
        self._mask_close_finalizing = True
        self.close()

    def _current_mask_context(self) -> MaskContext | None:
        settings = self.panel.smart_mask_panel.settings
        probability = self._mask_probability
        if (not settings.enabled or settings.target is MaskTarget.WHOLE_IMAGE
                or probability is None or self._base_rgba is None
                or probability.identity.source != self._mask_source):
            return None
        return MaskContext(self._mask_source, self._base_rgba, probability, settings)

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

        self.view_menu.addSeparator()
        self.view_menu.addAction(self._actions["full_quality_preview"])

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

    def _on_zoom_changed(self, _percent: int) -> None:
        """viewport.zoom_changed fires on every zoom/fit step; debounce bursts
        before evaluating whether a higher-res refinement render is due."""
        self._zoom_debounce.start(self._zoom_debounce_ms)

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
            self._provide_animation_base, self.timeline, seed=0,
            cap_provider=self._policy_cap,
            export_pipeline_provider=self._export_pipeline)
        self.anim_controller.on_frame = self._show_animation_frame
        self.timeline_panel.keyframe_requested.connect(self._add_keyframe_at)
        self.timeline_panel.export_requested.connect(self._export_animation)

    def _provide_animation_base(self):
        if self._base_gray is None:          # no image loaded yet
            return None
        return self._base_gray, self._collect_settings()

    def _show_animation_frame(self, rgb_u8) -> None:
        # rgb_u8 may be a capped raster (async, latest-wins, Task 4.1); pass
        # the full source-logical size so the viewport scales it correctly
        # instead of distorting geometry, same as the still-image capped path.
        qimg = numpy_to_qimage(rgb_u8)
        self.last_qimage = qimg
        self.viewport.set_pixmap(QPixmap.fromImage(qimg), logical_size=self._reference_size())

    def _add_keyframe_at(self, frame_index: int) -> None:
        from ..animation.timeline import Keyframe
        s = self._collect_settings()
        self.timeline.add(
            Keyframe(int(frame_index), "luminance_threshold", s.luminance_threshold))

    def _export_animation(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        if self._base_gray is None:
            return
        if not self._mask_media_allowed("animation"):
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
            cap_provider=self._policy_cap,
            export_pipeline_provider=self._export_pipeline,
            mask_settings_provider=lambda: self.panel.smart_mask_panel.settings,
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
        self.panel.mode_combo.setCurrentText("source")

    def _current_color_engine(self):
        """ColorEngine reflecting the panel's Palette + Mode, or None when off."""
        if self._color_mode() == "off":
            return None
        palette = self._current_palette()
        if palette is None:
            return None
        from ..color.engine import ColorEngine
        return ColorEngine(
            palette,
            self._color_mode(),
            context_cache=self._color_context_cache,
            source_rgb=self._base_rgb if self._color_mode() == "source" else None,
            source_dither=self.panel.state.get("source_dither", 100),
            source_dither_brighten=bool(
                self.panel.state.get("source_dither_brighten", False)),
        )

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

    def _export_pipeline(self) -> RenderPipeline:
        """A dedicated render pipeline snapshotting the current color engine and
        effects, decoupled from the live preview pipeline.

        Exports must render from an immutable context: a later UI edit (which
        reassigns ``self.pipeline.color_engine``/``effect_stack`` via
        ``_sync_pipeline``) or a preview render must never change an in-flight
        export. This builds a fresh pipeline with its own cache, so still, batch,
        video, and animation exports each own their snapshot.
        """
        return RenderPipeline(
            self._registry, self._current_color_engine(),
            self._current_effect_stack())

    def _rendered_rgb(self) -> np.ndarray:
        if self._base_gray is None:
            raise RuntimeError("No image loaded")
        # Capture every mutable editor input exactly once before rendering.
        source_gray = self._base_gray
        settings = self._collect_settings()
        engine = self._current_color_engine()
        effects = self._current_effect_stack()
        context = self._current_mask_context()
        pipeline = RenderPipeline(self._registry, engine, effects, cache_budget_bytes=0)
        if context is None:
            return pipeline.render(source_gray, settings)
        from ditherzam.render import render_context_signature, render_settings_signature
        rendered_identity = (
            context.source, render_settings_signature(settings),
            render_context_signature(engine, effects), source_gray.shape,
            "exact-export-v1",
        )
        return render_with_mask(
            lambda bake=None: pipeline.render(
                source_gray if bake is None else bake(source_gray), settings),
            context,
            caches=self._mask_caches, rendered_identity=rendered_identity,
            target_shape=source_gray.shape[:2])

    def _apply_preset(self, settings, palette, effects, smart_mask=None) -> None:
        # Drop queued renders before mutating several signal-producing controls.
        self._debounce.stop()
        self._settle.stop()
        self._zoom_debounce.stop()
        self._scheduler.invalidate()
        self._applying_preset = True
        panel = self.panel
        panel.blockSignals(True)
        self.glow_panel.blockSignals(True)
        try:
            for key in ("contrast", "midtones", "highlights", "luminance_threshold", "blur"):
                value = int(getattr(settings, key))
                panel.state[key] = value
                if key in panel._sliders:
                    panel._sliders[key].setValue(value)
            panel.saturation_slider.setValue(int(settings.saturation))
            panel.scale_slider.setValue(int(settings.scale))
            panel.invert_toggle.setChecked(bool(settings.invert))
            panel.preview_toggle.setChecked(bool(settings.preview_disabled))
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
            gp = self.glow_panel
            gp.enable_toggle.setChecked(bool(glow_state))
            if glow_state:
                for key, slider in gp._sliders.items():
                    slider.setValue(int(glow_state.get(key, GLOW_DEFAULTS[key])))
            if palette is not None:
                panel.set_working_palette(palette)
            panel.set_style(settings.style, settings.params)
            if smart_mask is not None:
                panel.smart_mask_panel.set_settings(smart_mask)
        finally:
            panel.blockSignals(False)
            self.glow_panel.blockSignals(False)
            self._applying_preset = False
        if smart_mask is not None:
            self._apply_mask_settings_lifecycle(smart_mask)
        self._refresh_mask_scope_actions()
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
            self.panel.smart_mask_panel.settings,
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
        contents = preset_to_settings(self._preset_manager.load(name))
        self._apply_preset(contents.settings, contents.palette, contents.effects,
                           contents.smart_mask)

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
            self.panel.smart_mask_panel.settings,
        )
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(preset, f, sort_keys=False, allow_unicode=True)

    def _on_export_raster(self, file_filter, ext):
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog
        if self._base_gray is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", file_filter)
        if not path:
            return
        rendered = self._rendered_rgb()
        selected_ext = Path(path).suffix.lower()
        if selected_ext in (".jpg", ".jpeg") and rendered.ndim == 3 and rendered.shape[2] == 4:
            if not getattr(self, "_jpeg_flatten_notice_shown", False):
                self.statusBar().showMessage(
                    "JPEG does not support transparency; transparent pixels are flattened onto white.",
                    8000,
                )
                self._jpeg_flatten_notice_shown = True
        save_raster(rendered, path)

    def _on_export_svg(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        if self._base_gray is None:
            return
        if not self._mask_media_allowed("SVG"):
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
        if not self._mask_media_allowed("batch"):
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        from pathlib import Path
        out = Path(folder) / "batch_processed"
        processed, skipped = batch_process(
            folder, out, self._collect_settings(),
            self._export_pipeline(), self._reference_size(),
        )
        QMessageBox.information(
            self, "Batch",
            f"Processed {processed} images. Skipped {skipped} images.",
        )

    # ---- public API ---------------------------------------------------------
    def load_array(self, gray_f32, rgb_u8=None, rgba_u8=None) -> None:
        """Atomically replace the source, defensively copying external RGBA.

        ``gray_f32`` and ``rgb_u8`` remain the render and Source Colors inputs
        respectively.  RGBA retention is additional source authority only.
        """
        self._replace_source_arrays(gray_f32, rgb_u8, rgba_u8, adopt_decoded_rgba=False)

    def _replace_source_arrays(self, gray_f32, rgb_u8, rgba_u8,
                               *, adopt_decoded_rgba: bool) -> None:
        """Validate a replacement and optionally accept private decoder ownership."""
        if not isinstance(gray_f32, np.ndarray) or gray_f32.dtype != np.float32:
            raise TypeError("gray_f32 must be a float32 ndarray")
        gray = gray_f32
        if gray.ndim != 2 or 0 in gray.shape:
            raise ValueError("gray_f32 must have non-empty shape (H, W)")
        if not np.isfinite(gray).all() or np.any(gray < 0) or np.any(gray > 255):
            raise ValueError("gray_f32 values must be finite and in [0, 255]")
        h, w = gray.shape

        rgb = rgb_u8
        if rgb is not None:
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
                raise TypeError("rgb_u8 must be a uint8 ndarray")
            if rgb.shape != (h, w, 3):
                raise ValueError("rgb_u8 must have non-empty shape (H, W, 3) matching gray_f32")

        if rgba_u8 is not None:
            if not isinstance(rgba_u8, np.ndarray) or rgba_u8.dtype != np.uint8:
                raise TypeError("rgba_u8 must be a uint8 ndarray")
            rgba_input = rgba_u8
            if rgba_input.shape != (h, w, 4):
                raise ValueError("rgba_u8 must have non-empty shape (H, W, 4) matching gray_f32")
            if rgb is not None and not np.array_equal(rgba_input[..., :3], rgb):
                raise ValueError("rgba_u8 RGB channels must match rgb_u8")
            # Only the private decode receiver may transfer ownership. Public
            # callers retain access to their arrays and can reverse NumPy's
            # writeable flag, so load_array always makes a defensive copy.
            decoded_transfer = (
                adopt_decoded_rgba
                and rgba_input.flags.owndata
                and not rgba_input.flags.writeable
                and rgba_input.flags.c_contiguous
                and rgba_input.base is None
            )
            rgba = rgba_input if decoded_transfer else np.array(
                rgba_input, dtype=np.uint8, order="C", copy=True)
        else:
            source_rgb = rgb
            if source_rgb is None:
                gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
                source_rgb = np.repeat(gray_u8[..., None], 3, axis=2)
            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = source_rgb
            rgba[..., 3] = 255
        rgba.setflags(write=False)

        # Invalidate the old identity before publishing any field of the new
        # source so an in-flight terminal can never attach to replacement data.
        old_source = self._mask_source
        self._mask_scheduler.invalidate_source(None)
        if old_source is not None:
            self._mask_caches.clear_source(old_source)
        self._mask_probability = None

        # All conversion and validation above must succeed before any source
        # field changes; callers never observe a partially replaced document.
        self._base_gray = gray
        self._base_rgb = rgb
        self._base_rgba = rgba
        self._mask_source = None
        self.pipeline.clear_cache()  # drop the previous image's cached intermediates
        self._pending_refit = True  # a new source: the next paint should fit
        mask_panel = self.panel.smart_mask_panel
        mask_panel.reset_for_source()
        mask_panel.set_availability(source=True, model=self._mask_dependencies_available())
        settings = mask_panel.settings
        self._configure_editor_caches(settings.enabled)
        if settings.enabled and settings.target is not MaskTarget.WHOLE_IMAGE:
            self._request_mask_detection()
        self._refresh_mask_scope_actions()

    def set_style(self, name: str) -> None:
        self.panel.set_style(name)

    def render_now(self) -> QImage:
        """Synchronous render (used by tests and the initial paint)."""
        if self._base_gray is None:
            raise RuntimeError("No image loaded")
        self._debounce.stop()
        self._settle.stop()
        self._scheduler.invalidate()  # supersede any in-flight background render
        self._sync_pipeline()
        request = self._build_request(
            RenderKind.FULL, target_max_side=max(self._reference_size()))
        pipeline = self.pipeline.snapshot_context(
            request.color_engine, request.effect_stack)
        renderer = lambda bake=None: (
            pipeline.render_cached(request.source_gray, request.settings)
            if bake is None else
            pipeline.render(bake(request.source_gray), request.settings))
        if request.mask_context is None:
            result = renderer()
        else:
            result = render_with_mask(
                renderer, request.mask_context, caches=self._mask_caches,
                rendered_identity=request.rendered_identity,
                target_shape=request.source_gray.shape[:2])
        if request.show_mask_overlay and request.mask_context is not None:
            context = request.mask_context
            s = context.settings
            mask = derive_master_mask(
                context.probability, sensitivity=s.sensitivity, target=s.target,
                invert=s.invert, expansion_px=s.expansion_px,
                feather_px=s.feather_px, source_shape=context.source_rgba.shape[:2])
            result = apply_mask_overlay(result, mask)
        qimg = numpy_to_qimage(result)
        self.last_qimage = qimg
        refit = self._pending_refit
        self._pending_refit = False
        self.viewport.set_pixmap(QPixmap.fromImage(qimg),
                                 logical_size=self._reference_size(), refit=refit)
        return qimg

    def schedule_render(self) -> None:
        if self._applying_preset:
            return
        self._full_preview_requested = False       # any edit returns to the cap
        self._last_zoom_bucket = None               # a normal edit re-settles the baseline
        self._debounce.start(self._debounce_ms)   # fast proxy
        self._settle.start(self._settle_ms)        # full-res once idle

    # ---- internals ----------------------------------------------------------
    def _policy_cap(self) -> int:
        """Settled longest-side cap from the current preview preference + viewport."""
        w, h = self._reference_size()
        source_longest = max(w, h)
        if source_longest <= 0:
            return 1440  # no image loaded yet; a reasonable default
        resolution = self.preview_preferences.resolution
        if resolution == "Auto":
            vp = self.viewport.viewport()
            vw, vh = vp.width(), vp.height()
            if vw <= 0 or vh <= 0:
                return preview_cap("Auto", source_longest)
            dpr = self.viewport.devicePixelRatioF()
            return auto_preview_resolution((h, w), (vw, vh), dpr)
        return preview_cap(resolution, source_longest)

    def _build_request(self, kind: RenderKind, target_max_side: int | None = None) -> RenderRequest:
        """Snapshot current UI state into an immutable request. ``generation``
        is a placeholder here -- the scheduler stamps the real value in
        ``request()``/``on_finished()``. A non-``None`` ``target_max_side``
        override wins over the kind-derived cap (used by zoom refinement to
        request a specific bucket)."""
        self._sync_pipeline()
        source_gray = self._base_gray
        settings = settings_from_controls(self.panel.state)
        logical_size = ((0, 0) if source_gray is None else
                        (int(source_gray.shape[1]), int(source_gray.shape[0])))
        if target_max_side is None:
            if kind is RenderKind.FULL:
                target_max_side = max(logical_size)
            elif kind is RenderKind.DRAG:
                target_max_side = min(self._proxy_max_side, self._policy_cap())
            else:  # SETTLE, ZOOM
                target_max_side = self._policy_cap()
        mask_context = self._current_mask_context()
        color_engine = self.pipeline.color_engine
        effect_stack = self.pipeline.effect_stack
        show_overlay = self.panel.smart_mask_panel.overlay_check.isChecked()
        return RenderRequest(
            generation=0,
            kind=kind,
            settings=settings,
            source_id=id(source_gray),
            target_max_side=target_max_side,
            logical_size=logical_size,
            color_engine=color_engine,
            effect_stack=effect_stack,
            mask_context=mask_context,
            source_gray=source_gray,
            show_mask_overlay=(show_overlay and mask_context is not None),
        )

    def _mask_media_allowed(self, kind: str) -> bool:
        """Defense-in-depth gate before any unsupported export work/dialog."""
        from PySide6.QtWidgets import QMessageBox
        from ditherzam.masking.scope import mask_allows_media, unsupported_mask_message
        if mask_allows_media(kind, self.panel.smart_mask_panel.settings):
            return True
        QMessageBox.warning(self, "Smart Mask", unsupported_mask_message(kind))
        return False

    def _refresh_mask_scope_actions(self) -> None:
        """Expose the scope policy in actions while retaining handler guards."""
        from ditherzam.masking.scope import mask_allows_media, unsupported_mask_message
        settings = self.panel.smart_mask_panel.settings
        blocked = not mask_allows_media("svg", settings)
        message = unsupported_mask_message("this media") if blocked else ""
        for key in ("export_svg", "batch_folder"):
            action = getattr(self, "_export_actions", {}).get(key)
            if action is not None:
                action.setEnabled(not blocked)
                action.setToolTip(message)
                action.setStatusTip(message)
        panel = getattr(self, "timeline_panel", None)
        if panel is not None:
            panel.export_btn.setEnabled(not blocked)
            panel.export_btn.setToolTip(message)
        controller = getattr(self, "video_controller", None)
        if controller is not None:
            controller.refresh_mask_scope()

    def _do_render(self) -> None:
        """Debounce tick: request a fast proxy render."""
        if self._base_gray is None:
            return
        req = self._scheduler.request(self._build_request(RenderKind.DRAG))
        if req is not None:
            self._launch_worker(req)

    def _do_full_render(self) -> None:
        """Settle tick: request the settled (policy-capped) render, or an exact
        full render if Full Quality Preview was requested."""
        if self._base_gray is None:
            return
        kind = RenderKind.FULL if self._full_preview_requested else RenderKind.SETTLE
        req = self._scheduler.request(self._build_request(kind))
        if req is not None:
            self._launch_worker(req)

    def _do_full_quality_preview(self) -> None:
        """View > Full Quality Preview: force one exact, uncapped render now."""
        if self._base_gray is None:
            return
        self._full_preview_requested = True
        req = self._scheduler.request(self._build_request(RenderKind.FULL))
        if req is not None:
            self._launch_worker(req)

    def _zoom_required_pixels(self) -> int:
        """Device pixels the full source spans at the current viewport zoom."""
        source_longest = max(self._reference_size())
        scale = self.viewport.transform().m11() * self.viewport.devicePixelRatioF()
        return int(round(source_longest * scale))

    def _on_zoom_debounced(self) -> None:
        """Zoom-debounce tick: optionally schedule one bucketed refinement
        render, at most once per bucket, respecting the resolution ceiling."""
        if not self.preview_preferences.rerender_on_zoom:
            return
        if self._base_gray is None:
            return
        source_longest = max(self._reference_size())
        ceiling = preview_cap(self.preview_preferences.resolution, source_longest)
        baseline = self._policy_cap()
        required = self._zoom_required_pixels()
        bucket = zoom_preview_bucket(baseline, required, ceiling, source_longest)
        if bucket <= baseline:
            self._last_zoom_bucket = None  # zoomed back out to the settled baseline
            return
        if bucket == self._last_zoom_bucket:
            return  # already rendered this bucket
        self._last_zoom_bucket = bucket
        req = self._scheduler.request(self._build_request(RenderKind.ZOOM, target_max_side=bucket))
        if req is not None:
            self._launch_worker(req)

    def _launch_worker(self, request: RenderRequest) -> None:
        """Start one background render for an already-stamped request."""
        worker = _RenderWorker(self.pipeline, self._base_gray, request,
                                is_cancelled=lambda: self._scheduler.should_cancel(request),
                                mask_caches=self._mask_caches)
        worker.signals.finished.connect(self._on_rendered)
        worker.signals.failed.connect(self._on_render_failed)
        worker.signals.cancelled.connect(self._on_render_cancelled)
        self._pool.start(worker)

    def _on_rendered(self, qimg: QImage, request: RenderRequest) -> None:
        # Drop stale/out-of-order results; only the most-recently-started render
        # is painted.
        if self._scheduler.is_current(request):
            self.last_qimage = qimg
            refit = self._pending_refit
            self._pending_refit = False
            self.viewport.set_pixmap(QPixmap.fromImage(qimg),
                                     logical_size=request.logical_size, refit=refit)
        # If state changed while this render was in flight, run one trailing render
        # with the freshest, highest-priority state.
        nxt = self._scheduler.on_finished()
        if nxt is not None:
            self._launch_worker(nxt)

    def _on_render_failed(self, request: RenderRequest) -> None:
        # A background render raised (already logged in the worker). Don't paint,
        # but release the scheduler so rendering recovers instead of freezing.
        nxt = self._scheduler.on_finished()
        if nxt is not None:
            self._launch_worker(nxt)

    def _on_render_cancelled(self, request: RenderRequest) -> None:
        # Obsolete by design, not an error -- don't paint, but still release the
        # scheduler and run the trailing request that made this one obsolete.
        nxt = self._scheduler.on_finished()
        if nxt is not None:
            self._launch_worker(nxt)

    def _on_image_dropped(self, path: str) -> None:
        worker = _DecodeWorker(path)
        worker.signals.finished.connect(self._on_image_decoded)
        self._decode_worker = worker  # keep the signals QObject alive until it fires
        self._pool.start(worker)

    def _on_image_decoded(self, gray_f32, rgb_u8, rgba_u8) -> None:
        """GUI-thread slot: decode finished off-thread; paint a capped preview
        immediately instead of blocking on a synchronous exact render."""
        self._replace_source_arrays(
            gray_f32, rgb_u8, rgba_u8, adopt_decoded_rgba=True)
        self.schedule_render()

    def _install_shortcuts(self) -> None:
        hk = get_hotkeys(sys.platform)
        self._actions: dict[str, QAction] = {}
        bindings = {
            "zoom_in": self.viewport.zoom_in,
            "zoom_out": self.viewport.zoom_out,
            "zoom_reset": self.viewport.reset_zoom,
            "full_quality_preview": self._do_full_quality_preview,
        }
        for action_name, slot in bindings.items():
            act = QAction(self)
            act.setShortcut(QKeySequence(hk[action_name]))
            act.triggered.connect(slot)
            self.addAction(act)
            self._actions[action_name] = act
