from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QObject, QRectF, QRunnable, QSettings, Qt, QThreadPool, QTimer, Signal,
)
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
from ditherzam.masking.render import derive_render_mask, render_with_mask

from .controls import ControlPanel
from .convert import numpy_to_qimage, qimage_to_numpy_rgba
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
from ..diagnostics import default_log_path, log_action, log_duration


_LOG = logging.getLogger(__name__)


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
        started = time.perf_counter()
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
                # Reuse the render's preview-resolution derivation + derived cache
                # so a capped overlay never re-derives the master at source res.
                mask = derive_render_mask(
                    context, result.shape[:2], caches=self._mask_caches,
                    is_cancelled=self._is_cancelled)
                if self._is_cancelled is not None and self._is_cancelled():
                    raise RenderCancelled
                result = apply_mask_overlay(result, mask)
            qimg = numpy_to_qimage(result)
        except RenderCancelled:
            self.signals.cancelled.emit(self._request)
            return
        except Exception:
            _LOG.exception(
                "render_failed generation=%s kind=%s mode=%s source_shape=%s",
                self._request.generation, self._request.kind.value,
                self._request.mode, tuple(self._base_gray.shape),
            )
            self.signals.failed.emit(self._request)
            return
        log_duration(
            _LOG, "render_complete", started,
            generation=self._request.generation,
            kind=self._request.kind.value,
            mode=self._request.mode,
            output=f"{qimg.width()}x{qimg.height()}",
        )
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
        started = time.perf_counter()
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
            _LOG.exception("image_decode_failed path=%s", self._path)
            return
        log_duration(
            _LOG, "image_decode_complete", started,
            path=self._path, size=f"{rgba.shape[1]}x{rgba.shape[0]}",
        )
        self.signals.finished.emit(gray, rgb, rgba)


class ImageEditor(QMainWindow):
    def __init__(self, registry=None, color_engine=None, effect_stack=None,
                 debounce_ms: int = 20, settle_ms: int = 160, zoom_debounce_ms: int = 150,
                 proxy_max_side: int = 640, parent=None, preference_store=None,
                 mask_adapter=None, mask_model: ModelIdentity | None = None,
                 mask_preprocessing_version: str = PREPROCESSING_VERSION,
                 diagnostic_log_path: str | Path | None = None):
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
        # QThreadPool owns the C++ QRunnable, not the Python wrapper or its
        # Python-owned signal object.  Keep every inference worker alive until
        # its queued terminal signal reaches the GUI thread.  Dropping the last
        # wrapper reference while ONNX Runtime is executing can tear down
        # Shiboken/Python state underneath the native call.
        self._mask_workers: set[InferenceWorker] = set()
        self._mask_closing = False
        self._mask_close_finalizing = False
        self._mask_close_timer: QTimer | None = None
        self._mask_source = None
        self._mask_probability: ProbabilityMap | None = None
        self._decode_generation = 0
        self._decode_workers: set[_DecodeWorker] = set()
        # QThreadPool does not retain the Python QRunnable wrapper reliably
        # through queued signal delivery.  Keep editor preview workers (and
        # their QObject signal sources) alive until a terminal signal arrives.
        self._render_workers: set[_RenderWorker] = set()
        self._render_closing = False
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
        self._diagnostic_log_path = Path(
            diagnostic_log_path or default_log_path()
        )
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
        mask_panel.overlay_changed.connect(self._on_mask_overlay_changed)
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
        self._wire_composition()
        self._wire_layers()
        self._wire_debug_menu()
        self._refresh_mask_scope_actions()
        self._action_settings_snapshot = self._control_action_snapshot()

    def _control_action_snapshot(self) -> dict[str, object]:
        """Return bounded semantic control state, never pixels or payloads."""
        state = self.panel.state
        scalar_keys = (
            "style", "scale", "contrast", "midtones", "highlights",
            "luminance_threshold", "blur", "preview_disabled", "invert",
            "palette", "color_mode", "source_dither",
            "source_dither_brighten", "color_mapping", "depth", "saturation",
        )
        snapshot = {key: state.get(key) for key in scalar_keys}
        snapshot["params"] = tuple(sorted(
            (str(key), int(value))
            for key, value in state.get("params", {}).items()
        ))
        snapshot["effects"] = tuple(str(value) for value in state.get("effects", ()))
        snapshot["glow"] = tuple(sorted(
            (str(key), value) for key, value in self.glow_panel.state.items()
        ))
        mask = self.panel.smart_mask_panel.settings
        snapshot["smart_mask"] = (
            mask.enabled, mask.target.value, mask.sensitivity, mask.feather_px,
            mask.expansion_px, mask.invert, mask.outside.value, mask.bake_fill,
        )
        return snapshot

    def _log_settled_control_changes(self) -> None:
        """Coalesce a burst of UI changes into one settled semantic action."""
        current = self._control_action_snapshot()
        previous = self._action_settings_snapshot
        changes = {
            key: {"from": previous.get(key), "to": value}
            for key, value in current.items()
            if previous.get(key) != value
        }
        if changes:
            log_action("controls.settled_change", changes=changes)
            self._action_settings_snapshot = current

    def _wire_debug_menu(self) -> None:
        self.debug_menu = self.menuBar().addMenu("&Debug")
        self.debug_log_action = QAction("Program Notes…", self)
        self.debug_log_action.setObjectName("program_notes_action")
        self.debug_log_action.triggered.connect(self._show_program_notes)
        self.debug_menu.addAction(self.debug_log_action)

    def _show_program_notes(self) -> None:
        from .diagnostics_dialog import DiagnosticsDialog
        DiagnosticsDialog(self._diagnostic_log_path, self).exec()

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

    def _on_mask_overlay_changed(self, _enabled: bool) -> None:
        """Refresh inspection pixels without invalidating layer-owned state."""
        controller = getattr(self, "layers_controller", None)
        if controller is not None and controller.document is not None:
            controller.request_preview()
            return
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
        log_action(
            "smart_mask.detect_requested",
            source_width=request.source.width,
            source_height=request.source.height,
            model=request.model.model_id,
            queued=launch is None,
        )
        if launch is not None:
            self._launch_mask_worker(launch)

    def _cancel_mask_detection(self) -> None:
        self._mask_scheduler.invalidate_source(self._mask_source)
        panel = self.panel.smart_mask_panel
        if panel.status is MaskPanelStatus.DETECTING:
            panel.set_status(MaskPanelStatus.CANCELLED)
            log_action("smart_mask.detect_cancelled")

    def _launch_mask_worker(self, request: InferenceRequest) -> None:
        worker = InferenceWorker(request, self._mask_adapter)
        self._mask_workers.add(worker)
        for signal in (worker.signals.succeeded, worker.signals.no_subject,
                       worker.signals.cancelled, worker.signals.model_unavailable,
                       worker.signals.failed):
            signal.connect(self._on_mask_terminal)
            signal.connect(
                lambda _outcome, retained=worker:
                self._mask_workers.discard(retained)
            )
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
        log_action(
            "smart_mask.detect_terminal",
            terminal=outcome.terminal.value,
            current=current,
            source_width=outcome.request.source.width,
            source_height=outcome.request.source.height,
            error=None if outcome.error is None else type(outcome.error).__name__,
        )
        if current and outcome.request.source == self._mask_source:
            panel = self.panel.smart_mask_panel
            if outcome.terminal is InferenceTerminal.SUCCESS:
                candidate = outcome.result.probability
                if self._mask_caches.put_inference(candidate):
                    if self._layer_transform_active():
                        # Retain the completed local inference in the bounded
                        # cache without mutating the active layer transaction.
                        trailing = self._mask_scheduler.on_terminal(outcome)
                        if trailing is not None:
                            self._launch_mask_worker(trailing)
                        return
                    self._mask_probability = candidate
                    panel.set_valid_mask_available(True)
                    panel.set_status(MaskPanelStatus.READY)
                    self.schedule_render()
                    controller = getattr(self, "composition_controller", None)
                    if controller is not None and controller.composition is not None:
                        controller.request_frame(
                            self.composition_panel.frame_slider.value())
                    layers_controller = getattr(
                        self, "layers_controller", None)
                    if (layers_controller is not None
                            and layers_controller.stack.layers):
                        layers_controller.request_preview()
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
        if self._layer_transform_active():
            self._cancel_layer_transform()
        controller = getattr(self, "composition_controller", None)
        if controller is not None:
            controller.shutdown()
        layers_controller = getattr(self, "layers_controller", None)
        if layers_controller is not None:
            layers_controller.shutdown()
        if self._mask_close_finalizing:
            super().closeEvent(event)
            return
        self._render_closing = True
        self._debounce.stop()
        self._settle.stop()
        self._zoom_debounce.stop()
        self._scheduler.invalidate()
        log_action(
            "app.close_requested",
            mask_workers=self._mask_pool.activeThreadCount(),
            render_workers=len(self._render_workers),
        )
        self._mask_closing = True
        self._mask_scheduler.invalidate_source(None)
        self._mask_pool.clear()
        if (
            self._mask_pool.activeThreadCount() == 0
            and not self._render_workers
        ):
            self._mask_close_finalizing = True
            log_action("app.close_completed", deferred=False)
            super().closeEvent(event)
            return
        event.ignore()
        if self._mask_close_timer is None:
            self._mask_close_timer = QTimer(self)
            self._mask_close_timer.setInterval(10)
            self._mask_close_timer.timeout.connect(self._poll_mask_pool_close)
        self._mask_close_timer.start()

    def _poll_mask_pool_close(self) -> None:
        if (
            self._mask_pool.activeThreadCount() != 0
            or self._render_workers
        ):
            return
        if self._mask_close_timer is not None:
            self._mask_close_timer.stop()
        self._mask_close_finalizing = True
        log_action("app.close_completed", deferred=True)
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
        log_action("preview.resolution_changed", resolution=resolution)
        self._save_preview_preferences_and_schedule()

    def _set_rerender_on_zoom(self, enabled: bool) -> None:
        self.preview_preferences = PreviewPreferences(
            self.preview_preferences.resolution, enabled
        )
        log_action("preview.zoom_rerender_changed", enabled=bool(enabled))
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
        self.animation_dock = QDockWidget("Animation", self)
        self.animation_dock.setWidget(self.timeline_panel)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.animation_dock)

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

    # ---- still-image Look Composer -----------------------------------------
    def _wire_composition(self) -> None:
        from PySide6.QtWidgets import QDockWidget
        from .composition_controller import CompositionController
        from .composition_panel import CompositionPanel

        self.composition_panel = CompositionPanel(parent=self)
        self.composition_dock = QDockWidget("Look Composer", self)
        self.composition_dock.setWidget(self.composition_panel)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.composition_dock)
        self.tabifyDockWidget(self.animation_dock, self.composition_dock)

        self.composition_controller = CompositionController(
            self.composition_panel,
            self._registry,
            preset_provider=self._current_composition_preset,
            source_provider=self._provide_composition_source,
            cap_provider=self._policy_cap,
            frame_sink=self._show_composition_frame,
            parent=self,
        )
        self.composition_panel.export_requested.connect(
            self._export_composition_frame)
        self.composition_dock.visibilityChanged.connect(
            lambda visible: (
                None if visible else self.composition_panel.stop_playback()))

    def _current_composition_preset(self) -> dict:
        return settings_to_preset(
            self._collect_settings(),
            self._current_palette(),
            self._current_effect_stack(),
            self._color_mode(),
            self.panel.smart_mask_panel.settings,
            source_dither=int(self.panel.state.get("source_dither", 100)),
            source_dither_brighten=bool(
                self.panel.state.get("source_dither_brighten", False)),
        )

    def _provide_composition_source(self):
        if self._base_gray is None or self._base_rgba is None:
            return None
        probability = self._mask_probability
        if probability is not None:
            identity = source_identity(self._base_rgba)
            if probability.identity.source != identity:
                probability = None
        return self._base_gray, self._base_rgba, probability

    def _show_composition_frame(self, rgb_or_rgba_u8) -> None:
        # Composer interaction owns the viewport until another explicit editor
        # request: retire queued editor work so it cannot paint over this frame.
        self._debounce.stop()
        self._settle.stop()
        self._zoom_debounce.stop()
        self._scheduler.invalidate()
        qimg = numpy_to_qimage(rgb_or_rgba_u8)
        self.last_qimage = qimg
        self.viewport.set_pixmap(
            QPixmap.fromImage(qimg),
            logical_size=self._reference_size(),
            refit=False,
        )

    def _export_composition_frame(self, _frame_index: int) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Current Composition Frame",
            "composition-frame.png",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)",
        )
        if path:
            self.composition_controller.export_current(path)

    # ---- same-canvas spatial layer stack -----------------------------------
    def _wire_layers(self) -> None:
        from PySide6.QtWidgets import QDockWidget
        from .layers_controller import LayersController
        from .layers_panel import LayersPanel

        self.layers_panel = LayersPanel(parent=self)
        self.layers_dock = QDockWidget("Layers", self)
        self.layers_dock.setWidget(self.layers_panel)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.layers_dock)

        self.layers_controller = LayersController(
            self.layers_panel,
            self._registry,
            preset_provider=self._current_composition_preset,
            source_provider=self._provide_composition_source,
            cap_provider=self._layer_policy_cap,
            frame_sink=self._show_layer_frame,
            apply_preset=self._apply_layer_preset,
            apply_source=self._apply_layer_source,
            mutation_guard=self._guard_layer_transform,
            proxy_sink=self._install_layer_transform_proxy,
            proxy_geometry_sink=self._update_layer_transform_proxy,
            proxy_clear=self.viewport.clear_transform_proxy,
            mask_caches_provider=lambda: self._mask_caches,
            smart_mask_readiness_provider=self._layer_smart_mask_readiness,
            mask_stroke_sink=self._install_mask_stroke_roi,
            mask_stroke_clear=self.viewport.clear_mask_stroke_overlay,
            refinement_preview_sink=self._show_smart_refinement_preview,
            refinement_preview_clear=lambda: None,
            selection_overlay_sink=self.viewport.install_selection_overlay,
            selection_overlay_clear=self.viewport.clear_selection_overlay,
            parent=self,
        )
        self.layers_panel.export_requested.connect(self._export_layers)
        self.layers_panel.transform_mode_requested.connect(
            self._start_layer_transform)
        self.layers_panel.transform_confirmed.connect(
            self._confirm_layer_transform)
        self.layers_panel.transform_cancelled.connect(
            self._cancel_layer_transform)
        self.layers_panel.name_changed.connect(
            lambda *_args: QTimer.singleShot(
                0, self._refresh_active_layer_mask_scope))
        self.layers_controller.place_requested.connect(self._place_layer_image)
        self.layers_panel.import_mask_requested.connect(
            self._import_active_raster_mask)
        self.layers_panel.export_mask_requested.connect(
            self._export_active_raster_mask)
        self.layers_panel.selection_tool_requested.connect(
            self._arm_selection_tool)
        self.layers_panel.selection_clear_requested.connect(
            self._clear_temporary_selection)
        self.layers_panel.selection_from_mask_requested.connect(
            self.layers_controller.create_mask_from_selection)
        self.layers_panel.selection_refine_requested.connect(
            self.layers_controller.refine_temporary_selection)
        self.layers_panel.color_range_pick_requested.connect(
            lambda: self._arm_selection_tool("color_range"))
        self.layers_panel.color_range_changed.connect(
            self.layers_controller.update_color_range_selection)
        self.layers_panel.color_range_confirmed.connect(
            self._confirm_color_range_selection)
        self.layers_panel.color_range_cancelled.connect(
            self._cancel_color_range_selection)
        self.layers_panel.brush_settings_changed.connect(
            self._set_mask_brush_settings)
        self.layers_panel.brush_mode_changed.connect(
            self._set_mask_brush_mode)
        self.layers_panel.mask_paint_requested.connect(
            self._set_mask_paint_enabled)
        self.layers_panel.pointer_requested.connect(
            self._activate_pointer_tool)
        self.layers_panel.gradient_tool_requested.connect(
            self._arm_gradient_tool)
        self.viewport.layer_drag_started.connect(self._begin_layer_drag)
        self.viewport.layer_dragged.connect(self._drag_active_layer)
        self.viewport.layer_drag_finished.connect(self._finish_layer_drag)
        self.viewport.layer_resize_started.connect(self._begin_layer_resize)
        self.viewport.layer_resized.connect(self._resize_active_layer)
        self.viewport.layer_resize_finished.connect(self._finish_layer_resize)
        self.viewport.layer_nudge_requested.connect(self._nudge_active_layer)
        self.viewport.layer_transform_confirm_requested.connect(
            self._confirm_layer_transform)
        self.viewport.layer_transform_cancel_requested.connect(
            self._cancel_layer_transform)
        self.layers_controller.active_geometry_changed.connect(
            self._sync_layer_drag_target)
        from ditherzam.layers import BrushMode, BrushSettings
        self._mask_brush_settings = BrushSettings(
            32.0, 100, 100, BrushMode.REVEAL)
        self.viewport.mask_brush_started.connect(
            self._begin_mask_brush_stroke)
        self.viewport.mask_brush_moved.connect(
            self.layers_controller.continue_mask_brush_stroke)
        self.viewport.mask_brush_finished.connect(
            self.layers_controller.finish_mask_brush_stroke)
        self.viewport.mask_brush_cancel_requested.connect(
            self.layers_controller.cancel_mask_brush_stroke)
        self.viewport.mask_brush_size_delta_requested.connect(
            self._change_mask_brush_size)
        self.viewport.mask_brush_mode_swap_requested.connect(
            self._swap_mask_brush_mode)
        self.viewport.selection_dragged.connect(
            self._apply_temporary_selection_drag)
        self.viewport.selection_path_completed.connect(
            self._apply_temporary_selection_path)
        self.viewport.color_range_picked.connect(
            self._pick_color_range_selection)
        self.viewport.selection_cancel_requested.connect(
            self._cancel_color_range_selection)
        self.viewport.gradient_dragged.connect(
            self._apply_gradient_drag)
        if hasattr(self.layers_panel, "edit_target_requested"):
            self.layers_panel.edit_target_requested.connect(
                self._set_viewport_edit_target)
        self.edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = QAction("Undo", self)
        self.undo_action.setObjectName("undo_layer_document_action")
        self.undo_action.setShortcut(
            QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.layers_controller.undo)
        self.edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setObjectName("redo_layer_document_action")
        self.redo_action.setShortcut(
            QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.layers_controller.redo)
        self.edit_menu.addAction(self.redo_action)
        self.layers_controller.history_changed.connect(
            self._refresh_layer_history_actions)
        self._refresh_layer_history_actions()
        self._layer_transform_session: (
            tuple[str, tuple[int, int, int, int]] | None
        ) = None
        self._layer_drag_origin: tuple[int, int] | None = None
        self._layer_drag_id: str | None = None
        self._layer_resize_origin: tuple[int, int, int, int] | None = None
        self._layer_resize_id: str | None = None

    def _set_viewport_edit_target(self, _layer_id: str, kind: str) -> None:
        mask = None
        document = self.layers_controller.document
        index = self.layers_controller.active_index
        if document is not None and index is not None:
            mask = document.layers[index].raster_mask
        enabled = (
            kind == "mask"
            and not self._layer_transform_active()
            and mask is not None
            and mask.enabled
        )
        self.viewport.set_mask_brush_mode(
            enabled, self._mask_brush_settings.size)
        self.layers_panel.set_mask_paint_active(enabled)

    def _activate_pointer_tool(self) -> None:
        self.layers_controller.cancel_color_range_selection()
        self.layers_controller.cancel_pending_selection_work()
        self.viewport.set_pointer_tool()
        self.layers_panel.set_mask_paint_active(False)
        self.layers_panel.set_status("Pointer ready.")

    def _set_mask_paint_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled) and not self._layer_transform_active()
        if enabled:
            self.layers_controller.cancel_color_range_selection()
            self.layers_controller.cancel_pending_selection_work()
            layer_id = self._active_layer_id()
            if layer_id is None or not self.layers_controller.set_edit_target(
                layer_id, "mask"
            ):
                self.layers_panel.set_mask_paint_active(False)
                return
        self.viewport.set_mask_brush_mode(
            enabled, self._mask_brush_settings.size)
        self.layers_panel.set_mask_paint_active(enabled)
        if enabled:
            self.layers_panel.set_status(
                f"Paint Mask · {self._mask_brush_settings.mode.value.capitalize()}.")

    def _arm_selection_tool(self, shape: str) -> None:
        self.layers_controller.cancel_color_range_selection()
        self.layers_controller.cancel_pending_selection_work()
        self.layers_panel.set_mask_paint_active(False)
        self.viewport.set_selection_tool(shape)
        self.layers_panel.set_status(
            f"{shape.replace('_', ' ').title()} selection · draw on canvas.")

    def _arm_gradient_tool(self, kind: str) -> None:
        self.layers_controller.cancel_color_range_selection()
        self.layers_controller.cancel_pending_selection_work()
        self.layers_panel.set_mask_paint_active(False)
        self.viewport.set_gradient_tool(kind or None)

    def _begin_mask_brush_stroke(self, x: float, y: float) -> None:
        if not self.layers_controller.begin_mask_brush_stroke(
            x, y, self._mask_brush_settings
        ):
            self.layers_panel.set_status(
                "Mask brush needs a selected, enabled mask.", True)

    def _apply_temporary_selection_drag(
        self, shape: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        from ditherzam.layers.selection import (
            SelectionOperation, SelectionShape)

        operation_text = getattr(
            self.layers_panel, "selection_operation_text",
            lambda: "Replace")()
        operation = SelectionOperation(operation_text.lower())
        self.layers_controller.update_temporary_selection(
            SelectionShape(shape), (x0, y0, x1, y1), operation)

    def _apply_temporary_selection_path(self, kind: str, points) -> None:
        from ditherzam.layers.selection import SelectionOperation

        operation = SelectionOperation(
            self.layers_panel.selection_operation_text().lower())
        self.layers_controller.update_path_selection(
            kind, points, operation,
            diameter=float(self.layers_panel.selection_radius_spin.value() * 2),
        )

    def _pick_color_range_selection(self, x: float, y: float) -> None:
        from ditherzam.layers.selection import SelectionOperation
        operation = SelectionOperation(
            self.layers_panel.selection_operation_text().lower())
        if not self.layers_controller.begin_color_range_selection(
            x, y, operation,
            tolerance=self.layers_panel.color_range_tolerance_spin.value(),
            softness=self.layers_panel.color_range_softness_spin.value(),
        ):
            self.layers_panel.set_status(
                "Pick a visible point inside the selected image layer.", True)

    def _confirm_color_range_selection(self) -> None:
        if self.layers_controller.confirm_color_range_selection():
            self.viewport.set_selection_tool(None)
            self.layers_panel.set_status(
                "Confirming exact Color Range…"
                if self.layers_controller.selection_work_pending
                else "Color Range selection confirmed.")

    def _cancel_color_range_selection(self) -> None:
        cancelled = self.layers_controller.cancel_color_range_selection()
        self.viewport.set_selection_tool(None)
        if cancelled:
            self.layers_panel.set_status("Color Range selection cancelled.")

    def _set_mask_brush_settings(
        self, tip: str, size: int, hardness: int,
        strength: int, spacing_percent: int,
    ) -> None:
        from dataclasses import replace
        from ditherzam.layers import BrushTip

        self._mask_brush_settings = replace(
            self._mask_brush_settings,
            tip=BrushTip(tip), size=float(size), hardness=hardness,
            strength=strength, spacing_percent=spacing_percent,
        )
        self.viewport.set_mask_brush_size(float(size))

    def _set_mask_brush_mode(self, mode: str) -> None:
        from dataclasses import replace
        from ditherzam.layers import BrushMode

        self._mask_brush_settings = replace(
            self._mask_brush_settings, mode=BrushMode(str(mode).lower()))
        self.layers_panel.set_status(
            f"Mask brush: {self._mask_brush_settings.mode.value.capitalize()}.")

    def _clear_temporary_selection(self) -> None:
        self.layers_controller.clear_temporary_selection()
        self.viewport.clear_selection_overlay()

    def _apply_gradient_drag(
        self, kind: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        from ditherzam.layers import GradientKind

        try:
            self.layers_controller.generate_gradient_from_document_points(
                GradientKind(kind), x0, y0, x1, y1)
        except ValueError:
            self.layers_panel.set_status(
                "Gradient drag must start and end on the active layer.", True)

    def _change_mask_brush_size(self, direction: int) -> None:
        from dataclasses import replace
        size = max(1.0, min(
            2048.0,
            self._mask_brush_settings.size
            * (1.1 if direction > 0 else 1.0 / 1.1)))
        self._mask_brush_settings = replace(
            self._mask_brush_settings, size=size)
        self.viewport.set_mask_brush_size(size)

    def _swap_mask_brush_mode(self) -> None:
        from dataclasses import replace
        from ditherzam.layers import BrushMode
        mode = (
            BrushMode.HIDE
            if self._mask_brush_settings.mode is BrushMode.REVEAL
            else BrushMode.REVEAL)
        self._mask_brush_settings = replace(
            self._mask_brush_settings, mode=mode)
        self.layers_panel.brush_mode_combo.setCurrentText(
            mode.value.capitalize())
        self.layers_panel.set_status(
            f"Mask brush: {mode.value.capitalize()}.")

    def _layer_smart_mask_readiness(self, layer) -> tuple[bool, str]:
        """Authorize Smart freeze from the live selected-editor lifecycle."""
        panel = self.panel.smart_mask_panel
        settings = layer.look.smart_mask
        if panel.settings != settings:
            return False, "Smart Mask settings are stale for the selected layer."
        if not settings.enabled:
            return False, "Enable Smart Mask before freezing it."
        if settings.target is MaskTarget.WHOLE_IMAGE:
            return True, ""
        if not self._mask_dependencies_available():
            return False, "Smart Mask model is unavailable."
        if panel.status is not MaskPanelStatus.READY:
            return False, "Smart Mask is not ready; wait for detection to finish."
        source = layer.source
        if source is None:
            return False, "Smart Mask source is unavailable."
        probability = source.probability
        current = self._mask_probability
        if (
            self._mask_source != source.source_identity
            or probability is None
            or current is None
            or current.identity.source != source.source_identity
            or current.identity != probability.identity
        ):
            return False, "Smart Mask result is stale for the selected layer."
        return True, ""

    def _refresh_layer_history_actions(self) -> None:
        controller = self.layers_controller
        undo_label = controller.undo_label
        redo_label = controller.redo_label
        self.undo_action.setText(
            "Undo" if undo_label is None else f"Undo {undo_label}")
        self.redo_action.setText(
            "Redo" if redo_label is None else f"Redo {redo_label}")
        blocked = self._layer_transform_active()
        self.undo_action.setEnabled(controller.can_undo and not blocked)
        self.redo_action.setEnabled(controller.can_redo and not blocked)
        self.undo_action.setToolTip(
            "Undo the last layer or mask change."
            if undo_label is None else f"Undo {undo_label}.")
        self.redo_action.setToolTip(
            "Redo the next layer or mask change."
            if redo_label is None else f"Redo {redo_label}.")

    def _apply_layer_source(self, gray, rgba, probability=None) -> None:
        """Switch the editor view to a layer-owned immutable source.

        This adapter deliberately does not call ``load_array``: selecting a row
        must never open a new document or rebuild the layer graph.
        """
        self._debounce.stop()
        self._settle.stop()
        self._zoom_debounce.stop()
        self._scheduler.invalidate()
        self._full_preview_requested = False
        self._last_zoom_bucket = None
        self._mask_scheduler.invalidate_source(None)

        self._base_gray = gray
        self._base_rgba = rgba
        self._base_rgb = rgba[..., :3]
        self._mask_probability = probability
        self._mask_source = (
            probability.identity.source
            if probability is not None and hasattr(probability, "identity")
            else None
        )
        self.pipeline.clear_cache()
        mask_panel = self.panel.smart_mask_panel
        mask_panel.set_availability(
            source=True, model=self._mask_dependencies_available())
        mask_panel.set_valid_mask_available(probability is not None)
        self._refresh_mask_scope_actions()

    def _place_layer_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Place Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)",
        )
        if path:
            self._start_layer_decode(path, intent="place")

    def _import_active_raster_mask(self) -> None:
        if self._layer_transform_active():
            self.layers_panel.set_status(
                "Confirm or cancel the active layer transform first.", True)
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Raster Mask", "", "PNG Images (*.png)")
        if path:
            self.layers_controller.import_active_raster_mask(path)

    def _export_active_raster_mask(self) -> None:
        if self._layer_transform_active():
            self.layers_panel.set_status(
                "Confirm or cancel the active layer transform first.", True)
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Raster Mask", "layer-mask.png", "PNG Images (*.png)")
        if path:
            self.layers_controller.export_active_raster_mask(path)

    def _apply_layer_preset(self, preset: dict) -> None:
        contents = preset_to_settings(preset)
        self._apply_preset(
            contents.settings,
            contents.palette,
            contents.effects,
            contents.smart_mask,
            color_mode=contents.color_mode,
            source_dither=contents.source_dither,
            source_dither_brighten=contents.source_dither_brighten,
            render_result=False,
        )

    def _layer_reference_size(self) -> tuple[int, int]:
        controller = getattr(self, "layers_controller", None)
        document = None if controller is None else controller.document
        if document is None:
            return self._reference_size()
        return document.canvas.width, document.canvas.height

    def _show_layer_frame(self, rgb_or_rgba_u8) -> None:
        """Publish a document composite using document, not active-layer, geometry."""
        # Do not stop the editor timers here. Layer edits intentionally use the
        # same two-stage lifecycle as the single-image editor: the proxy frame
        # published by the debounce tick must leave the pending settle tick
        # alive so it can replace the proxy with the policy-quality composite.
        self._zoom_debounce.stop()
        self._scheduler.invalidate()
        from ditherzam.layers import InspectionMode
        frame = (
            self._apply_active_layer_mask_overlay(rgb_or_rgba_u8)
            if self.layers_controller.inspection_mode is InspectionMode.NORMAL
            else rgb_or_rgba_u8
        )
        qimg = numpy_to_qimage(frame)
        self.last_qimage = qimg
        self.viewport.set_pixmap(
            QPixmap.fromImage(qimg),
            logical_size=self._layer_reference_size(),
            refit=False,
        )
        self._sync_layer_drag_target()

    def _install_layer_transform_proxy(
        self, proxy, document_size, geometry
    ) -> None:
        if geometry is None:
            return
        background = QPixmap.fromImage(
            numpy_to_qimage(proxy.background_rgba)).copy()
        selected = QPixmap.fromImage(
            numpy_to_qimage(proxy.layer_rgba)).copy()
        width, height = document_size
        self.viewport.install_transform_proxy(
            background,
            selected,
            QRectF(*geometry),
            document_rect=QRectF(0, 0, width, height),
            opacity=(proxy.opacity / 100.0 if proxy.visible else 0.0),
        )

    def _install_mask_stroke_roi(self, _authority, rect, rgba) -> None:
        x0, y0, x1, y1 = rect
        pixmap = QPixmap.fromImage(numpy_to_qimage(rgba)).copy()
        proxy = self.layers_controller._latest_proxy
        if proxy is None or proxy.prefix_rgba is None:
            return
        raster_h, raster_w = proxy.prefix_rgba.shape[:2]
        self.viewport.install_mask_stroke_roi(
            pixmap, QRectF(x0, y0, x1 - x0, y1 - y0),
            (raster_w, raster_h), self._layer_reference_size())

    def _show_smart_refinement_preview(self, mask, _token: str) -> None:
        """Show capped grayscale transaction coverage without publishing state."""
        alpha = np.full(mask.shape, 255, dtype=np.uint8)
        preview = np.dstack((mask, mask, mask, alpha))
        qimg = numpy_to_qimage(preview)
        self.last_qimage = qimg
        self.viewport.set_pixmap(
            QPixmap.fromImage(qimg),
            logical_size=self._layer_reference_size(),
            refit=False,
        )

    def _update_layer_transform_proxy(self, geometry) -> None:
        if (
            geometry is not None
            and getattr(self.viewport, "_transform_proxy_layer", None) is not None
        ):
            self.viewport.update_transform_proxy_geometry(QRectF(*geometry))

    def _apply_active_layer_mask_overlay(self, composite):
        """Tint only the active layer's transformed mask in a display copy."""
        panel = self.panel.smart_mask_panel
        document = self.layers_controller.document
        index = self.layers_controller.active_index
        settings = panel.settings
        if (
            not panel.overlay_check.isChecked()
            or document is None
            or index is None
            or not settings.enabled
            or settings.target is MaskTarget.WHOLE_IMAGE
        ):
            return composite
        layer = document.layers[index]
        if not layer.visible or layer.opacity == 0:
            return composite
        source = layer.source
        probability = None if source is None else source.probability
        if source is None or probability is None:
            return composite
        context = MaskContext(
            probability.identity.source,
            source.rgba,
            probability,
            settings,
        )
        frame = np.asarray(composite)
        frame_h, frame_w = frame.shape[:2]
        scale_x = frame_w / float(document.canvas.width)
        scale_y = frame_h / float(document.canvas.height)
        layer_h, layer_w = source.rgba.shape[:2]
        target_h = max(1, int(round(
            layer_h * layer.transform.scale_y * scale_y)))
        target_w = max(1, int(round(
            layer_w * layer.transform.scale_x * scale_x)))
        mask = derive_render_mask(
            context, (target_h, target_w), caches=self._mask_caches)
        alpha = np.asarray(
            Image.fromarray(source.rgba[..., 3], mode="L").resize(
                (target_w, target_h), resample=Image.Resampling.NEAREST),
            dtype=np.float32,
        )
        mask = mask * (alpha / np.float32(255.0))
        mask = mask * np.float32(layer.opacity / 100.0)
        coverage = np.zeros((frame_h, frame_w), dtype=np.float32)
        x = int(round(layer.transform.x * scale_x))
        y = int(round(layer.transform.y * scale_y))
        left, top = max(0, x), max(0, y)
        right, bottom = min(frame_w, x + target_w), min(frame_h, y + target_h)
        if left >= right or top >= bottom:
            return composite
        coverage[top:bottom, left:right] = mask[
            top - y:bottom - y, left - x:right - x]
        coverage.flags.writeable = False
        return apply_mask_overlay(frame, coverage)

    def _sync_layer_drag_target(self) -> None:
        geometry = (
            self.layers_controller.active_geometry()
            if self._layer_transform_session is not None
            else None
        )
        self.viewport.set_layer_drag_target(
            None if geometry is None else QRectF(*geometry))

    def _active_layer_id(self) -> str | None:
        document = self.layers_controller.document
        index = self.layers_controller.active_index
        return (
            None if document is None or index is None
            else document.layers[index].id
        )

    def _layer_transform_active(self) -> bool:
        return getattr(self, "_layer_transform_session", None) is not None

    def _guard_layer_transform(self) -> bool:
        """Return False and explain why an unrelated action cannot proceed."""
        if not self._layer_transform_active():
            return True
        self.layers_panel.set_status(
            "Confirm or cancel the active layer transform first.", error=True)
        return False

    def _set_layer_transform_ui_locked(self, active: bool) -> None:
        self.tabs.setEnabled(not active)
        self.menuBar().setEnabled(not active)
        for name in ("composition_panel", "timeline_panel"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not active)
        for action in getattr(self, "_export_actions", {}).values():
            action.setEnabled(not active)
        full_quality = getattr(self, "_actions", {}).get("full_quality_preview")
        if full_quality is not None:
            full_quality.setEnabled(not active)
        if not active:
            self._refresh_mask_scope_actions()

    def _start_layer_transform(self, index: int) -> None:
        if self._layer_transform_active():
            self._guard_layer_transform()
            return
        if self.layers_controller.active_index != int(index):
            self.layers_controller.activate_layer(int(index))
        geometry = self.layers_controller.active_geometry()
        layer_id = self._active_layer_id()
        if geometry is None or layer_id is None:
            return
        if not self.layers_controller.begin_transform_transaction():
            self.layers_panel.set_status(
                "Layer transform could not be started.", error=True)
            return
        # Transform owns pointer and keyboard input until Confirm/Cancel.
        # Masking tools otherwise win the viewport's mouse-event priority and
        # make the transform handles appear unresponsive.
        self.viewport.set_mask_brush_mode(False)
        self.layers_panel.set_mask_paint_active(False)
        self.layers_controller.cancel_color_range_selection()
        self.layers_controller.cancel_pending_selection_work()
        self.viewport.set_selection_tool(None)
        self.viewport.set_gradient_tool(None)
        self._layer_transform_session = (layer_id, geometry)
        self.layers_panel.set_transform_mode(True)
        self._set_layer_transform_ui_locked(True)
        self._sync_layer_drag_target()
        self.layers_panel.set_status(
            "Drag inside to move, or drag a blue handle to resize.")

    def _confirm_layer_transform(self) -> None:
        if self._layer_transform_session is None:
            return
        self._layer_transform_session = None
        self._clear_layer_gesture()
        self.layers_controller.confirm_transform_transaction()
        self.layers_panel.set_transform_mode(False)
        self._set_layer_transform_ui_locked(False)
        self._sync_layer_drag_target()
        self.layers_panel.set_status("Layer transform confirmed.")
        self._resume_mask_after_layer_transform()

    def _cancel_layer_transform(self) -> None:
        session = self._layer_transform_session
        if session is None:
            return
        self._layer_transform_session = None
        self._clear_layer_gesture()
        self.layers_controller.cancel_transform_transaction()
        self.layers_panel.set_transform_mode(False)
        self._set_layer_transform_ui_locked(False)
        self._sync_layer_drag_target()
        self.layers_panel.set_status("Layer transform cancelled.")
        self._resume_mask_after_layer_transform()

    def _resume_mask_after_layer_transform(self) -> None:
        settings = self.panel.smart_mask_panel.settings
        if settings.enabled and self._mask_probability is None:
            self._apply_mask_settings_lifecycle(settings)

    def _clear_layer_gesture(self) -> None:
        self.layers_controller.end_transform_gesture()
        self._layer_drag_origin = None
        self._layer_drag_id = None
        self._layer_resize_origin = None
        self._layer_resize_id = None

    def _begin_layer_drag(self) -> None:
        self.layers_controller.begin_transform_gesture()
        geometry = self.layers_controller.active_geometry()
        self._layer_drag_origin = (
            None if geometry is None else (geometry[0], geometry[1]))
        document = self.layers_controller.document
        index = self.layers_controller.active_index
        self._layer_drag_id = (
            None if document is None or index is None
            else document.layers[index].id
        )

    def _drag_active_layer(self, delta_x: float, delta_y: float) -> None:
        if self._layer_drag_origin is None:
            return
        document = self.layers_controller.document
        index = self.layers_controller.active_index
        if (
            document is None
            or index is None
            or document.layers[index].id != self._layer_drag_id
        ):
            return
        origin_x, origin_y = self._layer_drag_origin
        self.layers_controller.move_active_layer_to(
            int(round(origin_x + float(delta_x))),
            int(round(origin_y + float(delta_y))),
        )

    def _nudge_active_layer(self, delta_x: int, delta_y: int) -> None:
        if not self._layer_transform_active():
            return
        geometry = self.layers_controller.active_geometry()
        if geometry is None:
            return
        x, y, _width, _height = geometry
        self.layers_controller.move_active_layer_to(
            x + int(delta_x), y + int(delta_y))

    def _finish_layer_drag(self) -> None:
        self.layers_controller.end_transform_gesture()
        self._layer_drag_origin = None
        self._layer_drag_id = None
        self._sync_layer_drag_target()

    def _begin_layer_resize(self, _corner: str) -> None:
        if self._layer_transform_session is None:
            return
        self.layers_controller.begin_transform_gesture()
        self._layer_resize_origin = self.layers_controller.active_geometry()
        self._layer_resize_id = self._active_layer_id()

    def _resize_active_layer(
        self, corner: str, delta_x: float, delta_y: float
    ) -> None:
        origin = self._layer_resize_origin
        if origin is None or self._active_layer_id() != self._layer_resize_id:
            return
        x, y, width, height = origin
        dx = int(round(float(delta_x)))
        dy = int(round(float(delta_y)))
        if corner == "nw":
            nx, ny, nw, nh = x + dx, y + dy, width - dx, height - dy
        elif corner == "ne":
            nx, ny, nw, nh = x, y + dy, width + dx, height - dy
        elif corner == "sw":
            nx, ny, nw, nh = x + dx, y, width - dx, height + dy
        elif corner == "se":
            nx, ny, nw, nh = x, y, width + dx, height + dy
        elif corner == "n":
            nx, ny, nw, nh = x, y + dy, width, height - dy
        elif corner == "s":
            nx, ny, nw, nh = x, y, width, height + dy
        elif corner == "e":
            nx, ny, nw, nh = x, y, width + dx, height
        elif corner == "w":
            nx, ny, nw, nh = x + dx, y, width - dx, height
        else:
            return

        nw = max(1, nw)
        nh = max(1, nh)
        if self.layers_panel.lock_aspect_check.isChecked():
            ratio = width / float(max(1, height))
            if abs(nw - width) / max(1, width) >= abs(nh - height) / max(1, height):
                nh = max(1, int(round(nw / ratio)))
            else:
                nw = max(1, int(round(nh * ratio)))
            if "w" in corner:
                nx = x + width - nw
            if "n" in corner:
                ny = y + height - nh
        else:
            if "w" in corner:
                nx = min(nx, x + width - 1)
            if "n" in corner:
                ny = min(ny, y + height - 1)
        self.layers_controller.resize_active_layer_to(nx, ny, nw, nh)

    def _finish_layer_resize(self) -> None:
        self.layers_controller.end_transform_gesture()
        self._layer_resize_origin = None
        self._layer_resize_id = None
        self._sync_layer_drag_target()

    def _export_layers(self) -> None:
        """Export the flattened still stack without media-mask gating."""
        from PySide6.QtWidgets import QFileDialog
        if not self._guard_layer_transform():
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Layers",
            "layers.png",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)",
        )
        if path:
            result = self.layers_controller.export_current(path)
            log_action("layers.exported", path=path, success=result is not None)

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
                "export_preview": self._on_export_preview,
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
        algorithm = str(
            self.panel.state.get("extract_algorithm", "balanced"))
        palette = generate_palette(
            self._base_rgb,
            unit,
            value,
            algorithm=algorithm,
            min_coverage=(
                int(self.panel.state.get("extract_min_coverage", 5)) / 1000.0
            ),
            diversity=int(self.panel.state.get("extract_diversity", 100)),
        )
        self.panel.set_extraction_result(value, palette.colors.shape[0])
        if palette.colors.shape[0] == 0:
            return
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

    def _apply_preset(
        self, settings, palette, effects, smart_mask=None, *,
        color_mode: str = "off",
        source_dither: int = 100,
        source_dither_brighten: bool = False,
        render_result: bool = True,
    ) -> None:
        if not self._guard_layer_transform():
            return
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
            panel.depth_slider.setValue(int(settings.depth))
            panel.mapping_combo.setCurrentText(str(settings.color_mapping))
            panel.invert_toggle.setChecked(bool(settings.invert))
            panel.preview_toggle.setChecked(bool(settings.preview_disabled))
            panel.mode_combo.setCurrentText(str(color_mode))
            panel.source_dither_slider.setValue(int(source_dither))
            panel.source_dither_brighten_check.setChecked(
                bool(source_dither_brighten))
            # State is authoritative. Assign it explicitly instead of relying on
            # child-widget signals while the parent is in a guarded transaction.
            panel.state.update({
                "saturation": int(settings.saturation),
                "scale": int(settings.scale),
                "depth": int(settings.depth),
                "color_mapping": str(settings.color_mapping),
                "invert": bool(settings.invert),
                "preview_disabled": bool(settings.preview_disabled),
                "color_mode": str(color_mode),
                "source_dither": int(source_dither),
                "source_dither_brighten": bool(source_dither_brighten),
            })
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
        self._action_settings_snapshot = self._control_action_snapshot()
        self._refresh_mask_scope_actions()
        if render_result and self._base_gray is not None:
            self.render_now()

    # -- menu handlers --
    def _on_save_preset(self):
        if not self._guard_layer_transform():
            return
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")
        if not ok or not name:
            return
        preset = settings_to_preset(
            self._collect_settings(), self._current_palette(),
            self._current_effect_stack(), self._color_mode(),
            self.panel.smart_mask_panel.settings,
            source_dither=int(self.panel.state.get("source_dither", 100)),
            source_dither_brighten=bool(
                self.panel.state.get("source_dither_brighten", False)),
        )
        self._preset_manager.save(name, preset)
        log_action("preset.saved", name=name)
        QMessageBox.information(self, "Presets", f"Preset '{name}' saved successfully!")

    def _on_load_preset(self):
        from PySide6.QtWidgets import QInputDialog
        if not self._guard_layer_transform():
            return
        names = self._preset_manager.list()
        if not names:
            return
        name, ok = QInputDialog.getItem(self, "Load Preset", "Preset:", names, 0, False)
        if not ok:
            return
        contents = preset_to_settings(self._preset_manager.load(name))
        self._apply_preset(
            contents.settings,
            contents.palette,
            contents.effects,
            contents.smart_mask,
            color_mode=contents.color_mode,
            source_dither=contents.source_dither,
            source_dither_brighten=contents.source_dither_brighten,
        )
        log_action("preset.loaded", name=name)

    def _on_import_preset(self):
        if not self._guard_layer_transform():
            return
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(self, "Import Preset(s)", "",
                                              "Preset Files (*.yaml *.yml)")
        if not path:
            return
        try:
            name = self._preset_manager.import_file(path)
        except ValueError:
            log_action("preset.import_failed", path=path, error="invalid")
            QMessageBox.warning(self, "Presets", "Not a valid preset file.")
            return
        log_action("preset.imported", path=path, name=name)
        QMessageBox.information(self, "Presets", f"Imported preset '{name}'.")

    def _on_export_preset(self):
        if not self._guard_layer_transform():
            return
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
            source_dither=int(self.panel.state.get("source_dither", 100)),
            source_dither_brighten=bool(
                self.panel.state.get("source_dither_brighten", False)),
        )
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(preset, f, sort_keys=False, allow_unicode=True)
        log_action("preset.exported", path=path)

    def _on_export_raster(self, file_filter, ext):
        if not self._guard_layer_transform():
            return
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog
        if self._base_gray is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", file_filter)
        if not path:
            return
        document = self.layers_controller.document
        if document is not None and len(document.layers) > 1:
            selected_ext = Path(path).suffix.lower()
            if (
                selected_ext in (".jpg", ".jpeg")
                and not getattr(self, "_jpeg_flatten_notice_shown", False)
            ):
                self.statusBar().showMessage(
                    "JPEG does not support transparency; transparent pixels are flattened onto white.",
                    8000,
                )
                self._jpeg_flatten_notice_shown = True
            # The legacy editor renderer represents only the active layer.
            # Once a stack exists, every still-image export must use the exact
            # LayerDocument compositor.
            result = self.layers_controller.export_current(path)
            log_action(
                "image.exported" if result is not None else "image.export_failed",
                path=path,
                format=selected_ext.lstrip("."),
                layer_count=len(document.layers),
            )
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
        log_action(
            "image.exported",
            path=path,
            format=selected_ext.lstrip("."),
            width=rendered.shape[1],
            height=rendered.shape[0],
        )

    def _on_export_preview(self):
        """Save the preview raster exactly as displayed.

        Deliberately bypasses the exact-export pipeline: the saved file is the
        capped/proxy preview image itself, including every preview
        approximation, at the preview's resolution.
        """
        if not self._guard_layer_transform():
            return
        from PySide6.QtWidgets import QFileDialog
        image = self.last_qimage
        if image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Preview", "preview.png", "PNG Files (*.png)")
        if not path:
            return
        if not image.save(path):
            log_action("preview.export_failed", path=path)
            self.statusBar().showMessage(f"Could not save preview to {path}", 8000)
        else:
            log_action(
                "preview.exported",
                path=path,
                width=image.width(),
                height=image.height(),
            )

    def _on_export_svg(self):
        if not self._guard_layer_transform():
            return
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
        log_action("image.svg_exported", path=path)

    def _on_batch_folder(self):
        if not self._guard_layer_transform():
            return
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
        log_action(
            "batch.completed",
            folder=folder,
            processed=processed,
            skipped=skipped,
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
        if not self._guard_layer_transform():
            return
        self._replace_source_arrays(
            gray_f32, rgb_u8, rgba_u8, adopt_decoded_rgba=False)

    def _replace_source_arrays(self, gray_f32, rgb_u8, rgba_u8,
                               *, adopt_decoded_rgba: bool) -> None:
        """Validate a replacement and optionally accept private decoder ownership."""
        if not self._guard_layer_transform():
            return
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
        controller = getattr(self, "composition_controller", None)
        if controller is not None and controller.composition is not None:
            controller.request_frame(self.composition_panel.frame_slider.value())
        layers_controller = getattr(self, "layers_controller", None)
        if layers_controller is not None:
            layers_controller.initialize_source_layer()

    def set_style(self, name: str) -> None:
        if not self._guard_layer_transform():
            return
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
            mask = derive_render_mask(
                request.mask_context, result.shape[:2], caches=self._mask_caches)
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
        layers_controller = getattr(self, "layers_controller", None)
        if (
            layers_controller is not None
            and getattr(layers_controller, "document", None) is not None
        ):
            self._zoom_debounce.stop()
            self._full_preview_requested = False
            self._last_zoom_bucket = None
            # Layer documents retain the normal two-stage editor preview:
            # a fast capped composite shortly after input, followed by the
            # settled policy-quality composite. Rendering the full stack on
            # every slider event starves publication through cancellation and
            # makes controls appear to jump straight to their final result.
            if not self._debounce.isActive():
                self._debounce.start(self._debounce_ms)
            self._settle.start(self._settle_ms)
            return
        self._full_preview_requested = False       # any edit returns to the cap
        self._last_zoom_bucket = None               # a normal edit re-settles the baseline
        self._debounce.start(self._debounce_ms)   # fast proxy
        self._settle.start(self._settle_ms)        # full-res once idle

    # ---- internals ----------------------------------------------------------
    def _policy_cap(self) -> int:
        """Settled longest-side cap from the current preview preference + viewport."""
        w, h = self._reference_size()
        return self._policy_cap_for_size(w, h)

    def _layer_policy_cap(self) -> int:
        """Settled cap based on the document canvas, independent of row selection."""
        w, h = self._layer_reference_size()
        return self._policy_cap_for_size(w, h)

    def _policy_cap_for_size(self, w: int, h: int) -> int:
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

    def _build_request(
        self,
        kind: RenderKind,
        target_max_side: int | None = None,
        *,
        layer_preview_context=None,
    ) -> RenderRequest:
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
        show_overlay = (
            layer_preview_context is None
            and self.panel.smart_mask_panel.overlay_check.isChecked()
        )
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
            layer_preview_context=layer_preview_context,
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
        self._refresh_active_layer_mask_scope()

    def _refresh_active_layer_mask_scope(self) -> None:
        mask_panel = self.panel.smart_mask_panel
        controller = getattr(self, "layers_controller", None)
        document = None if controller is None else controller.document
        index = None if controller is None else controller.active_index
        name = (
            None
            if document is None or index is None
            else document.layers[index].name
        )
        mask_panel.set_mask_scope(name)

    def _do_render(self) -> None:
        """Debounce tick: request a fast proxy render."""
        if self._base_gray is None:
            return
        layers_controller = getattr(self, "layers_controller", None)
        if (
            layers_controller is not None
            and getattr(layers_controller, "document", None) is not None
        ):
            if layers_controller.has_active_layer:
                layers_controller.update_active_from_editor(
                    request_preview=False,
                )
                # While controls move, preview only the selected source through
                # the editor's lightweight pipeline. The full layer graph is
                # intentionally absent from this path and is recomposited by
                # the settle tick below.
                cap = min(self._proxy_max_side, self._layer_policy_cap())
                authority = layers_controller.freeze_interactive_preview(cap)
                req = self._scheduler.request(self._build_request(
                    RenderKind.DRAG,
                    target_max_side=cap,
                    layer_preview_context=authority,
                ))
                if req is not None:
                    self._launch_worker(req)
            return
        req = self._scheduler.request(self._build_request(RenderKind.DRAG))
        if req is not None:
            self._launch_worker(req)

    def _do_full_render(self) -> None:
        """Settle tick: request the settled (policy-capped) render, or an exact
        full render if Full Quality Preview was requested."""
        if self._base_gray is None:
            return
        self._log_settled_control_changes()
        layers_controller = getattr(self, "layers_controller", None)
        if (
            layers_controller is not None
            and getattr(layers_controller, "document", None) is not None
        ):
            # Keep an in-flight selected-layer proxy publishable while the
            # settled document composite renders.  The proxy may take longer
            # than the settle interval and is still the freshest frame ready
            # for display.  ``_show_layer_frame`` invalidates it only when the
            # replacement composite has actually arrived, preventing the
            # viewport from going visually silent if layer rendering stalls.
            cap = (
                max(self._layer_reference_size())
                if self._full_preview_requested
                else self._layer_policy_cap()
            )
            layers_controller.request_preview(cap)
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
        log_action("preview.full_quality_requested")
        layers_controller = getattr(self, "layers_controller", None)
        if (
            layers_controller is not None
            and getattr(layers_controller, "document", None) is not None
        ):
            layers_controller.request_preview(max(self._layer_reference_size()))
            return
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
        if self._render_closing:
            return
        worker = _RenderWorker(self.pipeline, self._base_gray, request,
                                is_cancelled=lambda: self._scheduler.should_cancel(request),
                                mask_caches=self._mask_caches)
        self._render_workers.add(worker)
        worker.signals.finished.connect(self._on_rendered)
        worker.signals.failed.connect(self._on_render_failed)
        worker.signals.cancelled.connect(self._on_render_cancelled)

        def release(*_args, retained=worker) -> None:
            self._render_workers.discard(retained)

        worker.signals.finished.connect(release)
        worker.signals.failed.connect(release)
        worker.signals.cancelled.connect(release)
        self._pool.start(worker)

    def _on_rendered(self, qimg: QImage, request: RenderRequest) -> None:
        # Drop stale/out-of-order results; only the most-recently-started render
        # is painted.
        if self._scheduler.is_current(request):
            authority = request.layer_preview_context
            if authority is None:
                published = qimg
                logical_size = request.logical_size
            else:
                composite = self.layers_controller.compose_interactive_preview(
                    qimage_to_numpy_rgba(qimg), authority)
                published = (
                    None if composite is None else
                    numpy_to_qimage(self._apply_active_layer_mask_overlay(composite))
                )
                logical_size = self._layer_reference_size()
            if published is not None:
                self.last_qimage = published
                refit = self._pending_refit
                self._pending_refit = False
                self.viewport.set_pixmap(
                    QPixmap.fromImage(published),
                    logical_size=logical_size,
                    refit=refit,
                )
                if authority is not None:
                    self._sync_layer_drag_target()
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
        document = getattr(self.layers_controller, "document", None)
        intent = "open" if document is None else "place"
        self._start_layer_decode(path, intent=intent)

    def _start_layer_decode(self, path: str, *, intent: str) -> None:
        """Decode with a frozen routing intent and reject superseded results."""
        if not self._guard_layer_transform():
            return
        if intent not in {"open", "place"}:
            raise ValueError("decode intent must be 'open' or 'place'")
        self._decode_generation += 1
        generation = self._decode_generation
        log_action("image.decode_requested", path=path, intent=intent)
        worker = _DecodeWorker(path)
        self._decode_workers.add(worker)
        worker.signals.finished.connect(
            lambda gray, _rgb, rgba, g=generation, i=intent:
            self._on_layer_decoded(gray, rgba, generation=g, intent=i))
        worker.signals.finished.connect(
            lambda *_args, w=worker: self._decode_workers.discard(w))
        self._pool.start(worker)

    def _on_layer_decoded(
        self, gray_f32, rgba_u8, *, generation: int, intent: str
    ) -> None:
        if generation != self._decode_generation:
            return
        self._accept_decoded_layer_source(gray_f32, rgba_u8, intent=intent)
        self.schedule_render()

    def _accept_decoded_layer_source(
        self, gray_f32, rgba_u8, *, intent: str
    ) -> None:
        """Route decoded pixels explicitly into open or place semantics."""
        if not self._guard_layer_transform():
            return
        if intent == "open":
            self.layers_controller.open_document(gray_f32, rgba_u8)
        elif intent == "place":
            self.layers_controller.place_source(gray_f32, rgba_u8)
        else:
            raise ValueError("decode intent must be 'open' or 'place'")

    def _on_image_decoded(self, gray_f32, rgb_u8, rgba_u8) -> None:
        """Compatibility slot for callers that provide an already-decoded image."""
        if not self._guard_layer_transform():
            return
        document = getattr(self.layers_controller, "document", None)
        if document is None:
            self._replace_source_arrays(
                gray_f32, rgb_u8, rgba_u8, adopt_decoded_rgba=True)
        else:
            self._accept_decoded_layer_source(gray_f32, rgba_u8, intent="place")
        self.schedule_render()

    def _install_shortcuts(self) -> None:
        hk = get_hotkeys(sys.platform)
        self._actions: dict[str, QAction] = {}
        bindings = {
            "zoom_in": self.viewport.zoom_in,
            "zoom_out": self.viewport.zoom_out,
            "zoom_reset": self.viewport.reset_zoom,
            "full_quality_preview": self._do_full_quality_preview,
            "toggle_mask_inspection": self._toggle_mask_inspection,
        }
        for action_name, slot in bindings.items():
            act = QAction(self)
            act.setShortcut(QKeySequence(hk[action_name]))
            act.triggered.connect(slot)
            self.addAction(act)
            self._actions[action_name] = act

    def _toggle_mask_inspection(self) -> None:
        from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QLineEdit

        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            return
        controller = getattr(self, "layers_controller", None)
        if controller is not None:
            controller.toggle_red_inspection()
