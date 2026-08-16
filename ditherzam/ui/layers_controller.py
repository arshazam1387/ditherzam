"""Qt controller for the still-image spatial layer stack."""
from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap

from ditherzam.composition import Look, export_frame
from ditherzam.diagnostics import log_action
from ditherzam.layers import (
    BrushMode,
    BrushSettings,
    BrushStroke,
    CanvasSpec,
    Layer,
    LayerDocument,
    LayerDocumentHistory,
    LayerSource,
    LayerStack,
    LayerTransform,
    LayerLookRenderCache,
    LowerLayerPreviewContext,
    InspectionMode,
    RasterLayerMask,
    RasterMaskIOError,
    RasterMaskOperationError,
    MaskRefinementError,
    MaskRefinementKind,
    SmartRefinementSpec,
    GradientSpec,
    GradientKind,
    LuminancePreset,
    LuminanceRange,
    MaskGeneratorCandidate,
    MaskPatternKind,
    MaskPatternSpec,
    MaskCandidate,
    MaskCandidateOrigin,
    MaskCombinationMode,
    capped_combination_preview,
    combine_mask_candidate,
    gradient_mask,
    luminance_mask,
    pattern_mask,
    dither_mask,
    derive_refined_smart_mask,
    derive_refined_smart_preview,
    export_raster_mask_png,
    fill_raster_mask,
    freeze_smart_mask,
    import_raster_mask_png,
    invert_raster_mask,
    composite_active_layer_preview,
    lower_layer_preview_context,
    render_layer_document,
    render_layer_document_with_proxy,
    composite_mask_stroke_roi,
    inspect_mask,
)


from ditherzam.layers.mask_contracts import MaskObjectIdentity, MaskPublicationKey
from ditherzam.layers.selection import (
    SelectionOperation,
    SelectionShape,
    TemporarySelection,
    combine_selection,
    rasterize_selection,
    rasterize_polygon_selection,
    rasterize_freehand_selection,
    select_all,
    invert_selection,
    grow_selection,
    shrink_selection,
    feather_selection,
    restrict_mask_edit,
    selection_to_source,
    select_color_range,
    source_selection_to_document,
)
from ditherzam.render import RenderCancelled
from ditherzam.ui.convert import numpy_to_qimage


# Window teardown destroys the controller QObject before global-pool runnables
# necessarily reach a cancellation checkpoint. Keep those runnables rooted
# independently of the window until their terminal signal has been emitted.
_SHUTDOWN_LAYER_WORKERS: set["_LayerWorker"] = set()
_SHUTDOWN_SELECTION_WORKERS: set["_SelectionWorker"] = set()

_ASYNC_SELECTION_PIXELS = 512 * 512


@dataclass(frozen=True)
class LayerEditTarget:
    """Stable controller-owned destination for subsequent pixel edits."""

    layer_id: str
    kind: str

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ValueError("edit target layer_id cannot be empty")
        if self.kind not in {"layer", "mask"}:
            raise ValueError("edit target kind must be layer or mask")


@dataclass(frozen=True)
class MaskReplacementProposal:
    """One-shot candidate bound to the exact state it may replace."""

    token: str
    layer_id: str
    source_identity: object
    mask_identity: object
    mask_revision: int
    pixels: np.ndarray
    label: str


@dataclass(frozen=True)
class MaskStrokeAuthority:
    """Exact immutable identity to which an isolated brush stroke is bound."""

    layer_id: str
    source_identity: object
    mask_identity: object
    mask_revision: int
    transform: LayerTransform
    document_revision: int
    generation: int


@dataclass(frozen=True)
class SmartRefinementAuthority:
    layer_id: str
    source_identity: object
    probability_identity: object
    smart_settings: object
    mask_identity: object
    mask_revision: int
    document_revision: int
    generation: int


@dataclass(frozen=True)
class MaskCombinationAuthority:
    token: str
    layer_id: str
    source_identity: object
    mask_identity: object
    mask_revision: int
    document_revision: int
    candidate: MaskCandidate
    mode: MaskCombinationMode
    preview: np.ndarray


@dataclass(frozen=True)
class InteractiveLayerPreviewAuthority:
    """Frozen active-layer publication bound to one reusable lower context."""

    context: LowerLayerPreviewContext
    active_layer: Layer
    graph_guard: tuple
    lower_key: tuple


@dataclass(frozen=True)
class _SelectionTask:
    """One exact selection calculation frozen against a document object."""

    generation: int
    document: LayerDocument
    kind: str
    base: TemporarySelection | None
    payload: tuple


def _compute_selection_task(task: _SelectionTask) -> TemporarySelection:
    """Run only Qt-free selection primitives for a frozen worker request."""
    shape = (task.document.canvas.height, task.document.canvas.width)
    if task.kind == "shape":
        selection_shape, bounds, operation = task.payload
        pixels = rasterize_selection(shape, selection_shape, bounds)
        return combine_selection(task.base, pixels, operation)
    if task.kind == "path":
        path_kind, points, operation, diameter = task.payload
        if path_kind == "polygon":
            pixels = rasterize_polygon_selection(shape, points)
        elif path_kind == "freehand":
            pixels = rasterize_freehand_selection(
                shape, points, diameter=diameter)
        else:
            raise ValueError("selection path must be polygon or freehand")
        return combine_selection(task.base, pixels, operation)
    if task.kind == "refine":
        operation, radius = task.payload
        if operation == "all":
            return select_all(shape)
        if task.base is None:
            raise ValueError("selection refinement needs an active selection")
        if operation == "invert":
            return invert_selection(task.base)
        if operation == "grow":
            return grow_selection(task.base, radius)
        if operation == "shrink":
            return shrink_selection(task.base, radius)
        if operation == "feather":
            return feather_selection(task.base, radius)
        raise ValueError("unknown selection refinement")
    if task.kind == "color_range":
        (
            rgba, target, tolerance, softness, transform, operation,
        ) = task.payload
        source_candidate = select_color_range(
            rgba, target, tolerance=tolerance, softness=softness)
        document_candidate = source_selection_to_document(
            source_candidate, shape,
            layer_x=transform.x, layer_y=transform.y,
            scale_x=transform.scale_x, scale_y=transform.scale_y,
        )
        return combine_selection(task.base, document_candidate.pixels, operation)
    raise ValueError("unknown selection worker task")


class _SelectionSignals(QObject):
    finished = Signal(object, object, str)
    failed = Signal(str, object, str)


class _SelectionWorker(QRunnable):
    """Bounded exact selection worker; publication remains controller-owned."""

    def __init__(self, task: _SelectionTask):
        super().__init__()
        self.task = task
        self.request_id = str(uuid4())
        self.signals = _SelectionSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = _compute_selection_task(self.task)
        except Exception as exc:  # noqa: BLE001 - worker/UI boundary
            self.signals.failed.emit(str(exc), self.task, self.request_id)
        else:
            self.signals.finished.emit(result, self.task, self.request_id)


class _WorkerSignals(QObject):
    finished = Signal(object, int)
    failed = Signal(str, int)
    cancelled = Signal(int)


class _LayerWorker(QRunnable):
    """Render one frozen layer-document preview request."""

    def __init__(self, request, registry, look_cache, mask_caches):
        super().__init__()
        (
            self.generation,
            self.document,
            self.cap,
            self.thumbnail_keys,
            self.publication_keys,
        ) = request
        self.registry = registry
        self.look_cache = look_cache
        self.mask_caches = mask_caches
        self.signals = _WorkerSignals()
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def _is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @Slot()
    def run(self) -> None:
        try:
            if self._is_cancelled():
                raise RenderCancelled
            selected_id = (
                self.document.selected_ids[0]
                if self.document.selected_ids else None
            )
            proxy = None
            if selected_id is None:
                composite = render_layer_document(
                    self.document,
                    self.registry,
                    target_max_side=self.cap,
                    is_cancelled=self._is_cancelled,
                    allow_pending_masks=True,
                    look_cache=self.look_cache,
                    mask_caches=self.mask_caches,
                )
            else:
                rendered = render_layer_document_with_proxy(
                    self.document,
                    self.registry,
                    layer_id=selected_id,
                    target_max_side=self.cap,
                    is_cancelled=self._is_cancelled,
                    allow_pending_masks=True,
                    look_cache=self.look_cache,
                    mask_caches=self.mask_caches,
                )
                composite = rendered.composite_rgba
                proxy = rendered.proxy
            if self._is_cancelled():
                raise RenderCancelled
            thumbnails = []
            by_key = {
                _thumbnail_key(layer): layer for layer in self.document.layers
            }
            for key in self.thumbnail_keys:
                if self._is_cancelled():
                    raise RenderCancelled
                layer = by_key.get(key)
                if layer is None:
                    continue
                thumbnail_document = LayerDocument(
                    self.document.canvas, (layer,), (), self.document.revision
                )
                rgba = render_layer_document(
                    thumbnail_document,
                    self.registry,
                    target_max_side=56,
                    is_cancelled=self._is_cancelled,
                    allow_pending_masks=True,
                    look_cache=self.look_cache,
                    mask_caches=self.mask_caches,
                )
                if self._is_cancelled():
                    raise RenderCancelled
                frozen = np.array(rgba, dtype=np.uint8, order="C", copy=True)
                frozen.flags.writeable = False
                thumbnails.append((key, frozen))
            payload = (
                composite, tuple(thumbnails), proxy, self.publication_keys)
        except RenderCancelled:
            self.signals.cancelled.emit(self.generation)
        except Exception as exc:  # noqa: BLE001 - worker/UI boundary
            self.signals.failed.emit(str(exc), self.generation)
        else:
            self.signals.finished.emit(payload, self.generation)


class _RefinementSignals(QObject):
    finished = Signal(object, str, bool, str)
    failed = Signal(str, str, bool, str)
    cancelled = Signal(str, bool, str)


class _RefinementWorker(QRunnable):
    def __init__(self, token, probability, spec, rgba, *, exact):
        super().__init__()
        self.token = token
        self.probability = probability
        self.spec = spec
        self.rgba = rgba
        self.exact = exact
        self.request_id = str(uuid4())
        self.signals = _RefinementSignals()
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    @Slot()
    def run(self):
        try:
            derive = (
                derive_refined_smart_mask if self.exact
                else derive_refined_smart_preview)
            pixels = derive(
                self.probability, self.spec, rgba=self.rgba,
                is_cancelled=self._cancelled.is_set)
        except RenderCancelled:
            self.signals.cancelled.emit(
                self.token, self.exact, self.request_id)
        except Exception as exc:
            self.signals.failed.emit(
                str(exc), self.token, self.exact, self.request_id)
        else:
            self.signals.finished.emit(
                pixels, self.token, self.exact, self.request_id)


def _thumbnail_key(layer: Layer) -> tuple:
    source = layer.source
    if source is None:
        raise ValueError("document layers must own a source")
    mask = layer.raster_mask
    return (
        layer.id,
        source.source_identity,
        layer.look.signature,
        layer.mask_revision,
        layer.transform,
        None if mask is None else MaskObjectIdentity(mask),
        0 if mask is None else mask.revision,
        False if mask is None else mask.enabled,
        0 if mask is None else mask.density,
        layer.visible,
        layer.opacity,
        layer.blend_mode,
    )


def _publication_key(layer, document, cap, generation):
    mask = layer.raster_mask
    return MaskPublicationKey(
        layer.id,
        layer.source.source_identity,
        None if mask is None else MaskObjectIdentity(mask),
        0 if mask is None else mask.revision,
        False if mask is None else mask.enabled,
        0 if mask is None else mask.density,
        layer.transform,
        document.revision,
        (document.canvas.width, document.canvas.height, cap),
        generation,
    )


def _raster_mask_key(layer: Layer) -> tuple:
    mask = layer.raster_mask
    return (
        None if mask is None else MaskObjectIdentity(mask),
        0 if mask is None else mask.revision,
        False if mask is None else mask.enabled,
        0 if mask is None else mask.density,
    )


def _layer_visual_key(layer: Layer) -> tuple:
    source = layer.source
    if source is None:
        raise ValueError("document layers must own sources")
    return (
        layer.id,
        source.source_identity,
        None if source.probability is None else source.probability.identity,
        layer.look.signature,
        layer.mask_revision,
        _raster_mask_key(layer),
        layer.transform,
        layer.visible,
        layer.opacity,
        layer.blend_mode,
    )


def _active_layer_guard(layer: Layer) -> tuple:
    """Everything except the active Look, which may advance during a drag."""
    source = layer.source
    if source is None:
        raise ValueError("document layers must own sources")
    return (
        layer.id,
        source.source_identity,
        None if source.probability is None else source.probability.identity,
        layer.mask_revision,
        _raster_mask_key(layer),
        layer.transform,
        layer.visible,
        layer.opacity,
        layer.blend_mode,
    )


def _lower_preview_key(document: LayerDocument, index: int) -> tuple:
    active = document.layers[index]
    return (
        (document.canvas.width, document.canvas.height),
        active.id,
        index,
        tuple(_layer_visual_key(layer) for layer in document.layers[:index]),
    )


def _interactive_graph_guard(document: LayerDocument, index: int) -> tuple:
    return (
        (document.canvas.width, document.canvas.height),
        document.selected_ids,
        index,
        tuple(
            _active_layer_guard(layer)
            if layer.id == document.layers[index].id
            else _layer_visual_key(layer)
            for layer in document.layers
        ),
    )


class LayersController(QObject):
    active_geometry_changed = Signal()
    history_changed = Signal()

    """Own immutable layer state and latest-wins asynchronous preview work."""

    place_requested = Signal()

    def __init__(
        self,
        panel,
        registry,
        preset_provider,
        source_provider,
        cap_provider,
        frame_sink,
        apply_preset,
        apply_source=None,
        mutation_guard=None,
        proxy_sink=None,
        proxy_geometry_sink=None,
        proxy_clear=None,
        mask_caches_provider=None,
        smart_mask_readiness_provider=None,
        mask_stroke_sink=None,
        mask_stroke_clear=None,
        refinement_preview_sink=None,
        refinement_preview_clear=None,
        selection_overlay_sink=None,
        selection_overlay_clear=None,
        parent=None,
    ):
        super().__init__(parent)
        self.panel = panel
        self.registry = registry
        self._preset_provider = preset_provider
        self._source_provider = source_provider
        self._cap_provider = cap_provider
        self._frame_sink = frame_sink
        self._apply_preset = apply_preset
        self._apply_source = apply_source or (
            lambda _gray, _rgba, _probability: None
        )
        self._mutation_guard = mutation_guard or (lambda: True)
        self._proxy_sink = proxy_sink or (lambda *_args: None)
        self._proxy_geometry_sink = proxy_geometry_sink or (lambda *_args: None)
        self._proxy_clear = proxy_clear or (lambda: None)
        self._mask_caches_provider = mask_caches_provider or (lambda: None)
        self._smart_mask_readiness_provider = (
            smart_mask_readiness_provider or self._default_smart_mask_readiness)
        self._mask_stroke_visual_enabled = mask_stroke_sink is not None
        self._mask_stroke_sink = mask_stroke_sink or (lambda *_args: None)
        self._mask_stroke_clear = mask_stroke_clear or (lambda: None)
        self._refinement_preview_sink = (
            refinement_preview_sink or (lambda *_args: None))
        self._refinement_preview_clear = (
            refinement_preview_clear or (lambda: None))
        self._selection_overlay_sink = (
            selection_overlay_sink or (lambda _pixels: None))
        self._selection_overlay_clear = (
            selection_overlay_clear or (lambda: None))
        self._look_cache = LayerLookRenderCache()
        self._history = LayerDocumentHistory()

        self._document: LayerDocument | None = None
        self._next_layer_number = 1
        self._pool = QThreadPool.globalInstance()
        self._workers: set[_LayerWorker] = set()
        self._busy = False
        self._pending_request = None
        self._generation = 0
        self._closing = False
        self._active_generation = None
        self._thumbnail_cache: dict[tuple[str, tuple], QPixmap] = {}
        self._transform_gesture: tuple[str, tuple] | None = None
        self._transform_transaction: LayerDocument | None = None
        self._latest_proxy = None
        self._latest_proxy_lower_key = None
        self._interactive_lower_key = None
        self._interactive_lower_context = None
        self._edit_target: LayerEditTarget | None = None
        self._mask_replacement: MaskReplacementProposal | None = None
        self._mask_combination: MaskCombinationAuthority | None = None
        self._mask_stroke = None
        self._mask_stroke_overlay_pending_exact = False
        self._inspection_mode = InspectionMode.NORMAL
        self._inspection_before_transform = None
        self._accepted_composite = None
        self._smart_refinement: tuple[
            str, SmartRefinementAuthority, SmartRefinementSpec] | None = None
        self._refinement_pool = QThreadPool(self)
        self._refinement_pool.setMaxThreadCount(1)
        self._refinement_worker: _RefinementWorker | None = None
        self._refinement_workers: dict[str, _RefinementWorker] = {}
        self._refinement_pending_spec: SmartRefinementSpec | None = None
        self._refinement_confirm_pending = False
        self._smart_replacement_authority = None
        # Selection is intentionally controller-session state, not document
        # state: it is neither serialized nor recorded by layer history.
        self._selection: TemporarySelection | None = None
        self._color_range_session = None
        self._color_range_preview: TemporarySelection | None = None
        self._color_range_confirm_pending = False
        self._selection_generation = 0
        self._selection_pool = QThreadPool(self)
        self._selection_pool.setMaxThreadCount(1)
        self._selection_worker: _SelectionWorker | None = None
        self._selection_workers: dict[str, _SelectionWorker] = {}
        self._selection_pending_task: _SelectionTask | None = None

        if hasattr(panel, "new_blank_requested"):
            panel.new_blank_requested.connect(self.new_blank_layer)
        else:
            panel.add_requested.connect(self.add_layer)
        if hasattr(panel, "place_image_requested"):
            panel.place_image_requested.connect(self.place_requested.emit)
        panel.duplicate_requested.connect(self.duplicate_layer)
        panel.delete_requested.connect(self.delete_layer)
        panel.move_requested.connect(self.move_layer)
        panel.visibility_changed.connect(self.set_visibility)
        panel.name_changed.connect(self.set_name)
        panel.blend_changed.connect(self.set_blend_mode)
        panel.opacity_changed.connect(self.set_opacity)
        panel.transform_changed.connect(self.set_transform)
        panel.center_requested.connect(self.center_layer)
        panel.fit_requested.connect(self.fit_layer)
        panel.selection_changed.connect(self.activate_layer)
        panel.reveal_mask_requested.connect(self._reveal_mask_requested)
        panel.hide_mask_requested.connect(self._hide_mask_requested)
        panel.transparency_mask_requested.connect(
            self._transparency_mask_requested)
        panel.mask_enabled_changed.connect(self._mask_enabled_requested)
        panel.mask_density_changed.connect(self._mask_density_requested)
        if hasattr(panel, "smart_mask_requested"):
            panel.smart_mask_requested.connect(self._smart_mask_requested)
            panel.invert_mask_requested.connect(self.invert_active_raster_mask)
            panel.fill_white_mask_requested.connect(
                lambda: self.fill_active_raster_mask(255))
            panel.fill_black_mask_requested.connect(
                lambda: self.fill_active_raster_mask(0))
            panel.reset_mask_requested.connect(self.reset_active_raster_mask)
            panel.delete_mask_requested.connect(self.delete_active_raster_mask)
        if hasattr(panel, "smart_refinement_started"):
            panel.smart_refinement_started.connect(
                self.start_smart_refinement)
            panel.smart_refinement_changed.connect(
                self.update_smart_refinement)
            panel.smart_refinement_confirmed.connect(
                self.confirm_smart_refinement)
            panel.smart_refinement_cancelled.connect(
                self.cancel_smart_refinement)
        if hasattr(panel, "luminance_mask_requested"):
            panel.luminance_mask_requested.connect(
                lambda name: self.generate_luminance_mask(
                    LuminancePreset(str(name)),
                    self._panel_mask_combination_mode()))
            panel.gradient_mask_requested.connect(
                self._gradient_mask_requested)
            panel.pattern_mask_requested.connect(
                self._pattern_mask_requested)
        if hasattr(panel, "edit_target_requested"):
            panel.edit_target_requested.connect(self.set_edit_target)
        if hasattr(panel, "inspection_mode_changed"):
            panel.inspection_mode_changed.connect(self.set_inspection_mode)
        if hasattr(panel, "mask_replacement_confirmed"):
            panel.mask_replacement_confirmed.connect(
                self.confirm_mask_replacement)
            panel.mask_replacement_cancelled.connect(
                self.cancel_mask_replacement)
        self._refresh_panel(None)

    def _graph_mutation_allowed(self) -> bool:
        """Second-line guard for non-geometry document changes and exports."""
        allowed = (
            not self._closing
            and self._mask_stroke is None
            and self._smart_refinement is None
            and bool(self._mutation_guard())
        )
        if not allowed and self._mask_stroke is not None:
            self.panel.set_status(
                "Finish or cancel the active mask brush stroke first.", True)
        return allowed

    @staticmethod
    def _log_layer_action(action: str, layer: Layer, **fields) -> None:
        """Record metadata-only semantic layer actions."""
        source = layer.source
        if source is not None:
            fields.setdefault("source_width", int(source.rgba.shape[1]))
            fields.setdefault("source_height", int(source.rgba.shape[0]))
        log_action(
            f"layer.{action}",
            layer_id=layer.id,
            **fields,
        )

    @property
    def smart_refinement_active(self) -> bool:
        return self._smart_refinement is not None

    def _smart_refinement_is_current(
        self, authority: SmartRefinementAuthority
    ) -> tuple[int, Layer] | None:
        if self._document is None:
            return None
        index = next((i for i, item in enumerate(self._document.layers)
                      if item.id == authority.layer_id), None)
        if index is None or index != self.active_index:
            return None
        layer = self._document.layers[index]
        source = layer.source
        mask = layer.raster_mask
        probability = None if source is None else source.probability
        if (
            source is None
            or source.source_identity != authority.source_identity
            or (None if probability is None else probability.identity)
               != authority.probability_identity
            or layer.look.smart_mask != authority.smart_settings
            or (None if mask is None else MaskObjectIdentity(mask))
               != authority.mask_identity
            or (0 if mask is None else mask.revision)
               != authority.mask_revision
            or self._document.revision != authority.document_revision
            or self._generation != authority.generation
        ):
            return None
        return index, layer

    @Slot()
    def start_smart_refinement(self) -> bool:
        if (
            self._closing or self._smart_refinement is not None
            or self._mask_replacement is not None
            or self._mask_stroke is not None
            or self._transform_transaction is not None
            or self._transform_gesture is not None
            or not self._mutation_guard()
        ):
            return False
        index = self.active_index
        if index is None or self._document is None:
            self.panel.set_status("Select a layer first.", True)
            return False
        layer = self._document.layers[index]
        ready, reason = self._smart_mask_readiness_provider(layer)
        if not ready:
            self.panel.set_status(
                reason or "Smart Mask is unavailable or pending.", True)
            return False
        source = layer.source
        assert source is not None
        settings = layer.look.smart_mask
        probability = source.probability
        spec = SmartRefinementSpec(
            settings.target, settings.sensitivity, settings.invert)
        mask = layer.raster_mask
        authority = SmartRefinementAuthority(
            layer.id, source.source_identity,
            None if probability is None else probability.identity,
            settings, None if mask is None else MaskObjectIdentity(mask),
            0 if mask is None else mask.revision,
            self._document.revision, self._generation)
        token = str(uuid4())
        self._smart_refinement = (token, authority, spec)
        if hasattr(self.panel, "show_smart_refinement"):
            self.panel.show_smart_refinement(
                token, settings.sensitivity, settings.invert)
        return self._publish_smart_refinement_preview()

    def _publish_smart_refinement_preview(self) -> bool:
        transaction = self._smart_refinement
        if transaction is None:
            return False
        token, authority, spec = transaction
        current = self._smart_refinement_is_current(authority)
        if current is None:
            self.cancel_smart_refinement(token)
            self.panel.set_status("Smart refinement became stale.", True)
            return False
        _index, layer = current
        self._refinement_pending_spec = spec
        if self._refinement_worker is not None:
            self._refinement_worker.cancel()
            return True
        self._start_refinement_worker(layer, spec, exact=False)
        return True

    def _start_refinement_worker(self, layer, spec, *, exact):
        token = self._smart_refinement[0]
        worker = _RefinementWorker(
            token, layer.source.probability, spec, layer.source.rgba,
            exact=exact)
        self._refinement_worker = worker
        self._refinement_workers[worker.request_id] = worker
        self._refinement_pending_spec = None
        worker.signals.finished.connect(self._refinement_finished)
        worker.signals.failed.connect(self._refinement_failed)
        worker.signals.cancelled.connect(self._refinement_cancelled)
        self._refinement_pool.start(worker)

    def _retire_refinement_worker(self, request_id: str) -> bool:
        worker = self._refinement_workers.pop(request_id, None)
        if worker is None or worker is not self._refinement_worker:
            return False
        self._refinement_worker = None
        return True

    def _advance_refinement_after_stale_terminal(self) -> None:
        transaction = self._smart_refinement
        if transaction is None or self._refinement_worker is not None:
            return
        current = self._smart_refinement_is_current(transaction[1])
        if current is None:
            return
        spec = self._refinement_pending_spec or transaction[2]
        self._start_refinement_worker(
            current[1], spec, exact=self._refinement_confirm_pending)

    @Slot(object, str, bool, str)
    def _refinement_finished(self, pixels, token, exact, request_id):
        if not self._retire_refinement_worker(request_id):
            return
        transaction = self._smart_refinement
        if transaction is None or token != transaction[0]:
            self._advance_refinement_after_stale_terminal()
            return
        _token, authority, spec = transaction
        current = self._smart_refinement_is_current(authority)
        if current is None:
            self.cancel_smart_refinement(token)
            return
        if exact:
            if not self._refinement_confirm_pending:
                return
            self._refinement_confirm_pending = False
            self._finish_smart_refinement_ui(retain_authority=authority)
            self.propose_active_raster_mask(
                pixels, "Frozen Smart refinement")
            if self._mask_replacement is not None:
                self._smart_replacement_authority = (
                    self._mask_replacement.token, authority, spec)
            return
        if self._refinement_confirm_pending:
            _index, layer = current
            self._start_refinement_worker(layer, transaction[2], exact=True)
            return
        if spec == transaction[2]:
            self._refinement_preview_sink(pixels, token)
        if self._refinement_pending_spec is not None:
            _index, layer = current
            self._start_refinement_worker(
                layer, self._refinement_pending_spec, exact=False)

    @Slot(str, str, bool, str)
    def _refinement_failed(self, message, token, exact, request_id):
        if not self._retire_refinement_worker(request_id):
            return
        if self._smart_refinement is not None and token == self._smart_refinement[0]:
            self.cancel_smart_refinement(token)
            self.panel.set_status(message, True)
        else:
            self._advance_refinement_after_stale_terminal()

    @Slot(str, bool, str)
    def _refinement_cancelled(self, token, exact, request_id):
        if not self._retire_refinement_worker(request_id):
            return
        if self._smart_refinement is None or token != self._smart_refinement[0]:
            self._advance_refinement_after_stale_terminal()
            return
        if self._refinement_confirm_pending:
            current = self._smart_refinement_is_current(
                self._smart_refinement[1])
            if current is not None:
                self._start_refinement_worker(
                    current[1], self._smart_refinement[2], exact=True)
        elif self._refinement_pending_spec is not None:
            current = self._smart_refinement_is_current(
                self._smart_refinement[1])
            if current is not None:
                self._start_refinement_worker(
                    current[1], self._refinement_pending_spec, exact=False)

    @Slot(int, str, int, int, bool)
    def update_smart_refinement(
        self, sensitivity: int, kind: str, morphology_radius: int,
        feather_radius: int, invert: bool
    ) -> bool:
        transaction = self._smart_refinement
        if transaction is None:
            return False
        token, authority, _old = transaction
        try:
            spec = SmartRefinementSpec(
                authority.smart_settings.target, sensitivity, invert,
                MaskRefinementKind(str(kind).lower()),
                morphology_radius, feather_radius)
        except (MaskRefinementError, ValueError):
            return False
        self._smart_refinement = (token, authority, spec)
        return self._publish_smart_refinement_preview()

    @Slot(str)
    def confirm_smart_refinement(self, token: str) -> bool:
        transaction = self._smart_refinement
        if transaction is None or token != transaction[0]:
            return False
        _token, authority, spec = transaction
        current = self._smart_refinement_is_current(authority)
        if current is None:
            self.cancel_smart_refinement(token)
            self.panel.set_status("Smart refinement became stale.", True)
            return False
        _index, layer = current
        ready, _reason = self._smart_mask_readiness_provider(layer)
        if not ready:
            self.cancel_smart_refinement(token)
            self.panel.set_status("Smart refinement became stale.", True)
            return False
        self._refinement_confirm_pending = True
        self._refinement_pending_spec = None
        if self._refinement_worker is not None:
            self._refinement_worker.cancel()
        else:
            self._start_refinement_worker(layer, spec, exact=True)
        if hasattr(self.panel, "set_smart_refinement_pending"):
            self.panel.set_smart_refinement_pending(True)
        return True

    def _finish_smart_refinement_ui(self, *, retain_authority=None) -> None:
        if self._refinement_worker is not None:
            self._refinement_worker.cancel()
        self._refinement_pending_spec = None
        self._refinement_confirm_pending = False
        self._smart_refinement = None
        self._refinement_preview_clear()
        if self._accepted_composite is not None:
            self._frame_sink(self._accepted_composite)
        if hasattr(self.panel, "show_smart_refinement"):
            self.panel.show_smart_refinement(None)

    @Slot(str)
    def cancel_smart_refinement(self, token: str = "") -> bool:
        transaction = self._smart_refinement
        if transaction is None or (token and token != transaction[0]):
            return False
        self._finish_smart_refinement_ui()
        self.panel.set_status("Smart refinement cancelled.")
        return True

    @property
    def mask_stroke_active(self) -> bool:
        return self._mask_stroke is not None

    @property
    def inspection_mode(self) -> InspectionMode:
        return self._inspection_mode

    def set_inspection_mode(self, mode) -> bool:
        if self._smart_refinement is not None:
            return False
        try:
            requested = (
                mode if isinstance(mode, InspectionMode)
                else InspectionMode(str(mode)))
        except ValueError:
            return False
        if self._mask_stroke is not None:
            self.panel.set_status(
                "Finish or cancel the active mask brush stroke first.", True)
            self.panel.set_inspection_mode(self._inspection_mode.value)
            return False
        if (
            self._transform_transaction is not None
            or self._transform_gesture is not None
        ):
            self.panel.set_status(
                "Confirm or cancel the active layer transform first.", True)
            self.panel.set_inspection_mode(self._inspection_mode.value)
            return False
        index = self.active_index
        layer = (
            None if index is None or self._document is None
            else self._document.layers[index])
        if layer is None or layer.raster_mask is None:
            requested = InspectionMode.NORMAL
        if requested is self._inspection_mode:
            self.panel.set_inspection_mode(requested.value)
            return False
        self._inspection_mode = requested
        self.panel.set_inspection_mode(requested.value)
        self._publish_accepted_inspection()
        return True

    def toggle_red_inspection(self) -> bool:
        index = self.active_index
        if (
            index is None or self._document is None
            or self._edit_target != LayerEditTarget(
                self._document.layers[index].id, "mask")
            or self._document.layers[index].raster_mask is None
        ):
            return False
        target = (
            InspectionMode.NORMAL
            if self._inspection_mode is InspectionMode.RED_OVERLAY
            else InspectionMode.RED_OVERLAY)
        return self.set_inspection_mode(target)

    def _publish_accepted_inspection(self) -> bool:
        composite = self._accepted_composite
        proxy = self._latest_proxy
        index = self.active_index
        layer = (
            None if index is None or self._document is None
            else self._document.layers[index])
        if composite is None:
            return False
        mode = self._inspection_mode
        if (
            layer is None or layer.raster_mask is None or proxy is None
            or proxy.layer_id != layer.id
        ):
            mode = InspectionMode.NORMAL
            self._inspection_mode = mode
            self.panel.set_inspection_mode(mode.value)
        frame = (
            composite if mode is InspectionMode.NORMAL else
            inspect_mask(composite, proxy, layer.raster_mask, mode))
        self._frame_sink(frame)
        return True

    def _live_stroke_layer(self, authority: MaskStrokeAuthority):
        if self._document is None:
            return None
        index = next((
            i for i, layer in enumerate(self._document.layers)
            if layer.id == authority.layer_id
        ), None)
        if index is None or index != self.active_index:
            return None
        layer = self._document.layers[index]
        mask = layer.raster_mask
        if (
            layer.source is None
            or layer.source.source_identity != authority.source_identity
            or mask is None
            or MaskObjectIdentity(mask) != authority.mask_identity
            or mask.revision != authority.mask_revision
            or not mask.enabled
            or layer.transform != authority.transform
            or self._document.revision != authority.document_revision
            or self._generation != authority.generation
        ):
            return None
        return index, layer

    def begin_mask_brush_stroke(
        self, document_x: float, document_y: float, settings: BrushSettings
    ) -> bool:
        """Start painting in an isolated mask copy; document state stays frozen."""
        if (
            self._closing or self._mask_stroke is not None
            or not bool(self._mutation_guard())
            or self._transform_transaction is not None
            or self._transform_gesture is not None
            or not isinstance(settings, BrushSettings)
            or self._inspection_mode is not InspectionMode.NORMAL
        ):
            return False
        index = self.active_index
        if index is None or self._document is None:
            return False
        layer = self._document.layers[index]
        mask = layer.raster_mask
        if (
            self._edit_target != LayerEditTarget(layer.id, "mask")
            or layer.source is None or mask is None or not mask.enabled
        ):
            self.panel.set_status(
                "Select an enabled layer mask before painting.", True)
            return False
        proxy = self._latest_proxy
        if self._mask_stroke_visual_enabled and (
            proxy is None or proxy.layer_id != layer.id
            or not proxy.supports_mask_stroke
        ):
            self.panel.set_status(
                "Preparing an exact mask brush preview; try again shortly.",
                True)
            if not self._busy:
                self.request_preview()
            return False
        # Retire any exact worker before binding the stroke. Its terminal may
        # still arrive, but the generation check prevents it from touching the
        # authoritative pixmap or clearing a stroke-owned overlay.
        self._generation += 1
        self._pending_request = None
        for worker in tuple(self._workers):
            worker.cancel()
        work = np.array(mask.pixels, dtype=np.uint8, order="C", copy=True)
        transform = layer.transform
        authority = MaskStrokeAuthority(
            layer.id, layer.source.source_identity, MaskObjectIdentity(mask),
            mask.revision, transform, self._document.revision, self._generation)
        stroke = BrushStroke(
            work, settings, layer_x=transform.x, layer_y=transform.y,
            scale_x=transform.scale_x, scale_y=transform.scale_y,
            rotation_degrees=transform.rotation_degrees,
            flip_x=transform.flip_x, flip_y=transform.flip_y)
        self._mask_stroke = (authority, stroke, mask)
        dirty = stroke.start(document_x, document_y)
        if dirty is not None:
            self._publish_mask_stroke_dirty(authority, stroke, mask, dirty)
        return True

    def _publish_mask_stroke_dirty(
        self, authority, stroke, original, dirty
    ) -> None:
        if not self._mask_stroke_visual_enabled:
            self._mask_stroke_sink(authority, stroke.buffer, dirty)
            return
        proxy = self._latest_proxy
        if (
            proxy is None or proxy.layer_id != authority.layer_id
            or not proxy.supports_mask_stroke
        ):
            self.cancel_mask_brush_stroke()
            return
        rect, rgba = composite_mask_stroke_roi(
            proxy, stroke.buffer, original.density, dirty)
        if rgba.size:
            self._mask_stroke_sink(authority, rect, rgba)

    def continue_mask_brush_stroke(
        self, document_x: float, document_y: float
    ) -> bool:
        active = self._mask_stroke
        if active is None:
            return False
        authority, stroke, _original = active
        if self._live_stroke_layer(authority) is None:
            self.cancel_mask_brush_stroke()
            return False
        dirty = stroke.add_point(document_x, document_y)
        if dirty is not None:
            self._publish_mask_stroke_dirty(
                authority, stroke, _original, dirty)
        return dirty is not None

    def cancel_mask_brush_stroke(self) -> bool:
        if self._mask_stroke is None:
            return False
        authority, stroke, _original = self._mask_stroke
        self._mask_stroke = None
        self._mask_stroke_clear()
        self.panel.set_status("Mask brush stroke cancelled.")
        dirty = stroke.dirty_rect
        log_action(
            "mask.brush_stroke_cancelled",
            layer_id=authority.layer_id,
            changed=dirty is not None,
        )
        return True

    def finish_mask_brush_stroke(self) -> bool:
        active = self._mask_stroke
        if active is None:
            return False
        authority, stroke, original = active
        live = self._live_stroke_layer(authority)
        self._mask_stroke = None
        if live is None:
            self._mask_stroke_clear()
            self.panel.set_status("Mask brush stroke became stale.", True)
            return False
        edited_pixels = stroke.buffer
        selection_coverage = self.selection_coverage_for_active_source()
        if selection_coverage is not None:
            edited_pixels = restrict_mask_edit(
                original.pixels, edited_pixels, selection_coverage)
        if stroke.dirty_rect is None or np.array_equal(
            edited_pixels, original.pixels
        ):
            self._mask_stroke_clear()
            self.panel.set_status("Mask brush stroke made no change.")
            return False
        index, layer = live
        self._thumbnail_cache.pop(_thumbnail_key(layer), None)
        new_mask = original.evolve(pixels=edited_pixels)
        self._mask_stroke_overlay_pending_exact = True
        self._replace(
            index, replace(layer, raster_mask=new_mask), "Brush Layer Mask")
        dirty = stroke.dirty_rect
        log_action(
            "mask.brush_stroke_completed",
            layer_id=layer.id,
            mode=stroke.settings.mode.value,
            size=float(stroke.settings.size),
            hardness=int(stroke.settings.hardness),
            strength=int(stroke.settings.strength),
            dirty_width=int(dirty.x1 - dirty.x0),
            dirty_height=int(dirty.y1 - dirty.y0),
        )
        return True

    @property
    def stack(self) -> LayerStack:
        return LayerStack(()) if self._document is None else LayerStack(
            self._document.layers
        )

    @property
    def document(self) -> LayerDocument | None:
        return self._document

    @property
    def active_index(self) -> int | None:
        return self._selected_index()

    @property
    def has_active_layer(self) -> bool:
        return self.active_index is not None

    def freeze_interactive_preview(
        self, target_max_side: int
    ) -> InteractiveLayerPreviewAuthority | None:
        """Capture active-layer metadata plus a cached lower-stack composite."""
        document = self._document
        index = self.active_index
        if document is None or index is None:
            return None
        if (
            isinstance(target_max_side, bool)
            or not isinstance(target_max_side, int)
            or target_max_side <= 0
        ):
            raise ValueError("Preview cap must be a positive integer.")
        lower_key = _lower_preview_key(document, index)
        cache_key = (lower_key, target_max_side)
        if (
            self._interactive_lower_context is None
            or self._interactive_lower_key != cache_key
        ):
            prefix = None
            proxy = self._latest_proxy
            if (
                proxy is not None
                and proxy.layer_id == document.layers[index].id
                and self._latest_proxy_lower_key == lower_key
                and proxy.prefix_rgba is not None
            ):
                prefix = proxy.prefix_rgba
            context = lower_layer_preview_context(
                document,
                self.registry,
                layer_id=document.layers[index].id,
                target_max_side=target_max_side,
                lower_rgba=prefix,
                allow_pending_masks=True,
                look_cache=self._look_cache,
                mask_caches=self._mask_caches_provider(),
            )
            self._interactive_lower_key = cache_key
            self._interactive_lower_context = context
        return InteractiveLayerPreviewAuthority(
            self._interactive_lower_context,
            document.layers[index],
            _interactive_graph_guard(document, index),
            lower_key,
        )

    def compose_interactive_preview(
        self, rendered, authority: InteractiveLayerPreviewAuthority
    ) -> np.ndarray | None:
        """Accept and composite a worker result only if cheap graph state is live."""
        if not isinstance(authority, InteractiveLayerPreviewAuthority):
            raise ValueError(
                "authority must be an InteractiveLayerPreviewAuthority")
        document = self._document
        index = self.active_index
        if (
            document is None
            or index is None
            or _lower_preview_key(document, index) != authority.lower_key
            or _interactive_graph_guard(document, index) != authority.graph_guard
        ):
            return None
        return composite_active_layer_preview(
            authority.context, authority.active_layer, rendered)

    @property
    def edit_target(self) -> LayerEditTarget | None:
        return self._edit_target

    @property
    def mask_replacement_proposal(self) -> MaskReplacementProposal | None:
        return self._mask_replacement

    @property
    def mask_combination_preview(self) -> MaskCombinationAuthority | None:
        return self._mask_combination

    @property
    def temporary_selection(self) -> TemporarySelection | None:
        return self._selection

    @property
    def selection_work_pending(self) -> bool:
        return (
            self._selection_worker is not None
            or self._selection_pending_task is not None)

    def _set_selection_pending_ui(self, pending: bool) -> None:
        if hasattr(self.panel, "set_selection_pending"):
            self.panel.set_selection_pending(bool(pending))

    def _next_selection_task(
        self, kind: str, base: TemporarySelection | None, payload: tuple
    ) -> _SelectionTask:
        assert self._document is not None
        self._selection_generation += 1
        return _SelectionTask(
            self._selection_generation, self._document, kind, base, payload)

    @staticmethod
    def _selection_task_is_heavy(task: _SelectionTask) -> bool:
        document_pixels = task.document.canvas.width * task.document.canvas.height
        if task.kind == "color_range":
            source_pixels = int(task.payload[0].shape[0] * task.payload[0].shape[1])
            return max(document_pixels, source_pixels) > _ASYNC_SELECTION_PIXELS
        return document_pixels > _ASYNC_SELECTION_PIXELS

    def _publish_selection_result(
        self, result: TemporarySelection, task: _SelectionTask
    ) -> None:
        if task.kind == "color_range":
            if self._color_range_session is None:
                return
            self._color_range_preview = result
            self._selection_overlay_sink(result.pixels)
            if self._color_range_confirm_pending:
                self._selection = result
                self._color_range_session = None
                self._color_range_preview = None
                self._color_range_confirm_pending = False
                log_action("selection.color_range_confirmed")
                self.panel.set_status("Color Range selection confirmed.")
            return
        self._selection = result
        self._selection_overlay_sink(result.pixels)
        operation = task.payload[0] if task.kind == "refine" else task.kind
        log_action("selection.published", operation=str(operation))

    def _dispatch_selection_task(self, task: _SelectionTask) -> bool:
        self._selection_pending_task = None
        if not self._selection_task_is_heavy(task):
            try:
                result = _compute_selection_task(task)
            except (TypeError, ValueError):
                return False
            if self._document is task.document and task.generation == self._selection_generation:
                self._publish_selection_result(result, task)
                return True
            return False
        self._set_selection_pending_ui(True)
        self.panel.set_status("Computing exact selection…")
        if self._selection_worker is not None:
            # One active calculation plus one replaceable trailing request keeps
            # slider/button bursts bounded while guaranteeing latest-wins.
            self._selection_pending_task = task
            return True
        self._start_selection_worker(task)
        return True

    def _start_selection_worker(self, task: _SelectionTask) -> None:
        worker = _SelectionWorker(task)
        self._selection_worker = worker
        self._selection_workers[worker.request_id] = worker
        worker.signals.finished.connect(self._selection_finished)
        worker.signals.failed.connect(self._selection_failed)
        release = lambda *_args, retained=worker: (
            _SHUTDOWN_SELECTION_WORKERS.discard(retained))
        worker.signals.finished.connect(release)
        worker.signals.failed.connect(release)
        self._selection_pool.start(worker)

    def _retire_selection_worker(self, request_id: str) -> bool:
        worker = self._selection_workers.pop(request_id, None)
        if worker is None or worker is not self._selection_worker:
            return False
        self._selection_worker = None
        return True

    def _advance_selection_queue(self) -> None:
        task = self._selection_pending_task
        self._selection_pending_task = None
        if task is not None and not self._closing:
            if (
                task.document is self._document
                and task.generation == self._selection_generation
            ):
                self._start_selection_worker(task)
                return
        self._set_selection_pending_ui(False)

    @Slot(object, object, str)
    def _selection_finished(self, result, task, request_id: str) -> None:
        if not self._retire_selection_worker(request_id):
            return
        if (
            not self._closing
            and task.document is self._document
            and task.generation == self._selection_generation
        ):
            self._publish_selection_result(result, task)
            if task.kind != "color_range" or self._color_range_session is not None:
                self.panel.set_status("Selection ready.")
        self._advance_selection_queue()

    @Slot(str, object, str)
    def _selection_failed(self, message: str, task, request_id: str) -> None:
        if not self._retire_selection_worker(request_id):
            return
        if task.generation == self._selection_generation and not self._closing:
            self.panel.set_status(message, True)
        self._advance_selection_queue()

    def cancel_pending_selection_work(self) -> None:
        self._selection_generation += 1
        self._selection_pending_task = None
        self._color_range_confirm_pending = False
        self._set_selection_pending_ui(False)

    def _invalidate_selection_for_document_change(self) -> None:
        if self._color_range_session is not None:
            self.cancel_color_range_selection()
        else:
            self.cancel_pending_selection_work()

    def clear_temporary_selection(self) -> bool:
        self.cancel_pending_selection_work()
        self._color_range_session = None
        self._color_range_preview = None
        changed = self._selection is not None
        self._selection = None
        self._selection_overlay_clear()
        if changed:
            log_action("selection.cleared")
        return changed

    def begin_color_range_selection(
        self, document_x: float, document_y: float,
        operation: SelectionOperation, *, tolerance: int, softness: int,
    ) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        layer = self._document.layers[index]
        source = layer.source
        if source is None or not isinstance(operation, SelectionOperation):
            return False
        transform = layer.transform
        sx = int(np.floor((float(document_x) - transform.x) / transform.scale_x))
        sy = int(np.floor((float(document_y) - transform.y) / transform.scale_y))
        if not (0 <= sx < source.rgba.shape[1] and 0 <= sy < source.rgba.shape[0]):
            return False
        self.cancel_pending_selection_work()
        target = tuple(int(value) for value in source.rgba[sy, sx, :3])
        self._color_range_session = (
            layer.id, self._selection, operation, target,
            int(tolerance), int(softness),
        )
        return self._preview_color_range()

    def _preview_color_range(self) -> bool:
        if self._color_range_session is None or self._document is None:
            return False
        layer_id, base, operation, target, tolerance, softness = self._color_range_session
        index = self.active_index
        if index is None or self._document.layers[index].id != layer_id:
            self.cancel_color_range_selection()
            return False
        layer = self._document.layers[index]
        if layer.source is None:
            self.cancel_color_range_selection()
            return False
        self._color_range_preview = None
        task = self._next_selection_task(
            "color_range", base,
            (
                layer.source.rgba, target, int(tolerance), int(softness),
                layer.transform, operation,
            ),
        )
        return self._dispatch_selection_task(task)

    def update_color_range_selection(self, tolerance: int, softness: int) -> bool:
        if self._color_range_session is None:
            return False
        layer_id, base, operation, target, _old_tolerance, _old_softness = (
            self._color_range_session)
        self._color_range_session = (
            layer_id, base, operation, target, int(tolerance), int(softness))
        return self._preview_color_range()

    def confirm_color_range_selection(self) -> bool:
        if self._color_range_session is None or self._document is None:
            return False
        layer_id, base, operation, target, tolerance, softness = self._color_range_session
        index = self.active_index
        if index is None or self._document.layers[index].id != layer_id:
            return self.cancel_color_range_selection()
        if self._color_range_preview is None:
            if self.selection_work_pending:
                self._color_range_confirm_pending = True
                self.panel.set_status("Confirming exact Color Range…")
                return True
            return self._preview_color_range()
        self._selection = self._color_range_preview
        self._color_range_session = None
        self._color_range_preview = None
        self._color_range_confirm_pending = False
        self._selection_overlay_sink(self._selection.pixels)
        log_action("selection.color_range_confirmed", operation=operation.value)
        return True

    def cancel_color_range_selection(self) -> bool:
        if self._color_range_session is None:
            return False
        _layer_id, base, _operation, _target, _tolerance, _softness = (
            self._color_range_session)
        self.cancel_pending_selection_work()
        self._color_range_session = None
        self._color_range_preview = None
        if base is None:
            self._selection_overlay_clear()
        else:
            self._selection_overlay_sink(base.pixels)
        log_action("selection.color_range_cancelled")
        return True

    def update_temporary_selection(
        self,
        shape: SelectionShape,
        bounds: tuple[float, float, float, float],
        operation: SelectionOperation = SelectionOperation.REPLACE,
    ) -> bool:
        self.cancel_color_range_selection()
        if self._document is None or self._closing:
            return False
        task = self._next_selection_task(
            "shape", self._selection, (shape, bounds, operation))
        accepted = self._dispatch_selection_task(task)
        if not accepted:
            return False
        log_action(
            "selection.updated",
            shape=shape.value,
            operation=operation.value,
            width=int(abs(bounds[2] - bounds[0])),
            height=int(abs(bounds[3] - bounds[1])),
        )
        return True

    def update_path_selection(
        self,
        kind: str,
        points,
        operation: SelectionOperation = SelectionOperation.REPLACE,
        *,
        diameter: float = 8.0,
    ) -> bool:
        self.cancel_color_range_selection()
        if self._document is None or self._closing:
            return False
        if kind not in {"polygon", "freehand"}:
            raise ValueError("selection path must be polygon or freehand")
        task = self._next_selection_task(
            "path", self._selection,
            (kind, tuple(points), operation, float(diameter)))
        if not self._dispatch_selection_task(task):
            return False
        log_action(
            "selection.updated", shape=kind, operation=operation.value,
            points=len(points),
        )
        return True

    def refine_temporary_selection(self, operation: str, radius: int = 0) -> bool:
        self.cancel_color_range_selection()
        if self._document is None or self._closing:
            return False
        if operation != "all" and self._selection is None:
            return False
        if operation not in {"all", "invert", "grow", "shrink", "feather"}:
            raise ValueError("unknown selection refinement")
        task = self._next_selection_task(
            "refine", self._selection, (operation, int(radius)))
        if not self._dispatch_selection_task(task):
            return False
        log_action("selection.refined", operation=operation, radius=radius)
        return True

    def selection_coverage_for_active_source(self) -> np.ndarray | None:
        index = self.active_index
        if (
            self._selection is None
            or index is None
            or self._document is None
        ):
            return None
        layer = self._document.layers[index]
        if source := layer.source:
            transform = layer.transform
            return selection_to_source(
                self._selection,
                source.rgba.shape[:2],
                layer_x=transform.x,
                layer_y=transform.y,
                scale_x=transform.scale_x,
                scale_y=transform.scale_y,
            )
        return None

    def create_mask_from_selection(self) -> bool:
        coverage = self.selection_coverage_for_active_source()
        if coverage is None:
            return False
        return self.propose_active_raster_mask(coverage, "Selection")

    def _cancel_mask_proposal(self) -> None:
        had_combination = self._mask_combination is not None
        self._mask_replacement = None
        self._mask_combination = None
        self._smart_replacement_authority = None
        if had_combination:
            self._refinement_preview_clear()
            if self._document is not None and not self._closing:
                self.request_preview()
        if hasattr(self.panel, "show_mask_replacement"):
            self.panel.show_mask_replacement(None)

    def _reconcile_edit_target(self) -> None:
        live_ids = set() if self._document is None else {
            layer.id for layer in self._document.layers
        }
        if self._edit_target is not None and (
            self._edit_target.layer_id not in live_ids
        ):
            self._edit_target = None
        if self._edit_target is None and self.active_index is not None:
            assert self._document is not None
            self._edit_target = LayerEditTarget(
                self._document.layers[self.active_index].id, "layer")
        if hasattr(self.panel, "set_edit_target"):
            target = self._edit_target
            self.panel.set_edit_target(
                None if target is None else target.layer_id,
                "layer" if target is None else target.kind)

    @Slot(str, str)
    def set_edit_target(self, layer_id: str, kind: str) -> bool:
        if self._closing or self._transform_transaction is not None:
            return False
        try:
            target = LayerEditTarget(str(layer_id), str(kind))
        except ValueError:
            return False
        if self._document is None:
            return False
        index = next(
            (i for i, layer in enumerate(self._document.layers)
             if layer.id == target.layer_id), None)
        if index is None:
            return False
        self._cancel_mask_proposal()
        if index != self.active_index:
            self.activate_layer(index)
        self._edit_target = target
        self._reconcile_edit_target()
        layer = self._document.layers[index]
        if target.kind == "mask":
            state = (
                "No raster mask; mask creation target selected."
                if layer.raster_mask is None else
                "Raster mask target selected."
            )
            if layer.raster_mask is not None and not layer.raster_mask.enabled:
                state = (
                    "Disabled raster mask target selected; enable it before "
                    "editing pixels.")
            self.panel.set_status(state)
        return True

    @property
    def can_edit_target_pixels(self) -> bool:
        """Whether the current target accepts direct pixel tools."""
        target = self._edit_target
        if (
            target is None or self._document is None
            or self._transform_transaction is not None
        ):
            return False
        layer = next(
            (item for item in self._document.layers
             if item.id == target.layer_id), None)
        if layer is None:
            return False
        if target.kind == "layer":
            return True
        return layer.raster_mask is None or layer.raster_mask.enabled

    def _valid_index(self, index: int) -> bool:
        return (
            not isinstance(index, bool)
            and isinstance(index, int)
            and self._smart_refinement is None
            and self._document is not None
            and 0 <= index < len(self._document.layers)
        )

    def _selected_index(self) -> int | None:
        if self._document is None or not self._document.selected_ids:
            return None
        selected_id = self._document.selected_ids[0]
        return next(
            (index for index, layer in enumerate(self._document.layers)
             if layer.id == selected_id),
            None,
        )

    def _refresh_panel(self, selected: int | None) -> None:
        if (
            selected is not None
            and self._document is not None
            and 0 <= selected < len(self._document.layers)
        ):
            selected_id = self._document.layers[selected].id
            if (
                self._edit_target is None
                or self._edit_target.layer_id != selected_id
            ):
                self._edit_target = LayerEditTarget(selected_id, "layer")
        rows = [
            (
                layer.id,
                layer.name,
                layer.visible,
                layer.opacity,
                layer.blend_mode,
                int(round(layer.transform.x)),
                int(round(layer.transform.y)),
                max(1, int(round(
                    layer.source.rgba.shape[1] * layer.transform.scale_x))),
                max(1, int(round(
                    layer.source.rgba.shape[0] * layer.transform.scale_y))),
                layer.raster_mask is not None,
                False if layer.raster_mask is None
                else layer.raster_mask.enabled,
            )
            for layer in (() if self._document is None else self._document.layers)
        ]
        self.panel.set_layers(rows, selected)
        self._reconcile_edit_target()
        active_mask = None
        if (
            selected is not None
            and self._document is not None
            and 0 <= selected < len(self._document.layers)
        ):
            active_mask = self._document.layers[selected].raster_mask
        self.panel.set_mask_state(
            active_mask is not None,
            enabled=False if active_mask is None else active_mask.enabled,
            density=100 if active_mask is None else active_mask.density,
        )
        if active_mask is None and self._inspection_mode is not InspectionMode.NORMAL:
            self._inspection_mode = InspectionMode.NORMAL
        if hasattr(self.panel, "set_inspection_mode"):
            self.panel.set_inspection_mode(self._inspection_mode.value)
        if hasattr(self.panel, "set_document_available"):
            self.panel.set_document_available(self._document is not None)
        for layer in (() if self._document is None else self._document.layers):
            pixmap = self._thumbnail_cache.get(_thumbnail_key(layer))
            if pixmap is not None:
                self.panel.set_layer_thumbnail(layer.id, pixmap)
        self.active_geometry_changed.emit()

    def _replace(
        self, index: int, layer: Layer, label: str = "Edit Layer"
    ) -> None:
        assert self._document is not None
        self._commit_document(
            self._document.replace(index, layer).select(index), label)
        self._refresh_panel(index)
        self.request_preview()

    def _commit_document(
        self, document: LayerDocument, label: str, *, record: bool = True
    ) -> bool:
        before = self._document
        if before is document:
            return False
        self._invalidate_selection_for_document_change()
        self._document = document
        self._cancel_mask_proposal()
        admitted = False
        if (
            record
            and before is not None
            and self._transform_transaction is None
        ):
            admitted = self._history.record(before, document, label)
            self.history_changed.emit()
        return admitted

    @property
    def can_undo(self) -> bool:
        return (
            self._history.can_undo
            and self._transform_transaction is None
            and self._smart_refinement is None)

    @property
    def can_redo(self) -> bool:
        return (
            self._history.can_redo
            and self._transform_transaction is None
            and self._smart_refinement is None)

    @property
    def undo_label(self) -> str | None:
        return self._history.undo_label

    @property
    def redo_label(self) -> str | None:
        return self._history.redo_label

    def _publish_history_move(self, document: LayerDocument) -> None:
        self._cancel_mask_proposal()
        self._invalidate_selection_for_document_change()
        self._document = document
        self._generation += 1
        self._pending_request = None
        for worker in tuple(self._workers):
            worker.cancel()
        self._thumbnail_cache.clear()
        self._latest_proxy = None
        self._latest_proxy_lower_key = None
        self._proxy_clear()
        index = self.active_index
        self._refresh_panel(index)
        if index is not None:
            layer = document.layers[index]
            assert layer.source is not None
            self._apply_source(
                layer.source.gray, layer.source.rgba, layer.source.probability)
            self._apply_preset(layer.look.preset)
        self.history_changed.emit()
        self.request_preview()

    @Slot()
    def undo(self) -> bool:
        if (
            not self._graph_mutation_allowed()
            or self._document is None
            or not self.can_undo
        ):
            return False
        label = self._history.undo_label
        self._publish_history_move(self._history.undo(self._document))
        log_action("history.undo", label=label)
        return True

    @Slot()
    def redo(self) -> bool:
        if (
            not self._graph_mutation_allowed()
            or self._document is None
            or not self.can_redo
        ):
            return False
        label = self._history.redo_label
        self._publish_history_move(self._history.redo(self._document))
        log_action("history.redo", label=label)
        return True

    def begin_transform_transaction(self) -> bool:
        if self._document is None or self._transform_transaction is not None:
            return False
        self._cancel_mask_proposal()
        self._inspection_before_transform = self._inspection_mode
        if self._inspection_mode is not InspectionMode.NORMAL:
            self._inspection_mode = InspectionMode.NORMAL
            self.panel.set_inspection_mode(InspectionMode.NORMAL.value)
            self._publish_accepted_inspection()
        self._transform_transaction = self._document
        index = self.active_index
        if index is not None:
            self._edit_target = LayerEditTarget(
                self._document.layers[index].id, "layer")
            self._reconcile_edit_target()
        self.history_changed.emit()
        return True

    def confirm_transform_transaction(self) -> bool:
        before = self._transform_transaction
        self._transform_transaction = None
        restore = self._inspection_before_transform
        self._inspection_before_transform = None
        if before is None or self._document is None:
            return False
        recorded = self._history.record(
            before, self._document, "Transform Layer")
        self.history_changed.emit()
        if restore is not None:
            self.set_inspection_mode(restore)
        index = self.active_index
        if recorded and index is not None:
            layer = self._document.layers[index]
            self._log_layer_action(
                "transform_confirmed", layer,
                x=int(round(layer.transform.x)),
                y=int(round(layer.transform.y)),
                width=self.active_geometry()[2],
                height=self.active_geometry()[3],
            )
        return recorded

    def cancel_transform_transaction(self) -> bool:
        before = self._transform_transaction
        self._transform_transaction = None
        restore = self._inspection_before_transform
        self._inspection_before_transform = None
        if before is None or self._document is None:
            return False
        if before is self._document:
            self.history_changed.emit()
            if restore is not None:
                self.set_inspection_mode(restore)
            return False
        # Preserve a fresh revision even though cancellation is not history.
        restored = LayerDocument(
            before.canvas, before.layers, before.selected_ids,
            self._document.revision + 1)
        self._publish_history_move(restored)
        if restore is not None:
            self.set_inspection_mode(restore)
        index = self.active_index
        if index is not None:
            self._log_layer_action(
                "transform_cancelled", restored.layers[index])
        return True

    def _commit_raster_mask(
        self, index: int, mask: RasterLayerMask | None
    ) -> None:
        """Publish one raster-mask mutation through the normal replace path."""
        assert self._document is not None
        existing = self._document.layers[index]
        self._thumbnail_cache.pop(_thumbnail_key(existing), None)
        self._replace(
            index, replace(existing, raster_mask=mask), "Edit Layer Mask")

    def propose_active_raster_mask(self, pixels, label: str) -> bool:
        """Commit a first mask, or publish a single stale-safe replacement."""
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            return False
        layer = self._document.layers[index]
        source = layer.source
        assert source is not None
        array = np.asarray(pixels)
        if array.shape != source.rgba.shape[:2]:
            raise ValueError("raster mask pixels must match the source dimensions")
        candidate = np.array(array, dtype=np.uint8, order="C", copy=True)
        candidate.flags.writeable = False
        existing = layer.raster_mask
        # Every new proposal atomically supersedes all earlier proposal kinds.
        self._cancel_mask_proposal()
        if existing is None:
            self._commit_raster_mask(
                index, RasterLayerMask(candidate, enabled=True, density=100))
            log_action(
                "mask.created", layer_id=layer.id, source=label,
                width=int(candidate.shape[1]), height=int(candidate.shape[0]))
            return True
        if (
            existing.enabled
            and existing.density == 100
            and np.array_equal(existing.pixels, candidate)
        ):
            self._cancel_mask_proposal()
            return False
        proposal = MaskReplacementProposal(
            str(uuid4()), layer.id, source.source_identity,
            MaskObjectIdentity(existing), existing.revision, candidate,
            str(label))
        self._mask_replacement = proposal
        if hasattr(self.panel, "show_mask_replacement"):
            self.panel.show_mask_replacement(
                proposal.token, f"Replace raster mask with {proposal.label}?")
        return False

    def propose_generated_mask(self, candidate: MaskGeneratorCandidate) -> bool:
        """Route every creative generator through one typed candidate boundary."""
        if not isinstance(candidate, MaskGeneratorCandidate):
            raise TypeError("generated masks require a MaskGeneratorCandidate")
        index = self.active_index
        if index is None or self._document is None:
            return False
        source = self._document.layers[index].source
        if source is None or source.source_identity != candidate.source_identity:
            return False
        return self.propose_active_raster_mask(candidate.pixels, candidate.label)

    def propose_mask_candidate(
        self,
        candidate: MaskCandidate,
        mode: MaskCombinationMode = MaskCombinationMode.REPLACE,
    ) -> bool:
        """Route all MT-15 origins through one typed, confirmed boundary."""
        if not isinstance(candidate, MaskCandidate):
            raise TypeError("mask operations require a MaskCandidate")
        if not isinstance(mode, MaskCombinationMode):
            raise TypeError("mode must be a MaskCombinationMode")
        index = self.active_index
        if index is None or self._document is None:
            return False
        layer = self._document.layers[index]
        source = layer.source
        if source is None or candidate.pixels.shape != source.rgba.shape[:2]:
            raise ValueError("mask candidate must match the active source")
        if candidate.source_identity != source.source_identity:
            return False
        coverage = self.selection_coverage_for_active_source()
        if layer.raster_mask is None:
            if mode is not MaskCombinationMode.REPLACE:
                return False
            pixels = candidate.pixels
            if coverage is not None:
                pixels = restrict_mask_edit(
                    np.full(pixels.shape, 255, dtype=np.uint8),
                    pixels,
                    coverage,
                )
            return self.propose_active_raster_mask(
                pixels, candidate.label)
        if coverage is not None:
            combined = combine_mask_candidate(
                layer.raster_mask.pixels, candidate, mode)
            restricted = restrict_mask_edit(
                layer.raster_mask.pixels, combined.pixels, coverage)
            candidate = MaskCandidate(
                restricted,
                candidate.origin,
                candidate.label,
                candidate.source_identity,
            )
            mode = MaskCombinationMode.REPLACE
        self._cancel_mask_proposal()
        token = self.begin_mask_combination(candidate, mode)
        if token is not None:
            mask = layer.raster_mask
            assert mask is not None
            # Compatibility view for the established confirmation widget. The
            # transaction authority, not these pixels, controls exact publish.
            self._mask_replacement = MaskReplacementProposal(
                token, layer.id, source.source_identity,
                MaskObjectIdentity(mask), mask.revision,
                candidate.pixels, f"{mode.value} · {candidate.label}")
            if hasattr(self.panel, "show_mask_replacement"):
                self.panel.show_mask_replacement(
                    token, f"{mode.value} raster mask with {candidate.label}?")
        return False

    def combine_generated_mask(
        self,
        candidate: MaskGeneratorCandidate,
        origin: MaskCandidateOrigin,
        mode: MaskCombinationMode,
    ) -> bool:
        if not isinstance(candidate, MaskGeneratorCandidate):
            raise TypeError("generated masks require a MaskGeneratorCandidate")
        return self.propose_mask_candidate(
            MaskCandidate(
                candidate.pixels, origin, candidate.label,
                candidate.source_identity), mode)

    def begin_mask_combination(
        self, candidate: MaskCandidate, mode: MaskCombinationMode
    ) -> str | None:
        """Publish only a capped display preview for a new latest request."""
        if (not isinstance(candidate, MaskCandidate)
                or not isinstance(mode, MaskCombinationMode)
                or not self._graph_mutation_allowed()):
            return None
        index = self.active_index
        if index is None or self._document is None:
            return None
        layer = self._document.layers[index]
        source, mask = layer.source, layer.raster_mask
        if (source is None or mask is None
                or candidate.source_identity != source.source_identity
                or candidate.pixels.shape != mask.pixels.shape):
            return None
        preview = capped_combination_preview(
            mask.pixels, candidate, mode, 720)
        token = str(uuid4())
        self._mask_combination = MaskCombinationAuthority(
            token, layer.id, source.source_identity,
            MaskObjectIdentity(mask), mask.revision,
            self._document.revision, candidate, mode, preview)
        self._refinement_preview_sink(preview, token)
        return token

    def _current_mask_combination(
        self, token: str
    ) -> tuple[int, Layer, MaskCombinationAuthority] | None:
        authority = self._mask_combination
        if (authority is None or token != authority.token
                or self._document is None
                or self._document.revision != authority.document_revision):
            return None
        index = next((i for i, layer in enumerate(self._document.layers)
                      if layer.id == authority.layer_id), None)
        if index is None or index != self.active_index:
            return None
        layer = self._document.layers[index]
        source, mask = layer.source, layer.raster_mask
        if (source is None or mask is None
                or source.source_identity != authority.source_identity
                or authority.candidate.source_identity
                != authority.source_identity
                or MaskObjectIdentity(mask) != authority.mask_identity
                or mask.revision != authority.mask_revision):
            return None
        return index, layer, authority

    def confirm_mask_combination(self, token: str) -> bool:
        """Recompute at source size and publish at most one exact history entry."""
        live = self._current_mask_combination(token)
        # A late A result must not clear a newer B transaction.
        if live is None:
            if (self._mask_combination is not None
                    and self._mask_combination.token == token):
                self._mask_combination = None
                self._refinement_preview_clear()
                if self._document is not None and not self._closing:
                    self.request_preview()
            return False
        index, layer, authority = live
        mask = layer.raster_mask
        assert mask is not None
        exact = combine_mask_candidate(
            mask.pixels, authority.candidate, authority.mode).pixels
        self._mask_combination = None
        self._refinement_preview_clear()
        if (mask.enabled and mask.density == 100
                and np.array_equal(mask.pixels, exact)):
            return False
        self._commit_raster_mask(
            index, mask.evolve(pixels=exact, enabled=True, density=100))
        log_action(
            "mask.combination_confirmed", layer_id=layer.id,
            mode=authority.mode.value,
            origin=authority.candidate.origin.value,
        )
        return True

    def cancel_mask_combination(self, token: str) -> bool:
        authority = self._mask_combination
        if authority is None or token != authority.token:
            return False
        self._mask_combination = None
        self._refinement_preview_clear()
        self.request_preview()
        log_action(
            "mask.combination_cancelled", layer_id=authority.layer_id,
            mode=authority.mode.value,
            origin=authority.candidate.origin.value,
        )
        return True

    def _panel_mask_combination_mode(self) -> MaskCombinationMode:
        combo = getattr(self.panel, "mask_combination_combo", None)
        text = combo.currentText() if combo is not None else "Replace"
        try:
            return MaskCombinationMode(text)
        except ValueError:
            return MaskCombinationMode.REPLACE

    def generate_luminance_mask(
        self, bounds: LuminanceRange | LuminancePreset,
        mode: MaskCombinationMode = MaskCombinationMode.REPLACE,
    ) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        source = self._document.layers[index].source
        if source is None:
            return False
        return self.combine_generated_mask(
            luminance_mask(source.rgba, bounds),
            MaskCandidateOrigin.LUMINANCE, mode)

    def generate_gradient_mask(
        self, spec: GradientSpec,
        mode: MaskCombinationMode = MaskCombinationMode.REPLACE,
    ) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        source = self._document.layers[index].source
        if source is None:
            return False
        return self.combine_generated_mask(
            gradient_mask(source.rgba, spec),
            MaskCandidateOrigin.GRADIENT, mode)

    def generate_gradient_from_document_points(
        self,
        kind: GradientKind,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        layer = self._document.layers[index]
        source = layer.source
        if source is None:
            return False
        transform = layer.transform
        width = max(1, source.rgba.shape[1] - 1)
        height = max(1, source.rgba.shape[0] - 1)
        spec = GradientSpec(
            kind,
            ((float(start_x) - transform.x) / transform.scale_x) / width,
            ((float(start_y) - transform.y) / transform.scale_y) / height,
            ((float(end_x) - transform.x) / transform.scale_x) / width,
            ((float(end_y) - transform.y) / transform.scale_y) / height,
        )
        return self.generate_gradient_mask(
            spec, self._panel_mask_combination_mode())

    def generate_pattern_mask(
        self, spec: MaskPatternSpec,
        mode: MaskCombinationMode = MaskCombinationMode.REPLACE,
    ) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        source = self._document.layers[index].source
        if source is None:
            return False
        return self.combine_generated_mask(
            pattern_mask(source.rgba, spec),
            MaskCandidateOrigin.PATTERN, mode)

    def dither_active_raster_mask(self, spec: MaskPatternSpec) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        layer = self._document.layers[index]
        mask = layer.raster_mask
        source = layer.source
        if mask is None or source is None:
            return False
        return self.combine_generated_mask(
            dither_mask(mask.pixels, spec, source.source_identity),
            MaskCandidateOrigin.PATTERN, MaskCombinationMode.REPLACE)

    @Slot(str, int, int, int, int, int, int, bool)
    def _pattern_mask_requested(
        self, kind: str, scale: int, orientation: int,
        offset_x: int, offset_y: int, mix: int, seed: int, dither: bool,
    ) -> bool:
        try:
            spec = MaskPatternSpec(
                MaskPatternKind(kind), scale, orientation,
                offset_x, offset_y, mix, seed)
        except (ValueError, TypeError):
            self.panel.set_status("Pattern settings are unavailable.", True)
            return False
        if dither:
            return self.dither_active_raster_mask(spec)
        return self.generate_pattern_mask(
            spec, self._panel_mask_combination_mode())

    @Slot(str, float, float, float, float)
    def _gradient_mask_requested(
        self, kind: str, start_x: float, start_y: float,
        end_x: float, end_y: float
    ) -> bool:
        try:
            spec = GradientSpec(
                GradientKind(kind), start_x, start_y, end_x, end_y)
        except (ValueError, TypeError):
            self.panel.set_status(
                "Gradient geometry needs two distinct points.", True)
            return False
        return self.generate_gradient_mask(
            spec, self._panel_mask_combination_mode())

    @Slot(str)
    def confirm_mask_replacement(self, token: str) -> bool:
        if (self._mask_combination is not None
                and token == self._mask_combination.token):
            result = self.confirm_mask_combination(token)
            self._mask_replacement = None
            if hasattr(self.panel, "show_mask_replacement"):
                self.panel.show_mask_replacement(None)
            return result
        proposal = self._mask_replacement
        if proposal is None or token != proposal.token:
            return False
        smart_authority = self._smart_replacement_authority
        if smart_authority is not None:
            smart_token, authority, _spec = smart_authority
            if (
                smart_token != token
                or self._smart_refinement_is_current(authority) is None
            ):
                self._cancel_mask_proposal()
                return False
        if not self._graph_mutation_allowed() or self._document is None:
            self._cancel_mask_proposal()
            return False
        index = next(
            (i for i, layer in enumerate(self._document.layers)
             if layer.id == proposal.layer_id), None)
        if index is None or index != self.active_index:
            self._cancel_mask_proposal()
            return False
        layer = self._document.layers[index]
        source = layer.source
        mask = layer.raster_mask
        if (
            source is None
            or source.source_identity != proposal.source_identity
            or mask is None
            or MaskObjectIdentity(mask) != proposal.mask_identity
            or mask.revision != proposal.mask_revision
        ):
            self._cancel_mask_proposal()
            return False
        self._mask_replacement = None
        self._smart_replacement_authority = None
        if hasattr(self.panel, "show_mask_replacement"):
            self.panel.show_mask_replacement(None)
        self._commit_raster_mask(
            index, mask.evolve(
                pixels=proposal.pixels, enabled=True, density=100))
        log_action(
            "mask.replacement_confirmed", layer_id=layer.id,
            source=proposal.label,
            width=int(proposal.pixels.shape[1]),
            height=int(proposal.pixels.shape[0]),
        )
        return True

    @Slot(str)
    def cancel_mask_replacement(self, token: str) -> bool:
        if (self._mask_combination is not None
                and token == self._mask_combination.token):
            result = self.cancel_mask_combination(token)
            self._mask_replacement = None
            if result and hasattr(self.panel, "show_mask_replacement"):
                self.panel.show_mask_replacement(None)
            return result
        proposal = self._mask_replacement
        if proposal is None or token != proposal.token:
            return False
        log_action(
            "mask.replacement_cancelled", layer_id=proposal.layer_id,
            source=proposal.label,
        )
        self._cancel_mask_proposal()
        return True

    def replace_active_raster_mask(
        self, pixels, *, replace_existing: bool
    ) -> bool:
        """Compatibility entry point routed through the confirmation boundary.

        ``replace_existing`` is retained only for source compatibility; it never
        authorizes replacement.
        """
        if type(replace_existing) is not bool:
            raise ValueError("replace_existing must be a bool")
        return self.propose_active_raster_mask(pixels, "replacement")

    def reveal_all_raster_mask(self, *, replace_existing: bool) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        source = self._document.layers[index].source
        assert source is not None
        if type(replace_existing) is not bool:
            raise ValueError("replace_existing must be a bool")
        return self.propose_active_raster_mask(
            np.full(source.rgba.shape[:2], 255, dtype=np.uint8), "Reveal All")

    def hide_all_raster_mask(self, *, replace_existing: bool) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        source = self._document.layers[index].source
        assert source is not None
        if type(replace_existing) is not bool:
            raise ValueError("replace_existing must be a bool")
        return self.propose_active_raster_mask(
            np.zeros(source.rgba.shape[:2], dtype=np.uint8), "Hide All")

    def raster_mask_from_transparency(
        self, *, replace_existing: bool
    ) -> bool:
        index = self.active_index
        if index is None or self._document is None:
            return False
        source = self._document.layers[index].source
        assert source is not None
        if type(replace_existing) is not bool:
            raise ValueError("replace_existing must be a bool")
        return self.propose_active_raster_mask(
            source.rgba[..., 3], "From Transparency")

    def set_active_raster_mask_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        changed = self._evolve_active_raster_mask(enabled=enabled)
        if changed:
            layer = self._document.layers[self.active_index]
            log_action(
                "mask.enabled_changed", layer_id=layer.id,
                enabled=enabled)
        return changed

    def set_active_raster_mask_density(self, density: int) -> bool:
        if (
            isinstance(density, bool)
            or not isinstance(density, int)
            or not 0 <= density <= 100
        ):
            raise ValueError("density must be an integer within 0..100")
        changed = self._evolve_active_raster_mask(density=density)
        if changed:
            layer = self._document.layers[self.active_index]
            log_action(
                "mask.density_changed", layer_id=layer.id,
                density=density)
        return changed

    def _evolve_active_raster_mask(self, **changes) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            return False
        mask = self._document.layers[index].raster_mask
        if mask is None:
            return False
        if all(getattr(mask, key) == value for key, value in changes.items()):
            return False
        self._commit_raster_mask(index, mask.evolve(**changes))
        return True

    def invert_active_raster_mask(self) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            self.panel.set_status("Select a layer with a raster mask.", True)
            return False
        mask = self._document.layers[index].raster_mask
        if mask is None:
            self.panel.set_status("No raster mask to invert.", True)
            return False
        pixels = invert_raster_mask(mask.pixels)
        coverage = self.selection_coverage_for_active_source()
        if coverage is not None:
            pixels = restrict_mask_edit(mask.pixels, pixels, coverage)
        self._commit_raster_mask(index, mask.evolve(pixels=pixels))
        log_action(
            "mask.inverted",
            layer_id=self._document.layers[index].id,
            selection_limited=coverage is not None,
        )
        return True

    def fill_active_raster_mask(self, value: int) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            self.panel.set_status("Select a layer with a raster mask.", True)
            return False
        mask = self._document.layers[index].raster_mask
        if mask is None:
            self.panel.set_status("No raster mask to fill.", True)
            return False
        try:
            pixels = fill_raster_mask(mask.pixels.shape, value)
        except RasterMaskOperationError as exc:
            self.panel.set_status(str(exc), True)
            return False
        coverage = self.selection_coverage_for_active_source()
        if coverage is not None:
            pixels = restrict_mask_edit(mask.pixels, pixels, coverage)
        self._commit_raster_mask(index, mask.evolve(pixels=pixels))
        log_action(
            "mask.filled", layer_id=self._document.layers[index].id,
            value=value, selection_limited=coverage is not None)
        return True

    def reset_active_raster_mask(self) -> bool:
        return self.reveal_all_raster_mask(replace_existing=False)

    def delete_active_raster_mask(self) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            self.panel.set_status("Select a layer with a raster mask.", True)
            return False
        if self._document.layers[index].raster_mask is None:
            self.panel.set_status("No raster mask to delete.", True)
            return False
        self._cancel_mask_proposal()
        layer_id = self._document.layers[index].id
        self._commit_raster_mask(index, None)
        log_action("mask.deleted", layer_id=layer_id)
        return True

    def raster_mask_from_smart(self) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            self.panel.set_status("Select a layer first.", True)
            return False
        layer = self._document.layers[index]
        source = layer.source
        assert source is not None
        ready, reason = self._smart_mask_readiness_provider(layer)
        if not ready:
            self.panel.set_status(
                reason or "Smart Mask is unavailable or pending.", True)
            return False
        try:
            pixels = freeze_smart_mask(
                source.probability, layer.look.smart_mask, rgba=source.rgba)
        except RasterMaskOperationError as exc:
            self.panel.set_status(str(exc), True)
            return False
        return self.propose_mask_candidate(
            MaskCandidate(
                pixels, MaskCandidateOrigin.SMART, "From Smart Mask",
                source.source_identity),
            self._panel_mask_combination_mode())

    @staticmethod
    def _default_smart_mask_readiness(layer: Layer) -> tuple[bool, str]:
        """Honest headless authority for controller-only tests/integrations."""
        settings = layer.look.smart_mask
        if not settings.enabled:
            return False, "Enable Smart Mask before freezing it."
        probability = layer.source.probability
        if settings.target.value == "whole_image":
            return True, ""
        if probability is None:
            return False, "Smart Mask probability is unavailable or pending."
        if probability.identity.source != layer.source.source_identity:
            return False, "Smart Mask probability is stale for this layer source."
        return True, ""

    def import_active_raster_mask(self, path: str | Path) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            self.panel.set_status("Select a layer before importing a mask.", True)
            return False
        source = self._document.layers[index].source
        assert source is not None
        try:
            pixels = import_raster_mask_png(path, source.rgba.shape[:2])
        except RasterMaskIOError as exc:
            self.panel.set_status(str(exc), True)
            return False
        return self.propose_mask_candidate(
            MaskCandidate(
                pixels, MaskCandidateOrigin.IMPORTED, "Imported PNG",
                source.source_identity),
            self._panel_mask_combination_mode())

    def export_active_raster_mask(self, path: str | Path) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None or self._document is None:
            self.panel.set_status("Select a layer with a raster mask.", True)
            return False
        mask = self._document.layers[index].raster_mask
        if mask is None:
            self.panel.set_status("No raster mask to export.", True)
            return False
        try:
            export_raster_mask_png(path, mask.pixels)
        except RasterMaskIOError as exc:
            self.panel.set_status(str(exc), True)
            return False
        self.panel.set_status("Raster mask exported.")
        log_action(
            "mask.exported",
            layer_id=self._document.layers[index].id,
            format=Path(path).suffix.lower().lstrip("."),
            width=int(mask.pixels.shape[1]),
            height=int(mask.pixels.shape[0]),
        )
        return True

    @Slot(int, bool)
    def _reveal_mask_requested(
        self, index: int, replace_existing: bool
    ) -> None:
        if index == self.active_index:
            source = self._document.layers[index].source
            self.propose_active_raster_mask(
                np.full(source.rgba.shape[:2], 255, dtype=np.uint8),
                "Reveal All")

    @Slot(int, bool)
    def _hide_mask_requested(self, index: int, replace_existing: bool) -> None:
        if index == self.active_index:
            source = self._document.layers[index].source
            self.propose_active_raster_mask(
                np.zeros(source.rgba.shape[:2], dtype=np.uint8), "Hide All")

    @Slot(int, bool)
    def _transparency_mask_requested(
        self, index: int, replace_existing: bool
    ) -> None:
        if index == self.active_index:
            source = self._document.layers[index].source
            self.propose_active_raster_mask(
                source.rgba[..., 3], "From Transparency")

    @Slot()
    def _smart_mask_requested(self) -> None:
        self.raster_mask_from_smart()

    @Slot(int, bool)
    def _mask_enabled_requested(self, index: int, enabled: bool) -> None:
        if index == self.active_index:
            self.set_active_raster_mask_enabled(enabled)

    @Slot(int, int)
    def _mask_density_requested(self, index: int, density: int) -> None:
        if index == self.active_index:
            self.set_active_raster_mask_density(density)

    def begin_transform_gesture(self) -> bool:
        """Defer list and thumbnail churn during direct canvas manipulation."""
        if self._closing or self._transform_gesture is not None:
            return False
        index = self.active_index
        if index is None or self._document is None:
            return False
        layer = self._document.layers[index]
        self._transform_gesture = (layer.id, _thumbnail_key(layer))
        self._generation += 1
        self._pending_request = None
        for worker in tuple(self._workers):
            worker.cancel()
        proxy = self._latest_proxy
        if proxy is not None and proxy.layer_id == layer.id:
            self._proxy_sink(
                proxy,
                (self._document.canvas.width, self._document.canvas.height),
                self.active_geometry(),
            )
        return True

    def end_transform_gesture(self) -> bool:
        """Reconcile deferred panel and thumbnail state after a canvas gesture."""
        gesture = self._transform_gesture
        if gesture is None:
            return False
        self._transform_gesture = None
        layer_id, original_key = gesture
        if self._document is None:
            return False
        current = next(
            (layer for layer in self._document.layers if layer.id == layer_id),
            None,
        )
        if current is not None and _thumbnail_key(current) != original_key:
            self._thumbnail_cache.pop(original_key, None)
        self._refresh_panel(self.active_index)
        self.request_preview()
        return True

    @Slot()
    def initialize_source_layer(self) -> None:
        """Start a fresh still-image document with one selected base layer."""
        if self._closing:
            return
        self._cancel_mask_proposal()
        source = self._source_provider()
        if source is None:
            self.panel.set_status(
                "Open an image before creating a layer document.", error=True
            )
            return
        try:
            self.open_document(*source, activate=False)
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)

    @Slot()
    def add_layer(self) -> None:
        self.new_blank_layer()

    def open_document(
        self, gray, rgba, probability=None, *, activate: bool = True
    ) -> None:
        if not self._graph_mutation_allowed():
            return
        try:
            source = LayerSource(gray, rgba, probability)
            look = Look("Layer 1", self._preset_provider())
            layer = Layer(str(uuid4()), "Layer 1", look, source=source)
            document = LayerDocument(
                CanvasSpec(source.rgba.shape[1], source.rgba.shape[0]),
                (layer,),
                (layer.id,),
            )
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)
            return
        self._invalidate_selection_for_document_change()
        self._document = document
        self._selection = None
        self._color_range_session = None
        self._selection_overlay_clear()
        self._history.clear()
        self.history_changed.emit()
        self._look_cache.clear()
        self._generation += 1
        self._thumbnail_cache.clear()
        self._next_layer_number = 2
        self._refresh_panel(0)
        self.panel.set_status("Editing Layer 1.")
        if activate:
            # Opening through a decoded drop does not pass through
            # ImageEditor.load_array. Activate explicitly so editor fields are views
            # of the selected source. initialize_source_layer passes activate=False
            # because _replace_source_arrays has already published those fields and
            # must preserve its public identity/atomicity contract.
            self.activate_layer(0)
        else:
            self.request_preview()
        self._log_layer_action(
            "document_opened", layer,
            canvas_width=document.canvas.width,
            canvas_height=document.canvas.height,
            layer_count=1,
        )

    def place_source(
        self, gray, rgba, probability=None, name=None, *,
        _action: str = "placed",
    ) -> None:
        if not self._graph_mutation_allowed():
            return
        if self._document is None:
            self.panel.set_status(
                "Open an image before placing another source.", error=True
            )
            return
        try:
            source = LayerSource(gray, rgba, probability)
            number = self._next_layer_number
            layer_name = name if name is not None else f"Layer {number}"
            transform = LayerTransform(
                x=(self._document.canvas.width - source.rgba.shape[1]) / 2.0,
                y=(self._document.canvas.height - source.rgba.shape[0]) / 2.0,
            )
            layer = Layer(
                str(uuid4()),
                layer_name,
                Look(layer_name, self._preset_provider()),
                source=source,
                transform=transform,
            )
            document = self._document.add(layer, select=True)
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)
            return
        self._commit_document(document, "Place Layer")
        self._next_layer_number += 1
        selected = len(self._document.layers) - 1
        self._refresh_panel(selected)
        self.panel.set_status(f"Placed {layer.name}.")
        self._log_layer_action(
            _action, layer, index=selected,
            layer_count=len(self._document.layers))
        self.activate_layer(selected)

    @Slot()
    def new_blank_layer(self) -> None:
        if self._closing:
            return
        if self._document is None:
            self.panel.set_status(
                "Open an image before creating a blank layer.", error=True
            )
            return
        height = self._document.canvas.height
        width = self._document.canvas.width
        self.place_source(
            np.zeros((height, width), dtype=np.float32),
            np.zeros((height, width, 4), dtype=np.uint8),
            _action="blank_added",
        )

    @Slot(int)
    def duplicate_layer(self, index: int) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        assert self._document is not None
        original = self._document.layers[index]
        self._commit_document(
            self._document.duplicate(
                index, str(uuid4()), f"{original.name} copy"
            ).select(index + 1),
            "Duplicate Layer",
        )
        self._refresh_panel(index + 1)
        self.panel.set_status(f"Duplicated {original.name}.")
        duplicate = self._document.layers[index + 1]
        self._log_layer_action(
            "duplicated", duplicate, source_layer_id=original.id,
            index=index + 1, layer_count=len(self._document.layers))
        self.request_preview()

    @Slot(int)
    def delete_layer(self, index: int) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        assert self._document is not None
        removed = self._document.layers[index]
        self._thumbnail_cache = {
            key: pixmap
            for key, pixmap in self._thumbnail_cache.items()
            if key[0] != removed.id
        }
        document = self._document.remove(index)
        selected = (
            min(index, len(document.layers) - 1)
            if document.layers else None
        )
        if selected is not None:
            document = document.select(selected)
        self._commit_document(document, "Delete Layer")
        self._refresh_panel(selected)
        self.panel.set_status(f"Deleted {removed.name}.")
        self._log_layer_action(
            "deleted", removed, index=index,
            layer_count=len(self._document.layers))
        if selected is None:
            self.request_preview()
        else:
            self.activate_layer(selected)

    @Slot(int, int)
    def move_layer(self, index: int, new_index: int) -> None:
        if (
            not self._graph_mutation_allowed()
            or not self._valid_index(index)
            or not self._valid_index(new_index)
            or index == new_index
        ):
            return
        assert self._document is not None
        self._commit_document(
            self._document.move(index, new_index).select(new_index),
            "Move Layer",
        )
        self._refresh_panel(new_index)
        self._log_layer_action(
            "reordered", self._document.layers[new_index],
            from_index=index, to_index=new_index,
            layer_count=len(self._document.layers))
        self.request_preview()

    @Slot(int, bool)
    def set_visibility(self, index: int, visible: bool) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        assert self._document is not None
        layer = self._document.layers[index]
        if layer.visible == bool(visible):
            return
        self._replace(
            index, replace(layer, visible=bool(visible)),
            "Show Layer" if visible else "Hide Layer",
        )
        self._log_layer_action(
            "visibility_changed", self._document.layers[index],
            visible=bool(visible))

    @Slot(int, str)
    def set_name(self, index: int, name: str) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        try:
            assert self._document is not None
            layer = replace(self._document.layers[index], name=name)
        except ValueError as exc:
            self.panel.set_status(str(exc), error=True)
            return
        self._replace(index, layer, "Rename Layer")

    @Slot(int, str)
    def set_blend_mode(self, index: int, mode: str) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        try:
            assert self._document is not None
            if self._document.layers[index].blend_mode == mode:
                return
            layer = replace(self._document.layers[index], blend_mode=mode)
        except ValueError as exc:
            self.panel.set_status(str(exc), error=True)
            return
        self._replace(index, layer, "Change Blend Mode")
        self._log_layer_action(
            "blend_changed", layer, blend_mode=layer.blend_mode)

    @Slot(int, int)
    def set_opacity(self, index: int, opacity: int) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        try:
            assert self._document is not None
            if self._document.layers[index].opacity == opacity:
                return
            layer = replace(self._document.layers[index], opacity=opacity)
        except ValueError as exc:
            self.panel.set_status(str(exc), error=True)
            return
        self._replace(index, layer, "Change Layer Opacity")
        self._log_layer_action(
            "opacity_changed", layer, opacity=int(layer.opacity))

    @Slot(int, int, int, int, int)
    def set_transform(
        self, index: int, x: int, y: int, width: int, height: int
    ) -> None:
        if self._closing or not self._valid_index(index):
            return
        if (
            any(isinstance(value, bool) or not isinstance(value, int)
                for value in (x, y, width, height))
            or width <= 0
            or height <= 0
        ):
            self.panel.set_status(
                "Layer position must be integers and size must be positive.",
                error=True,
            )
            return
        assert self._document is not None
        existing = self._document.layers[index]
        source = existing.source
        assert source is not None
        transform = replace(
            existing.transform,
            x=float(x),
            y=float(y),
            scale_x=width / float(source.rgba.shape[1]),
            scale_y=height / float(source.rgba.shape[0]),
        )
        if (
            self._transform_gesture is not None
            and self._transform_gesture[0] == existing.id
        ):
            self._commit_document(
                self._document.replace(
                    index, replace(existing, transform=transform)
                ).select(index),
                "Transform Layer",
            )
            self.active_geometry_changed.emit()
            self._proxy_geometry_sink(self.active_geometry())
            return
        self._thumbnail_cache.pop(_thumbnail_key(existing), None)
        self._replace(
            index, replace(existing, transform=transform), "Transform Layer")

    @Slot(int)
    def center_layer(self, index: int) -> None:
        if self._closing or not self._valid_index(index):
            return
        assert self._document is not None
        layer = self._document.layers[index]
        source = layer.source
        assert source is not None
        width = max(1, int(round(
            source.rgba.shape[1] * layer.transform.scale_x)))
        height = max(1, int(round(
            source.rgba.shape[0] * layer.transform.scale_y)))
        self.set_transform(
            index,
            x=int(round((self._document.canvas.width - width) / 2.0)),
            y=int(round((self._document.canvas.height - height) / 2.0)),
            width=width,
            height=height,
        )

    @Slot(int)
    def fit_layer(self, index: int) -> None:
        if self._closing or not self._valid_index(index):
            return
        assert self._document is not None
        layer = self._document.layers[index]
        source = layer.source
        assert source is not None
        source_h, source_w = source.rgba.shape[:2]
        scale = min(
            self._document.canvas.width / float(source_w),
            self._document.canvas.height / float(source_h),
        )
        width = max(1, int(round(source_w * scale)))
        height = max(1, int(round(source_h * scale)))
        self.set_transform(
            index,
            x=int(round((self._document.canvas.width - width) / 2.0)),
            y=int(round((self._document.canvas.height - height) / 2.0)),
            width=width,
            height=height,
        )

    def active_geometry(self) -> tuple[int, int, int, int] | None:
        index = self.active_index
        if index is None or self._document is None:
            return None
        layer = self._document.layers[index]
        source = layer.source
        assert source is not None
        return (
            int(round(layer.transform.x)),
            int(round(layer.transform.y)),
            max(1, int(round(
                source.rgba.shape[1] * layer.transform.scale_x))),
            max(1, int(round(
                source.rgba.shape[0] * layer.transform.scale_y))),
        )

    def move_active_layer_to(self, x: int, y: int) -> None:
        geometry = self.active_geometry()
        index = self.active_index
        if geometry is None or index is None:
            return
        _old_x, _old_y, width, height = geometry
        self.set_transform(
            index, x=int(x), y=int(y), width=width, height=height)

    def resize_active_layer_to(
        self, x: int, y: int, width: int, height: int
    ) -> None:
        index = self.active_index
        if index is None:
            return
        self.set_transform(
            index,
            x=int(x),
            y=int(y),
            width=max(1, int(width)),
            height=max(1, int(height)),
        )

    @Slot(int)
    def activate_layer(self, index: int) -> None:
        if self._mask_stroke is not None:
            self.cancel_mask_brush_stroke()
        if self._smart_refinement is not None:
            self.cancel_smart_refinement(self._smart_refinement[0])
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        if index != self.active_index:
            self.cancel_color_range_selection()
            self.cancel_pending_selection_work()
        self._cancel_mask_proposal()
        try:
            assert self._document is not None
            self._document = self._document.select(index)
            layer = self._document.layers[index]
            assert layer.source is not None
            self._apply_source(
                layer.source.gray, layer.source.rgba, layer.source.probability
            )
            self._apply_preset(layer.look.preset)
        except Exception as exc:  # noqa: BLE001 - callback/UI boundary
            self.panel.set_status(str(exc), error=True)
            return
        self._refresh_panel(index)
        self.panel.set_status(f"Editing {layer.name}.")
        self._log_layer_action(
            "selected", layer, index=index,
            layer_count=len(self._document.layers))
        self.request_preview()

    def update_active_from_editor(
        self, target_max_side: int | None = None, *,
        request_preview: bool = True,
    ) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None:
            return False
        assert self._document is not None
        existing = self._document.layers[index]
        self._cancel_mask_proposal()
        try:
            look = Look(existing.look.name, self._preset_provider())
            source = existing.source
            current = self._source_provider()
            if source is not None and current is not None:
                _gray, _rgba, probability = current
                if probability is not None and probability is not source.probability:
                    source = LayerSource(
                        source.gray,
                        source.rgba,
                        probability,
                        source.source_identity,
                    )
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)
            return True
        self._thumbnail_cache.pop(_thumbnail_key(existing), None)
        mask_changed = source is not existing.source
        self._document = self._document.replace(
            index,
            replace(
                existing,
                look=look,
                source=source,
                mask_revision=(
                    existing.mask_revision + 1
                    if mask_changed else existing.mask_revision
                ),
            ),
        ).select(index)
        self._refresh_panel(index)
        if not request_preview:
            return True
        if target_max_side is None:
            self.request_preview()
        else:
            self.request_preview(target_max_side)
        return True

    def _freeze_request(self, target_max_side: int | None = None):
        if self._document is None:
            raise ValueError("Open an image before previewing layers.")
        cap = (
            self._cap_provider()
            if target_max_side is None else target_max_side
        )
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ValueError("Preview cap must be a positive integer.")
        self._generation += 1
        generation = self._generation
        missing_thumbnail_keys = tuple(
            key
            for layer in self._document.layers
            for key in (_thumbnail_key(layer),)
            if (
                self._transform_gesture is None
                or layer.id != self._transform_gesture[0]
            )
            if key not in self._thumbnail_cache
        )
        return (
            generation,
            self._document,
            cap,
            missing_thumbnail_keys,
            tuple(
                _publication_key(
                    layer, self._document, cap, generation)
                for layer in self._document.layers
            ),
        )

    @Slot()
    def request_preview(self, target_max_side: int | None = None) -> None:
        if self._closing:
            return
        try:
            request = self._freeze_request(target_max_side)
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)
            return
        if self._busy:
            self._pending_request = request
            for worker in tuple(self._workers):
                worker.cancel()
            return
        self._start_request(request)

    def _start_request(self, request) -> None:
        if self._closing:
            return
        worker = _LayerWorker(
            request, self.registry, self._look_cache,
            self._mask_caches_provider())
        self._workers.add(worker)
        self._busy = True
        self._active_generation = request[0]
        worker.signals.finished.connect(self._preview_finished)
        worker.signals.failed.connect(self._preview_failed)
        worker.signals.cancelled.connect(self._preview_cancelled)

        def release(*_args) -> None:
            self._workers.discard(worker)

        worker.signals.finished.connect(release)
        worker.signals.failed.connect(release)
        worker.signals.cancelled.connect(release)

        def release_shutdown_retention(*_args) -> None:
            _SHUTDOWN_LAYER_WORKERS.discard(worker)

        worker.signals.finished.connect(release_shutdown_retention)
        worker.signals.failed.connect(release_shutdown_retention)
        worker.signals.cancelled.connect(release_shutdown_retention)
        self.panel.set_status("Rendering layers…")
        self._pool.start(worker)

    def _terminal(self, generation: int) -> None:
        if generation != self._active_generation:
            return
        self._busy = False
        self._active_generation = None
        pending = self._pending_request
        self._pending_request = None
        if pending is not None and not self._closing:
            self._start_request(pending)

    @Slot(object, int)
    def _preview_finished(self, payload, generation: int) -> None:
        if not self._closing and generation == self._generation:
            if len(payload) == 2:
                composite, thumbnails = payload
                proxy = None
                publication_keys = None
            elif len(payload) == 3:
                composite, thumbnails, proxy = payload
                publication_keys = None
            else:
                composite, thumbnails, proxy, publication_keys = payload
            live_publication_keys = (
                None if self._document is None else tuple(
                    _publication_key(
                        layer, self._document,
                        (
                            publication_keys[0].target[2]
                            if publication_keys else self._cap_provider()
                        ),
                        generation)
                    for layer in self._document.layers
                )
            )
            if (
                publication_keys is not None
                and publication_keys != live_publication_keys
            ):
                self._terminal(generation)
                return
            live_keys = {
                _thumbnail_key(layer)
                for layer in (() if self._document is None
                              else self._document.layers)
            }
            for key, rgba in thumbnails:
                if key not in live_keys:
                    continue
                pixmap = QPixmap.fromImage(numpy_to_qimage(rgba)).copy()
                self._thumbnail_cache[key] = pixmap
                self.panel.set_layer_thumbnail(key[0], pixmap)
            self._latest_proxy = proxy
            selected_index = self.active_index
            self._latest_proxy_lower_key = (
                None
                if proxy is None or self._document is None or selected_index is None
                else _lower_preview_key(self._document, selected_index)
            )
            accepted = np.array(
                composite, dtype=np.uint8, order="C", copy=True)
            accepted.flags.writeable = False
            self._accepted_composite = accepted
            if self._smart_refinement is None:
                self._publish_accepted_inspection()
            if self._mask_stroke_overlay_pending_exact:
                self._mask_stroke_overlay_pending_exact = False
                self._mask_stroke_clear()
            self._proxy_clear()
            if self._smart_refinement is None:
                self.panel.set_status("Layers ready.")
        self._terminal(generation)

    @Slot(str, int)
    def _preview_failed(self, message: str, generation: int) -> None:
        if not self._closing and generation == self._generation:
            self.panel.set_status(message, error=True)
        self._terminal(generation)

    @Slot(int)
    def _preview_cancelled(self, generation: int) -> None:
        self._terminal(generation)

    def export_current(self, path) -> Path | None:
        if not self._graph_mutation_allowed():
            return None
        if self._document is None:
            self.panel.set_status(
                "Open an image before exporting layers.", error=True)
            return None
        try:
            result = render_layer_document(
                self._document, self.registry,
                look_cache=self._look_cache,
                mask_caches=self._mask_caches_provider())
            destination = export_frame(result, path)
        except Exception as exc:  # noqa: BLE001 - export/UI boundary
            self.panel.set_status(str(exc), error=True)
            return None
        self.panel.set_status(f"Exported {destination.name}.")
        return destination

    @Slot()
    def shutdown(self) -> None:
        if self._closing:
            return
        self.cancel_mask_brush_stroke()
        if self._smart_refinement is not None:
            self.cancel_smart_refinement(self._smart_refinement[0])
        for refinement_worker in tuple(self._refinement_workers.values()):
            refinement_worker.cancel()
        self.cancel_pending_selection_work()
        for selection_worker in tuple(self._selection_workers.values()):
            _SHUTDOWN_SELECTION_WORKERS.add(selection_worker)
        self._closing = True
        self._cancel_mask_proposal()
        self._transform_gesture = None
        self._transform_transaction = None
        self._history.clear()
        self.history_changed.emit()
        self._latest_proxy = None
        self._latest_proxy_lower_key = None
        self._interactive_lower_key = None
        self._interactive_lower_context = None
        self._accepted_composite = None
        self._proxy_clear()
        self._generation += 1
        self._pending_request = None
        for worker in tuple(self._workers):
            _SHUTDOWN_LAYER_WORKERS.add(worker)
            worker.cancel()
        self._look_cache.clear()
