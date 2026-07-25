"""Qt controller for the still-image spatial layer stack."""
from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtGui import QPixmap

from ditherzam.composition import Look, export_frame
from ditherzam.layers import (
    CanvasSpec,
    Layer,
    LayerDocument,
    LayerSource,
    LayerStack,
    LayerTransform,
    render_layer_document,
    render_layer_document_with_proxy,
)
from ditherzam.render import RenderCancelled
from ditherzam.ui.convert import numpy_to_qimage


class _WorkerSignals(QObject):
    finished = Signal(object, int)
    failed = Signal(str, int)
    cancelled = Signal(int)


class _LayerWorker(QRunnable):
    """Render one frozen layer-document preview request."""

    def __init__(self, request, registry):
        super().__init__()
        (
            self.generation,
            self.document,
            self.cap,
            self.thumbnail_keys,
        ) = request
        self.registry = registry
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
                )
            else:
                rendered = render_layer_document_with_proxy(
                    self.document,
                    self.registry,
                    layer_id=selected_id,
                    target_max_side=self.cap,
                    is_cancelled=self._is_cancelled,
                    allow_pending_masks=True,
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
                )
                if self._is_cancelled():
                    raise RenderCancelled
                frozen = np.array(rgba, dtype=np.uint8, order="C", copy=True)
                frozen.flags.writeable = False
                thumbnails.append((key, frozen))
            payload = (composite, tuple(thumbnails), proxy)
        except RenderCancelled:
            self.signals.cancelled.emit(self.generation)
        except Exception as exc:  # noqa: BLE001 - worker/UI boundary
            self.signals.failed.emit(str(exc), self.generation)
        else:
            self.signals.finished.emit(payload, self.generation)


def _thumbnail_key(layer: Layer) -> tuple:
    source = layer.source
    if source is None:
        raise ValueError("document layers must own a source")
    return (
        layer.id,
        source.source_identity,
        layer.look.signature,
        layer.mask_revision,
        layer.transform,
    )


class LayersController(QObject):
    active_geometry_changed = Signal()

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

        self._document: LayerDocument | None = None
        self._next_layer_number = 1
        self._pool = QThreadPool.globalInstance()
        self._workers: set[_LayerWorker] = set()
        self._busy = False
        self._pending_request = None
        self._generation = 0
        self._closing = False
        self._thumbnail_cache: dict[tuple[str, tuple], QPixmap] = {}
        self._transform_gesture: tuple[str, tuple] | None = None
        self._latest_proxy = None

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
        self._refresh_panel(None)

    def _graph_mutation_allowed(self) -> bool:
        """Second-line guard for non-geometry document changes and exports."""
        return not self._closing and bool(self._mutation_guard())

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

    def _valid_index(self, index: int) -> bool:
        return (
            not isinstance(index, bool)
            and isinstance(index, int)
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
            )
            for layer in (() if self._document is None else self._document.layers)
        ]
        self.panel.set_layers(rows, selected)
        if hasattr(self.panel, "set_document_available"):
            self.panel.set_document_available(self._document is not None)
        for layer in (() if self._document is None else self._document.layers):
            pixmap = self._thumbnail_cache.get(_thumbnail_key(layer))
            if pixmap is not None:
                self.panel.set_layer_thumbnail(layer.id, pixmap)
        self.active_geometry_changed.emit()

    def _replace(self, index: int, layer: Layer) -> None:
        assert self._document is not None
        self._document = self._document.replace(index, layer).select(index)
        self._refresh_panel(index)
        self.request_preview()

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
            self._document = LayerDocument(
                CanvasSpec(source.rgba.shape[1], source.rgba.shape[0]),
                (layer,),
                (layer.id,),
            )
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)
            return
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

    def place_source(self, gray, rgba, probability=None, name=None) -> None:
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
            self._document = self._document.add(layer, select=True)
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)
            return
        self._next_layer_number += 1
        selected = len(self._document.layers) - 1
        self._refresh_panel(selected)
        self.panel.set_status(f"Placed {layer.name}.")
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
        )

    @Slot(int)
    def duplicate_layer(self, index: int) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        assert self._document is not None
        original = self._document.layers[index]
        self._document = self._document.duplicate(
            index, str(uuid4()), f"{original.name} copy"
        ).select(index + 1)
        self._refresh_panel(index + 1)
        self.panel.set_status(f"Duplicated {original.name}.")
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
        self._document = self._document.remove(index)
        selected = (
            min(index, len(self._document.layers) - 1)
            if self._document.layers else None
        )
        if selected is not None:
            self._document = self._document.select(selected)
        self._refresh_panel(selected)
        self.panel.set_status(f"Deleted {removed.name}.")
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
        self._document = self._document.move(index, new_index).select(new_index)
        self._refresh_panel(new_index)
        self.request_preview()

    @Slot(int, bool)
    def set_visibility(self, index: int, visible: bool) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        assert self._document is not None
        self._replace(
            index, replace(self._document.layers[index], visible=bool(visible))
        )

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
        self._replace(index, layer)

    @Slot(int, str)
    def set_blend_mode(self, index: int, mode: str) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        try:
            assert self._document is not None
            layer = replace(self._document.layers[index], blend_mode=mode)
        except ValueError as exc:
            self.panel.set_status(str(exc), error=True)
            return
        self._replace(index, layer)

    @Slot(int, int)
    def set_opacity(self, index: int, opacity: int) -> None:
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
        try:
            assert self._document is not None
            layer = replace(self._document.layers[index], opacity=opacity)
        except ValueError as exc:
            self.panel.set_status(str(exc), error=True)
            return
        self._replace(index, layer)

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
            self._document = self._document.replace(
                index, replace(existing, transform=transform)
            ).select(index)
            self.active_geometry_changed.emit()
            self._proxy_geometry_sink(self.active_geometry())
            return
        self._thumbnail_cache.pop(_thumbnail_key(existing), None)
        self._replace(index, replace(existing, transform=transform))

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
        if not self._graph_mutation_allowed() or not self._valid_index(index):
            return
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
        self.request_preview()

    def update_active_from_editor(self) -> bool:
        if not self._graph_mutation_allowed():
            return False
        index = self.active_index
        if index is None:
            return False
        assert self._document is not None
        existing = self._document.layers[index]
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
        self.request_preview()
        return True

    def _freeze_request(self):
        if self._document is None:
            raise ValueError("Open an image before previewing layers.")
        cap = self._cap_provider()
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ValueError("Preview cap must be a positive integer.")
        self._generation += 1
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
            self._generation,
            self._document,
            cap,
            missing_thumbnail_keys,
        )

    @Slot()
    def request_preview(self) -> None:
        if self._closing:
            return
        try:
            request = self._freeze_request()
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
        worker = _LayerWorker(request, self.registry)
        self._workers.add(worker)
        self._busy = True
        worker.signals.finished.connect(self._preview_finished)
        worker.signals.failed.connect(self._preview_failed)
        worker.signals.cancelled.connect(self._preview_cancelled)

        def release(*_args) -> None:
            self._workers.discard(worker)

        worker.signals.finished.connect(release)
        worker.signals.failed.connect(release)
        worker.signals.cancelled.connect(release)
        self.panel.set_status("Rendering layers…")
        self._pool.start(worker)

    def _terminal(self) -> None:
        self._busy = False
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
            else:
                composite, thumbnails, proxy = payload
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
            self._frame_sink(composite)
            self._proxy_clear()
            self.panel.set_status("Layers ready.")
        self._terminal()

    @Slot(str, int)
    def _preview_failed(self, message: str, generation: int) -> None:
        if not self._closing and generation == self._generation:
            self.panel.set_status(message, error=True)
        self._terminal()

    @Slot(int)
    def _preview_cancelled(self, _generation: int) -> None:
        self._terminal()

    def export_current(self, path) -> Path | None:
        if not self._graph_mutation_allowed():
            return None
        if self._document is None:
            self.panel.set_status(
                "Open an image before exporting layers.", error=True)
            return None
        try:
            result = render_layer_document(self._document, self.registry)
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
        self._closing = True
        self._transform_gesture = None
        self._latest_proxy = None
        self._proxy_clear()
        self._generation += 1
        self._pending_request = None
        for worker in tuple(self._workers):
            worker.cancel()
