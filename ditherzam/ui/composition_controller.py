"""Qt controller for the still-image Look Composer."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ditherzam.diagnostics import log_action
from ditherzam.composition import (
    Composition,
    Compositor,
    Look,
    LookClip,
    LookRenderer,
    TransitionSpec,
    export_frame,
)
from ditherzam.render import RenderCancelled


_DEFAULT_DURATION = 24
_DEFAULT_TRANSITION = "crossfade"
_DEFAULT_TRANSITION_DURATION = 6
_TRANSITION_KINDS = frozenset(
    {"crossfade", "dither-dissolve", "spatial-wipe", "param-morph"}
)


class _WorkerSignals(QObject):
    finished = Signal(object, int)
    failed = Signal(str, int)
    cancelled = Signal(int)


class _PreviewLookRenderer(LookRenderer):
    """Apply one frozen preview cap to every Look branch."""

    def __init__(self, registry, source_gray, source_rgba, *, probability, cap):
        super().__init__(
            registry, source_gray, source_rgba, probability=probability)
        self._cap = cap

    def render(self, look, *, settings=None, smart_mask=None,
               target_max_side=None, temporal_field=None, is_cancelled=None):
        return super().render(
            look,
            settings=settings,
            smart_mask=smart_mask,
            target_max_side=self._cap,
            temporal_field=temporal_field,
            is_cancelled=is_cancelled,
        )


class _CompositionWorker(QRunnable):
    """Render one immutable preview request off the GUI thread."""

    def __init__(self, request, registry):
        super().__init__()
        (
            self.generation,
            self.frame,
            self.composition,
            self.source_gray,
            self.source_rgba,
            self.probability,
            self.cap,
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
            renderer = _PreviewLookRenderer(
                self.registry,
                self.source_gray,
                self.source_rgba,
                probability=self.probability,
                cap=self.cap,
            )
            compositor = Compositor(self.composition, renderer)
            result = compositor.render_frame(self.frame)
            if self._is_cancelled():
                raise RenderCancelled
        except RenderCancelled:
            self.signals.cancelled.emit(self.generation)
        except Exception as exc:  # noqa: BLE001 - worker boundary surfaces errors
            self.signals.failed.emit(str(exc), self.generation)
        else:
            self.signals.finished.emit(result, self.generation)


class CompositionController(QObject):
    """Own Look-track state and asynchronous, latest-wins preview rendering."""

    def __init__(
        self,
        panel,
        registry,
        preset_provider,
        source_provider,
        cap_provider,
        frame_sink,
        parent=None,
    ):
        super().__init__(parent)
        self.panel = panel
        self.registry = registry
        self._preset_provider = preset_provider
        self._source_provider = source_provider
        self._cap_provider = cap_provider
        self._frame_sink = frame_sink

        self._looks: list[Look] = []
        self._durations: list[int] = []
        self._transitions: list[TransitionSpec | None] = []
        self._composition: Composition | None = None

        self._pool = QThreadPool.globalInstance()
        self._workers: set[_CompositionWorker] = set()
        self._busy = False
        self._pending_request = None
        self._generation = 0
        self._closing = False

        panel.capture_requested.connect(self.capture_current)
        panel.remove_requested.connect(self.remove_look)
        panel.clip_duration_changed.connect(self.set_clip_duration)
        panel.transition_changed.connect(self.set_transition)
        panel.frame_changed.connect(self.request_frame)
        panel.play_toggled.connect(self.set_playing)

        self._refresh_panel(None)

    @property
    def composition(self) -> Composition | None:
        return self._composition

    def _selected_index(self) -> int | None:
        index = self.panel.selected_index()
        if index is None or not 0 <= index < len(self._looks):
            return None
        return index

    def _refresh_panel(self, selected: int | None) -> None:
        rows = []
        start = 0
        for look, duration in zip(self._looks, self._durations):
            rows.append((look.name, start, start + duration))
            start += duration
        self.panel.set_clips(rows, selected)
        current = min(self.panel.frame_slider.value(), max(0, start - 1))
        self.panel.set_frame_range(start, current)

    def _rebuild(self, selected: int | None = None) -> None:
        if not self._looks:
            self._composition = None
            self._transitions.clear()
            self._refresh_panel(None)
            return

        while len(self._transitions) < len(self._looks):
            self._transitions.append(
                None if not self._transitions
                else TransitionSpec(_DEFAULT_TRANSITION, _DEFAULT_TRANSITION_DURATION)
            )
        del self._transitions[len(self._looks):]
        self._transitions[0] = None

        total = sum(self._durations)
        composition = Composition(total)
        start = 0
        for look, duration in zip(self._looks, self._durations):
            composition.add_clip(LookClip(look, start, start + duration))
            start += duration
        for index in range(1, len(self._looks)):
            spec = self._transitions[index]
            if spec is None:
                spec = TransitionSpec(
                    _DEFAULT_TRANSITION, _DEFAULT_TRANSITION_DURATION)
            maximum = min(self._durations[index - 1], self._durations[index])
            duration = min(max(1, int(spec.duration)), maximum)
            spec = TransitionSpec(spec.kind, duration, spec.params)
            self._transitions[index] = spec
            composition.set_transition(index, spec)
        composition.validate()
        self._composition = composition
        if selected is None:
            selected = self._selected_index()
        self._refresh_panel(selected)

    @Slot()
    def capture_current(self) -> None:
        if self._closing:
            return
        try:
            preset = self._preset_provider()
            look = Look(f"Look {len(self._looks) + 1}", preset)
        except Exception as exc:  # noqa: BLE001 - provider/UI boundary
            self.panel.set_status(str(exc), error=True)
            log_action("composition.capture_failed", error=str(exc))
            return
        self._looks.append(look)
        self._durations.append(_DEFAULT_DURATION)
        self._transitions.append(
            None if len(self._looks) == 1
            else TransitionSpec(_DEFAULT_TRANSITION, _DEFAULT_TRANSITION_DURATION)
        )
        self._rebuild(len(self._looks) - 1)
        self.panel.set_status(f"Captured {look.name}.")
        log_action("composition.look_captured", look=look.name, look_count=len(self._looks))

    @Slot(int)
    def remove_look(self, index: int) -> None:
        if isinstance(index, bool) or not 0 <= index < len(self._looks):
            return
        removed = self._looks[index]
        del self._looks[index]
        del self._durations[index]
        del self._transitions[index]
        selected = min(index, len(self._looks) - 1) if self._looks else None
        self._rebuild(selected)
        self.panel.set_status("Look removed.")
        log_action("composition.look_removed", index=index, look=removed.name,
                   look_count=len(self._looks))

    @Slot(int, int)
    def set_clip_duration(self, index: int, duration: int) -> None:
        if (isinstance(index, bool) or isinstance(duration, bool)
                or not 0 <= index < len(self._looks)):
            return
        self._durations[index] = min(3600, max(1, int(duration)))
        self._rebuild(index)
        log_action("composition.clip_duration_changed", index=index,
                   duration=self._durations[index])

    @Slot(int, str, int)
    def set_transition(self, index: int, kind: str, duration: int) -> None:
        if (isinstance(index, bool) or index <= 0
                or index >= len(self._looks) or kind not in _TRANSITION_KINDS):
            return
        maximum = min(self._durations[index - 1], self._durations[index])
        duration = min(maximum, max(1, int(duration)))
        self._transitions[index] = TransitionSpec(kind, duration)
        self._rebuild(index)
        log_action("composition.transition_changed", index=index, kind=kind,
                   duration=duration)

    @Slot(bool)
    def set_playing(self, playing: bool) -> None:
        if not playing:
            self.panel.stop_playback()

    def _freeze_request(self, frame: int):
        source = self._source_provider()
        if source is None:
            raise ValueError("Open an image before previewing a composition.")
        gray, rgba, probability = source
        cap = self._cap_provider()
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ValueError("Preview cap must be a positive integer.")
        gray = np.array(gray, dtype=np.float32, order="C", copy=True)
        rgba = np.array(rgba, dtype=np.uint8, order="C", copy=True)
        gray.flags.writeable = False
        rgba.flags.writeable = False
        self._generation += 1
        return (
            self._generation, frame, self._composition, gray, rgba,
            probability, cap,
        )

    @Slot(int)
    def request_frame(self, frame: int) -> None:
        if self._closing or self._composition is None:
            return
        frame = min(max(0, int(frame)), self._composition.length - 1)
        try:
            request = self._freeze_request(frame)
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
        worker = _CompositionWorker(request, self.registry)
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
        self.panel.set_status("Rendering preview…")
        self._pool.start(worker)

    def _terminal(self) -> None:
        self._busy = False
        pending = self._pending_request
        self._pending_request = None
        if pending is not None and not self._closing:
            self._start_request(pending)

    @Slot(object, int)
    def _preview_finished(self, frame, generation: int) -> None:
        if not self._closing and generation == self._generation:
            self._frame_sink(frame)
            self.panel.set_status("Preview ready.")
        self._terminal()

    @Slot(str, int)
    def _preview_failed(self, message: str, generation: int) -> None:
        if not self._closing and generation == self._generation:
            self.panel.set_status(message, error=True)
            self.panel.stop_playback()
        self._terminal()

    @Slot(int)
    def _preview_cancelled(self, _generation: int) -> None:
        self._terminal()

    def export_current(self, path) -> Path | None:
        if self._closing:
            log_action("composition.export_failed", reason="closing")
            return None
        if self._composition is None:
            self.panel.set_status("Capture a Look before exporting.", error=True)
            log_action("composition.export_failed", reason="no_composition")
            return None
        source = self._source_provider()
        if source is None:
            self.panel.set_status(
                "Open an image before exporting a composition.", error=True)
            log_action("composition.export_failed", reason="no_source")
            return None
        frame = min(
            max(0, int(self.panel.frame_slider.value())),
            self._composition.length - 1,
        )
        try:
            gray, rgba, probability = source
            renderer = LookRenderer(
                self.registry, gray, rgba, probability=probability)
            result = Compositor(self._composition, renderer).render_frame(frame)
            destination = export_frame(result, path)
        except Exception as exc:  # noqa: BLE001 - export/UI boundary
            self.panel.set_status(str(exc), error=True)
            log_action("composition.export_failed", error=str(exc), path=path)
            return None
        self.panel.set_status(f"Exported {destination.name}.")
        log_action("composition.export_succeeded", frame=frame, path=destination)
        return destination

    @Slot()
    def shutdown(self) -> None:
        """Stop playback and make all queued/running preview work unpublishable."""
        if self._closing:
            return
        self._closing = True
        self.panel.stop_playback()
        self._generation += 1
        self._pending_request = None
        for worker in tuple(self._workers):
            worker.cancel()
