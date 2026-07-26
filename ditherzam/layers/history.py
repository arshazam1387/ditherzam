from __future__ import annotations

from dataclasses import dataclass, replace

from .model import Layer, LayerDocument


DEFAULT_HISTORY_MAX_ENTRIES = 64
DEFAULT_HISTORY_MAX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class LayerDocumentHistoryEntry:
    before: LayerDocument
    after: LayerDocument
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.before, LayerDocument):
            raise ValueError("before must be a LayerDocument")
        if not isinstance(self.after, LayerDocument):
            raise ValueError("after must be a LayerDocument")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")


def _payloads(document: LayerDocument):
    for layer in document.layers:
        source = layer.source
        if source is not None:
            yield source.gray
            yield source.rgba
            if source.probability is not None:
                yield source.probability.values
        if layer.raster_mask is not None:
            yield layer.raster_mask.pixels


def _retained_bytes(entries: list[LayerDocumentHistoryEntry]) -> int:
    seen: set[int] = set()
    total = 0
    for entry in entries:
        for document in (entry.before, entry.after):
            for payload in _payloads(document):
                identity = id(payload)
                if identity not in seen:
                    seen.add(identity)
                    total += int(payload.nbytes)
    return total


def _controlled_signature(document: LayerDocument) -> tuple:
    return (
        document.canvas,
        tuple(
            (
                layer.id,
                layer.name,
                layer.visible,
                layer.opacity,
                layer.blend_mode,
                layer.transform,
                layer.raster_mask,
            )
            for layer in document.layers
        ),
    )


def _rebase(
    target: LayerDocument, current: LayerDocument, revision: int
) -> LayerDocument:
    live = {layer.id: layer for layer in current.layers}
    layers: list[Layer] = []
    for stored in target.layers:
        current_layer = live.get(stored.id)
        if current_layer is None:
            layers.append(stored)
        else:
            layers.append(
                replace(
                    stored,
                    look=current_layer.look,
                    source=current_layer.source,
                    mask_revision=current_layer.mask_revision,
                )
            )
    ids = {layer.id for layer in layers}
    selected = (
        current.selected_ids
        if current.selected_ids and current.selected_ids[0] in ids
        else tuple(value for value in target.selected_ids if value in ids)
    )
    return LayerDocument(target.canvas, tuple(layers), selected, revision)


class LayerDocumentHistory:
    """Bounded Qt-free history for layer graph and raster-mask mutations."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_HISTORY_MAX_ENTRIES,
        max_bytes: int = DEFAULT_HISTORY_MAX_BYTES,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise ValueError("max_entries must be an integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise ValueError("max_bytes must be an integer")
        if max_entries <= 0 or max_bytes <= 0:
            raise ValueError("history bounds must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: list[LayerDocumentHistoryEntry] = []
        self._cursor = 0
        self._next_revision = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def retained_bytes(self) -> int:
        return _retained_bytes(self._entries)

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._entries)

    @property
    def undo_label(self) -> str | None:
        return self._entries[self._cursor - 1].label if self.can_undo else None

    @property
    def redo_label(self) -> str | None:
        return self._entries[self._cursor].label if self.can_redo else None

    def clear(self) -> None:
        self._entries.clear()
        self._cursor = 0
        self._next_revision = 0

    def record(
        self, before: LayerDocument, after: LayerDocument, label: str
    ) -> bool:
        entry = LayerDocumentHistoryEntry(before, after, label)
        self._next_revision = max(
            self._next_revision, before.revision, after.revision)
        if _controlled_signature(before) == _controlled_signature(after):
            return False
        candidate = self._entries[:self._cursor] + [entry]
        if _retained_bytes([entry]) > self.max_bytes:
            self._entries = self._entries[:self._cursor]
            return False
        while (
            len(candidate) > self.max_entries
            or _retained_bytes(candidate) > self.max_bytes
        ):
            candidate.pop(0)
        self._entries = candidate
        self._cursor = len(candidate)
        return True

    def undo(self, current: LayerDocument) -> LayerDocument:
        if not self.can_undo:
            return current
        self._cursor -= 1
        self._next_revision = max(self._next_revision, current.revision) + 1
        return _rebase(
            self._entries[self._cursor].before, current, self._next_revision)

    def redo(self, current: LayerDocument) -> LayerDocument:
        if not self.can_redo:
            return current
        entry = self._entries[self._cursor]
        self._cursor += 1
        self._next_revision = max(self._next_revision, current.revision) + 1
        return _rebase(entry.after, current, self._next_revision)
