"""Small, bounded caches for Smart Mask inference and outer compositing.

The staged :class:`RenderCache` intentionally remains mask-unaware.  These
three stores share one byte budget so an editor cannot exceed the mask cache
allowance by filling every partition independently.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
import threading

import numpy as np

from ditherzam.masking.contracts import (
    InferenceIdentity, MaskIdentity, ProbabilityMap, SourceIdentity,
    validate_confidence_array,
)
from ditherzam.masking.settings import OutsideMode
from ditherzam.render_cache import MAX_EDITOR_RETAINED_CACHE_BYTES, MIB, _group_storage


DEFAULT_MASK_CACHE_BUDGET_BYTES = 64 * MIB
DEFAULT_MASKED_RENDER_CACHE_BUDGET_BYTES = 128 * MIB


@dataclass(frozen=True)
class EditorCacheAllocation:
    """Budgets for the two cache instances owned by one image editor."""
    render_bytes: int
    mask_bytes: int

    def __post_init__(self) -> None:
        values = (self.render_bytes, self.mask_bytes)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("cache allocations must be integer byte counts")
        if any(value < 0 for value in values):
            raise ValueError("cache allocations must be non-negative")
        if sum(values) > MAX_EDITOR_RETAINED_CACHE_BYTES:
            raise ValueError("combined cache allocation exceeds the 192 MiB editor ceiling")


def editor_cache_allocation(mask_enabled: bool, *, render_bytes: int | None = None,
                            mask_bytes: int | None = None) -> EditorCacheAllocation:
    """Return the validated split an editor must use for its owned caches.

    SM-12 must instantiate both actual caches from this result and verify their
    summed budgets. Standalone RenderCache behavior is intentionally unchanged.
    """
    if not isinstance(mask_enabled, bool):
        raise ValueError("mask_enabled must be bool")
    allocation = EditorCacheAllocation(
        (DEFAULT_MASKED_RENDER_CACHE_BUDGET_BYTES if mask_enabled else MAX_EDITOR_RETAINED_CACHE_BYTES)
        if render_bytes is None else render_bytes,
        (DEFAULT_MASK_CACHE_BUDGET_BYTES if mask_enabled else 0)
        if mask_bytes is None else mask_bytes,
    )
    if not mask_enabled and allocation.mask_bytes != 0:
        raise ValueError("mask-disabled editors must allocate zero mask-cache bytes")
    return allocation


@dataclass(frozen=True)
class CompositeIdentity:
    """Everything which can change an outer-composite result."""

    rendered_identity: object
    mask: MaskIdentity
    outside_mode: OutsideMode
    source: SourceIdentity
    alpha_algorithm_version: str
    baked: bool = False

    def __post_init__(self) -> None:
        try:
            hash(self.rendered_identity)
        except TypeError as exc:
            raise ValueError("rendered_identity must be hashable") from exc
        if not isinstance(self.mask, MaskIdentity):
            raise TypeError("mask must be a MaskIdentity")
        if not isinstance(self.outside_mode, OutsideMode):
            raise TypeError("outside_mode must be an OutsideMode")
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be a SourceIdentity")
        if not isinstance(self.alpha_algorithm_version, str) or not self.alpha_algorithm_version.strip():
            raise ValueError("alpha_algorithm_version must be non-empty")
        if not isinstance(self.baked, bool):
            raise TypeError("baked must be a bool")


class MaskCaches:
    """Thread-safe, globally bounded inference/derived/composite LRUs.

    Entries are evicted atomically.  NumPy views which share an ultimate owner
    are charged once across all three stores.  Published arrays are owned and
    read-only, preventing a cache hit from exposing cross-call mutable state.
    """

    _KINDS = ("inference", "derived", "composite")

    def __init__(self, budget_bytes: int = DEFAULT_MASK_CACHE_BUDGET_BYTES) -> None:
        budget = int(budget_bytes)
        if budget < 0 or budget > MAX_EDITOR_RETAINED_CACHE_BYTES:
            raise ValueError("budget_bytes must be within the 192 MiB editor ceiling")
        self._budget = budget
        self._stores = {kind: OrderedDict() for kind in self._KINDS}
        self._lru: OrderedDict[tuple[str, object], None] = OrderedDict()
        self._refs: dict[int, tuple[int, int]] = {}
        self._retained = 0
        self._evictions = 0
        self._lock = threading.RLock()

    @property
    def budget_bytes(self) -> int: return self._budget
    @property
    def retained_bytes(self) -> int:
        with self._lock: return self._retained
    @property
    def entry_count(self) -> int:
        with self._lock: return len(self._lru)
    @property
    def eviction_count(self) -> int:
        with self._lock: return self._evictions

    @property
    def metrics(self):
        with self._lock:
            counts = {kind: len(self._stores[kind]) for kind in self._KINDS}
            return MappingProxyType({
                "budget_bytes": self._budget, "retained_bytes": self._retained,
                "entry_count": len(self._lru), "eviction_count": self._evictions,
                "inference_entries": counts["inference"],
                "derived_entries": counts["derived"],
                "composite_entries": counts["composite"],
            })

    @staticmethod
    def _readonly_owned(array: np.ndarray, *, confidence: bool = False) -> np.ndarray:
        if confidence:
            validate_confidence_array(array)
        owned = np.array(array, order="C", copy=True)
        owned.flags.writeable = False
        return owned

    def _get(self, kind: str, key):
        with self._lock:
            entry = self._stores[kind].get(key)
            if entry is None: return None
            marker = (kind, key)
            self._lru.move_to_end(marker)
            self._stores[kind].move_to_end(key)
            return entry[0]

    def _release(self, storage: dict[int, int]) -> None:
        for marker in storage:
            size, refs = self._refs[marker]
            if refs == 1:
                del self._refs[marker]
                self._retained -= size
            else:
                self._refs[marker] = (size, refs - 1)

    def _remove(self, kind: str, key, *, eviction: bool = False) -> None:
        entry = self._stores[kind].pop(key, None)
        self._lru.pop((kind, key), None)
        if entry is not None:
            self._release(entry[1])
            if eviction: self._evictions += 1

    def _put(self, kind: str, key, value) -> bool:
        # ProbabilityMap is a value object rather than a container understood
        # by RenderCache's walker.  Charge its actual immutable payload
        # explicitly; otherwise inference entries would incorrectly cost zero.
        storage = _group_storage(value.values if isinstance(value, ProbabilityMap) else value)
        if sum(storage.values()) > self._budget:
            return False
        with self._lock:
            self._remove(kind, key)
            additional = sum(size for marker, size in storage.items() if marker not in self._refs)
            while self._lru and self._retained + additional > self._budget:
                old_kind, old_key = next(iter(self._lru))
                self._remove(old_kind, old_key, eviction=True)
                additional = sum(size for marker, size in storage.items() if marker not in self._refs)
            if self._retained + additional > self._budget:
                return False
            for marker, size in storage.items():
                current = self._refs.get(marker)
                if current is None:
                    self._refs[marker] = (size, 1); self._retained += size
                else:
                    self._refs[marker] = (size, current[1] + 1)
            self._stores[kind][key] = (value, storage)
            self._lru[(kind, key)] = None
            return True

    def get_inference(self, identity: InferenceIdentity) -> ProbabilityMap | None:
        return self._get("inference", identity)

    def put_inference(self, probability: ProbabilityMap) -> bool:
        if not isinstance(probability, ProbabilityMap): raise TypeError("probability must be a ProbabilityMap")
        return self._put("inference", probability.identity, probability)

    def get_derived(self, identity: MaskIdentity) -> np.ndarray | None:
        return self._get("derived", identity)

    def put_derived(self, identity: MaskIdentity, mask: np.ndarray) -> bool:
        if not isinstance(identity, MaskIdentity): raise TypeError("identity must be a MaskIdentity")
        return self._put("derived", identity, self._readonly_owned(mask, confidence=True))

    def get_composite(self, identity: CompositeIdentity) -> np.ndarray | None:
        return self._get("composite", identity)

    def put_composite(self, identity: CompositeIdentity, image: np.ndarray) -> bool:
        if not isinstance(identity, CompositeIdentity): raise TypeError("identity must be a CompositeIdentity")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("composite image must be uint8 (H,W,3|4)")
        return self._put("composite", identity, self._readonly_owned(image))

    def clear_source(self, source: SourceIdentity) -> None:
        """Discard only entries derived from ``source``."""
        if not isinstance(source, SourceIdentity): raise TypeError("source must be a SourceIdentity")
        with self._lock:
            for key in tuple(self._stores["inference"]):
                if key.source == source: self._remove("inference", key)
            for key in tuple(self._stores["derived"]):
                if key.inference.source == source: self._remove("derived", key)
            for key in tuple(self._stores["composite"]):
                if key.source == source or key.mask.inference.source == source:
                    self._remove("composite", key)

    def clear(self) -> None:
        with self._lock:
            for kind in self._KINDS: self._stores[kind].clear()
            self._lru.clear(); self._refs.clear(); self._retained = 0; self._evictions = 0
