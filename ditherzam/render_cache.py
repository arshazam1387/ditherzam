"""Memory-bounded storage for complete staged-render dependency groups."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from types import MappingProxyType
import threading

import numpy as np


MIB = 1024 * 1024
MAX_EDITOR_RETAINED_CACHE_BYTES = 192 * MIB
# Standalone and mask-disabled pipelines retain the established full budget.
# Mask-enabled editors use masking.cache.editor_cache_allocation to split it.
DEFAULT_CACHE_BUDGET_BYTES = MAX_EDITOR_RETAINED_CACHE_BYTES


def _iter_arrays(value, seen_containers: set[int]):
    if isinstance(value, np.ndarray):
        yield value
        return
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen_containers:
            return
        seen_containers.add(marker)
        for item in value.values():
            yield from _iter_arrays(item, seen_containers)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        marker = id(value)
        if marker in seen_containers:
            return
        seen_containers.add(marker)
        for item in value:
            yield from _iter_arrays(item, seen_containers)


def _backing(array: np.ndarray) -> np.ndarray:
    """Return the ultimate NumPy owner of an array view."""
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    return owner


def _group_storage(group) -> dict[int, int]:
    storage: dict[int, int] = {}
    for array in _iter_arrays(group, set()):
        owner = _backing(array)
        storage.setdefault(id(owner), int(owner.nbytes))
    return storage


class RenderCache:
    """LRU of atomic groups, bounded by unique NumPy backing storage."""

    def __init__(self, budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES) -> None:
        budget = int(budget_bytes)
        if budget < 0:
            raise ValueError("budget_bytes must be non-negative")
        self._budget_bytes = budget
        self._entries: OrderedDict[object, tuple[object, dict[int, int]]] = OrderedDict()
        self._storage_refs: dict[int, tuple[int, int]] = {}
        self._retained_bytes = 0
        self._eviction_count = 0
        self._lock = threading.RLock()

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._retained_bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def eviction_count(self) -> int:
        with self._lock:
            return self._eviction_count

    @property
    def metrics(self):
        with self._lock:
            return MappingProxyType({
                "budget_bytes": self._budget_bytes,
                "retained_bytes": self._retained_bytes,
                "entry_count": len(self._entries),
                "eviction_count": self._eviction_count,
            })

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key, group) -> bool:
        storage = _group_storage(group)
        group_bytes = sum(storage.values())
        if group_bytes > self._budget_bytes:
            return False

        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._release(old[1])

            additional = sum(
                size for marker, size in storage.items()
                if marker not in self._storage_refs
            )
            while self._entries and self._retained_bytes + additional > self._budget_bytes:
                _, (_, evicted_storage) = self._entries.popitem(last=False)
                self._release(evicted_storage)
                self._eviction_count += 1
                additional = sum(
                    size for marker, size in storage.items()
                    if marker not in self._storage_refs
                )

            # A replacement can become inadmissible after releasing its old
            # shared storage even though its standalone size fits the budget.
            if self._retained_bytes + additional > self._budget_bytes:
                return False

            for marker, size in storage.items():
                current = self._storage_refs.get(marker)
                if current is None:
                    self._storage_refs[marker] = (size, 1)
                    self._retained_bytes += size
                else:
                    self._storage_refs[marker] = (current[0], current[1] + 1)
            self._entries[key] = (group, storage)
            return True

    def _release(self, storage: dict[int, int]) -> None:
        for marker in storage:
            size, refs = self._storage_refs[marker]
            if refs == 1:
                del self._storage_refs[marker]
                self._retained_bytes -= size
            else:
                self._storage_refs[marker] = (size, refs - 1)

    def keys(self):
        with self._lock:
            return tuple(self._entries.keys())

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._storage_refs.clear()
            self._retained_bytes = 0
            self._eviction_count = 0
