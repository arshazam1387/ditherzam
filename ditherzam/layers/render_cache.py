"""Bounded single-flight cache for completed spatial-layer Looks."""

from __future__ import annotations

import threading

import numpy as np

from ..render import RenderCancelled
from ..render_cache import MIB, RenderCache


DEFAULT_LAYER_LOOK_CACHE_BUDGET_BYTES = 32 * MIB


class LayerLookRenderCache:
    def __init__(
        self, budget_bytes: int = DEFAULT_LAYER_LOOK_CACHE_BUDGET_BYTES
    ) -> None:
        self._cache = RenderCache(budget_bytes)
        self._condition = threading.Condition(threading.RLock())
        self._inflight: set[object] = set()
        self._epoch = 0

    @property
    def metrics(self):
        return self._cache.metrics

    def clear(self) -> None:
        with self._condition:
            self._epoch += 1
            self._cache.clear()
            self._condition.notify_all()

    def get_or_render(self, key, render, *, is_cancelled=None):
        while True:
            with self._condition:
                cached = self._cache.get(key)
                if cached is not None:
                    return cached
                if key not in self._inflight:
                    self._inflight.add(key)
                    render_epoch = self._epoch
                    break
                if is_cancelled is not None and is_cancelled():
                    raise RenderCancelled
                self._condition.wait(0.01)
        try:
            value = render()
            if is_cancelled is not None and is_cancelled():
                raise RenderCancelled
            frozen = np.array(value, dtype=np.uint8, order="C", copy=True)
            frozen.flags.writeable = False
            with self._condition:
                if render_epoch == self._epoch:
                    self._cache.put(key, frozen)
            return frozen
        finally:
            with self._condition:
                self._inflight.discard(key)
                self._condition.notify_all()
