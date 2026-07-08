"""Render coalescing state machine (Qt-free, GUI-thread-only).

Keeps at most one render in flight and tags each with a monotonic generation
token. While a render runs, further requests are marked *pending* instead of
spawning more workers; when the in-flight render finishes, a single trailing
render is started with the latest state. Stale/out-of-order results are dropped
via ``is_current``.

This turns a slider drag from "one full render per debounce tick, all painted
out of order" into "one render, then one trailing render with the freshest
state" — far less CPU and no wasted paints.
"""
from __future__ import annotations


class RenderCoalescer:
    def __init__(self) -> None:
        self._gen = 0
        self._busy = False
        self._pending = False

    def _begin(self) -> int:
        self._busy = True
        self._pending = False
        self._gen += 1
        return self._gen

    def request(self) -> int | None:
        """A render was scheduled. Return a token to launch a worker with, or
        ``None`` if one is already in flight (the request is coalesced)."""
        if self._busy:
            self._pending = True
            return None
        return self._begin()

    def is_current(self, token: int) -> bool:
        """True if ``token`` is the most recently started render (should paint)."""
        return token == self._gen

    def invalidate(self) -> None:
        """Mark any in-flight render stale (e.g. a synchronous render_now painted).
        Their delivered results will fail ``is_current`` and not paint."""
        self._gen += 1

    def on_finished(self) -> int | None:
        """A worker finished. Return a new token if a trailing render should start
        (state changed while busy), else ``None``."""
        self._busy = False
        if self._pending:
            return self._begin()
        return None
