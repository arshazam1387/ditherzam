"""Priority-aware render scheduling (Qt-free, GUI-thread-only).

Keeps at most one render in flight and tags each launched request with a
monotonic generation. While a render runs, further requests are coalesced
into a single *pending* trailing ``RenderRequest`` instead of spawning more
workers; when the in-flight render finishes, that trailing request launches
with the latest state. Stale/out-of-order results are dropped via
``is_current``.

Coalescing is priority-aware: among requests arriving during one busy window,
the freshest generation always wins, and within one generation an explicit
Full request outranks capped (drag/settle/zoom) work, with settle/zoom
outranking drag. See ``render_request.supersedes``.

This turns a slider drag from "one full render per debounce tick, all painted
out of order" into "one render, then one trailing render with the freshest,
highest-priority state" -- far less CPU and no wasted paints.
"""
from __future__ import annotations

from dataclasses import replace

from .render_request import RenderRequest, supersedes


class RenderScheduler:
    def __init__(self) -> None:
        self._gen = 0
        self._busy = False
        self._pending: RenderRequest | None = None

    def _begin(self, req: RenderRequest) -> RenderRequest:
        self._gen += 1
        self._busy = True
        self._pending = None
        return replace(req, generation=self._gen)

    def request(self, req: RenderRequest) -> RenderRequest | None:
        """A render was asked for. Returns the stamped request to launch a
        worker with, or ``None`` if one is already in flight (the request is
        coalesced into the pending trailing request per priority)."""
        if self._busy:
            stamped = replace(req, generation=self._gen)
            if self._pending is None or supersedes(stamped, self._pending):
                self._pending = stamped
            return None
        return self._begin(req)

    def is_current(self, req: RenderRequest) -> bool:
        """True if ``req`` is the most-recently-started render (should paint)."""
        return req.generation == self._gen

    def should_cancel(self, req: RenderRequest) -> bool:
        """True if ``req`` is in flight but a newer trailing request already
        obsoletes it -- the worker should stop at its next stage boundary."""
        return self.is_current(req) and self._pending is not None

    def invalidate(self) -> None:
        """Mark any in-flight render stale (e.g. a synchronous render_now
        painted). Its delivered result will fail ``is_current`` and not paint.

        Also drop any coalesced pending request: a synchronous full render
        already satisfies whatever was queued, and because requests are frozen
        at build time a stale pending would otherwise be promoted by the next
        ``on_finished`` and paint older state over the fresh render."""
        self._gen += 1
        self._pending = None

    def on_finished(self) -> RenderRequest | None:
        """A worker finished. Returns the stamped trailing request to launch
        if state changed while busy, else ``None``."""
        self._busy = False
        if self._pending is None:
            return None
        return self._begin(self._pending)
