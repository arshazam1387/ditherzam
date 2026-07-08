---
type: gotcha
phase: 8
status: done
date: 2026-07-08
---

The app "gets stuck the more you use it" was NOT a memory leak — it was the render
coalescer wedging. `_RenderWorker.run()` (QThreadPool) had no try/except, so any render
that raised on the background thread never emitted `signals.finished`, so
`RenderCoalescer.on_finished()` never ran and `_busy` stuck `True` forever. From then on
every `request()` coalesced to `None`, no worker launched, and the preview froze until
restart. Pre-existing since the optimization pass; the Sub-project C hover-preview fires
far more renders, so a session hits a throwing render sooner (hence "the more you use it").

**Fix (commit `8240941`, TDD, tests/test_render_resilience.py):** `run()` now always reports
an outcome — `finished(qimg, token)` on success, a new `failed(token)` signal on exception
(logged) — and `ImageEditor._on_render_failed` calls `coalescer.on_finished()` so rendering
recovers instead of freezing. Verified: after a throwing render, `_busy` returns to False and
the image updates again.

**Apply:** any Qt worker whose completion drives a state machine (coalescer/`_busy`) MUST emit
a terminal signal on EVERY path incl. exceptions, or the state machine deadlocks silently
(Qt only prints "Error calling Python override of QRunnable::run()").

**Throwing input FOUND + FIXED (2026-07-08, session 2):** it was a TOCTOU race, confirmed by
headless repro (`sys.setswitchinterval(1e-6)` + a thread reassigning `pipeline.color_engine`
while another loops `render()` → `AttributeError("'NoneType' object has no attribute 'map'")`).
`render()`/`render_cached()` each read `self.color_engine` and `self.effect_stack` **multiple
times** per call (the `is not None` check, then deref). The GUI thread's `_sync_pipeline()`
reassigns those attrs (to a fresh engine/stack, or `None` when Color/Effects is toggled off)
with no lock, so a reassignment landing between two reads deref'd `None` on the render worker.
The `_cache_lock` never covered it (proxy `render()` skips the lock; and it guards the cache
dict, not the attribute reads). **Fix:** snapshot each attr into a local **once** at the top of
its stage and use the local throughout — both methods. Atomic read under the GIL ⇒ a concurrent
reassignment can't split a single render. Output byte-identical (STAGE_ORDER + `test_render_cache`
still green). Regression: `tests/test_render_thread_safety.py` (4 read-count tests, deterministic,
no timing). Full suite **584 green** JIT-off. `render()` was edited but only to bind locals —
STAGE_ORDER and RenderSettings fields untouched. Related: [[021-color-palette-library-c-shipped]].
