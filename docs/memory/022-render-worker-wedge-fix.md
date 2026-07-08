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

**Still open (not fixed, lower priority):** what actually throws is not pinned down — the pure
pipeline is exception-free across a wide setting/palette/effect/size matrix. Suspected latent
trigger is a thread race: `render_now()` (GUI thread) can run concurrently with an in-flight
background worker; the proxy path `render()` reads `pipeline.color_engine`/`effect_stack`
without the cache lock while `_sync_pipeline()` reassigns them. The wedge fix makes any such
throw recoverable regardless of cause. Related: [[021-color-palette-library-c-shipped]].
