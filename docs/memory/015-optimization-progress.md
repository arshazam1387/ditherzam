---
type: progress
phase: 8
status: done
date: 2026-07-07
---

**Performance optimization pass — 5 wins, MERGED to `main` + pushed** (origin
at bb1fbf6; branched off [[005-project-state]]'s 90a9497). All TDD'd, committed per step,
green JIT-on AND JIT-off (**374 passed**, was 333). Full write-up + reproducible
harness in `benchmarks/` (`RESULTS.md`, `bench.py`, `bench_cache.py`,
`ui_latency.py`).

**Key measurement that reordered the handoff:** the palette **color map**
(`nearest_indices`), not effects, was the #1 per-render cost — 370 ms even for a
4-color palette (a `(H,W,K,3)` broadcast temp).

Landed (each output-preserving unless noted):
1. **`nearest_indices` → njit** parallel per-pixel loop. 370→98 ms @1080p,
   bit-identical (same squared-dist argmin, first-min tie-break). `color/engine.py`.
2. **Render coalescing** — Qt-free `ui/render_scheduler.RenderCoalescer`: single
   in-flight render + generation token; drag 20 full renders → 1 + trailing, no
   stale/out-of-order paints. Also killed the shared-pipeline data race.
3. **Staged cache** — `RenderPipeline.render_cached()` (SEPARATE from the frozen
   `render()`, which is untouched so `test_render_order` still holds). Caches 6
   layers by content signature; downstream tweak @1080p: invert 59 ms, effects
   192 ms, saturation 340 ms (vs 592 ms full). Bit-identical to `render()`.
4. **Preview proxy** — `ui/preview.render_preview()`: during a drag render a
   ≤640px proxy (dither scale reduced to match), full-res on 160 ms settle. 1080p
   534→99 ms, 4K 2274→217 ms. Approximate + display-only; settled/exported image
   is always full-res (verified: settled display == exact `render_cached`).
5. **Background JIT warmup** — `warmup.py` daemon thread from `app.main`; first
   render 715→354 ms.

**Gotchas found:** (a) `settings_map.settings_from_controls` still defaults
`blur` to **50** (not 0) if the key is absent — latent, out of scope, panel
provides 0 in practice; see [[014-live-app-bugs-fixed]]. (b) Offscreen Qt
QThreadPool segfaults with unbounded concurrent workers / `.close()` / bound-method
monkeypatching — the coalescer's single-in-flight fixed the real cause; probes
patch module-level funcs instead. (c) Can't naively short-circuit `apply_saturation`
at value 50 — `lum+(rgb-lum)*1.0` shifts ±1 LSB vs committed pixels; handled via
cache/proxy, not a math change.

**Not done (lower ROI):** #4 effects PIL-gaussian speedup (proxy+cache cover the
interactive case), #6 numba threading (single-in-flight already avoids
oversubscription), #7 deferring numba's ~900 ms *import* (needs a Qt-free
name/category registry — larger refactor; only affects time-to-window),
#8 batch/video parallelism.

**Next:** optionally the deferred items above (effects gaussian, numba import
deferral, batch parallelism), packaging/dist, real-photo QA.
