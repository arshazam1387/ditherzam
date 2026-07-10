# High-Resolution Preview and Render Performance — Implementation Plan

**Date:** 2026-07-09  
**Design:** `docs/superpowers/specs/2026-07-09-high-resolution-preview-performance-design.md`  
**Method:** Every task is red → green → refactor, followed by focused tests and a
commit boundary. Preserve unrelated Glow-tab work in the current dirty tree.

## Global gates

After each task, run its focused JIT-off tests. After each wave, run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:NUMBA_DISABLE_JIT='1'
.venv/Scripts/python.exe -m pytest -q
```

Then run the wave’s exactness targets with JIT enabled. The historical seven
`special.py` float-index failures are reported separately if still present; no new
JIT-on regression is accepted. Always include render order, cached equality,
thread-safety, depth promotion, and worker recovery in integration gates.

## Wave 0 — Evidence and exactness contracts

### Task 0.1 — Reproducible high-resolution harness

- Files: `benchmarks/high_res.py`, `benchmarks/common.py`,
  `benchmarks/HIGH_RES_BASELINE_2026-07-09.md`.
- Cover 1080p/4K/5K, all color modes, palette sizes, preview caps, cached
  mutations, first/warm timing, allocation peak, and optional RSS.
- Keep large Python diffusion behind `--large-diffused`.
- Verify `--help` and a quick cache run.
- Commit: `bench: add high-resolution render matrix`.

### Task 0.2 — Freeze differential references

- Add `tests/test_color_engine_exactness.py` and focused adjustment/effect
  reference fixtures before changing algorithms.
- Cover 1×1, 1×N, N×1, random and out-of-range float32, non-contiguous input,
  palette ties, K=1/2/4/16/64, depth 1/2/64, and built-ins.
- Assert final `uint8` array equality and palette first-minimum ties.
- Commit: `test(perf): lock exact render references`.

## Wave 1 — Capped preview foundation and first hotspot

### Task 1.1 — Pure preview-resolution policy

- Modify `ditherzam/ui/preview.py`; extend `tests/test_preview.py`.
- Add Auto/480/720/1080/1440/2160/Full parsing, aspect-preserving target
  geometry, Auto 720–1440 buckets, resize/zoom bucket helpers, and exact-if-fit.
- Return capped RGB directly; remove source-sized proxy upscaling.
- Preserve proxy dither-scale behavior and exact factor-one output.
- Expected red: policy APIs absent; current proxy output is source-sized.
- Commit: `perf(preview): add deterministic capped preview policy`.

### Task 1.2 — Persistent preview preferences

- Add `ditherzam/ui/preview_preferences.py` and
  `tests/test_preview_preferences.py`.
- Store resolution=Auto and rerender-on-zoom=False via QSettings-compatible
  application preferences; normalize corrupt values.
- Prove neither field enters creative presets.
- Commit: `feat(ui): persist preview quality preferences`.

### Task 1.3 — Exact compiled RGB Floyd–Steinberg

- Modify `ditherzam/color/engine.py`; add exactness/performance tests.
- Replace per-pixel Python vector allocations with a sequential scalar `@njit`
  kernel. Preserve scan order, neighbor order, float32 operations, boundaries,
  strict tie behavior, and final pixels.
- Do not mark diffusion parallel.
- Acceptance: ≥5× faster than 5.56 s at 480p, byte-identical corpus.
- Commit: `perf(color): compile exact RGB Floyd-Steinberg`.

### Task 1.4 — Source-logical viewport geometry

- Modify `ditherzam/ui/viewport_math.py`, `ui/viewport.py`, and viewport tests.
- Display a capped pixmap scaled to source-logical scene bounds; preserve
  source-relative fit, pan, zoom, cursor, and 100% semantics.
- Ordinary preview replacement must not refit/reset zoom; source replacement may.
- Commit: `perf(viewport): display capped rasters in source coordinates`.

## Wave 2 — Lifecycle, immutable ownership, and color allocation wins

### Task 2.1 — Visible preview controls

- Modify `ui/main_window.py`; add `tests/test_preview_ui.py`.
- Add exclusive View-menu resolution actions and checkable zoom-rerender action.
- Preference changes invalidate one-off Full state and schedule policy work.
- Commit: `feat(ui): add preview resolution controls`.

### Task 2.2 — Immutable prioritized render requests

- Add `ui/render_request.py`; modify `ui/render_scheduler.py`, worker wiring, and
  scheduler/resilience/thread-safety tests.
- Frozen request owns generation, kind, settings, source identity, target,
  logical geometry, and snapshotted color/effect context.
- One in flight plus one complete latest trailing request. Newest state wins;
  Full outranks capped only within identical state.
- Preserve exactly one terminal outcome on success/failure.
- Commit: `refactor(render-ui): schedule immutable preview requests`.

### Task 2.3 — Unified capped lifecycle and Full action

- Modify `ui/main_window.py`, `ui/hotkeys.py`, and lifecycle tests.
- Both drag and settle use policy. Initial load/decode is asynchronous.
- Add View → Full Quality Preview (`Ctrl+Enter`); later edit returns to cap.
- Retain synchronous `render_now()` only for tests/compatibility.
- Commit: `feat(preview): cap settled and initial renders`.

### Task 2.4 — Optional bucketed zoom refinement

- Modify viewport/main-window wiring and tests.
- Debounce zoom bursts; Off schedules nothing; On schedules once per bucket.
- Respect numeric, Auto, Full, and source ceilings.
- Commit: `feat(preview): add optional zoom refinement`.

### Task 2.5 — Fused ordered palette mapping

- Modify `color/engine.py` and exactness/allocation tests.
- Compute Bayer offset and nearest lookup inside one compiled mapper; remove
  H×W offset, H×W×3 biased, and H×W int64 index intermediates.
- Preserve spread precision, rounding, clipping, and first-minimum ties.
- Commit: `perf(color): fuse ordered palette mapping`.

### Task 2.6 — Fused ramp mapping

- Modify `color/engine.py` and ramp exactness tests.
- Accept 2D gray without RGB repeat; fuse Rec.601 luminance, banker-rounded
  level selection, clipping, and ramp lookup.
- Cover depth 1/2/64, mappings, RGB/non-contiguous input, and half boundaries.
- Commit: `perf(color): fuse ramp luminance and lookup`.

### Task 2.7 — Stable content-keyed color context

- Add content-key utilities/context cache; modify `color/engine.py`, `render.py`,
  and main-window engine ownership with dedicated tests.
- Key by algorithm version + palette shape/dtype/ordered bytes + mode/depth/
  mapping/phase; never palette name.
- Reuse immutable palette arrays, luminance order, and ramps; same-name edits miss.
- Stop mutating shared engine fields during render.
- Commit: `perf(color): reuse immutable palette-derived context`.

## Wave 3 — Core memory traffic, cache bounds, effects, and cancellation

### Task 3.1 — Exact saturation/RGB output fusion

- Modify `adjustments.py`/`render.py`; extend saturation/render/cache tests.
- Avoid grayscale `np.repeat`; produce final RGB uint8 while preserving current
  neutral-50 ±1-LSB behavior and clamp/round semantics.
- Commit: `perf(render): reduce saturation and RGB temporaries`.

### Task 3.2 — Profile-guided tonal adjustment fusion

- Benchmark contrast/midtones/highlights separately and together.
- Fuse only if exact dither-facing float output is proven. Keep public frozen
  stage calls/order; reject algebraic rewrites that move threshold pixels.
- Commit only a proven win: `perf(adjustments): reduce tonal stage traffic`.

### Task 3.3 — 192 MiB cache accounting and eviction

- Add `render_cache.py`; modify `render.py`; add budget tests.
- Count unique NumPy backing storage once; aliases do not double-count.
- Evict complete dependency groups LRU; skip oversized entries; expose read-only
  bytes/count/eviction metrics; clear on source replacement.
- Commit: `perf: bound staged render cache by memory`.

### Task 3.4 — Workspace/buffer reuse

- Add request/context-local or exclusively leased scratch buffers.
- Never return/cache an array later overwritten; never use a module-global
  mutable workspace; account retained buffers under the same budget.
- Test consecutive-output stability and cached/plain concurrency.
- Commit: `perf(render): reuse exact scratch buffers safely`.

### Task 3.5 — Effects profile and exact optimizations

- Extend benchmark effects section before production edits.
- Measure blur, sharpen, chromatic aberration, JPEG glitch, and Epsilon Glow at
  1080p/4K. Optimize only proven hotspots, one effect per commit.
- Freeze output fixtures first; input is never mutated.
- Candidate: reduce Epsilon Glow resize/blur/full-frame intermediates.
- Commits: `bench: profile high-resolution effects`, then per proven effect.

### Task 3.6 — Cooperative stage-boundary cancellation

- Add request cancellation predicate/outcome across render/UI scheduler.
- Check only between expensive stages; never interrupt inside NumPy/PIL/Numba.
- Cancel emits a terminal outcome, never paints/fails/wedges/caches partial work,
  and launches the newest trailing request. Exports are not cancelled by UI.
- Commit: `perf: stop obsolete renders between stages`.

### Task 3.7 — QImage ownership/copy experiment

- Modify `ui/convert.py` only if lifetime-safe ownership across queued signals is
  proven after `gc.collect()` and source deletion/mutation.
- Benchmark conversion allocations. If Qt detaches anyway or lifetime safety is
  uncertain, retain the safe copy and document the rejected optimization.
- Commit only a measured safe win.

## Wave 4 — Media/export isolation, threading, startup, and validation

### Task 4.1 — Capped animation and video screen previews

- Modify `ui/timeline_panel.py`, `ui/video_controller.py`, and media UI tests.
- Animation screen frames render asynchronously/latest-wins at selected cap;
  video display frames are capped before conversion. Preserve source-logical
  geometry and exact exported frames.
- Commit: `perf(media-ui): cap animation and video previews`.

### Task 4.2 — Dedicated exact export contexts

- Snapshot source/settings/palette/effects into dedicated still/batch/video/
  animation contexts at launch. UI edits cannot change an in-flight export.
- Preview preferences never enter export APIs. SVG retains its existing exact
  source/threshold contract.
- Commit: `fix(export): isolate exact rendering from preview state`.

### Task 4.3 — Thread scaling benchmark

- Add a reproducible 1/2/4/8-thread subprocess matrix for independent Numba
  kernels, capped previews, effects-heavy paths, UI heartbeat, memory, and hashes.
- Diffusion remains labeled sequential. Restore thread settings after probes.
- Commit: `bench: characterize render thread scaling`.

### Task 4.4 — Bounded threading policy

- Add Qt-free `threading_policy.py`; wire interactive/export/video workers.
- Choose Auto only from measured results; clamp to available 1/2/4/8 counts.
- Install per-worker Numba masks; bound total Qt jobs × Numba threads; keep one
  interactive render and reserve UI capacity during exports.
- Commit: `perf: bound render and export thread budgets`.

### Task 4.5 — JIT warmup refinement

- Warm selected nearest/ordered/ramp/saturation/diffusion paths with tiny inputs
  on the existing daemon thread; do not warm every kernel or block the GUI.
- Test dispatch and best-effort termination; measure cold separately.
- Commit: `perf(startup): warm selected render kernels`.

### Task 4.6 — Acceptance matrix and real-photo QA

- Run all caps, modes, K=4/16/64, cached mutations, effects, conversions,
  1/2/4/8 threads, cold/warm, peak/RSS/retained bytes, and output hashes.
- Verify preview latency targets, ≥5× diffusion, ≤192 MiB retained cache, no
  source-sized capped allocation, exact Full/exports, no stale paints/wedges.
- Perform the approved 4K+ manual QA checklist and append before/after results.
- Commit: `docs: record high-resolution performance results`.

## Parallel execution and merge discipline

- Wave 1 tasks 1.1, 1.2, and 1.3 are parallel-safe; 1.4 follows the policy API.
- Ordered and ramp kernel work must use isolated commits because both edit
  `color/engine.py`; stable context integrates them afterward.
- All `main_window.py` lifecycle tasks are serialized and must preserve the
  existing Glow-tab changes.
- All `render.py` cache/saturation/cancellation tasks are serialized.
- Media preview and effects profiling may proceed in parallel after immutable
  request/context contracts stabilize.
- An integration reviewer runs global gates after every wave before the next
  wave begins.
