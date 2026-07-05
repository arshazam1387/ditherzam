# HANDOFF PROMPT — Optimize the ditherzam app (performance)

> Paste everything below the line into a fresh Claude Code session (or a subagent
> prompt) that has the `ditherzam` repo as its working directory. It carries all
> context needed to profile and optimize the app without re-deriving anything.

---

## Your role & goal

You are optimizing **ditherzam**, a finished clean-room PySide6 pixel-dither studio.
All 8 subsystems are built and green (**333 tests pass**). The app runs, previews,
dithers, applies color/effects, and exports. **Your job is performance** — make the
live editing experience fast and the app snappy — **without changing render output
(golden-array tests must still pass) or breaking any test, and without violating the
architecture invariants below.**

Optimize behavior, not correctness: every change is measure → change → re-measure →
prove tests still pass.

## Measured baseline (establish your own before changing anything)

On this machine, JIT enabled, a 1080p (1920×1080) grayscale input:

| Path | Time |
|---|---|
| cold `import` of core modules | ~750 ms |
| first Floyd-Steinberg render (includes Numba JIT compile) | ~760 ms |
| **warm** Floyd-Steinberg render @1080p | **~230 ms** |
| warm FS + Game Boy palette + Chromatic Aberration + Epsilon Glow | **~960 ms** |

230 ms–1 s per render is too slow for smooth slider dragging (target <50 ms
perceived). Effects (PIL-based) and the full-pipeline-recompute-per-change are the
big costs. Re-measure with your own harness (see "Measurement" below) — don't trust
these numbers blindly; they set the order of magnitude.

## Architecture invariants (never violate)

- **Clean-room:** no Dither Boy / Studio AAA code/strings/URLs/binaries; no network/
  telemetry/licensing code.
- **Qt-free core:** only `ditherzam/ui/`, `ditherzam/app.py`, and
  `ditherzam/video/workers.py` may import PySide6. Keep all optimization of the
  render core (`render.py`, `dithering/`, `color/`, `effects/`, `adjustments.py`,
  `imaging.py`, `animation/`) pure and headlessly testable.
- **Frozen render order & contracts** (spec §8.1): contrast → midtones → highlights →
  blur → dither(downscale→kernel→upscale) → color → saturation → effects → invert.
  `RenderPipeline.render(base_gray_f32, settings, temporal_field=None) -> uint8 HxWx3`,
  `apply_dither(..., threshold_field=None)`, `ColorEngine.map`, `EffectStack.apply`
  — signatures stay verbatim. `test_render_order.py` locks the stage order.
- **Kernels** are `@njit(cache=True, parallel=True)` except error-diffusion kernels,
  which MUST stay serial `@njit(cache=True)` (order-dependent). float32, 0..255.
- **TDD** every behavior change; **commit after each green step**; DRY, YAGNI.
- **No output changes:** golden-array kernel tests and the render tests pin exact
  pixels. Optimizations must be output-preserving. If an optimization is inherently
  approximate (e.g. a fast preview proxy), it must be a SEPARATE path that does not
  alter the full-quality render, and gated/labeled as such.

## Environment (this machine)

- Repo/cwd: `C:\Users\arsha\Desktop\custom dither` (git; `origin` =
  `github.com/arshazam1387/ditherzam`, private; `gh` authed as `arshazam1387`).
- **Python 3.12.13 venv already provisioned at `.venv/`** (gitignored) with
  numpy/numba/pillow/PySide6/pytest/pyyaml. Use `.venv/Scripts/python.exe` for
  everything; GUI: `.venv/Scripts/pythonw.exe -m ditherzam.app`.
- Tests: `NUMBA_DISABLE_JIT=1 .venv/Scripts/python.exe -m pytest -q`; Qt tests need
  `QT_QPA_PLATFORM=offscreen`. **But profile with JIT ENABLED** (unset
  `NUMBA_DISABLE_JIT`) — JIT-off numbers are meaningless for perf.
- ffmpeg is installed (used by video/animation export).

## Measurement first (Phase 0 — do before any change)

You cannot optimize what you don't measure. Build a small, committed benchmark
harness under `benchmarks/` (Qt-free where possible):

- Per-stage timing of `RenderPipeline.render` (wrap each of the 9 stages, print ms).
- Warm vs cold (JIT) timings at 512², 1080p, and 4K, for FS (error-diffusion),
  Bayer (ordered/parallel), and a heavy style.
- `cProfile` + a `py-spy`-style sampling run (`.venv/Scripts/python.exe -m cProfile
  -s cumtime bench.py`) to find the real hot spots — do NOT assume.
- A UI-interaction latency probe (offscreen): simulate a slider drag (N rapid
  `panel.state` changes → `schedule_render`) and measure time-to-updated-pixmap and
  how many renders actually ran vs were superseded.

Report the baseline table, then attack the biggest wins first.

## Candidate optimizations (verify each against a profile; don't do blindly)

Ordered by likely payoff — confirm with your measurements:

1. **Cancel superseded background renders.** `main_window._do_render` starts a fresh
   `_RenderWorker` (QRunnable on the global QThreadPool) on every debounce tick, but
   in-flight workers run to completion and can deliver stale results out of order.
   Add a generation/epoch token: tag each worker, and in `_on_rendered` drop results
   whose token != current. Optionally support cooperative cancel. Big CPU win while
   dragging.

2. **Incremental / cached staged render.** `RenderPipeline.render` recomputes ALL
   stages from `base_gray` every call. Cache intermediates keyed by the inputs that
   affect them: (a) the post-adjustment grayscale (contrast/midtones/highlights/blur)
   — reuse when only style/scale/color/effects/saturation/invert change; (b) the
   dithered array — reuse when only color/saturation/effects/invert change. This
   turns a color/effect tweak from a full ~230 ms+ render into a few ms. Keep it
   output-identical (hash/compare the relevant settings subset). Consider a
   `render(..., dirty_from_stage=...)` fast path rather than mutating the frozen
   `render()` contract; add tests proving cached == full.

3. **Interactive preview proxy.** While a control is actively changing, render a
   downscaled proxy (e.g. longest side ≤ ~720 px) for instant feedback, then a
   full-resolution pass on idle/release. Must not change the committed/exported
   full-res output. Tune the debounce (currently 20 ms) accordingly.

4. **Effect cost.** Effects dominate (~960 ms with 2 effects). Post effects
   (`effects/post.py`) use PIL; Gaussian-blur-based ones (Blur, Epsilon Glow, Sharpen
   via unsharp) are the cost. Options: operate at preview resolution for the proxy
   path, use separable/np-vectorized blur, cache blur results, or `@njit` the pixel
   ops. Keep full-res output identical (add golden tests if you change math).

5. **JIT warmup.** First render pays JIT compile (~500 ms delta). `cache=True`
   persists to disk, so it's mostly a first-ever-run cost — but still warm up common
   kernels (FS, Atkinson, Bayer) in a background thread at app start so the user's
   first interaction isn't janky. Ensure the on-disk numba cache dir is writable/
   shipped appropriately.

6. **Numba threading.** `parallel=True` kernels spawn their own thread pool; running
   them inside a Qt QThreadPool worker can oversubscribe cores. Evaluate
   `numba.set_num_threads`, the threading layer (`tbb`/`omp`/`workqueue`), and whether
   a single dedicated render thread beats the global pool. Measure, don't guess.

7. **Micro:** avoid redundant array copies / dtype churn in `render()` and
   `imaging.py`; `_sync_pipeline()` rebuilds `ColorEngine`/`EffectStack` every render
   — cache by a state signature; lazy-import heavy modules to cut the ~750 ms import/
   startup; consider a splash while warming.

8. **Batch/video/animation throughput.** Per-frame and per-file rendering is
   sequential; once the single-render path is fast, consider parallelizing across
   frames/files (process/thread pool) for export — but that's lower priority than
   interactive latency.

## Guardrails

- After every change: `NUMBA_DISABLE_JIT=1 .venv/Scripts/python.exe -m pytest -q`
  must stay green (currently **333 passed**), plus the offscreen Qt suite.
- Re-run your benchmark harness and record before/after in the commit message.
- Keep the core Qt-free (grep `PySide6` under the core packages must stay empty).
- Any new fast/preview path must be provably output-identical to the full path where
  it claims to be, or clearly separated where it's an approximation.
- Don't regress the golden-array kernel tests or `test_render_order.py`.

## Deliverables

1. `benchmarks/` harness + a short `benchmarks/RESULTS.md` (baseline vs optimized).
2. The optimizations themselves, each its own TDD'd, committed change with
   before/after numbers.
3. A final summary: interactive latency (drag-to-preview) before vs after, full-res
   render time before vs after, startup time before vs after.
4. `git push` when done; update `docs/memory/` (via the `zam-memory` skill) with an
   optimization progress entry.

Start with Phase 0 measurement, then do #1 (cancel superseded renders) and #3
(preview proxy) for the biggest perceived-latency win, then #2 (staged cache), then
the rest as the profile justifies.
