# Architecture

This document describes the runtime architecture of ditherzam 0.1.0, the data
flow through the renderer, and the rules that keep previews responsive without
changing exported pixels.

## System overview

```text
PySide6 application
  main window, controls, viewport, palette editor, timeline, video UI
                         |
                         v
immutable render request + latest-wins scheduler
                         |
                         v
RenderPipeline ------------------------------------+
  adjustments -> dither registry -> color -> effects|
         |                 |                        |
         +------ bounded staged render cache -------+
                         |
                 NumPy / Numba arrays
                         |
          +--------------+---------------+
          |                              |
     screen preview                 exact export
     capped, async             raster / SVG / media
```

The package keeps image processing independent from Qt. PySide6 is confined to
the application entry point, UI modules, and media workers. The render, color,
dithering, effects, animation math, presets, batch, masking contracts, and export
logic can be exercised without creating a window.

## Module map

| Area | Primary modules | Responsibility |
|---|---|---|
| Application | `ditherzam/app.py`, `ditherzam/ui/` | Window lifecycle, controls, viewport, background scheduling, export actions |
| Render orchestration | `render.py`, `render_cache.py` | Fixed pipeline order, immutable settings, cancellation boundaries, bounded staged cache |
| Dithering | `dithering/registry.py`, `dithering/pipeline.py`, `dithering/kernels/` | Style discovery, parameter mapping, binary/multilevel dispatch, compiled algorithms |
| Color | `color/engine.py`, `color/context.py`, `color/ramp.py`, `color/palette_store.py` | Palette loading/editing, extracted colors, reusable lookup contexts, tone mapping |
| Effects | `effects/stack.py`, `effects/post.py` | Ordered post-processing effects and reusable effect buffers |
| Animation | `animation/temporal.py`, `animation/timeline.py`, `ui/timeline_panel.py` | Temporal threshold fields, easing, frame requests, preview and export |
| Video | `video/ffmpeg.py`, `video/frames.py`, `video/workers.py`, `ui/video_controller.py` | Safe FFmpeg commands, frame processing, assembly, playback coordination |
| Masking | `masking/`, `ui/mask_workers.py`, `ui/smart_mask_panel.py` | Optional offline inference, geometry, cache, compositing, fail-closed asset validation |
| Output | `export/raster.py`, `export/vector.py`, `batch.py`, `presets.py` | Exact image export, SVG run merging, batch jobs, serializable settings |

## Render data model

Decoded images are converted to owned arrays before asynchronous work begins.
The base render representation is grayscale `float32` in the range 0..255.
Settings are frozen values, and a worker receives a snapshot of the pipeline
dependencies it needs. This avoids a control change replacing a palette or
effect stack halfway through an existing render.

`RenderPipeline` applies stages in a stable order:

1. Brightness
2. Contrast
3. Midtones
4. Highlights
5. Dithering and depth handling
6. Palette color mapping
7. Saturation
8. Inversion
9. Ordered effects

Contrast, midtones, and highlights share a render-private scratch buffer across
their separate passes. They are intentionally not fused into one formula because
that changes rounding and therefore output pixels.

## Dither dispatch

The global `DitherRegistry` contains the user-facing name, category, dimensional
requirements, controls, implementation function, and multilevel capability for
each style. Kernel modules register themselves when `ditherzam.dithering` is
imported.

The library currently contains 77 styles:

- 18 error-diffusion styles
- 14 ordered styles
- 12 patterned styles
- 15 glitch styles
- 18 special and generative styles

Universal controls are normalized before dispatch. Style-specific parameter
functions translate UI slider values into algorithm values, keeping presentation
logic out of the kernels. Seeds make stochastic styles reproducible.

Native multilevel kernels return their depth information directly. Binary-only
kernels retain their exact two-level behavior and use a generic promotion step
when the selected depth is greater than two.

## Color engine

Palettes are immutable value objects loaded from built-in YAML files or the user
palette store. The editor can fork a built-in palette before changing it, so
packaged defaults remain stable. Image extraction provides source-derived colors.

Reusable color contexts precompute palette arrays and lookup information. This
avoids rebuilding identical data for every preview. Ordered-color and depth-ramp
paths use fused compiled passes where doing so preserves exact output.

## Preview scheduling and caching

Interactive rendering follows a latest-wins policy:

1. The UI captures an immutable request.
2. If no render is running, it starts a worker in the Qt thread pool.
3. While that worker runs, new changes replace one pending request.
4. The worker emits exactly one terminal signal, including on failure.
5. The scheduler publishes a result only if it is still current, then starts the
   pending request if one exists.

This bounds work during slider drags: the application finishes at most the active
render and the newest request instead of queueing every intermediate state.
Cancellation checks between expensive stages let obsolete work exit earlier.

The staged LRU cache is bounded by memory rather than entry count. Cached arrays
must never be reused as mutable scratch space; render-private buffers are the only
safe place for in-place operations.

## Preview versus export

Screen previews and exports have different performance contracts:

- Preview rendering may downsample the logical source according to the configured
  cap. The viewport continues to use source-space coordinates for stable zoom and
  pan behavior.
- A settled zoom can request a more detailed region without changing source data.
- Animation and video screen previews use the same capped, asynchronous policy.
- Image, animation, batch, and video exports create exact render contexts and never
  read the preview cap.

This separation is an invariant: display preferences must not change exported
dimensions or pixels.

## Threading and compilation

Numba compiles hot dither, color, morphology, and adjustment paths. The application
installs a measured interactive thread budget before background warm-up so kernels
compile with the same policy used by normal renders. FFmpeg jobs and export workers
use bounded resources to avoid multiplying CPU-heavy pools.

Performance work is accepted only when output hashes remain stable where exactness
is required. Benchmark scripts under `benchmarks/` cover high-resolution rendering,
cache behavior, UI latency, QImage conversion, and thread scaling.

## Animation and video

Animation generates temporal threshold fields from deterministic patterns and a
timeline with easing. Each frame goes through the standard render pipeline, so
still and animated output share style and color semantics.

Video processing probes media with FFprobe, decodes frames, renders them, and asks
FFmpeg to assemble the result. Command builders keep probing, decoding, encoding,
and muxing concerns separate. Playback shown after export is distinct from the
capped animation preview in the editor.

## Optional masking

Masking is an optional branch around the normal renderer:

```text
owned source -> offline inference -> probability map -> geometry -> mask
                                                        |
normal render ------------------------------------------+-> alpha composite
```

Inference requests and results carry source/model identities so stale masks cannot
attach to a different image. Inference and composite caches are bounded. Model
loading is fail-closed: the app performs no network download, and a missing or
invalid local asset leaves masking disabled.

## Testing invariants

The suite protects the behaviors most likely to regress:

- Core modules remain importable without Qt.
- Two-level rendering remains byte-stable when multilevel support is added.
- Preview caps never leak into exact exports.
- Mutable UI state is snapshotted before concurrent rendering.
- Worker success and failure paths always release the scheduler.
- Caches remain bounded and cached outputs are never corrupted by buffer reuse.
- Stochastic styles are deterministic for the same seed.
- Compiled and non-compiled kernel paths agree.

Use Python 3.12. Fast development runs set `NUMBA_DISABLE_JIT=1`; kernel changes
must also receive a JIT-enabled pass.
