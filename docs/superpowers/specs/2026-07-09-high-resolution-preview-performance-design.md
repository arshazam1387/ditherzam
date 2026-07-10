# High-Resolution Preview and Render Performance — Design

**Date:** 2026-07-09  
**Status:** Approved  
**Scope:** Interactive still-image preview performance, shared render efficiency,
and exact-output safeguards.

## Goal and evidence

Keep the editor responsive on 4K+ sources without changing creative output.
On-screen work becomes quality-bounded; exports remain exact.

| case | baseline | consequence |
|---|---:|---|
| Exact 4K ramp/nearest/ordered | 0.86–1.08 s | No automatic exact settled pass |
| 4K color off | ~615 ms | Optimize and cap the whole pipeline |
| Diffused RGB, 480p | 5.56 s | First proven compute hotspot |
| Unchanged cached render | effectively free | Preserve staged caching |
| 4K saturation-only | ~412 ms | Reuse upstream and render fewer pixels |
| 4K upstream change | ~773 ms | Resolution policy is still required |
| First drag feedback | ~67 ms | Preserve coalescing and priority |
| Full-size proxy display allocation | ~24 MiB at 4K | Return display-sized pixels |
| Retained 4K staged cache | ~182 MiB | Add a budget and eviction |

## Approved UX

Add visible **Preview Resolution** choices: **Auto** (default), **480**, **720**,
**1080**, **1440**, **2160**, and **Full**. Numeric values are longest-side
pixels. The choice applies to both drag and settled previews, persists across
restarts, stays outside creative presets, and never changes source/export size or
dither scale.

Auto is deterministic and viewport-aware: use drawable viewport device pixels,
preserve aspect ratio, choose the smallest listed cap covering the fitted image,
and clamp the result to 720–1440. Resize rerenders only when crossing a bucket.

Add **View → Full Quality Preview** with **Ctrl+Enter**. It performs one exact
source-resolution preview. A subsequent creative edit, palette hover, source or
preference change, or qualifying zoom returns to the selected policy.

Persist **Rerender preview when zooming in**, default Off and excluded from
presets. When enabled, a settled zoom crossing a quality bucket requests the next
allowed preview. Numeric modes never exceed their selected cap; Auto never
exceeds 1440; Full may reach the source. Initial load schedules an asynchronous
capped preview; images already within the cap render exactly.

## Render lifecycle

| event | quality | execution | preview preference |
|---|---|---|---|
| Initial load | selected cap; exact if it fits | asynchronous | applied |
| Drag and idle settle | selected cap | asynchronous/coalesced | applied |
| Full Quality Preview | exact source | asynchronous one-off | ignored |
| Zoom refinement | next allowed bucket | async after settle | ceiling only |
| PNG/JPEG/SVG | exact source/output | dedicated exact snapshot | ignored |
| Batch | exact per source | dedicated exact snapshot | ignored |
| Video/animation screen preview | selected cap | asynchronous | applied |
| Video/animation export | exact configured frames | dedicated snapshot | ignored |

Preview pixels never feed exports.

## Architecture

### Pure policy and viewport sizing

Qt-free helpers own resolution parsing, cap geometry, Auto selection, proxy
dither-scale adjustment, and zoom refinement. Qt supplies logical viewport size
and device-pixel ratio. Preview rendering returns only capped RGB pixels. The
viewport presents the smaller pixmap at source-logical scene bounds so fit,
pan, zoom, cursor, and 100% semantics remain source-relative without allocating
a source-sized raster.

### Immutable requests and quality priority

Frozen render requests contain generation, kind, source identity, immutable
RenderSettings, target dimensions, snapshotted color/effect context, logical
source dimensions, and display metadata. Workers never reread mutable UI state.
Every worker emits exactly one success or failure terminal signal.

Keep at most one interactive worker and one latest trailing request. Priority is
newest state, then explicit Full over capped work for the same generation, then
settled/zoom over drag for identical state. A new edit invalidates an older Full
result. Export uses a separate exact snapshot and is never downgraded by preview
invalidation.

### Color ownership and exact algorithms

A stable render context reuses palette arrays, luminance ordering, ramps, and
other exact derived data. Keys use palette shape/dtype/ordered color bytes plus
mode, depth, mapping, phase, and algorithm version—never palette name alone.
Cached arrays are immutable.

Compile or restructure RGB Floyd–Steinberg while preserving scan direction,
boundaries, float32 precision/casting points, palette first-minimum ties, and
final pixels. Fuse ramp luminance/index lookup and ordered offset/color lookup to
avoid large intermediates, but require differential equality to the reference.

### Memory, cancellation, and threading

Do not upscale previews to source dimensions. Avoid grayscale-to-RGB and ordered
temporaries where exact output permits. Account retained array `nbytes` under a
**192 MiB per-editor cache budget**, evicting least-recently-used complete entries
before insertion. Oversized active intermediates may render but are not retained.

Use selective internal parallelism only for independent work such as pointwise
adjustments, nearest mapping, ramp lookup, ordered mapping, and effects proven
safe by profiling. Benchmark 1/2/4/8 threads. Floyd–Steinberg remains sequential
unless a byte-identical strategy is proven. Keep one interactive render in
flight, prevent Qt/Numba/export oversubscription, and give exports a bounded
thread budget. Obsolete jobs may stop between expensive stages, never through
unsafe thread termination.

## Optimization task areas

1. Viewport-sized capped preview and source-logical viewport geometry.
2. Preview preferences, Full action, zoom policy, and asynchronous initial load.
3. Immutable request/context ownership and stale-result priority.
4. Exact compiled RGB diffusion.
5. Fused ramp and ordered mappings with fewer allocations.
6. Stable content-keyed ColorEngine derived-data reuse.
7. Exact grayscale/RGB, saturation, and adjustment temporary reduction.
8. Bounded staged cache and reusable buffers.
9. Profile-guided effects and adjustment fusion.
10. QImage/QPixmap copy reduction with lifetime safety.
11. Export isolation and capped animation/video screen previews.
12. Selective threading, stage-boundary cancellation, and JIT warmup refinement.

## Compatibility and rejected options

Preserve clean-room rules, Qt-free core, frozen `STAGE_ORDER`, exact full
`render_cached()==render`, nearest tie behavior, two-level pixels, depth
promotion/tone bias, palette hover/editing, effects, presets, batch, video,
animation, and exports. Preserve terminal worker signals and one-read
engine/effect snapshots. Validate JIT-off and JIT-on while separately accounting
for the historical special-kernel JIT failures.

Rejected: automatic exact settle, drag-only caps, timing-adaptive Auto, preview
quality in presets, mandatory zoom rerenders, full-source proxy upscaling,
approximate LUTs, changed diffusion scan order, palette-name cache keys, unbounded
caching, simultaneous interactive renders, and preview-derived exports.

## Quantitative acceptance

1. Warm preview worker median: ≤100 ms at 480, ≤150 ms at 720, ≤250 ms at
   1080, ≤400 ms at 1440, and ≤1.1 s at 2160.
2. 4K slider first feedback ≤100 ms median and ≤150 ms p95; no stale paints.
3. Exact diffused RGB at least 5× faster than the 5.56 s 480p baseline while
   byte-identical on the reference corpus.
4. Unchanged cache remains effectively free; equal-size saturation/upstream
   benchmarks regress no more than 10%.
5. Retained staged/derived cache ≤192 MiB; capped display results allocate only
   target-sized RGB plus at most 10% bookkeeping.
6. Full preview and still/batch/video/animation exports remain independent of
   preview preferences and exact against the reference path.
7. Report cold JIT separately and do not compile on the GUI thread.
8. Thread-count benchmarks record latency, throughput, memory, and UI heartbeat;
   the selected default must beat one thread without oversubscription.

## Manual 4K+ QA

- Load 4K+ portrait, landscape, noisy, gradient, saturated, and alpha-flattened
  photos; confirm asynchronous initial display and responsive UI.
- Cycle all caps and verify source-relative fit/pan/zoom/cursor behavior.
- Cross standard/high-DPI displays; Auto changes only at bucket boundaries.
- Exercise rapid adjustments, palette hover/edit, color/effect toggles, and a
  forced worker error; latest state paints and recovery succeeds.
- Invoke Ctrl+Enter during capped work, then edit; stale results never overwrite.
- Verify zoom-rerender Off and On behavior and resolution ceilings.
- Compare all color modes, especially diffusion, against reference hashes.
- Export still, batch, video, and animation while capped; verify exact dimensions
  and pixels.
- Run 50+ edits/source replacements while monitoring retained memory.
