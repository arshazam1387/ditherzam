# Handoff — High-Resolution Performance Planning

Paste this into a fresh Codex or Claude Code session opened at `C:\Users\arsha\Desktop\custom dither`.

---

We need to **plan a new optimization pass for ditherzam**. The app increasingly slows down as source-image resolution rises. This session is for investigation, benchmarking, design, and a detailed TDD implementation plan; do not begin implementation until the design is approved.

Start by invoking `zam-memory` in recall mode. Read `docs/memory/INDEX.md`, every constraint, the latest progress, and performance/render/color entries 015, 017, 018, 020, 022, and 023. Verify remembered claims against the tree.

## Goals

1. Keep interaction responsive with high-resolution images.
2. Add a visible user option for preview resolution/quality.
3. Optimize rendering generally, especially the color engine.
4. Preserve exact full-resolution export unless the user explicitly chooses otherwise. Preview resolution must never silently reduce export quality.

## Existing implementation

- `ditherzam/ui/preview.py` caps the interaction proxy's longest side at `proxy_max_side` (currently 640), adjusts dither scale, renders small, then nearest-upscales to source dimensions.
- `ditherzam/ui/main_window.py` schedules a proxy after 20 ms and exact full-resolution `render_cached()` after 160 ms. The settled pass can still stall on large images. `_RenderWorker` plus `RenderCoalescer` allows one in-flight render and one trailing request.
- `ditherzam/render.py::RenderPipeline.render_cached()` caches staged intermediates. `render()` order is frozen: contrast → midtones → highlights → blur → dither → color → saturation → effects → invert.
- `ditherzam/color/engine.py` already uses `_nearest_indices_njit(..., parallel=True)` instead of an `(H,W,K,3)` broadcast. Preserve exact first-minimum tie behavior and avoid large palette-distance temporaries.
- `ColorEngine._get_ramp()` caches ramps, but `ImageEditor._sync_pipeline()` creates a new engine before every render, bypassing this cache in the live path.
- `ramp` computes luminance and indexed ramp mapping; `nearest` and `ordered` use the Numba mapper; `diffused` still uses Python `_floyd_steinberg_rgb` nested loops and is a likely high-resolution hotspot.
- Extend the existing reproducible material in `benchmarks/` rather than replacing it.

## Measure before designing

Profile deterministic cold and warm renders at 1080p, 4K, and one larger size if memory permits. Cover color off/ramp/nearest/ordered/diffused, small and large palettes, several proxy caps, exact settled and repeated cached renders, and changes to dither-only, palette/mode, saturation, effects, and invert. Measure time, peak memory, UI-visible latency where practical, and JIT cold start separately. Report baseline numbers and rank proven costs; do not assume color is the only bottleneck in every mode.

## Preview-resolution UX to design with the user

- A clear **Preview Resolution** or **Preview Quality** control.
- Auto plus measured explicit longest-side choices (candidate values: 480, 720, 1080, 1440, 2160, Full).
- Decide whether it controls only drag previews or also settled on-screen previews.
- Consider Auto based on source size and/or a latency budget.
- Decide whether it is session-only or persisted as an app preference; avoid putting machine-specific preview quality in creative presets without a strong reason.
- Avoid allocating a full-source-size upscaled proxy when the viewport only needs display size.
- Provide an obvious one-off full-quality preview if settled previews may stay capped.
- Keep preview quality distinct from dither `scale`, source size, and export size.

State exactly when full-resolution work occurs for initial load, idle settle, explicit full preview, still export, batch, video, and animation.

## Candidates to benchmark, not assume

- Reuse a stable `ColorEngine` safely so ramp and derived palette state survive renders; invalidate by palette content plus mode/depth/mapping/phase, never palette name alone.
- Numba-compile or restructure RGB Floyd–Steinberg while preserving scan order, boundaries, precision, and pixels.
- Avoid grayscale-to-RGB allocations where a mode can operate on grayscale/indexed levels.
- Optimize ramp luminance/indexing and ordered offset construction without visual changes unless approved.
- Cache palette arrays/derived data by color content; palette editing and hover preview can change colors under the same name.
- Consider LUTs only with specified error and memory cost; exact output is the default.
- Prevent unnecessary exact settled renders during rapid changes.
- Render proxies for actual viewport needs instead of upscaling to source size if profiling confirms the allocation matters.
- Bound 4K+ cache memory; do not retain unlimited large intermediates.
- Follow profiling into effects/dither hotspots too.

## Constraints

- Clean-room; no Studio AAA/Dither Boy code, strings, binaries, licensing, telemetry, or network behavior.
- Core stays Qt-free. Qt remains under `ui/`, `app.py`, and `video/workers.py`.
- Python 3.12 via `.venv/Scripts/python.exe`; TDD per task.
- Preserve `RenderPipeline.STAGE_ORDER`; full-resolution `render_cached()` remains byte-identical to `render()`.
- Preserve entry 022's terminal worker signals, one-read engine/effect snapshots, and coalescer recovery. Do not reintroduce the wedge or TOCTOU race.
- Preserve depth promotion, threshold tone-bias semantics, palette editing/hover preview, effects, animation, video, batch, and exports.
- Inspect `git status` and diffs; do not overwrite unrelated local work.
- Validate JIT-off and JIT-on. Establish the current baseline and separate entry 020's historical JIT failures from regressions.

## Planning deliverables

1. Extend the benchmark harness and record baselines without mixing benchmark-only changes into product behavior.
2. Write an approval-gated design in `docs/superpowers/specs/` covering UX, render lifecycle, architecture, cache invalidation/ownership, concurrency, memory bounds, compatibility, and rejected options.
3. After approval, write a detailed plan in `docs/superpowers/plans/` with small red → green → refactor tasks, exact files/tests/commands, expected failures, and commit boundaries.
4. Define quantitative acceptance criteria: preview latency by selected size, improvement for proven color hotspots, bounded peak/cache memory, exact-output tests, stale-render tests, and full-quality export tests.
5. Include a before/after benchmark matrix and manual real-photo 4K+ QA checklist.
6. Record durable findings and the approved next task through `zam-memory`, deduplicating the index.

Begin with recall and measurement. Then show the baseline and ask focused UX questions one at a time before proposing the design.
