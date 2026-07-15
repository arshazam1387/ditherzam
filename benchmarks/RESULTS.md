# ditherzam performance — benchmark results

Machine: this box (Windows 11, CPython 3.12.13, JIT **enabled**). Reproduce:

```
.venv/Scripts/python.exe -m benchmarks.bench        # render-core table
.venv/Scripts/python.exe -m benchmarks.ui_latency   # offscreen drag-to-pixmap probe
```

All render-core numbers are **warm** (JIT compiled) unless a "cold" column is shown.
Times are median-of-5 wall-clock ms. Inputs are deterministic synthetic grayscale
(`benchmarks/common.make_gray`).

## Final summary (baseline `90a9497` → `perf/optimize-app`)

| dimension | before | after |
|-----------|--------|-------|
| **drag-to-first-feedback** (1080p, upstream control) | full ~250 ms+ render every tick, 19/20 painted then instantly replaced | **~98 ms** proxy; 12 cheap proxies + **1** exact full render, no stale paints |
| **downstream tweak** (1080p, cached full render) | ~592 ms (full recompute) | invert **59 ms**, effects **192 ms**, saturation **340 ms** |
| **full-res render** (1080p, FS + palette + 2 effects) | 756 ms | **497 ms** |
| **color palette map** (1080p, per render) | 370 ms | **98 ms** (bit-identical) |
| **first render after launch** (fresh process) | 715 ms (cold JIT) | **354 ms** (background warmup) |
| tests | 333 | **374** (all new work TDD'd; green JIT-on **and** JIT-off) |

Output is unchanged: `render()` and the frozen stage order are untouched, golden
kernel/render tests pass, and the new `render_cached`/`nearest_indices` paths are
proven bit-identical. The preview proxy is the only approximate path and is
display-only during an active drag; every committed/exported image and the
settled on-screen image come from the full-resolution pipeline (verified
end-to-end: settled display == exact `render_cached`).

## Baseline (before any optimization) — commit `90a9497`

### Warm vs cold render (ms)

| size  | style            | cold  | warm  |
|-------|------------------|-------|-------|
| 512   | Floyd-Steinberg  | 271   | 31    |
| 512   | Atkinson         | 36    | 32    |
| 512   | Bayer 4x4        | 54    | 29    |
| 1080p | Floyd-Steinberg  | 231   | 257   |
| 1080p | Atkinson         | 210   | 211   |
| 1080p | Bayer 4x4        | 204   | 203   |
| 4K    | Floyd-Steinberg  | 903   | 852   |
| 4K    | Atkinson         | 833   | 834   |
| 4K    | Bayer 4x4        | 822   | 834   |

### Heavy path: FS + Game Boy (4-color) + Chromatic Aberration + Epsilon Glow (warm)

| size  | time  |
|-------|-------|
| 512   | 93 ms |
| 1080p | 756 ms |

### UI drag latency probe (offscreen, 720p, 20-step luminance drag)

| metric            | baseline |
|-------------------|----------|
| full renders run  | 20       |
| pixmaps delivered | 20       |
| **wasted paints** | **19**   (each immediately superseded) |
| drag+drain wall   | ~1.6 s   |

Every drag tick spawns a fresh worker that runs to completion; 19 of 20 results are
painted then instantly replaced. This is what optimization #1 (cancel superseded)
targets.

**After #1 + #2 + #3** (same probe, now at 1080p; a luminance drag is an
upstream-control change the cache can't help, so the proxy carries it):

| metric                    | after |
|---------------------------|-------|
| proxy renders (feedback)  | ~12 (cheap, ~99 ms each) |
| full-res renders          | **1** (on settle) |
| time to first feedback    | **~98 ms** (was a full ~250 ms+ render) |

20 full renders → 12 cheap proxies + 1 exact full render.

### Per-stage breakdown @1080p, heavy path (warm)

| stage      | ms/render |
|------------|-----------|
| **color**  | **370**   |
| effects    | 176       |
| saturation | 115       |
| clamp      | 20        |
| midtones   | 17        |
| dither     | 14        |
| contrast   | 8         |
| highlights | 8         |
| blur       | 0 (identity) |

## Findings that reorder the optimization priorities

The working assumption was that **effects** dominate. Measurement disagrees:

1. **`ColorEngine.map` (nearest) is the #1 cost — 370 ms even for a 4-color palette.**
   `nearest_indices` builds a `(H, W, K, 3)` broadcast temp (~100 MB at 1080p) then
   squares/sums/argmins it. Memory-bound; scales with palette size K. Fixing this is
   output-preserving (same squared-distance argmin) and the single biggest safe win.
2. **`apply_saturation` is 115 ms and runs on EVERY render**, even at the default
   value 50 (identity). It dominates a plain (no-palette) 1080p render. Cannot be
   naively short-circuited (would shift ±1 LSB vs the currently-committed pixels), so
   it is addressed via the staged cache + preview proxy rather than a math change.
3. Effects (176 ms) are real but third. PIL Gaussian-blur based (Epsilon Glow).

## Optimizations (this table fills in as each lands)

| # | optimization | before | after | notes |
|---|--------------|--------|-------|-------|
| A | `nearest_indices` njit | 370 ms | 98 ms | color stage, bit-identical (3.8×); heavy path 756→497 ms |
| 1 | cancel superseded + single-in-flight coalescing | 20 renders / 19 wasted | 8 renders / 7 wasted | 20-step 720p drag; no stale out-of-order paints |
| 2 | staged render cache (`render_cached`) | 592 ms (full) | invert **59**, effects **192**, saturation **340** | @1080p heavy path, single-control tick; bit-identical to `render()`. Upstream changes (contrast/luminance) stay ~full — see #3. |
| 3 | interactive preview proxy | 534 ms (1080p) / 2274 ms (4K) full | **99 ms** / **217 ms** proxy | upstream-control drag; proxy is display-only, full-res on settle |
| 5 | background JIT warmup | first render 715 ms | first render **354 ms** | ~360 ms cold-JIT compile moved to a daemon thread that overlaps window-show |

### Startup / first-interaction (fresh process)

| | without warmup | with background warmup |
|---|---|---|
| first user render (1080p FS + palette) | 715 ms (cold JIT link) | **354 ms** (pre-warmed) |
| warmup thread duration | — | ~360 ms (off the critical path) |

Note: importing `ditherzam.ui.main_window` costs ~1.3 s, ~900 ms of which is
Numba's *import* (pulled in by kernel registration, which the dither combo needs
at construction). Deferring that would need a registry that lists kernel
names/categories without importing Numba — a larger refactor left as future work;
it only affects time-to-window, not time-to-first-render (which any render pays).

### Staged cache: single-control cached tick @1080p heavy path (warm)

| control changed | full uncached | cached tick |
|-----------------|---------------|-------------|
| invert          | 592 ms        | **59 ms**   |
| effects param   | 592 ms        | **192 ms**  |
| saturation      | 592 ms        | **340 ms**  |
| luminance       | 592 ms        | 484 ms (reuses adjustments only) |
| contrast (top)  | 592 ms        | 520 ms (~full: nothing upstream to reuse) |

## 2026-07-09 — Task 3.2 tonal adjustment fusion (JIT on)

Command: `.venv/Scripts/python.exe -m benchmarks.adjustment_fusion`

Deterministic float32 input; median of 5 warm runs at 1080p and 3 at 4K.
Peak allocation is Python `tracemalloc` (useful for candidate comparison, not
whole-process RSS). Hashes and the complete reproducible sweep are emitted as JSON.

| operation/candidate | 1080p | 4K | 1080p peak | 4K peak | exact together output |
|---|---:|---:|---:|---:|---|
| contrast | 15.6 ms | 38.9 ms | 15.82 MiB | 63.28 MiB | n/a |
| midtones | 62.4 ms | 183.0 ms | 15.82 MiB | 63.28 MiB | n/a |
| highlights | 20.4 ms | 36.3 ms | 15.82 MiB | 63.28 MiB | n/a |
| production three-call chain | 84.7 ms | 281.4 ms | 23.73 MiB | 94.92 MiB | reference |
| in-place, three-pass candidate | **49.0 ms** | **162.2 ms** | **7.91 MiB** | **31.64 MiB** | **yes** |
| compiled true one-pass candidate | 16.9 ms | 63.8 ms | 7.91 MiB | 31.64 MiB | **no** |

The exact in-place candidate is 1.73× faster at both resolutions and cuts
traced peak allocation by two thirds. It matched the production SHA-256 at
1080p and 4K and matched all 100,009 adversarial/random values under two setting
triples, including NaNs produced by negative inputs. The true one-pass Numba
candidate is faster, but changed 1,482,787 / 2,073,600 values at 1080p and
5,931,888 / 8,294,400 at 4K (maximum finite sweep error 0.0001220703125).

**Recommendation:** implement the allocation-reduced three-pass helper, retaining
the current float32 rounding boundaries and frozen stage semantics. Reject the
true algebraic/compiled one-pass fusion because it is not byte-exact. Before
production integration, preserve cancellation boundaries and public stage-call
observability required by render-order tests.

Downstream-of-color controls become cheap; top-of-pipeline controls need the
preview proxy (#3) since there is no upstream intermediate to reuse.
