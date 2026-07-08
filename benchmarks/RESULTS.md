# ditherzam performance — benchmark results

Machine: this box (Windows 11, CPython 3.12.13, JIT **enabled**). Reproduce:

```
.venv/Scripts/python.exe -m benchmarks.bench        # render-core table
.venv/Scripts/python.exe -m benchmarks.ui_latency   # offscreen drag-to-pixmap probe
```

All render-core numbers are **warm** (JIT compiled) unless a "cold" column is shown.
Times are median-of-5 wall-clock ms. Inputs are deterministic synthetic grayscale
(`benchmarks/common.make_gray`).

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

## Findings that reorder the handoff's priorities

The handoff assumed **effects** dominate. Measurement disagrees:

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
| 2 | staged render cache | — | — | pending |
| 3 | interactive preview proxy | — | — | pending |
