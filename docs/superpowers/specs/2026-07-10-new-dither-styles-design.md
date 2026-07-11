# New Dither Styles — Design (2026-07-10)

## Problem

The catalog has 66 kernels but low visual variety: the 16 Error-Diffusion kernels
(Floyd-Steinberg, JJN, Stucki, Burkes, Sierra family, …) all produce nearly
identical grainy raster diffusion, and there are ~5 near-duplicate Bayer/ordered
variants. Users perceive "a lot of styles have similar outputs."

## Goal

Add 8 new kernels that each look **distinct from one another and from the existing
set**, spanning four structural ideas that break the square-grid / raster-diffusion
mold, plus two genuinely novel generative techniques.

## Scope

In: 8 new dither kernels, one new source module, golden-fixture coverage, both-JIT-mode
verification. Out: new UI widgets, new categories, video/animation-specific work,
changes to existing kernels.

## The 8 kernels

All register into **existing** categories (no new category). All output **binary
0/255** float32 — the pipeline's `_binary_to_levels` promoter (pipeline.py:31)
auto-gives them multi-tone/palette support, so no `supports_levels` plumbing.
All use the universally-present **`dither_parameter_slider`** as their one control
(mapped per-kernel below); `luminance_threshold_value` (always passed, 0..255) is a
secondary knob where noted.

| # | Name | Registry category | Slider = main control |
|---|------|-------------------|-----------------------|
| 1 | Hilbert (Riemersma) | Error Diffusion | queue length / decay |
| 2 | Spiral Path | Error Diffusion | error retention |
| 3 | Flow Hatch | Patterned | line spacing |
| 4 | Hex Bayer | Ordered Dither | cell size |
| 5 | Triangular | Ordered Dither | cell size |
| 6 | Spiral Engrave | Special Effects | spiral pitch |
| 7 | Reaction-Diffusion | Special Effects | iteration count |
| 8 | Quasicrystal | Special Effects | number of waves |

### 1. Hilbert (Riemersma)
Walk a Hilbert space-filling curve over the bounding square of side
`S = 2^ceil(log2(max(h,w)))`. Iterate index `d` from `0 .. S*S-1`, convert to `(x,y)`
via an iterative `d2xy`, skip out-of-bounds cells. Maintain a Riemersma decaying
error queue of length `L` (from slider, e.g. 8..32): the running error is a weighted
sum with geometric decay `r` so old error fades. At each visited pixel: `val = img +
carried_err`; emit 255 if `val >= thr` else 0; push residual into the queue.
**Sequential** (inherently) → byte-identical across JIT modes.
Look: soft, grain-free, isotropic, no raster streaks.

### 2. Spiral Path
Same 1-D diffusion idea but the visiting order is a center-out square spiral instead
of a Hilbert curve. Precompute the spiral visiting order iteratively (ring by ring).
Slider controls error retention fraction. Sequential → deterministic.
Look: swirling grain that radiates from the center.

### 3. Flow Hatch
Per-pixel, parallel. Compute Sobel gradient `(gx, gy)`; the isophote direction is
perpendicular to the gradient. Project pixel position onto the gradient axis to get a
phase `p = (x*gx + y*gy)/mag`; draw a hatch line where `frac(p / spacing)` is near 0.
Gate lines by darkness (`1 - img/255`) against `luminance_threshold`. Flat regions
(low `mag`) fall back to a fixed diagonal so they aren't blank.
Look: engraving whose lines bend around the shapes.

### 4. Hex Bayer
Per-pixel, parallel. Map `(x,y)` to a hexagonal (offset-row) lattice cell of size `c`
(slider): odd rows shifted by `c/2`. Index a small ordered threshold table by
`(cell_row % m, cell_col % m)` and compare. Result reads as honeycomb clustering.

### 5. Triangular
Per-pixel, parallel. Subdivide the plane into a triangular grid of size `c`; determine
which of the two triangles in each square cell a pixel falls in (via the diagonal
test) and combine with a Bayer threshold offset per triangle → faceted texture.

### 6. Spiral Engrave
Per-pixel, parallel. Polar coords about the image center: `r`, `ang`. Archimedean
spiral phase `phi = r / pitch - ang / (2π)`; distance to nearest spiral arm
`darm = |frac(phi) - 0.5| ` scaled. Ink the pixel (0) when `darm < thickness`, where
`thickness` grows with local darkness → a single continuous line, fat in shadows.
Look: banknote / vinyl-record portrait.

### 7. Reaction-Diffusion  ⭐ novel
Gray-Scott model. Two fields `U`, `V` over the (downscaled) preview buffer. Initialize
`U=1`, `V=0`; inject `V` using **deterministic hash noise** seeded by pixel coords
(NOT `np.random` — see determinism rules). Feed rate `F` and kill rate `k` are
**modulated by image luminance** so pattern density tracks brightness. Run `N`
iterations (slider, capped e.g. 10..60) of the discrete Laplacian update with a
double buffer. Threshold final `V` to 0/255. Per-iteration cells are independent
(double-buffered) → deterministic; iteration loop is sequential.
Look: organic coral / maze / leopard-spot texture that encodes the image. This is the
"never used as a dither" showcase.

### 8. Quasicrystal  ⭐ novel
Per-pixel, parallel. Sum `n` plane waves (slider, e.g. 3..7) rotated by the golden
angle: `s = Σ cos(kx·x + ky·y + phase_i)` with fixed per-wave orientation. Modulate
threshold by image luminance. Emit 255 where `s` (normalized) `>= img_norm`. Fixed
loop order over waves → identical float sum across JIT modes.
Look: aperiodic shimmering interference moiré.

## Determinism rules (hard constraint)

The golden fixture is seeded in whichever JIT mode runs first, so **JIT-on output MUST
equal JIT-off output byte-for-byte**. Therefore:

- **No `np.random` inside `parallel=True` kernels.** Use an integer hash of pixel
  coords for any stochastic seeding (mode- and thread-invariant). Sequential kernels
  may use a seeded RNG but hash noise is preferred for consistency.
- **No cross-thread float reductions.** Every per-pixel value is computed from a
  fixed-order local loop (waves, neighbors), never a parallel reduction.
- **`int()` at every derived array index** (`img[int(yd), x]`) — the parfor float64
  typing bug fixed on 2026-07-10 (see memory 020). Applies especially to Flow Hatch,
  Hex, Triangular, Spiral neighbor/warp lookups.
- float32 I/O, values clamped to `[0, 255]`.

## Architecture / files

- New module `ditherzam/dithering/kernels/generative.py` holding all 8 (isolated,
  reviewable, ~one screen each). njit inner `_fn`, thin registered wrapper following
  the existing pattern in `special.py`.
- Register the module in `ditherzam/dithering/kernels/__init__.py` (import for side
  effects, same as the other kernel modules).
- No changes to registry, pipeline, or UI — categories and sliders already exist.

## Testing (TDD per constraint 003)

- `tests/test_kernels_all.py` auto-parametrizes every registered kernel: it asserts
  dtype/shape/range and **auto-seeds** a golden `.npy` on first run. New kernels get
  coverage for free.
- Per kernel: red = kernel not yet registered / import fails; green = registered,
  golden seeds, dtype/shape/range pass.
- **Both-mode gate:** after implementing, delete any freshly-seeded goldens, run the
  golden suite once under `NUMBA_DISABLE_JIT=1` (seeds), then again under
  `NUMBA_DISABLE_JIT=0` (must match). This proves byte-identical determinism. This is
  the real acceptance test for these kernels.
- Full suite green both modes; `test_total_count_at_least_63` rises to 74.
- Add a `tests/test_kernels_generative.py` with a couple of structural sanity checks
  (e.g. Quasicrystal produces both black and white pixels on a gradient; Spiral
  Engrave is symmetric-ish about center) to catch "all one color" regressions the
  auto-golden can't.

## Performance notes

- 7 of 8 are per-pixel `parallel=True`, O(H·W) — negligible at preview scale.
- Reaction-Diffusion is O(N·H·W); N capped via slider and it runs on the already
  downscaled preview buffer (capped-preview policy, memory 030), so it stays bounded.
  Warm-compile cost is one-time (`cache=True`).
- Hilbert/Spiral are sequential O(S²) with `S` the next pow2 ≥ max(h,w) at preview
  scale — fine; no parallelism needed.

## Out of scope / non-goals

- No new UI sliders, categories, or per-kernel custom panels.
- No changes to existing kernels or the promoter.
- Native multi-tone (`supports_levels`) is deferred — the binary→levels promoter
  already covers palette/depth for all 8.
