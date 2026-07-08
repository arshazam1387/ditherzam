# Color System Core (Depth Ramp) — Design

**Date:** 2026-07-07
**Status:** Approved (brainstorming) → ready for implementation plan
**Sub-project:** A of 3 (A = engine core, B = palette editing UX, C = palette library)

## Goal

Bring ditherzam's color system to parity with Dither Boy 6.0's signature look. The
reverse-engineered mechanic: **color is inserted *with* the dither** — the image is
dithered on **luminosity** into N tonal levels, then a palette is mapped across those
levels as a **tone ramp**. A single **depth** control (1–64) sets how many tonal
levels/shades are used, independent of how many colors the palette holds. Mismatching
depth against the palette (or choosing a non-luminance mapping) produces the intentional
**color glitch** aesthetic.

Today ditherzam's kernels output only **2 tones** (binary threshold) and the color engine
maps that grayscale to the nearest palette color. There is nothing for a "depth" control
to bite on, and no palette-as-ramp behaviour. This sub-project adds both.

### In scope (A)
- Multi-level (N-level) tone quantization in the dither stage.
- `depth` control (1–64) and `mapping` (ramp style) knob.
- New `color/ramp.py` producing a depth-length RGB ramp from a palette.
- New `ramp` mode in `ColorEngine`.
- Settings, cache-signature, preset round-trip, and minimal UI (Depth slider + Mapping
  dropdown) wiring.

### Out of scope (deferred to B/C, YAGNI here)
- In-app swatch editing, per-swatch lock/shuffle surfaced in UI (`Palette.shuffle`
  already exists at the model layer).
- Palette library: categories, hover/scroll preview, import/share.
- CMYK halftone (separate deferred spec item §17.3).
- Animating `phase` (the ramp exposes a `phase` arg now, but no timeline/keyframe wiring
  in A).

## Invariants (must hold)

- **Clean-room** — no Studio/Dither-Boy code, strings, or binaries; this is our own
  implementation of a publicly-described behaviour.
- **Qt-free core** — `color/ramp.py`, `color/engine.py`, and the kernels stay Qt-free;
  only `ui/` imports PySide6.
- **Frozen `RenderPipeline.render()` stage order** unchanged:
  `contrast, midtones, highlights, blur, dither, color, saturation, effects, invert`.
- **Golden fixtures byte-identical** for all existing kernels (see §3).
- Python 3.12; TDD per task; tests run `NUMBA_DISABLE_JIT=1` and Qt tests
  `QT_QPA_PLATFORM=offscreen`. Full suite must stay green (currently 375) in **both**
  JIT-on and JIT-off modes.

## Architecture

### Data flow

`RenderPipeline` stage order is unchanged. Two existing stages become richer:

1. **Dither stage** (`dithering/pipeline.apply_dither`): the generalized error-diffusion
   and ordered kernels quantize luminance into `depth` evenly-spaced levels instead of 2.
   Output remains a grayscale float32 (0..255) but now carries up to `depth` distinct
   dithered tones.
2. **Color stage** (`ColorEngine.map`, `ramp` mode): the grayscale is binned into `depth`
   levels and each level is recolored via a precomputed `depth`-length palette ramp.

The **same `depth`** value feeds both the kernel `levels` and the ramp length, so a
`match` mapping at `depth == K` gives the clean authentic gradient, while any other
mapping — or `depth != K` — yields the glitch. There is no separate "mismatch" toggle;
mismatch is a property of the chosen `mapping` and the depth/palette relationship.

### Component 1 — `ditherzam/color/ramp.py` (new)

Pure, Numba-free (arrays are tiny: `depth ≤ 64`).

```
build_ramp(palette: Palette, depth: int, mapping: str, phase: float = 0.0) -> np.ndarray
    # returns float32[depth, 3] in 0..255
```

`depth` is clamped to `[1, 64]`. `phase` in `[0, 1]` cyclically rotates the ramp entries
(no-op default; reserved for animation in a later cycle). Luminance uses Rec.601 weights
`0.299/0.587/0.114`, consistent with `adjustments.apply_saturation`.

Mapping modes:

| mode | behaviour |
|---|---|
| `match` | palette sorted ascending by luminance, then **nearest-swatch** sampled to `depth` entries (`ramp[i] = sorted[round(i/(depth-1)*(K-1))]`). Hard palette colors, no blending — authentic retro posterization. |
| `interpolated` | palette sorted ascending by luminance, then **linearly interpolated** between anchors across `depth` entries. Smooth gradient (blended colors), unlike `match`'s hard steps. |
| `glitch` | raw palette order (as stored), `ramp[i] = palette[i % K]`. No luminance sort → signature color jumps when `depth != K`. |
| `reverse` | `match`, then reversed — dark tones get light colors. |
| `hue_cycle` | palette-independent: `ramp[i]` = HSV hue swept `phase..phase+1` across `depth`, full S/V. Rainbow. |
| `banded` | sawtooth repeat of the luminance-sorted palette every `K` levels → hard glitch stripes. |

Single source of truth for the mode list: a module-level tuple `RAMP_MODES` that the UI
dropdown and validation read.

### Component 2 — Kernel generalization

Add `levels: int = 2` to the **error-diffusion** (`kernels/error_diffusion.py`) and
**ordered** (`kernels/ordered.py`) families only.

Dispatch rule (guarantees byte-identity):
- `levels <= 2` → call the **existing njit function unchanged**. Current outputs — at any
  `luminance_threshold` value — are preserved exactly.
- `levels >= 3` → call a **new** N-level njit path.

N-level quantization helper (njit): `quantize_to_levels(v, L)` maps a value in 0..255 to
the nearest of `L` evenly-spaced levels:
`round(clip(v,0,255)/255*(L-1)) / (L-1) * 255`.
- Error diffusion: `new = quantize_to_levels(old, levels)`; diffuse residual `old - new`
  with the kernel's existing weights; final pass re-quantizes.
- Ordered: add the Bayer/screen threshold offset (recentered to `[-0.5,0.5]*step`) then
  `quantize_to_levels`.

In N-level mode the `luminance_threshold` slider acts as a **global tone bias**: the value
is subtracted (as `tval - 127.5`) from the input before quantization, so the slider still
shifts overall brightness of the banding. Documented behaviour, not a bug.

Plumbing:
- `registry` entry gains `supports_levels: bool` (default `False`); the error-diffusion
  and ordered registrations set it `True`.
- `apply_dither(..., levels: int = 2)` receives depth-as-levels from the pipeline. For a
  level-capable entry it calls `entry.func(small, param, tval, levels)`; otherwise the
  existing `entry.func(small, param, tval)`. Level-capable registered funcs gain a 4th
  parameter `levels=2`.
- Kernels in the **pattern**, **glitch**, and **special** families are unchanged and stay
  2-tone; with them the depth control only affects the color ramp length, not the dither.

### Component 3 — `ColorEngine` + settings + UI

- `ColorEngine(palette, mode="ramp", depth=2, mapping="match", phase=0.0)`. New `ramp`
  branch in `map()`:
  1. `ramp = build_ramp(self.palette, depth, mapping, phase)` (cached on the engine keyed
     by `(palette signature, depth, mapping, phase)` to avoid rebuilding every frame).
  2. Bin the incoming grayscale: `level = clip(round(gray/255*(depth-1)), 0, depth-1)`
     (for `depth == 1`, all pixels → level 0).
  3. `out = ramp[level]`; `clamp_u8`.
  Existing `off/nearest/ordered/diffused` modes untouched.
- `RenderSettings`: add `depth: int = 2` and `color_mapping: str = "match"`. `depth` is
  forwarded to `apply_dither` as `levels` and to the color engine.
- Cache signatures: extend `_color_sig` to include `(depth, mapping, phase)`; extend the
  dither `dith_sig` to include `levels`. `render_cached` must stay byte-identical to
  `render`.
- UI (`ui/controls.py`, `ui/settings_map.py`): one **Depth** slider (1–64) and a
  **Mapping** dropdown populated from `RAMP_MODES`. Preset save/load
  (`presets.py`) round-trips `depth` and `color_mapping`.

## Testing (TDD)

Every task writes tests first. All tests pass in both JIT modes; Qt tests offscreen.

- **ramp.py**: for each mode — output shape `[depth,3]`, dtype float32, values in 0..255;
  `match`/`interpolated` luminance-monotonic non-decreasing; `reverse` = `match[::-1]`;
  `glitch` deterministic and equals `palette[i % K]`; `depth==1` → single row;
  `depth` clamped to `[1,64]`.
- **Kernel equivalence**: `levels=2` output byte-identical to the committed golden
  fixtures for representative error-diffusion and ordered kernels, tested across several
  `luminance_threshold` values.
- **N-level kernels**: distinct tone count `<= depth`; error-diffusion output mean ≈ input
  mean (within tolerance); deterministic.
- **ColorEngine ramp**: pipeline byte-stable across repeated renders; `render_cached`
  byte-identical to `render` with depth/mapping set; a 2-color palette at `depth=2`,
  `mapping=match` reproduces the expected two-tone recolor.
- **Settings/preset**: `depth` and `color_mapping` survive a preset save→load round-trip;
  `settings_map` maps UI state to `RenderSettings`.
- **Full suite** stays green (≥375).

## Risks / gotchas

- **Threshold semantics in N-level mode** — the `luminance_threshold` slider changes
  meaning (binary threshold → tone bias). Called out in UI help/spec; not applied to the
  `levels<=2` path, so no regression.
- **Cache-signature omissions** — forgetting `depth`/`mapping` in a signature would show
  stale frames. Covered by the `render_cached == render` test.
- **Ramp rebuild cost** — mitigated by caching the ramp on the engine keyed by its inputs.
- **`match` sorting of near-equal-luminance swatches** — use a stable sort so output is
  deterministic across runs and platforms.

## Build order note

A ships first; B (editing UX) and C (library) are later cycles and depend on A existing.
