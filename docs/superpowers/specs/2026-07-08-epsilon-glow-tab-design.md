# Epsilon Glow — Dedicated, Fully-Customizable Tab (Design Spec)

**Date:** 2026-07-08
**Status:** Approved, ready for planning
**Sub-project:** Epsilon Glow (post-effect upgrade + dedicated UI tab)

## Goal

Turn the existing 2-parameter `epsilon_glow` placeholder into a full-fidelity,
threshold-driven bloom tuned for dithered art, and give it its own dedicated,
fully-customizable **Glow** tab in the UI. Reverse-engineered from the public
Dither Boy "Epsilon Glow" tutorial's *described behaviour* — our own algorithm,
no Studio AAA source.

## Clean-room note

The name "Epsilon Glow" already exists in the tree (`_EFFECTS`, `effects/post.py`,
`_EFFECT_DEFAULTS`); we reuse it, we do not introduce new Studio AAA branding or
copy any Dither Boy code/strings/binaries. The algorithm below is derived only
from the publicly-described behaviour of a luminance-threshold glow. See
constraint [[001-clean-room]].

## Constraints (invariants)

- **Qt-free core.** `ditherzam/effects/**` must NOT import PySide6. Any UI→param
  mapping logic must be a pure function with its own unit test. See [[002-qt-free-core]].
- **Python 3.12; TDD per task** (red → green → refactor, commit each green). Suite
  green under `NUMBA_DISABLE_JIT=1`. Runner: `./.venv/Scripts/python.exe -m pytest`.
- **No new deps.** Available: numpy, numba, Pillow, PyYAML, platformdirs, PySide6.
  Anisotropy is achieved via non-uniform resize→isotropic-blur→resize (no scipy).
- **`RenderPipeline` stage order and `RenderSettings` fields are FROZEN.** Glow is a
  post-effect in the existing `EffectStack`; do not touch `render.py`.

## Part 1 — Headless core (`ditherzam/effects/post.py`)

Replace the placeholder with:

```python
def epsilon_glow(rgb_u8, threshold, smoothing, radius, intensity,
                 epsilon, falloff, distance_scale, aspect) -> np.ndarray
```

All params are floats (except where noted); returns uint8 RGB, same shape.

### Algorithm

1. **Weighted luminance** (Rec. 601): `lum = 0.299*R + 0.587*G + 0.114*B`, range
   0..255. Reproduces the tutorial's "green builds luminance fastest, then red,
   then blue."
2. **Soft-knee threshold mask**: `mask = smoothstep(threshold, threshold + smoothing, lum)`
   in [0,1]. Higher `threshold` ⇒ fewer qualifying pixels. `smoothing`=0 ⇒ hard
   cutoff; larger `smoothing` ⇒ glow eases in gently. When `smoothing`==0 use a
   plain `lum >= threshold` step to avoid a divide-by-zero.
3. **Emissive source**: `src = base_rgb.astype(float32) * mask[..., None]`. Only
   qualifying pixels emit light.
4. **Radial spread**: sum of a small fixed number of PIL Gaussian blurs of `src`
   at increasing sigmas derived from `r = radius * distance_scale`, each weighted
   by a `falloff` curve (higher `falloff` ⇒ weight concentrates on the tight
   blurs ⇒ tighter/more intense; lower ⇒ light travels further). **Aspect**:
   before blurring, resize `src` non-uniformly (stretch one axis by the aspect
   factor), blur isotropically, resize back — yields an anisotropic (directional)
   glow at the source with no scipy.
5. **Epsilon hot core**: `core = base_rgb * (mask ** k)` where `k` grows with
   `epsilon`; added at a high weight. Produces the "brilliant hot centre" and, at
   low `distance_scale`, the star/ray look.
6. **Additive combine**: `out = clip(base + glow*intensity + core, 0, 255).astype(uint8)`.

### Identity / edge behaviour

- `intensity == 0` ⇒ output byte-identical to input (glow contributes nothing;
  core is gated behind intensity too, or returned early).
- `radius <= 0` and `intensity <= 0` ⇒ early-return the input unchanged.
- Grayscale (R==G==B) input still works: luminance == the gray value.

### `_EFFECT_DEFAULTS`

Update `main_window.py`'s `_EFFECT_DEFAULTS["Epsilon Glow"]` to the new full param
set with sensible, visible defaults.

## Part 2 — Glow tab (`ditherzam/ui/glow_panel.py` + `QTabWidget`)

- Introduce a `QTabWidget` in the right pane of `ImageEditor`, wrapping the
  existing `ControlPanel` (tab **"Editor"**) and a new `GlowPanel` (tab **"Glow"**).
  The `QScrollArea` currently wrapping the panel wraps the tab widget instead (or
  each tab scrolls); existing `self.panel` reference is preserved.
- `GlowPanel(QWidget)`:
  - An **Enable Glow** `QCheckBox`.
  - Eight `ResettableGlowSlider` + `InvisibleSpinBox` rows: **Threshold,
    Smoothing, Radius, Intensity, Epsilon, Falloff, Distance Scale, Aspect**.
  - A **Reset all** button.
  - Holds its own `self.state` dict and emits a `changed` Signal on any edit
    (mirrors `ControlPanel` conventions).
- **Pure helper** (Qt-free, own unit test), e.g. `glow_params_from_state(state) -> dict`
  living in a Qt-free module (`ditherzam/effects/glow_params.py` or similar),
  mapping slider integer ranges to the core function's float params. `GlowPanel`
  calls it; the mapping is unit-tested without a GUI.

### Slider ranges → params (indicative; finalise in plan)

| Control        | Slider range | Maps to                |
|----------------|--------------|------------------------|
| Threshold      | 0..255       | `threshold` (float)    |
| Smoothing      | 0..128       | `smoothing` (float)    |
| Radius         | 0..200       | `radius` (float)       |
| Intensity      | 0..100       | `intensity` = v/100 * scale |
| Epsilon        | 0..100       | `epsilon` (0..1)       |
| Falloff        | 0..100       | `falloff` (0..1)       |
| Distance Scale | 1..100       | `distance_scale` (0.x..N) |
| Aspect         | 0..100       | `aspect` (0.25..4, 50=1:1) |

## Part 3 — Wiring (`main_window.py`)

- `_collect_effect_stack` (the code around `main_window.py:261-267`) appends
  `Epsilon Glow` with the Glow tab's live params **only when Enable Glow is on**.
- **Single source of truth**: remove `"Epsilon Glow"` from the name-only
  `_EFFECTS` combo list in `controls.py` so there are not two conflicting control
  paths for the same effect.
- Params round-trip through presets automatically because the effect stack already
  serializes `(name, params)`; a preset test confirms the glow params survive
  save→load and re-apply to the Glow tab state.

## Part 4 — Chromatic-effects interaction

The tutorial's "move CA before the glow so channel shifts blend new hues that the
threshold then picks up" is **emergent** from stack order — CA already runs before
Glow when ordered that way in the `EffectStack`. No new code; a test verifies the
effect (CA-then-Glow differs from Glow-then-CA, and both run without error).

## Testing (TDD, JIT-off)

- **Core** (`tests/test_effects_post.py` additions):
  - `intensity == 0` ⇒ identity.
  - Threshold monotonicity: higher `threshold` ⇒ fewer non-zero-glow pixels.
  - `smoothing == 0` hard-cut path has no NaNs / no divide error.
  - `epsilon` up ⇒ brighter core at qualifying pixels.
  - Grayscale input path works.
  - Output dtype uint8, shape preserved, values clipped 0..255.
- **Pure helper**: `glow_params_from_state` maps every slider to the expected float.
- **UI**: `GlowPanel` builds, has the 8 sliders + enable + reset, `changed` fires on
  edit, reset restores defaults (Qt widget test, offscreen).
- **Tab**: `ImageEditor` right pane has a `QTabWidget` with Editor + Glow tabs.
- **Preset round-trip**: glow params survive save→load.

## Out of scope (YAGNI)

- Generic per-effect parameter UI for the other effects (Blur/Sharpen/CA/JPEG stay
  name-only for now).
- Animating glow params over the timeline.
- GPU acceleration of the blur.
