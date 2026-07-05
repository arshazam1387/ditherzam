# Phase 8 — Animation & Temporal — Completion Task List

Deterministic per-frame temporal noise (9 seeded patterns), a backward-compatible
threshold-field hook, a keyframe timeline with easing, a `render_animation`
generator, and a Qt timeline/playback/MP4-export UI — all TDD, red → green →
commit, core Qt-free.

## Prereqs (must be green first)

- **Phase 1** — `ditherzam/dithering/{registry,pipeline}.py`, `imaging.py`,
  `adjustments.py`, the shared `registry`, and kernels `Floyd-Steinberg`,
  `Atkinson`, `Bayer-Matrix 4x4`.
- **Phase 3** — `apply_saturation`, `ditherzam/color/{palette,engine}.py`
  (only needed so `RenderPipeline` composes; palette path is optional at run time).
- **Phase 4** — `ditherzam/render.py` (`RenderSettings`, `RenderPipeline`) and
  `ditherzam/effects/stack.py`.
- **Phase 5** — `ditherzam/ui/` package exists and the app boots offscreen
  (needed only for Task 8.5).
- **Phase 7** — `ditherzam/video/ffmpeg.py::assemble_video` (needed only for the
  MP4-export path in Task 8.5; UI smoke tests skip if absent).

**Env:** Python **3.12**. If no 3.12 is on `PATH`, use the pinned interpreter at
`…/scratchpad/dbwork/py312/python.exe` and prefix `pytest` invocations with it.
All test runs below use `NUMBA_DISABLE_JIT=1` (Git Bash). None of Tasks 8.1–8.4
import Qt; only Task 8.5 (`ditherzam/ui/…`) may import PySide6.

**FROZEN CONTRACTS honored verbatim in this phase:**

```python
# ditherzam/animation/temporal.py
PATTERNS  # exactly 9
def temporal_noise(frame, shape, pattern, amplitude, seed=0) -> 'float32 HxW ~[-amp,amp]'
# ditherzam/animation/timeline.py
def ease(t, kind); class Keyframe(frame, field, value); class Timeline(length, add, value_at, settings_at)
# ditherzam/animation/__init__.py
def render_animation(pipeline, base_gray_f32, base_settings, timeline,
                     temporal_pattern, temporal_amplitude, seed=0) -> Iterator[uint8 HxWx3]
# threshold-field integration (backward compatible when None):
def apply_dither(..., threshold_field=None)
def RenderPipeline.render(self, base_gray_f32, settings, temporal_field=None)
```

The 9 pattern names (retro-display inspired), in order:
`"static"`, `"scanline-drift"`, `"interlace"`, `"rolling-bar"`, `"vhs-jitter"`,
`"blue-noise"`, `"bayer-cycle"`, `"plasma"`, `"film-grain"`.

---

### Task 8.0: Package scaffold (env sanity + empty package)

**Files:**
- Create: `ditherzam/animation/__init__.py` (empty package marker for now — filled in Task 8.4)

**Interfaces:** Produces the importable `ditherzam.animation` package.

- [ ] **Step 1: Confirm the interpreter is 3.12**

Run: `python -c "import sys; print(sys.version_info[:2])"`
Expected: `(3, 12)`. If not, substitute the pinned 3.12 interpreter for every
`python`/`pytest` call below and note it here.

- [ ] **Step 2: Create the empty package marker — `ditherzam/animation/__init__.py`**

```python
"""ditherzam.animation — temporal noise, keyframe timeline, animated rendering."""
```

- [ ] **Step 3: Verify it imports**

Run: `python -c "import ditherzam.animation; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add ditherzam/animation/__init__.py
git commit -m "feat(anim): animation package scaffold"
```

---

### Task 8.1: Nine deterministic temporal noise patterns

**Files:**
- Create: `ditherzam/animation/temporal.py`
- Test: `tests/test_temporal.py`

**Interfaces:**
- Produces:
  ```python
  PATTERNS: tuple[str, ...]   # exactly 9 names, verbatim order above
  def temporal_noise(frame: int, shape: tuple[int, int], pattern: str,
                     amplitude: float, seed: int = 0) -> np.ndarray  # float32 HxW, ~[-amp, amp]
  ```
- Determinism: identical `(frame, seed)` → byte-identical array. Stochastic patterns
  (`static`, `vhs-jitter`, `blue-noise`, `film-grain`) seed a fresh
  `np.random.default_rng(seed * 1_000_003 + frame)`; the rest are analytic in
  `frame`. Every pattern's base output is in `[-1, 1]`, so the returned field is in
  `[-amplitude, amplitude]`.

- [ ] **Step 1: Write failing test — `tests/test_temporal.py`**

```python
import numpy as np
from ditherzam.animation.temporal import PATTERNS, temporal_noise


def test_exactly_nine_patterns():
    assert len(PATTERNS) == 9


def test_pattern_names_exact():
    assert PATTERNS == (
        "static", "scanline-drift", "interlace", "rolling-bar", "vhs-jitter",
        "blue-noise", "bayer-cycle", "plasma", "film-grain",
    )


def test_shape_dtype_and_determinism():
    a = temporal_noise(3, (8, 8), "static", 10.0, seed=0)
    b = temporal_noise(3, (8, 8), "static", 10.0, seed=0)
    assert a.shape == (8, 8) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)                       # deterministic


def test_all_patterns_deterministic_and_bounded():
    for p in PATTERNS:
        a = temporal_noise(4, (12, 10), p, 7.0, seed=3)
        b = temporal_noise(4, (12, 10), p, 7.0, seed=3)
        assert a.shape == (12, 10) and a.dtype == np.float32
        np.testing.assert_array_equal(a, b)                   # per-pattern determinism
        assert a.max() <= 7.0 + 1e-3, p                        # amplitude upper bound
        assert a.min() >= -7.0 - 1e-3, p                       # amplitude lower bound


def test_frames_differ_over_time():
    a = temporal_noise(0, (8, 8), "vhs-jitter", 10.0)
    b = temporal_noise(1, (8, 8), "vhs-jitter", 10.0)
    assert not np.array_equal(a, b)                            # animates


def test_analytic_patterns_animate():
    for p in ("scanline-drift", "interlace", "rolling-bar", "bayer-cycle", "plasma"):
        a = temporal_noise(0, (16, 16), p, 4.0)
        b = temporal_noise(3, (16, 16), p, 4.0)
        assert not np.array_equal(a, b), p


def test_different_seed_differs():
    a = temporal_noise(2, (8, 8), "film-grain", 5.0, seed=0)
    b = temporal_noise(2, (8, 8), "film-grain", 5.0, seed=1)
    assert not np.array_equal(a, b)


def test_unknown_pattern_raises():
    import pytest
    with pytest.raises(KeyError):
        temporal_noise(0, (4, 4), "nope", 1.0)
```

- [ ] **Step 2: Run — verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_temporal.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ditherzam.animation.temporal'`)

- [ ] **Step 3: Implement `ditherzam/animation/temporal.py`**

```python
from __future__ import annotations

import numpy as np

PATTERNS: tuple[str, ...] = (
    "static",
    "scanline-drift",
    "interlace",
    "rolling-bar",
    "vhs-jitter",
    "blue-noise",
    "bayer-cycle",
    "plasma",
    "film-grain",
)

_TWO_PI = 2.0 * np.pi


def _rng(frame: int, seed: int) -> np.random.Generator:
    """Deterministic generator keyed on (frame, seed)."""
    return np.random.default_rng(int(seed) * 1_000_003 + int(frame))


def _bayer4() -> np.ndarray:
    b2 = np.array([[0, 2], [3, 1]], dtype=np.float32)
    return np.block([
        [4 * b2 + 0, 4 * b2 + 2],
        [4 * b2 + 3, 4 * b2 + 1],
    ]).astype(np.float32)  # values 0..15


def _static(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # TV "snow": full-field white noise, re-rolled every frame. -> [-1, 1)
    u = _rng(frame, seed).random((h, w), dtype=np.float32)
    return u * 2.0 - 1.0


def _scanline_drift(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Horizontal scanlines whose phase drifts vertically over time. -> [-1, 1]
    y = np.arange(h, dtype=np.float32)[:, None]
    row = np.sin(_TWO_PI * (y / 8.0 + frame * 0.05))
    return np.broadcast_to(row, (h, w)).astype(np.float32)


def _interlace(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Alternating even/odd lines that flip parity each frame. -> {-1, +1}
    y = np.arange(h, dtype=np.int64)[:, None]
    val = ((y + frame) % 2) * 2 - 1
    return np.broadcast_to(val.astype(np.float32), (h, w)).astype(np.float32)


def _rolling_bar(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # A bright horizontal hum-bar rolling vertically (rolling-shutter look). -> [0, 1]
    y = np.arange(h, dtype=np.float32)[:, None]
    center = (frame * 2.0) % max(h, 1)
    d = np.abs(y - center)
    d = np.minimum(d, h - d)                       # wrap vertically
    sigma = max(h / 8.0, 1.0)
    prof = np.exp(-(d * d) / (2.0 * sigma * sigma))
    return np.broadcast_to(prof, (h, w)).astype(np.float32)


def _vhs_jitter(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Per-row horizontal jitter offset, re-rolled each frame. -> [-1, 1)
    row = _rng(frame, seed).random(h, dtype=np.float32) * 2.0 - 1.0
    return np.broadcast_to(row[:, None], (h, w)).astype(np.float32)


def _blue_noise(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # High-frequency noise: white noise minus its 4-neighbour blur, then
    # normalised to exactly [-1, 1]. -> [-1, 1]
    n = _rng(frame, seed).random((h, w), dtype=np.float32)
    blur = 0.25 * (np.roll(n, 1, 0) + np.roll(n, -1, 0)
                   + np.roll(n, 1, 1) + np.roll(n, -1, 1))
    hp = n - blur
    m = float(np.max(np.abs(hp)))
    if m <= 0.0:
        return np.zeros((h, w), dtype=np.float32)
    return (hp / m).astype(np.float32)


def _bayer_cycle(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # 4x4 Bayer threshold field, phase-rolled each frame, tiled to HxW. -> ~[-0.94, 0.94]
    b = _bayer4()
    shift = frame % 4
    b = np.roll(np.roll(b, shift, axis=0), shift, axis=1)
    norm = (b + 0.5) / 16.0 * 2.0 - 1.0
    tiled = np.tile(norm, (h // 4 + 1, w // 4 + 1))[:h, :w]
    return tiled.astype(np.float32)


def _plasma(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Classic plasma: mean of four moving sines. -> [-1, 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = frame * 0.1
    p = (np.sin(xx / 6.0 + t)
         + np.sin(yy / 8.0 - t)
         + np.sin((xx + yy) / 10.0 + t)
         + np.sin(np.sqrt(xx * xx + yy * yy) / 7.0 + t))
    return (p / 4.0).astype(np.float32)


def _film_grain(frame: int, h: int, w: int, seed: int) -> np.ndarray:
    # Gaussian grain clipped to +/-3 sigma, scaled to [-1, 1]. -> [-1, 1]
    g = _rng(frame, seed).standard_normal((h, w)).astype(np.float32)
    return np.clip(g, -3.0, 3.0) / 3.0


_DISPATCH = {
    "static": _static,
    "scanline-drift": _scanline_drift,
    "interlace": _interlace,
    "rolling-bar": _rolling_bar,
    "vhs-jitter": _vhs_jitter,
    "blue-noise": _blue_noise,
    "bayer-cycle": _bayer_cycle,
    "plasma": _plasma,
    "film-grain": _film_grain,
}


def temporal_noise(frame: int, shape: tuple[int, int], pattern: str,
                   amplitude: float, seed: int = 0) -> np.ndarray:
    """Deterministic per-frame noise field, float32 HxW in ~[-amplitude, amplitude]."""
    if pattern not in _DISPATCH:
        raise KeyError(f"Unknown temporal pattern: {pattern!r}")
    h, w = int(shape[0]), int(shape[1])
    base = _DISPATCH[pattern](int(frame), h, w, int(seed))
    return (base * float(amplitude)).astype(np.float32)
```

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_temporal.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/animation/temporal.py tests/test_temporal.py
git commit -m "feat(anim): 9 deterministic temporal noise patterns"
```

---

### Task 8.2: Threshold-field hook (backward-compatible when None)

**Files:**
- Modify: `ditherzam/dithering/pipeline.py`
- Modify: `ditherzam/render.py`
- Test: `tests/test_threshold_field.py`

**Interfaces:**
- `apply_dither(..., threshold_field: np.ndarray | None = None)`. When a field is
  given, per-pixel threshold becomes `tval + threshold_field`. Because
  `pixel >= tval + field  ⟺  (pixel - field) >= tval`, we implement it generically
  by **subtracting the field from the downscaled image before the kernel runs** — no
  kernel signature changes, and it works for both ordered and error-diffusion
  kernels. The field is resized (float-safe nearest) to the downscaled grid.
- `RenderPipeline.render(self, base_gray_f32, settings, temporal_field=None)`
  forwards `temporal_field` to `apply_dither`.
- **Backward compatibility:** when `threshold_field`/`temporal_field` is `None`, the
  downscaled image is untouched and output is **byte-identical** to the Phase-1 path.

- [ ] **Step 1: Write failing test — `tests/test_threshold_field.py`**

```python
import numpy as np
from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.animation.temporal import temporal_noise


def _ramp():
    return np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))


def _flat():
    return np.full((16, 16), 128.0, dtype=np.float32)


def test_apply_dither_none_field_is_byte_identical():
    img = _ramp()
    base = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                        luminance_threshold=50, params={}, registry=registry)
    with_none = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                             luminance_threshold=50, params={}, registry=registry,
                             threshold_field=None)
    np.testing.assert_array_equal(base, with_none)             # backward compatible


def test_field_changes_dither_output():
    img = _flat()
    fld = temporal_noise(0, (16, 16), "static", 80.0, seed=1)
    base = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                        luminance_threshold=50, params={}, registry=registry)
    perturbed = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                             luminance_threshold=50, params={}, registry=registry,
                             threshold_field=fld)
    assert not np.array_equal(base, perturbed)


def test_field_resized_to_downscaled_grid():
    # scale=4 -> 16/4 = 4x4 small grid; pass a full-res 16x16 field, must not error.
    img = _ramp()
    fld = temporal_noise(0, (16, 16), "plasma", 40.0)
    out = apply_dither(img, style="Bayer-Matrix 4x4", scale=4,
                       luminance_threshold=50, params={}, registry=registry,
                       threshold_field=fld)
    assert out.shape == img.shape


def test_render_none_matches_no_field():
    p = RenderPipeline(registry)
    img = _ramp()
    s = RenderSettings(style="Bayer-Matrix 4x4", scale=1)
    a = p.render(img, s)
    b = p.render(img, s, temporal_field=None)
    np.testing.assert_array_equal(a, b)                        # backward compatible


def test_render_field_changes_output():
    p = RenderPipeline(registry)
    img = _flat()
    s = RenderSettings(style="Bayer-Matrix 4x4", scale=1)
    fld = temporal_noise(0, (16, 16), "static", 80.0, seed=1)
    a = p.render(img, s)
    b = p.render(img, s, temporal_field=fld)
    assert not np.array_equal(a, b)
```

- [ ] **Step 2: Run — verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_threshold_field.py -v`
Expected: FAIL (`TypeError: apply_dither() got an unexpected keyword argument 'threshold_field'`)

- [ ] **Step 3a: Implement — replace `ditherzam/dithering/pipeline.py` in full**

```python
from __future__ import annotations

import numpy as np

from ..imaging import nearest_downscale, nearest_upscale_to


def _luminance_to_255(luminance_threshold: float) -> float:
    return float(luminance_threshold / 100.0 * 255.0)


def _build_param(entry, params: dict):
    vals = [params[name] for name in entry.param_sliders if name in params]
    if len(vals) <= 1:
        return vals[0] if vals else 0
    return tuple(vals)


def _resize_field_nearest(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Float-safe nearest-neighbour resize that preserves negative values."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    fh, fw = field.shape[:2]
    if (fh, fw) == (th, tw):
        return field
    ys = np.minimum((np.arange(th) * fh) // max(th, 1), fh - 1)
    xs = np.minimum((np.arange(tw) * fw) // max(tw, 1), fw - 1)
    return field[ys][:, xs]


def apply_dither(gray_f32, *, style, scale, luminance_threshold,
                 params, registry, preview_disabled=False,
                 threshold_field=None) -> np.ndarray:
    entry = registry.get_entry(style)
    if style == "None" or entry is None or preview_disabled:
        return gray_f32

    tval = _luminance_to_255(luminance_threshold)
    factor = max(1, int(scale))
    h, w = gray_f32.shape[:2]

    small = nearest_downscale(gray_f32, factor)

    if threshold_field is not None:
        fld = _resize_field_nearest(
            np.asarray(threshold_field, dtype=np.float32), small.shape[:2])
        # per-pixel threshold tval + fld  <=>  compare (small - fld) against tval
        small = (small - fld).astype(np.float32)

    param = _build_param(entry, params)
    out = entry.func(small, param, tval)

    return nearest_upscale_to(out, (w, h))
```

- [ ] **Step 3b: Implement — replace `ditherzam/render.py` in full (adds `temporal_field` threading)**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .imaging import clamp_u8
from .adjustments import (
    apply_contrast, apply_midtones, apply_highlights, apply_blur,
    apply_invert, apply_saturation,
)
from .dithering.pipeline import apply_dither


@dataclass
class RenderSettings:
    contrast: float = 50
    midtones: float = 50
    highlights: float = 50
    blur: float = 50
    luminance_threshold: float = 50
    invert: bool = False
    saturation: float = 50
    style: str = "None"
    scale: int = 5
    preview_disabled: bool = False
    params: dict = field(default_factory=dict)


class RenderPipeline:
    def __init__(self, registry, color_engine=None, effect_stack=None) -> None:
        self.registry = registry
        self.color_engine = color_engine
        self.effect_stack = effect_stack

    def render(self, base_gray_f32, settings, temporal_field=None) -> np.ndarray:
        # 1. tonal adjustments on grayscale float32
        g = base_gray_f32.astype(np.float32)
        g = apply_contrast(g, settings.contrast)
        g = apply_midtones(g, settings.midtones)
        g = apply_highlights(g, settings.highlights)
        g = apply_blur(g, settings.blur)

        # 2. dither (downscale -> kernel -> upscale), optional temporal threshold field
        dithered = apply_dither(
            g, style=settings.style, scale=settings.scale,
            luminance_threshold=settings.luminance_threshold,
            params=settings.params, registry=self.registry,
            preview_disabled=settings.preview_disabled,
            threshold_field=temporal_field,
        )

        # 3. color: palette map, else broadcast grayscale to RGB
        if self.color_engine is not None:
            rgb = self.color_engine.map(dithered).astype(np.float32)
        else:
            rgb = np.repeat(dithered[:, :, None], 3, axis=2).astype(np.float32)

        # 4. saturation then clamp to u8
        rgb = apply_saturation(rgb, settings.saturation)
        rgb_u8 = clamp_u8(rgb)

        # 5. effects stack
        if self.effect_stack is not None:
            rgb_u8 = self.effect_stack.apply(rgb_u8)

        # 6. invert last
        if settings.invert:
            rgb_u8 = clamp_u8(apply_invert(rgb_u8.astype(np.float32), True))

        return rgb_u8
```

> Note: the only Phase-8 change to `render.py` is the `temporal_field=None`
> parameter and its forwarding into `apply_dither`. Everything else is the frozen
> Phase-4 composition (order: contrast → midtones → highlights → blur → dither →
> color → saturation → effects → invert). If your Phase-4 `render.py` already
> differs (e.g. extra fields), apply only the `temporal_field` addition rather than
> pasting this file wholesale.

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_threshold_field.py -v`
Expected: `5 passed`

Also re-run the Phase-1/Phase-4 suites to prove nothing regressed:
Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_pipeline.py tests/test_render.py -v`
Expected: all previously-green tests still pass.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/pipeline.py ditherzam/render.py tests/test_threshold_field.py
git commit -m "feat(anim): backward-compatible temporal threshold-field hook"
```

---

### Task 8.3: Keyframe timeline + easing

**Files:**
- Create: `ditherzam/animation/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
```python
def ease(t: float, kind: str) -> float          # "linear"|"ease-in"|"ease-out"|"ease-in-out"
@dataclass
class Keyframe:
    frame: int; field: str; value: float; kind: str = "linear"   # kind is optional, default linear
class Timeline:
    length: int
    def add(self, kf: Keyframe) -> None
    def value_at(self, field: str, frame: int) -> float           # per-segment eased interpolation
    def settings_at(self, base: RenderSettings, frame: int) -> RenderSettings
```
Interpolation is piecewise between the two surrounding keyframes of the same field;
the normalized segment position `t` is passed through `ease(t, kind)` where `kind`
is taken from the **ending** keyframe of the segment. Frames outside the keyframe
range clamp to the nearest endpoint. `settings_at` clones `base` via
`dataclasses.replace`, rounding integer fields (`scale`).

- [ ] **Step 1: Write failing test — `tests/test_timeline.py`**

```python
import numpy as np
from ditherzam.animation.timeline import Timeline, Keyframe, ease
from ditherzam.render import RenderSettings


def test_ease_endpoints_exact():
    for k in ("linear", "ease-in", "ease-out", "ease-in-out"):
        assert abs(ease(0.0, k) - 0.0) < 1e-9
        assert abs(ease(1.0, k) - 1.0) < 1e-9


def test_ease_shapes():
    assert abs(ease(0.5, "linear") - 0.5) < 1e-9
    assert ease(0.5, "ease-in") < 0.5           # slow start
    assert ease(0.5, "ease-out") > 0.5          # fast start
    assert abs(ease(0.5, "ease-in-out") - 0.5) < 1e-9


def test_ease_clamps_out_of_range():
    assert ease(-1.0, "linear") == 0.0
    assert ease(2.0, "linear") == 1.0


def test_ease_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        ease(0.5, "bounce")


def test_linear_interpolation_midpoint():
    t = Timeline(length=10)
    t.add(Keyframe(0, "contrast", 0))
    t.add(Keyframe(10, "contrast", 100))
    assert abs(t.value_at("contrast", 5) - 50) < 1e-6


def test_value_clamps_outside_range():
    t = Timeline(length=10)
    t.add(Keyframe(2, "contrast", 30))
    t.add(Keyframe(8, "contrast", 90))
    assert t.value_at("contrast", 0) == 30       # before first key
    assert t.value_at("contrast", 20) == 90      # after last key


def test_eased_segment_uses_end_keyframe_kind():
    t = Timeline(length=10)
    t.add(Keyframe(0, "contrast", 0))
    t.add(Keyframe(10, "contrast", 100, kind="ease-in"))
    # ease-in at t=0.5 -> 0.25 -> value 25
    assert abs(t.value_at("contrast", 5) - 25) < 1e-6


def test_add_replaces_same_frame_same_field():
    t = Timeline(length=10)
    t.add(Keyframe(5, "scale", 3))
    t.add(Keyframe(5, "scale", 9))               # overwrite
    assert t.value_at("scale", 5) == 9


def test_settings_at_applies_and_rounds_int_fields():
    t = Timeline(length=10)
    t.add(Keyframe(0, "scale", 2))
    t.add(Keyframe(10, "scale", 12))
    s = t.settings_at(RenderSettings(), 5)
    assert s.scale == 7 and isinstance(s.scale, int)


def test_settings_at_leaves_unkeyed_fields_alone():
    t = Timeline(length=10)
    t.add(Keyframe(0, "contrast", 10))
    t.add(Keyframe(10, "contrast", 90))
    base = RenderSettings(style="Floyd-Steinberg", saturation=42)
    s = t.settings_at(base, 5)
    assert s.style == "Floyd-Steinberg" and s.saturation == 42
    assert abs(s.contrast - 50) < 1e-6
```

- [ ] **Step 2: Run — verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_timeline.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ditherzam.animation.timeline'`)

- [ ] **Step 3: Implement `ditherzam/animation/timeline.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, replace

from ..render import RenderSettings

_INT_FIELDS = frozenset({"scale"})


def ease(t: float, kind: str) -> float:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
    if kind == "linear":
        return t
    if kind == "ease-in":
        return t * t
    if kind == "ease-out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if kind == "ease-in-out":
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0
    raise ValueError(f"Unknown easing kind: {kind!r}")


@dataclass
class Keyframe:
    frame: int
    field: str
    value: float
    kind: str = "linear"


class Timeline:
    def __init__(self, length: int) -> None:
        self.length = int(length)
        self._keys: dict[str, list[Keyframe]] = {}

    def add(self, kf: Keyframe) -> None:
        lst = self._keys.setdefault(kf.field, [])
        for i, existing in enumerate(lst):
            if existing.frame == kf.frame:
                lst[i] = kf
                break
        else:
            lst.append(kf)
        lst.sort(key=lambda k: k.frame)

    def fields(self) -> list[str]:
        return list(self._keys.keys())

    def value_at(self, field: str, frame: int) -> float:
        keys = self._keys.get(field)
        if not keys:
            raise KeyError(field)
        if frame <= keys[0].frame:
            return float(keys[0].value)
        if frame >= keys[-1].frame:
            return float(keys[-1].value)
        for i in range(len(keys) - 1):
            a, b = keys[i], keys[i + 1]
            if a.frame <= frame <= b.frame:
                span = b.frame - a.frame
                if span == 0:
                    return float(b.value)
                t = (frame - a.frame) / span
                e = ease(t, b.kind)
                return float(a.value + (b.value - a.value) * e)
        return float(keys[-1].value)

    def settings_at(self, base: RenderSettings, frame: int) -> RenderSettings:
        updates: dict = {}
        for fld in self._keys:
            v = self.value_at(fld, frame)
            if fld in _INT_FIELDS:
                v = int(round(v))
            updates[fld] = v
        return replace(base, **updates)
```

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_timeline.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/animation/timeline.py tests/test_timeline.py
git commit -m "feat(anim): keyframe timeline with easing"
```

---

### Task 8.4: `render_animation` generator

**Files:**
- Modify: `ditherzam/animation/__init__.py`
- Test: `tests/test_render_animation.py`

**Interfaces:**
```python
def render_animation(pipeline, base_gray_f32, base_settings, timeline,
                     temporal_pattern, temporal_amplitude, seed=0) -> Iterator[np.ndarray]
# yields exactly timeline.length frames of uint8[H, W, 3]
```
For each frame `f`: `settings = timeline.settings_at(base_settings, f)`; compute the
downscaled grid `small_shape` for `settings.scale`; if a pattern is active
(`temporal_pattern not in (None, "", "none")` and `temporal_amplitude > 0`) build
`field = temporal_noise(f, small_shape, pattern, amplitude, seed)` else `field=None`;
`yield pipeline.render(base_gray_f32, settings, temporal_field=field)`. The field is
generated at the downscaled grid size so it aligns 1:1 with the kernel input.

- [ ] **Step 1: Write failing test — `tests/test_render_animation.py`**

```python
import numpy as np
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.animation import render_animation
from ditherzam.animation.timeline import Timeline, Keyframe


def _flat():
    return np.full((16, 16), 128.0, dtype=np.float32)


def test_yields_length_rgb_frames():
    p = RenderPipeline(registry)
    tl = Timeline(length=5)
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Bayer-Matrix 4x4", scale=1),
        tl, "static", 90.0, seed=2))
    assert len(frames) == 5
    for fr in frames:
        assert fr.shape == (16, 16, 3) and fr.dtype == np.uint8


def test_temporal_motion_visible():
    p = RenderPipeline(registry)
    tl = Timeline(length=3)
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Bayer-Matrix 4x4", scale=1),
        tl, "vhs-jitter", 90.0, seed=0))
    assert not np.array_equal(frames[0], frames[1])


def test_no_pattern_is_static_across_frames():
    p = RenderPipeline(registry)
    tl = Timeline(length=3)                     # no keyframes, no temporal pattern
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Bayer-Matrix 4x4", scale=1),
        tl, "none", 0.0))
    np.testing.assert_array_equal(frames[0], frames[2])


def test_timeline_animates_settings():
    p = RenderPipeline(registry)
    tl = Timeline(length=5)
    tl.add(Keyframe(0, "luminance_threshold", 10))
    tl.add(Keyframe(4, "luminance_threshold", 90))
    frames = list(render_animation(
        p, _flat(), RenderSettings(style="Floyd-Steinberg", scale=1),
        tl, "none", 0.0))
    assert not np.array_equal(frames[0], frames[-1])
```

- [ ] **Step 2: Run — verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_render_animation.py -v`
Expected: FAIL (`ImportError: cannot import name 'render_animation' from 'ditherzam.animation'`)

- [ ] **Step 3: Implement — replace `ditherzam/animation/__init__.py`**

```python
"""ditherzam.animation — temporal noise, keyframe timeline, animated rendering."""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Iterator, Optional

import numpy as np

from .temporal import PATTERNS, temporal_noise
from .timeline import Keyframe, Timeline, ease

__all__ = [
    "PATTERNS", "temporal_noise", "ease", "Keyframe", "Timeline",
    "render_animation", "export_animation",
]

_INACTIVE_PATTERNS = {None, "", "none", "None", "off"}


def render_animation(pipeline, base_gray_f32, base_settings, timeline,
                     temporal_pattern, temporal_amplitude, seed=0) -> Iterator[np.ndarray]:
    """Yield timeline.length rendered uint8[H, W, 3] frames."""
    h, w = base_gray_f32.shape[:2]
    active = (temporal_pattern not in _INACTIVE_PATTERNS
              and float(temporal_amplitude) > 0.0)
    for f in range(timeline.length):
        settings = timeline.settings_at(base_settings, f)
        factor = max(1, int(settings.scale))
        small_shape = (max(1, h // factor), max(1, w // factor))
        field = None
        if active:
            field = temporal_noise(f, small_shape, temporal_pattern,
                                   float(temporal_amplitude), int(seed))
        yield pipeline.render(base_gray_f32, settings, temporal_field=field)


def export_animation(pipeline, base_gray_f32, base_settings, timeline,
                     temporal_pattern, temporal_amplitude, out_path,
                     fps: int = 24, seed: int = 0,
                     progress: Optional[Callable[[int, int], None]] = None) -> str:
    """Render every frame to PNG, then encode to MP4 via Phase-7's assemble_video.

    Qt-free. Raises ImportError only if the Phase-7 video module is absent.
    """
    from PIL import Image
    from ..video.ffmpeg import assemble_video

    frames_dir = tempfile.mkdtemp(prefix="ditherzam_anim_")
    n = int(timeline.length)
    for i, frame in enumerate(render_animation(
            pipeline, base_gray_f32, base_settings, timeline,
            temporal_pattern, temporal_amplitude, seed)):
        Image.fromarray(frame).save(os.path.join(frames_dir, f"frame{i:06d}.png"))
        if progress is not None:
            progress(i + 1, n)
    assemble_video(frames_dir, fps, None, out_path)
    return out_path
```

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_render_animation.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/animation/__init__.py tests/test_render_animation.py
git commit -m "feat(anim): render_animation frame generator + export_animation"
```

---

### Task 8.5: UI timeline panel, playback, MP4 export (Qt)

**Files:**
- Create: `ditherzam/ui/timeline_panel.py`  (only file in this phase that imports PySide6)
- Modify: `ditherzam/ui/main_window.py`  (dock the panel + wire signals)
- Test: `tests/test_anim_ui_smoke.py`

**Interfaces:**
- `TimelinePanel(QWidget)` — a pattern picker (`"none"` + the 9 `PATTERNS`), an
  amplitude slider, a length spinbox, a frame scrubber, a play/pause toggle
  (`QTimer`), an "Add Keyframe" button, and an "Export Animation (MP4)" button.
  Signals: `keyframe_requested(int)`, `frame_changed(int)`, `export_requested()`,
  `play_toggled(bool)`. Helpers: `pattern() -> str`, `amplitude() -> float`,
  `length() -> int`.
- `AnimationController` — wires a `TimelinePanel` to a `RenderPipeline`; on each
  `frame_changed` it renders that frame (applying `timeline.settings_at` + the
  active temporal field) and pushes it to an `on_frame` sink; `export(out_path, fps)`
  delegates to `animation.export_animation`.

- [ ] **Step 1: Write failing smoke test — `tests/test_anim_ui_smoke.py`**

```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from ditherzam.animation.temporal import PATTERNS
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.animation.timeline import Timeline
from ditherzam.ui.timeline_panel import TimelinePanel, AnimationController

_app = QApplication.instance() or QApplication([])


def test_panel_lists_none_plus_nine_patterns():
    p = TimelinePanel(length=10)
    items = [p.pattern_combo.itemText(i) for i in range(p.pattern_combo.count())]
    assert items[0] == "none"
    assert tuple(items[1:]) == PATTERNS
    assert len(items) == 10


def test_panel_signals_exist():
    p = TimelinePanel()
    for sig in ("keyframe_requested", "frame_changed", "export_requested", "play_toggled"):
        assert hasattr(p, sig)


def test_length_updates_frame_slider_range():
    p = TimelinePanel(length=10)
    p.length_spin.setValue(20)
    assert p.frame_slider.maximum() == 19


def test_play_toggle_drives_timer():
    p = TimelinePanel(length=5)
    p.play_btn.setChecked(True)
    assert p._timer.isActive()
    p.play_btn.setChecked(False)
    assert not p._timer.isActive()


def test_controller_renders_frame_to_sink():
    p = TimelinePanel(length=5)
    p.pattern_combo.setCurrentText("static")
    p.amp_slider.setValue(60)
    pipeline = RenderPipeline(registry)
    tl = Timeline(length=5)
    base = np.full((16, 16), 128.0, np.float32)
    ctrl = AnimationController(
        p, pipeline, lambda: (base, RenderSettings(style="Bayer-Matrix 4x4", scale=1)),
        tl, seed=0)
    captured = {}
    ctrl.on_frame = lambda img: captured.setdefault("img", img)
    out = ctrl.render_frame(2)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert captured["img"].shape == (16, 16, 3)
```

- [ ] **Step 2: Run — verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_anim_ui_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ditherzam.ui.timeline_panel'`)

- [ ] **Step 3a: Implement `ditherzam/ui/timeline_panel.py`**

```python
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..animation.temporal import PATTERNS, temporal_noise


class TimelinePanel(QWidget):
    """Temporal pattern picker, keyframe editor trigger, scrubber and playback."""

    keyframe_requested = Signal(int)   # current frame index
    frame_changed = Signal(int)        # scrub / playback frame index
    export_requested = Signal()
    play_toggled = Signal(bool)

    def __init__(self, length: int = 30, parent=None) -> None:
        super().__init__(parent)

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItem("none")
        self.pattern_combo.addItems(list(PATTERNS))

        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(20)

        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 3600)
        self.length_spin.setValue(length)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, length - 1))

        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.key_btn = QPushButton("Add Keyframe")
        self.export_btn = QPushButton("Export Animation (MP4)")

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // 24)   # ~24 fps preview

        self._build_layout()
        self._connect()

    def _build_layout(self) -> None:
        top = QHBoxLayout()
        top.addWidget(QLabel("Pattern"))
        top.addWidget(self.pattern_combo)
        top.addWidget(QLabel("Amplitude"))
        top.addWidget(self.amp_slider)
        top.addWidget(QLabel("Frames"))
        top.addWidget(self.length_spin)

        mid = QHBoxLayout()
        mid.addWidget(self.play_btn)
        mid.addWidget(self.frame_slider)

        bottom = QHBoxLayout()
        bottom.addWidget(self.key_btn)
        bottom.addWidget(self.export_btn)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid)
        root.addLayout(bottom)

    def _connect(self) -> None:
        self.length_spin.valueChanged.connect(self._on_length)
        self.frame_slider.valueChanged.connect(self.frame_changed.emit)
        self.play_btn.toggled.connect(self._on_play)
        self.key_btn.clicked.connect(
            lambda: self.keyframe_requested.emit(self.frame_slider.value()))
        self.export_btn.clicked.connect(self.export_requested.emit)
        self._timer.timeout.connect(self._advance)

    def _on_length(self, n: int) -> None:
        self.frame_slider.setRange(0, max(0, int(n) - 1))

    def _on_play(self, on: bool) -> None:
        self.play_btn.setText("Pause" if on else "Play")
        if on:
            self._timer.start()
        else:
            self._timer.stop()
        self.play_toggled.emit(on)

    def _advance(self) -> None:
        n = max(1, self.length_spin.value())
        self.frame_slider.setValue((self.frame_slider.value() + 1) % n)

    # --- read-only accessors for controllers ---
    def pattern(self) -> str:
        return self.pattern_combo.currentText()

    def amplitude(self) -> float:
        return float(self.amp_slider.value())

    def length(self) -> int:
        return int(self.length_spin.value())


class AnimationController:
    """Bridges a TimelinePanel to a RenderPipeline and an image sink."""

    _INACTIVE = {"", "none", "None", "off"}

    def __init__(self, panel: TimelinePanel, pipeline, provide_base, timeline,
                 seed: int = 0) -> None:
        self.panel = panel
        self.pipeline = pipeline
        self.provide_base = provide_base    # () -> (gray_f32, RenderSettings) | None
        self.timeline = timeline
        self.seed = int(seed)
        self.on_frame = None                # callable(np.uint8 HxWx3) | None
        panel.frame_changed.connect(self.render_frame)
        panel.export_requested.connect(self._on_export)

    def render_frame(self, frame_index: int) -> "np.ndarray | None":
        base = self.provide_base()
        if base is None:
            return None
        gray, settings = base
        settings = self.timeline.settings_at(settings, int(frame_index))
        h, w = gray.shape[:2]
        factor = max(1, int(settings.scale))
        small_shape = (max(1, h // factor), max(1, w // factor))
        pattern = self.panel.pattern()
        amp = self.panel.amplitude()
        field = None
        if pattern not in self._INACTIVE and amp > 0.0:
            field = temporal_noise(int(frame_index), small_shape, pattern, amp, self.seed)
        img = self.pipeline.render(gray, settings, temporal_field=field)
        if self.on_frame is not None:
            self.on_frame(img)
        return img

    def export(self, out_path: str, fps: int = 24) -> "str | None":
        from ..animation import export_animation
        base = self.provide_base()
        if base is None:
            return None
        gray, settings = base
        return export_animation(
            self.pipeline, gray, settings, self.timeline,
            self.panel.pattern(), self.panel.amplitude(),
            out_path, fps=fps, seed=self.seed)

    def _on_export(self) -> None:
        # Actual file dialog + worker is wired by MainWindow; this default is a no-op
        # hook so the panel's export button is always connected to a live slot.
        pass
```

- [ ] **Step 3b: Wire into `ditherzam/ui/main_window.py`** (add these fragments to the existing `MainWindow`; do not rewrite the whole file)

```python
# --- imports (top of main_window.py) ---
from PySide6.QtWidgets import QDockWidget, QFileDialog
from PySide6.QtCore import Qt
from ..animation.timeline import Timeline, Keyframe
from .timeline_panel import TimelinePanel, AnimationController

# --- inside MainWindow.__init__, after the viewport/controls are built ---
self.timeline = Timeline(length=30)
self.timeline_panel = TimelinePanel(length=30, parent=self)
dock = QDockWidget("Animation", self)
dock.setWidget(self.timeline_panel)
self.addDockWidget(Qt.BottomDockWidgetArea, dock)

self.anim_controller = AnimationController(
    self.timeline_panel, self.render_pipeline,
    self._provide_animation_base, self.timeline, seed=0)
self.anim_controller.on_frame = self._show_animation_frame
self.timeline_panel.keyframe_requested.connect(self._add_keyframe_at)
self.timeline_panel.export_requested.connect(self._export_animation)

# --- new MainWindow methods ---
def _provide_animation_base(self):
    if self.base_gray_f32 is None:          # no image loaded yet
        return None
    return self.base_gray_f32, self.current_settings()

def _show_animation_frame(self, rgb_u8):
    # push the rendered frame into the same viewport used for stills
    self.viewport.set_image_rgb(rgb_u8)

def _add_keyframe_at(self, frame_index: int):
    # snapshot the currently-focused control into a keyframe
    s = self.current_settings()
    self.timeline.add(Keyframe(frame_index, "luminance_threshold", s.luminance_threshold))

def _export_animation(self):
    out, _ = QFileDialog.getSaveFileName(self, "Export Animation", "animation.mp4",
                                         "MP4 Video (*.mp4)")
    if not out:
        return
    self.anim_controller.export(out, fps=self.timeline_panel.length() // 1 or 24)
```

> `current_settings()`, `render_pipeline`, `base_gray_f32`, and
> `viewport.set_image_rgb(...)` are Phase-5 members — reuse whatever your Phase-5
> `MainWindow` already exposes; adjust these names to match. For production, run
> `export()` inside a `QRunnable` with a progress dialog (see Phase-7 workers)
> rather than on the GUI thread.

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 QT_QPA_PLATFORM=offscreen pytest tests/test_anim_ui_smoke.py -v`
Expected: `5 passed` (or `skipped` if PySide6 is unavailable in the environment)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/timeline_panel.py ditherzam/ui/main_window.py tests/test_anim_ui_smoke.py
git commit -m "feat(anim): timeline UI, playback, MP4 export wiring"
```

---

## Subsystem Definition of Done (checklist)

- [ ] `PATTERNS` has **exactly 9** names, verbatim and in order:
  `static, scanline-drift, interlace, rolling-bar, vhs-jitter, blue-noise, bayer-cycle, plasma, film-grain`.
- [ ] `temporal_noise(frame, shape, pattern, amplitude, seed=0)` returns float32 `HxW`,
  deterministic per `(frame, seed)` for **every** pattern, and bounded to `[-amp, amp]`.
- [ ] Analytic patterns animate across frames; stochastic patterns differ across
  frames and across seeds; unknown pattern raises `KeyError`.
- [ ] `apply_dither(..., threshold_field=None)` is **byte-identical** to Phase 1 when
  the field is `None`; a non-`None` field perturbs the per-pixel threshold and is
  auto-resized to the downscaled grid.
- [ ] `RenderPipeline.render(..., temporal_field=None)` forwards the field and is
  backward-compatible when `None`.
- [ ] `ease` endpoints exact for all four kinds; midpoint correct; out-of-range
  clamps; unknown kind raises `ValueError`.
- [ ] `Keyframe(frame, field, value)` signature honored (optional `kind` default
  `"linear"`); `Timeline.add/value_at/settings_at` interpolate per-segment with
  easing and round integer fields.
- [ ] `render_animation` yields exactly `timeline.length` `uint8[H,W,3]` frames;
  temporal motion and keyframe animation both visible.
- [ ] MP4 export reuses Phase-7 `assemble_video`; **core stays Qt-free** — Qt appears
  only in `ditherzam/ui/timeline_panel.py` and the `main_window.py` wiring.
- [ ] `NUMBA_DISABLE_JIT=1 pytest tests/test_temporal.py tests/test_threshold_field.py
  tests/test_timeline.py tests/test_render_animation.py` all green.

## Self-Review

**Spec coverage (§17.5 Animation & temporal):**

| Spec item | Implemented by |
|---|---|
| 9 controllable animated noise patterns | Task 8.1 (`PATTERNS`, `temporal_noise`) |
| Per-frame varying noise perturbs the dither | Task 8.2 (`threshold_field` / `temporal_field`) |
| Animation timeline sequencing fields over time | Task 8.3 (`Timeline`, `Keyframe`) |
| Easing: ease-in / ease-out / linear (+ ease-in-out) | Task 8.3 (`ease`) |
| Live preview of the animation | Task 8.5 (`TimelinePanel` playback `QTimer` + `AnimationController`) |
| Export to MP4 with the animation pipeline | Task 8.4 `export_animation` + Task 8.5 export action (reuses Phase-7 `assemble_video`) |

**FROZEN-CONTRACT / type consistency check:**

- `PATTERNS` tuple of 9 str — matches contract.
- `temporal_noise(frame, shape, pattern, amplitude, seed=0) -> float32 HxW` — matches.
- `ease(t, kind)`, `Keyframe(frame, field, value)`,
  `Timeline(length, add, value_at, settings_at)` — names/signatures verbatim
  (`Keyframe.kind` is an *optional* extra with a default; required positional args
  unchanged).
- `apply_dither(..., threshold_field=None)` and
  `RenderPipeline.render(self, base_gray_f32, settings, temporal_field=None)` — added
  keyword-only/default params; all existing calls remain valid.
- `render_animation(pipeline, base_gray_f32, base_settings, timeline, temporal_pattern,
  temporal_amplitude, seed=0)` — matches the roadmap generator signature.

**Placeholder scan:** no `TODO`, no `pass`-only bodies except the deliberate
`AnimationController._on_export` hook (documented; real export runs via
`AnimationController.export`). Every code step is complete and runnable.

**Clean-room / Qt-free check:** no proprietary strings, URLs, or binaries; algorithms
(TV static, scanlines, interlace, rolling bar, VHS jitter, blue noise, Bayer, plasma,
film grain) are public-domain techniques. Only Task 8.5 imports PySide6.
