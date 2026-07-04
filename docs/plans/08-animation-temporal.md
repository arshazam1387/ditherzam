# Phase 8 — Animation & Temporal — Implementation Plan

> REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Read
> [`00-ROADMAP.md`](00-ROADMAP.md); complete Phases 1–5 (7 recommended for MP4 export).

**Goal:** The 6.0 animation feature set (spec §17.5): **9 controllable temporal
noise patterns** that vary per frame, and a **keyframe timeline** with easing
(ease-in / ease-out / linear) that animates any `RenderSettings` field, with a
live animated preview and MP4 export.

**Architecture:** `animation/temporal.py` produces a deterministic per-frame noise
field `noise(frame_index, shape, pattern, seed)` used to perturb the dither
threshold. `animation/timeline.py` holds keyframes `(frame, field, value)` and
interpolates each field per frame with an easing function. A `render_animation`
generator yields one rendered `uint8[H,W,3]` per frame; the UI plays it and Phase 7
encodes it to MP4.

**Tech Stack:** NumPy · pytest · (Phase 5 UI, Phase 7 encode).

## Global Constraints
See roadmap. Temporal noise is **deterministic** given `(frame_index, seed)` so
renders are reproducible and testable. Qt-free except the preview player.

---

## File structure (this phase)

- Create `ditherzam/animation/__init__.py`, `temporal.py`, `timeline.py`
- Modify `ditherzam/render.py` — accept an optional `temporal_field` to perturb threshold
- Tests: `tests/test_temporal.py`, `tests/test_timeline.py`, `tests/test_render_animation.py`

---

### Task 1: Temporal noise patterns (9)

**Files:** Create `ditherzam/animation/temporal.py`; Test `tests/test_temporal.py`
**Interfaces:**
```python
PATTERNS: tuple[str, ...]   # exactly 9 names
def temporal_noise(frame: int, shape: tuple[int,int], pattern: str,
                   amplitude: float, seed: int = 0) -> np.ndarray  # float32 HxW, ~[-amp,amp]
```

Nine patterns (retro-display inspired): `"static"`, `"scanline-drift"`,
`"interlace"`, `"rolling-bar"`, `"vhs-jitter"`, `"blue-noise"`, `"bayer-cycle"`,
`"plasma"`, `"film-grain"`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.animation.temporal import PATTERNS, temporal_noise

def test_nine_patterns():
    assert len(PATTERNS) == 9

def test_shape_and_determinism():
    a = temporal_noise(3, (8,8), "static", 10.0, seed=0)
    b = temporal_noise(3, (8,8), "static", 10.0, seed=0)
    assert a.shape == (8,8) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)                     # deterministic

def test_frames_differ_over_time():
    a = temporal_noise(0, (8,8), "vhs-jitter", 10.0)
    b = temporal_noise(1, (8,8), "vhs-jitter", 10.0)
    assert not np.array_equal(a, b)                          # animates

def test_amplitude_bounds():
    a = temporal_noise(2, (16,16), "film-grain", 5.0)
    assert a.max() <= 5.0 + 1e-3 and a.min() >= -5.0 - 1e-3
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement** each pattern as a deterministic function of `frame`+`seed`
  (`np.random.default_rng(seed*10007 + frame)` for stochastic ones; analytic
  sine/phase for `rolling-bar`, `interlace`, `plasma`, `scanline-drift`,
  `bayer-cycle`). Scale each to `[-amplitude, amplitude]`.
- [ ] **Step 4: Pass. Step 5: Commit** `feat(anim): 9 deterministic temporal noise patterns`

---

### Task 2: Threshold perturbation hook in the pipeline

**Files:** Modify `ditherzam/dithering/pipeline.py`, `ditherzam/render.py`;
Test `tests/test_render_animation.py` (part A)
**Interfaces:** `apply_dither(..., threshold_field: np.ndarray | None = None)` —
when given, the per-pixel threshold becomes `tval + threshold_field` (resized to the
downscaled grid). `RenderPipeline.render(..., temporal_field=None)` forwards it.

- [ ] **Step 1: Failing test** — two renders of the same input with different
  `temporal_field`s produce different dithered output; with `None` it matches the
  Phase 1 result exactly (backward compatible).
- [ ] **Step 2–3:** thread `threshold_field` through: resize field to small grid, add to
  `tval` inside kernels that accept a per-pixel threshold (add a
  `_ordered_field`/`_diffuse_field` variant, or add the field to the image before
  thresholding for diffusion kernels). Keep default path identical when `None`.
- [ ] **Step 4: pass. Step 5: commit** `feat(anim): temporal threshold perturbation in pipeline`

---

### Task 3: Keyframe timeline + easing

**Files:** Create `ditherzam/animation/timeline.py`; Test `tests/test_timeline.py`
**Interfaces:**
```python
def ease(t: float, kind: str) -> float          # "linear"|"ease-in"|"ease-out"|"ease-in-out"
@dataclass
class Keyframe: frame: int; field: str; value: float
class Timeline:
    length: int
    def add(self, kf: Keyframe) -> None
    def value_at(self, field: str, frame: int) -> float   # interpolated with per-segment easing
    def settings_at(self, base: RenderSettings, frame: int) -> RenderSettings
```

- [ ] **Step 1: Failing test**

```python
from ditherzam.animation.timeline import Timeline, Keyframe, ease
from ditherzam.render import RenderSettings

def test_ease_endpoints():
    for k in ("linear","ease-in","ease-out","ease-in-out"):
        assert abs(ease(0,k)-0) < 1e-9 and abs(ease(1,k)-1) < 1e-9

def test_linear_interpolation_midpoint():
    t = Timeline(length=10)
    t.add(Keyframe(0, "contrast", 0)); t.add(Keyframe(10, "contrast", 100))
    assert abs(t.value_at("contrast", 5) - 50) < 1e-6

def test_settings_at_applies_field():
    t = Timeline(length=10)
    t.add(Keyframe(0,"scale",2)); t.add(Keyframe(10,"scale",12))
    s = t.settings_at(RenderSettings(), 5)
    assert abs(s.scale - 7) < 1.0
```

- [ ] **Step 2–3:** implement piecewise interpolation between surrounding keyframes of
  the same field, applying `ease` to the normalized segment position; `settings_at`
  clones `base` and sets each animated field. Round integer fields (`scale`).
- [ ] **Step 4: pass. Step 5: commit** `feat(anim): keyframe timeline with easing`

---

### Task 4: `render_animation` generator

**Files:** Modify `ditherzam/render.py` (or `animation/__init__.py`);
Test `tests/test_render_animation.py` (part B)
**Interfaces:**
```python
def render_animation(pipeline, base_gray_f32, base_settings, timeline,
                     temporal_pattern, temporal_amplitude, seed=0) -> Iterator[np.ndarray]
# yields timeline.length frames of uint8[H,W,3]
```

- [ ] **Step 1: Failing test** — a 5-frame timeline yields 5 arrays, each
  `uint8[H,W,3]`; at least two consecutive frames differ when a temporal pattern is active.
- [ ] **Step 2–3:** implement generator: for each frame, `settings = timeline.settings_at(base, f)`,
  `field = temporal_noise(f, small_shape, pattern, amplitude, seed)`,
  `yield pipeline.render(base_gray_f32, settings, temporal_field=field)`.
- [ ] **Step 4: pass. Step 5: commit** `feat(anim): render_animation frame generator`

---

### Task 5: UI timeline + playback + MP4 export

**Files:** Modify `ditherzam/ui/main_window.py` (+ a small timeline widget);
Modify `ditherzam/video/ffmpeg.py` usage; (smoke) `tests/test_anim_ui_smoke.py`
- [ ] Add a temporal-pattern picker (9 options + amplitude slider), a minimal
  keyframe editor (add keyframe at current frame for the focused control), a
  play/pause `QTimer` that steps `render_animation` into the viewport, and an
  "Export Animation (MP4)" action that writes frames via `render_animation` then
  encodes with Phase 7's `assemble_video`. Smoke-test the actions exist. Commit
  `feat(anim): timeline UI, playback, MP4 export`.

---

## Phase 8 Self-Review
- [ ] Exactly 9 temporal patterns; each deterministic per `(frame, seed)` and animates over frames.
- [ ] `temporal_field=None` keeps Phase 1 output byte-identical (backward compatible).
- [ ] Easing endpoints exact; linear midpoint correct; `settings_at` animates fields.
- [ ] `render_animation` yields `length` RGB frames; temporal motion visible.
- [ ] MP4 export path reuses Phase 7 encoder; core Qt-free.

---

## Project completion gate (after all 8 phases)
- [ ] `pytest` green across all phases (`NUMBA_DISABLE_JIT=1` for speed).
- [ ] App launches, loads an image, applies a **color** dither with **stacked effects**, exports **PNG/SVG**.
- [ ] Video import → per-frame dither → MP4 with audio works (ffmpeg present).
- [ ] Animated temporal render exports to MP4.
- [ ] `len(registry.list_dithers()) >= 63`; color engine, effects, presets, batch all covered by tests.
- [ ] No proprietary code/strings/binaries anywhere in the tree.
