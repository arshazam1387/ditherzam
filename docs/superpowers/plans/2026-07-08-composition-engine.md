# Composition Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless, Qt-free `ditherzam/composition/` package that renders a timeline of look-clips with transitions/morphs between them into RGB frames and exports them to video, composing on top of the existing `RenderPipeline`.

**Architecture:** A `Look` wraps a preset dict and builds its own `RenderSettings`/`ColorEngine`/`EffectStack`. A `Composition` is a contiguous track of `LookClip`s with `TransitionSpec`s at boundaries; `resolve(idx)` says whether a frame is a plain hold or an A→B transition at parameter `t`. `Transition` strategies (registry, like the kernel registry) combine two rendered looks; the `Compositor` owns a per-look `RenderPipeline` cache, renders each frame, and drives whole-composition render/export via the existing ffmpeg helpers.

**Tech Stack:** Python 3.12, numpy, PIL, pyyaml. Reuses `ditherzam.presets`, `ditherzam.color`, `ditherzam.effects`, `ditherzam.render`, `ditherzam.animation`, `ditherzam.video.ffmpeg`.

## Global Constraints

- **Clean-room.** Our own code only; Dither Boy is behaviour inspiration, never source. No Studio AAA code/strings/binaries.
- **Qt-free core.** `ditherzam/composition/**` must NEVER import PySide6 (same rule as `color/**`, `dithering/**`). Only `ui/`, `app.py`, `video/workers.py` import Qt.
- **`RenderPipeline.render()` stage order and `RenderSettings` fields are FROZEN.** Do not modify `render.py`. Add NO fields to `RenderSettings`. Compose on top only.
- **Python 3.12; TDD per task.** Every task: failing test → see it fail → minimal code → see it pass → commit.
- **Suite green in JIT-off mode.** Tests run under `NUMBA_DISABLE_JIT=1` (set by `tests/conftest.py`). Runner: `./.venv/Scripts/python.exe -m pytest`. The 7 kernel tests that fail JIT-**on** (`special.py` float-index) are pre-existing and NOT this feature's concern.
- **Real names.** Valid dither styles include `"None"`, `"Floyd-Steinberg"`, `"Atkinson"`, `"Bayer-Matrix 4x4"` (regular hyphens). The kernel registry singleton is `from ditherzam.dithering import registry`.

---

## File Structure

```
ditherzam/composition/
  __init__.py      # public exports (Task 8)
  look.py          # Look                                  (Task 1)
  clip.py          # LookClip, TransitionSpec, Composition, Resolution, resolve()  (Task 2)
  transitions.py   # _LookContext, Transition, register/TRANSITIONS,
                   #   Crossfade (Task 3), DitherDissolve + SpatialWipe (Task 4), ParamMorph (Task 5)
  compositor.py    # Compositor, render_composition, export_composition  (Tasks 6-7)
  serialize.py     # composition_to_dict / composition_from_dict         (Task 8)
tests/
  test_composition_look.py            (Task 1)
  test_composition_clip.py            (Task 2)
  test_composition_transitions.py     (Tasks 3-4)
  test_composition_morph.py           (Task 5)
  test_composition_compositor.py      (Task 6)
  test_composition_render.py          (Task 7)
  test_composition_serialize.py       (Task 8)
```

---

### Task 1: `Look` — full-pipeline config from a preset dict

**Files:**
- Create: `ditherzam/composition/__init__.py` (empty for now; real exports land in Task 8)
- Create: `ditherzam/composition/look.py`
- Test: `tests/test_composition_look.py`

**Interfaces:**
- Consumes: `ditherzam.presets.preset_to_settings(preset) -> (RenderSettings, Palette|None, list[(name, params)])`; `ditherzam.color.engine.ColorEngine(palette, mode)`; `ditherzam.effects.stack.EffectStack()` with `.add(name, **params)` and `.items`.
- Produces: `Look(name: str, preset: dict)` with `.name`, `.preset`, `.settings -> RenderSettings` (memoized), `.build_color_engine() -> ColorEngine|None`, `.build_effect_stack() -> EffectStack`, `.color_effect_key() -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition_look.py
import numpy as np
from ditherzam.composition.look import Look


def _preset(style="Floyd-Steinberg", mode="off"):
    p = {
        "adjustments": {"contrast": 60, "midtones": 50, "highlights": 40,
                        "luminance_threshold": 55, "blur": 0, "saturation": 50,
                        "invert": False},
        "dither": {"style": style, "scale": 4, "depth": 2,
                   "color_mapping": "match", "preview_disabled": False, "params": {}},
    }
    if mode != "off":
        p["color"] = {"mode": mode,
                      "palette": {"name": "duo", "category": "",
                                  "colors": [[0, 0, 0], [255, 255, 255]]}}
    return p


def test_look_settings_from_preset():
    look = Look("A", _preset(style="Atkinson"))
    s = look.settings
    assert s.style == "Atkinson"
    assert s.contrast == 60
    assert s.scale == 4
    # memoized: same object returned each call
    assert look.settings is s


def test_look_color_engine_off_is_none():
    assert Look("A", _preset(mode="off")).build_color_engine() is None


def test_look_color_engine_built_when_mode_set():
    eng = Look("A", _preset(mode="nearest")).build_color_engine()
    assert eng is not None
    assert eng.mode == "nearest"
    assert eng.palette.colors.shape == (2, 3)


def test_look_effect_stack_from_preset():
    p = _preset()
    p["effects"] = [{"name": "Blur", "params": {"radius": 3}}]
    stack = Look("A", p).build_effect_stack()
    assert stack.items == [("Blur", {"radius": 3})]


def test_color_effect_key_differs_on_color_change():
    a = Look("A", _preset(mode="off"))
    b = Look("B", _preset(mode="nearest"))
    assert a.color_effect_key() != b.color_effect_key()
    assert a.color_effect_key() == Look("A2", _preset(mode="off")).color_effect_key()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_look.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.composition'`

- [ ] **Step 3: Write minimal implementation**

Create `ditherzam/composition/__init__.py` as an empty file (one newline).

```python
# ditherzam/composition/look.py
from __future__ import annotations

from ..presets import preset_to_settings
from ..render import RenderSettings


class Look:
    """A named, complete pipeline configuration built from a preset dict.

    A Look IS a preset: any saved preset (adjustments + dither + optional color +
    effects) becomes a reusable look. Qt-free.
    """

    def __init__(self, name: str, preset: dict) -> None:
        self.name = str(name)
        self.preset = dict(preset or {})
        self._settings: RenderSettings | None = None
        self._palette = None
        self._effects: list[tuple[str, dict]] | None = None

    def _decode(self) -> None:
        if self._settings is None:
            self._settings, self._palette, self._effects = preset_to_settings(self.preset)

    @property
    def settings(self) -> RenderSettings:
        self._decode()
        return self._settings

    def build_color_engine(self):
        self._decode()
        mode = str((self.preset.get("color") or {}).get("mode", "off"))
        if mode == "off" or self._palette is None:
            return None
        from ..color.engine import ColorEngine
        return ColorEngine(self._palette, mode)

    def build_effect_stack(self):
        self._decode()
        from ..effects.stack import EffectStack
        stack = EffectStack()
        for name, params in self._effects:
            stack.add(name, **params)
        return stack

    def color_effect_key(self) -> str:
        """Opaque signature of the color + effects config (NOT the tonal/dither
        numerics). ParamMorph uses it to decide render-once vs crossfade."""
        return repr((self.preset.get("color"), self.preset.get("effects")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_look.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/composition/__init__.py ditherzam/composition/look.py tests/test_composition_look.py
git commit -m "feat(composition): Look — full pipeline config from a preset dict"
```

---

### Task 2: `LookClip`, `TransitionSpec`, `Composition`, `resolve()`

**Files:**
- Create: `ditherzam/composition/clip.py`
- Test: `tests/test_composition_clip.py`

**Interfaces:**
- Consumes: `Look` (Task 1); `ditherzam.animation.timeline.Timeline` with `.settings_at(base, frame) -> RenderSettings`.
- Produces:
  - `LookClip(look, start: int, end: int, automation: Timeline|None = None)` — half-open `[start, end)`; `.settings_at(frame) -> RenderSettings` (applies automation to `look.settings`, else returns `look.settings`).
  - `TransitionSpec(kind: str, duration: int, params: dict)`.
  - `Composition(length: int)` with `.clips: list[LookClip]`, `.transitions: dict[int, TransitionSpec]`, `.add_clip(clip)`, `.set_transition(later_clip_index, spec)`, `.validate()`, `.resolve(idx) -> Resolution`.
  - `Resolution` — a namedtuple `(kind, clip, clip_b, t, spec)`; `kind` is `"hold"` or `"transition"`. For hold: `clip` set, others None/0.0. For transition: `clip`=A, `clip_b`=B, `t` in `[0,1)`, `spec` set.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition_clip.py
import numpy as np
import pytest
from ditherzam.composition.look import Look
from ditherzam.composition.clip import LookClip, TransitionSpec, Composition
from ditherzam.animation.timeline import Timeline, Keyframe


def _preset(style="None"):
    return {
        "adjustments": {"contrast": 50, "midtones": 50, "highlights": 50,
                        "luminance_threshold": 50, "blur": 0, "saturation": 50,
                        "invert": False},
        "dither": {"style": style, "scale": 4, "depth": 2,
                   "color_mapping": "match", "preview_disabled": False, "params": {}},
    }


def _comp_two():
    comp = Composition(length=200)
    comp.add_clip(LookClip(Look("A", _preset("Floyd-Steinberg")), 0, 100))
    comp.add_clip(LookClip(Look("B", _preset("Atkinson")), 100, 200))
    return comp


def test_validate_rejects_gap():
    comp = Composition(length=200)
    comp.add_clip(LookClip(Look("A", _preset()), 0, 90))   # gap 90..100
    comp.add_clip(LookClip(Look("B", _preset()), 100, 200))
    with pytest.raises(ValueError):
        comp.validate()


def test_resolve_hold_inside_first_clip():
    comp = _comp_two()
    comp.validate()
    r = comp.resolve(40)
    assert r.kind == "hold"
    assert r.clip.look.name == "A"


def test_resolve_transition_window():
    comp = _comp_two()
    comp.set_transition(1, TransitionSpec("crossfade", duration=20, params={}))
    comp.validate()
    r0 = comp.resolve(100)          # first frame of clip B
    assert r0.kind == "transition"
    assert r0.clip.look.name == "A" and r0.clip_b.look.name == "B"
    assert r0.t == pytest.approx(0.0)
    r_mid = comp.resolve(110)
    assert r_mid.t == pytest.approx(0.5)
    r_after = comp.resolve(120)     # transition window ended (100+20)
    assert r_after.kind == "hold" and r_after.clip.look.name == "B"


def test_set_transition_duration_capped_by_validate():
    comp = _comp_two()
    comp.set_transition(1, TransitionSpec("crossfade", duration=500, params={}))
    with pytest.raises(ValueError):
        comp.validate()


def test_automation_overrides_settings():
    tl = Timeline(length=100)
    tl.add(Keyframe(0, "contrast", 0.0))
    tl.add(Keyframe(99, "contrast", 100.0))
    clip = LookClip(Look("A", _preset()), 0, 100, automation=tl)
    # automation is relative to the clip: frame within clip
    assert clip.settings_at(0).contrast == pytest.approx(0.0)
    assert clip.settings_at(99).contrast == pytest.approx(100.0)
    # no automation -> base settings
    plain = LookClip(Look("A", _preset()), 0, 100)
    assert plain.settings_at(50).contrast == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_clip.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.composition.clip'`

- [ ] **Step 3: Write minimal implementation**

```python
# ditherzam/composition/clip.py
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field

from ..render import RenderSettings
from .look import Look

Resolution = namedtuple("Resolution", ["kind", "clip", "clip_b", "t", "spec"])


@dataclass
class LookClip:
    look: Look
    start: int          # inclusive
    end: int            # exclusive
    automation: object = None   # optional animation.Timeline

    def settings_at(self, frame: int) -> RenderSettings:
        base = self.look.settings
        if self.automation is None:
            return base
        local = int(frame) - int(self.start)
        return self.automation.settings_at(base, local)


@dataclass
class TransitionSpec:
    kind: str
    duration: int
    params: dict = field(default_factory=dict)


class Composition:
    """Contiguous track of LookClips covering [0, length) with transitions at
    boundaries. A transition attached to clip index i occupies the first
    `duration` frames of clips[i], cross-blending clips[i-1] -> clips[i]."""

    def __init__(self, length: int) -> None:
        self.length = int(length)
        self.clips: list[LookClip] = []
        self.transitions: dict[int, TransitionSpec] = {}

    def add_clip(self, clip: LookClip) -> None:
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.start)

    def set_transition(self, later_clip_index: int, spec: TransitionSpec) -> None:
        self.transitions[int(later_clip_index)] = spec

    def validate(self) -> None:
        if not self.clips:
            raise ValueError("composition has no clips")
        if self.clips[0].start != 0:
            raise ValueError("first clip must start at 0")
        if self.clips[-1].end != self.length:
            raise ValueError("last clip must end at composition length")
        for c in self.clips:
            if c.end <= c.start:
                raise ValueError(f"clip end {c.end} <= start {c.start}")
        for a, b in zip(self.clips, self.clips[1:]):
            if a.end != b.start:
                raise ValueError(f"non-contiguous clips: {a.end} != {b.start}")
        for idx, spec in self.transitions.items():
            if idx <= 0 or idx >= len(self.clips):
                raise ValueError(f"transition index {idx} out of range")
            clip = self.clips[idx]
            if spec.duration <= 0:
                raise ValueError("transition duration must be > 0")
            if spec.duration > (clip.end - clip.start):
                raise ValueError("transition duration exceeds clip length")

    def _clip_index_at(self, idx: int) -> int:
        for i, c in enumerate(self.clips):
            if c.start <= idx < c.end:
                return i
        # idx == length falls into the last clip's boundary; clamp
        return len(self.clips) - 1

    def resolve(self, idx: int) -> Resolution:
        i = self._clip_index_at(idx)
        clip = self.clips[i]
        spec = self.transitions.get(i)
        if spec is not None and idx < clip.start + spec.duration:
            t = (idx - clip.start) / float(spec.duration)
            return Resolution("transition", self.clips[i - 1], clip, t, spec)
        return Resolution("hold", clip, None, 0.0, spec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_clip.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/composition/clip.py tests/test_composition_clip.py
git commit -m "feat(composition): LookClip/Composition/resolve — the clip track model"
```

---

### Task 3: Transition base, registry, `_LookContext`, and `crossfade`

**Files:**
- Create: `ditherzam/composition/transitions.py`
- Test: `tests/test_composition_transitions.py`

**Interfaces:**
- Produces:
  - `_LookContext(settings: RenderSettings, render_fn, key=None)` — `render_fn(settings) -> uint8 HxWx3`; `.frame()` memoizes `render_fn(self.settings)`; `.render(settings)` calls `render_fn(settings)`; `.key` used by ParamMorph.
  - `Transition` base: `.render_frame(t, a: _LookContext, b: _LookContext, base_gray, params: dict) -> uint8 HxWx3`.
  - `TRANSITIONS: dict[str, Transition]` and `register(name)` decorator (instantiates and stores). `get_transition(name) -> Transition`.
  - `_lerp_u8(a_u8, b_u8, t) -> uint8` helper.
  - `CrossfadeTransition` registered as `"crossfade"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition_transitions.py
import numpy as np
from ditherzam.composition.transitions import (
    _LookContext, get_transition, TRANSITIONS, _lerp_u8,
)


def _const_ctx(color, settings=None):
    """A context whose render_fn ignores settings and returns a solid HxWx3 frame."""
    frame = np.full((8, 8, 3), color, dtype=np.uint8)
    return _LookContext(settings=settings, render_fn=lambda s: frame)


def test_lerp_endpoints():
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 200, np.uint8)
    assert np.array_equal(_lerp_u8(a, b, 0.0), a)
    assert np.array_equal(_lerp_u8(a, b, 1.0), b)
    assert int(_lerp_u8(a, b, 0.5)[0, 0, 0]) == 100


def test_crossfade_endpoints_and_mid():
    tr = get_transition("crossfade")
    a = _const_ctx(0)
    b = _const_ctx(200)
    base = np.zeros((8, 8), np.float32)
    assert np.array_equal(tr.render_frame(0.0, a, b, base, {}), np.zeros((8, 8, 3), np.uint8))
    assert np.array_equal(tr.render_frame(1.0, a, b, base, {}), np.full((8, 8, 3), 200, np.uint8))
    mid = tr.render_frame(0.5, a, b, base, {})
    assert int(mid[0, 0, 0]) == 100


def test_context_frame_memoized():
    calls = {"n": 0}
    def rf(s):
        calls["n"] += 1
        return np.zeros((2, 2, 3), np.uint8)
    ctx = _LookContext(settings=None, render_fn=rf)
    ctx.frame(); ctx.frame()
    assert calls["n"] == 1


def test_registry_has_crossfade():
    assert "crossfade" in TRANSITIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_transitions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.composition.transitions'`

- [ ] **Step 3: Write minimal implementation**

```python
# ditherzam/composition/transitions.py
from __future__ import annotations

import numpy as np


class _LookContext:
    """A look ready to render at one frame index. `render_fn(settings)` produces a
    uint8 HxWx3 frame (the Compositor bakes in base_gray + temporal field)."""

    def __init__(self, settings, render_fn, key=None) -> None:
        self.settings = settings
        self._render_fn = render_fn
        self.key = key
        self._frame = None

    def frame(self) -> np.ndarray:
        if self._frame is None:
            self._frame = self._render_fn(self.settings)
        return self._frame

    def render(self, settings) -> np.ndarray:
        return self._render_fn(settings)


def _lerp_u8(a_u8: np.ndarray, b_u8: np.ndarray, t: float) -> np.ndarray:
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
    out = a_u8.astype(np.float32) * (1.0 - t) + b_u8.astype(np.float32) * t
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


class Transition:
    """Combine two look contexts at parameter t in [0,1]. Subclasses override
    render_frame; t=0 must yield look A, t=1 must yield look B."""

    def render_frame(self, t, a: _LookContext, b: _LookContext,
                     base_gray, params: dict) -> np.ndarray:
        raise NotImplementedError


TRANSITIONS: dict[str, Transition] = {}


def register(name: str):
    def deco(cls):
        TRANSITIONS[name] = cls()
        return cls
    return deco


def get_transition(name: str) -> Transition:
    try:
        return TRANSITIONS[name]
    except KeyError:
        raise KeyError(f"unknown transition: {name!r}")


@register("crossfade")
class CrossfadeTransition(Transition):
    def render_frame(self, t, a, b, base_gray, params):
        return _lerp_u8(a.frame(), b.frame(), t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_transitions.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/composition/transitions.py tests/test_composition_transitions.py
git commit -m "feat(composition): Transition base + registry + crossfade"
```

---

### Task 4: `dither-dissolve` and `spatial-wipe` transitions

**Files:**
- Modify: `ditherzam/composition/transitions.py` (append masks + two strategies)
- Test: `tests/test_composition_transitions.py` (append)

**Interfaces:**
- Consumes: `_LookContext`, `Transition`, `register`, `_lerp_u8` (Task 3); `ditherzam.animation.temporal.temporal_noise(frame, shape, pattern, amplitude, seed=0) -> float32 HxW` (only when `params["mask"]=="noise"`).
- Produces: `DitherDissolveTransition` (`"dither-dissolve"`), `SpatialWipeTransition` (`"spatial-wipe"`), and private helpers `_bayer_mask(h, w)`, `_wipe_coord(h, w, mode, base_gray)`.

- [ ] **Step 1: Write the failing test (append to tests/test_composition_transitions.py)**

```python
def _ramp_ctx(base_val, top_val, settings=None):
    """8x8 frame that is `base_val` everywhere; helper only needs distinct A/B."""
    frame = np.full((8, 8, 3), base_val, dtype=np.uint8)
    return _LookContext(settings=settings, render_fn=lambda s: frame)


def test_dissolve_endpoints_all_a_all_b():
    from ditherzam.composition.transitions import get_transition
    tr = get_transition("dither-dissolve")
    a = _ramp_ctx(0); b = _ramp_ctx(255)
    base = np.zeros((8, 8), np.float32)
    assert np.array_equal(tr.render_frame(0.0, a, b, base, {}), np.zeros((8, 8, 3), np.uint8))
    assert np.array_equal(tr.render_frame(1.0, a, b, base, {}), np.full((8, 8, 3), 255, np.uint8))


def test_dissolve_b_fraction_monotonic():
    from ditherzam.composition.transitions import get_transition
    tr = get_transition("dither-dissolve")
    a = _ramp_ctx(0); b = _ramp_ctx(255)
    base = np.zeros((16, 16), np.float32)
    fracs = []
    for t in (0.2, 0.5, 0.8):
        out = tr.render_frame(t, a, b, base, {})
        fracs.append(float((out[..., 0] == 255).mean()))
    assert fracs[0] < fracs[1] < fracs[2]


def test_dissolve_deterministic():
    from ditherzam.composition.transitions import get_transition
    tr = get_transition("dither-dissolve")
    a = _ramp_ctx(0); b = _ramp_ctx(255)
    base = np.zeros((16, 16), np.float32)
    o1 = tr.render_frame(0.5, a, b, base, {})
    o2 = tr.render_frame(0.5, a, b, base, {})
    assert np.array_equal(o1, o2)


def test_wipe_linear_left_to_right():
    from ditherzam.composition.transitions import get_transition
    tr = get_transition("spatial-wipe")
    a = _ramp_ctx(0); b = _ramp_ctx(255)
    base = np.zeros((8, 8), np.float32)
    out = tr.render_frame(0.5, a, b, base, {"mode": "linear"})
    left = out[:, :4, 0]
    right = out[:, 4:, 0]
    assert left.mean() > right.mean()   # left half already revealed B
    assert np.array_equal(tr.render_frame(0.0, a, b, base, {"mode": "linear"}),
                          np.zeros((8, 8, 3), np.uint8))
    assert np.array_equal(tr.render_frame(1.0, a, b, base, {"mode": "linear"}),
                          np.full((8, 8, 3), 255, np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_transitions.py -q`
Expected: FAIL — `KeyError: 'unknown transition: dither-dissolve'`

- [ ] **Step 3: Write minimal implementation (append to transitions.py)**

```python
# --- Task 4: dither-dissolve + spatial-wipe -------------------------------

_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32)


def _bayer_mask(h: int, w: int) -> np.ndarray:
    """Tiled 8x8 Bayer threshold matrix normalized to [0,1) with shape (h, w)."""
    reps_y = (h + 7) // 8
    reps_x = (w + 7) // 8
    tiled = np.tile((_BAYER8 + 0.5) / 64.0, (reps_y, reps_x))
    return tiled[:h, :w]


def _wipe_coord(h: int, w: int, mode: str, base_gray) -> np.ndarray:
    """Per-pixel coordinate in [0,1]; a pixel reveals B once coord <= t."""
    if mode == "radial":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        m = d.max()
        return d / m if m > 0 else np.zeros((h, w), np.float32)
    if mode == "luma":
        g = np.asarray(base_gray, np.float32)
        return np.clip(g / 255.0, 0.0, 1.0)
    # linear (default): left-to-right
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    return np.broadcast_to(xs, (h, w)).copy()


@register("dither-dissolve")
class DitherDissolveTransition(Transition):
    def render_frame(self, t, a, b, base_gray, params):
        fa, fb = a.frame(), b.frame()
        h, w = fa.shape[:2]
        if params.get("mask") == "noise":
            from ..animation.temporal import temporal_noise
            n = temporal_noise(0, (h, w), "blue-noise", 1.0, int(params.get("seed", 0)))
            mask = (n - n.min()) / (np.ptp(n) or 1.0)
        else:
            mask = _bayer_mask(h, w)
        select = (mask < float(t))[..., None]
        return np.where(select, fb, fa).astype(np.uint8)


@register("spatial-wipe")
class SpatialWipeTransition(Transition):
    def render_frame(self, t, a, b, base_gray, params):
        fa, fb = a.frame(), b.frame()
        h, w = fa.shape[:2]
        coord = _wipe_coord(h, w, str(params.get("mode", "linear")), base_gray)
        edge = float(params.get("edge", 0.0))
        if edge <= 0.0:
            select = (coord <= float(t))[..., None]
            return np.where(select, fb, fa).astype(np.uint8)
        alpha = np.clip((float(t) - coord) / edge + 0.5, 0.0, 1.0)[..., None]
        out = fa.astype(np.float32) * (1.0 - alpha) + fb.astype(np.float32) * alpha
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_transitions.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/composition/transitions.py tests/test_composition_transitions.py
git commit -m "feat(composition): dither-dissolve + spatial-wipe transitions"
```

---

### Task 5: `param-morph` transition

**Files:**
- Modify: `ditherzam/composition/transitions.py` (append)
- Test: `tests/test_composition_morph.py`

**Interfaces:**
- Consumes: `_LookContext`, `Transition`, `register`, `_lerp_u8` (Task 3); `dataclasses.replace`; `RenderSettings` fields (contrast, midtones, highlights, blur, luminance_threshold, saturation, scale, depth, style).
- Produces: `ParamMorphTransition` (`"param-morph"`); private `_morph_settings(a_settings, b_settings, t) -> RenderSettings` (numeric-interpolated, discrete fields taken from `a_settings`).
- Behaviour: render-once (via `a.render`) only when `a.settings.style == b.settings.style AND a.key == b.key AND a.key is not None`; otherwise crossfade the two independently numeric-morphed renders.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition_morph.py
import numpy as np
import pytest
from dataclasses import replace
from ditherzam.render import RenderSettings
from ditherzam.composition.transitions import _LookContext, get_transition


def _settings(**kw):
    base = dict(style="None", scale=4, depth=2, contrast=50, midtones=50,
                highlights=50, blur=0, luminance_threshold=50, saturation=50)
    base.update(kw)
    return RenderSettings(**base)


def _recording_ctx(settings, key):
    """render_fn returns a frame encoding settings.contrast so we can read it back."""
    def rf(s):
        return np.full((4, 4, 3), int(round(s.contrast)), dtype=np.uint8)
    ctx = _LookContext(settings=settings, render_fn=rf, key=key)
    ctx.calls = []
    orig = ctx._render_fn
    ctx._render_fn = lambda s: (ctx.calls.append(s) or orig(s))
    return ctx


def test_morph_render_once_same_style_same_key():
    tr = get_transition("param-morph")
    a = _recording_ctx(_settings(contrast=0), key="k")
    b = _recording_ctx(_settings(contrast=100), key="k")
    base = np.zeros((4, 4), np.float32)
    out = tr.render_frame(0.5, a, b, base, {})
    assert int(out[0, 0, 0]) == 50            # contrast interpolated 0->100 at t=0.5
    assert len(a.calls) == 1 and len(b.calls) == 0   # rendered ONCE, via A


def test_morph_endpoints_same_style():
    tr = get_transition("param-morph")
    a = _recording_ctx(_settings(contrast=0), key="k")
    b = _recording_ctx(_settings(contrast=100), key="k")
    base = np.zeros((4, 4), np.float32)
    assert int(tr.render_frame(0.0, a, b, base, {})[0, 0, 0]) == 0
    assert int(tr.render_frame(1.0, a, b, base, {})[0, 0, 0]) == 100


def test_morph_fallback_crossfade_on_style_diff():
    tr = get_transition("param-morph")
    a = _recording_ctx(_settings(style="None", contrast=0), key="k")
    b = _recording_ctx(_settings(style="Atkinson", contrast=0), key="k")
    base = np.zeros((4, 4), np.float32)
    out = tr.render_frame(0.5, a, b, base, {})
    # both looks rendered (crossfade fallback)
    assert len(a.calls) == 1 and len(b.calls) == 1


def test_morph_scale_interpolates_to_int():
    from ditherzam.composition.transitions import _morph_settings
    s = _morph_settings(_settings(scale=2), _settings(scale=8), 0.5)
    assert s.scale == 5 and isinstance(s.scale, int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_morph.py -q`
Expected: FAIL — `KeyError: 'unknown transition: param-morph'` (and `_morph_settings` import error)

- [ ] **Step 3: Write minimal implementation (append to transitions.py)**

```python
# --- Task 5: param-morph --------------------------------------------------

from dataclasses import replace as _dc_replace  # noqa: E402

_MORPH_FLOAT = ("contrast", "midtones", "highlights", "blur",
                "luminance_threshold", "saturation")
_MORPH_INT = ("scale", "depth")


def _lerp(x: float, y: float, t: float) -> float:
    return float(x) * (1.0 - t) + float(y) * t


def _morph_settings(a_settings, b_settings, t: float):
    """Numeric-interpolated settings; discrete fields (style, color_mapping,
    params, invert, preview_disabled) are taken from a_settings."""
    vals = {f: _lerp(getattr(a_settings, f), getattr(b_settings, f), t)
            for f in _MORPH_FLOAT}
    for f in _MORPH_INT:
        vals[f] = int(round(_lerp(getattr(a_settings, f), getattr(b_settings, f), t)))
    return _dc_replace(a_settings, **vals)


@register("param-morph")
class ParamMorphTransition(Transition):
    def render_frame(self, t, a, b, base_gray, params):
        same_style = a.settings.style == b.settings.style
        same_look = a.key is not None and a.key == b.key
        if same_style and same_look:
            return np.asarray(a.render(_morph_settings(a.settings, b.settings, t)),
                              dtype=np.uint8)
        # fallback: numeric-morph each look independently, crossfade by t
        fa = a.render(_morph_settings(a.settings, b.settings, t))
        fb = b.render(_morph_settings(b.settings, a.settings, 1.0 - t))
        return _lerp_u8(fa, fb, t)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_morph.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/composition/transitions.py tests/test_composition_morph.py
git commit -m "feat(composition): param-morph transition (render-once + crossfade fallback)"
```

---

### Task 6: `Compositor.render_frame` + byte-identical parity

**Files:**
- Create: `ditherzam/composition/compositor.py`
- Test: `tests/test_composition_compositor.py`

**Interfaces:**
- Consumes: `Composition`/`Resolution` (Task 2); `get_transition`, `_LookContext` (Task 3); `ditherzam.render.RenderPipeline`; `ditherzam.dithering.registry`; `ditherzam.animation.temporal.temporal_noise`.
- Produces: `Compositor(composition, registry=None, seed=0)` with `.render_frame(idx, base_gray_f32, temporal_pattern=None, temporal_amplitude=0.0) -> uint8 HxWx3` and `_pipeline_for(look) -> RenderPipeline` (cached per `id(look)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition_compositor.py
import numpy as np
import pytest
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.dithering import registry
from ditherzam.composition.look import Look
from ditherzam.composition.clip import LookClip, TransitionSpec, Composition
from ditherzam.composition.compositor import Compositor


def _preset(style="Floyd-Steinberg"):
    return {
        "adjustments": {"contrast": 55, "midtones": 50, "highlights": 45,
                        "luminance_threshold": 50, "blur": 0, "saturation": 50,
                        "invert": False},
        "dither": {"style": style, "scale": 4, "depth": 2,
                   "color_mapping": "match", "preview_disabled": False, "params": {}},
    }


def _gray():
    rng = np.random.default_rng(7)
    return (rng.random((32, 32)) * 255).astype(np.float32)


def test_single_clip_byte_identical_to_pipeline():
    look = Look("A", _preset("Floyd-Steinberg"))
    comp = Composition(length=10)
    comp.add_clip(LookClip(look, 0, 10))
    comp.validate()
    cz = Compositor(comp)

    ref = RenderPipeline(registry, look.build_color_engine(), look.build_effect_stack())
    g = _gray()
    got = cz.render_frame(3, g)
    exp = ref.render(g, look.settings)
    assert np.array_equal(got, exp)


def test_transition_frame_differs_from_both_looks():
    a = Look("A", _preset("Floyd-Steinberg"))
    b = Look("B", _preset("Atkinson"))
    comp = Composition(length=40)
    comp.add_clip(LookClip(a, 0, 20))
    comp.add_clip(LookClip(b, 20, 40))
    comp.set_transition(1, TransitionSpec("crossfade", duration=10, params={}))
    comp.validate()
    cz = Compositor(comp)
    g = _gray()

    mid = cz.render_frame(25, g)              # t = 0.5 inside transition
    pure_a = RenderPipeline(registry, a.build_color_engine(),
                            a.build_effect_stack()).render(g, a.settings)
    pure_b = RenderPipeline(registry, b.build_color_engine(),
                            b.build_effect_stack()).render(g, b.settings)
    assert not np.array_equal(mid, pure_a)
    assert not np.array_equal(mid, pure_b)


def test_pipeline_cached_per_look():
    look = Look("A", _preset())
    comp = Composition(length=5)
    comp.add_clip(LookClip(look, 0, 5))
    comp.validate()
    cz = Compositor(comp)
    p1 = cz._pipeline_for(look)
    p2 = cz._pipeline_for(look)
    assert p1 is p2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_compositor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.composition.compositor'`

- [ ] **Step 3: Write minimal implementation**

```python
# ditherzam/composition/compositor.py
from __future__ import annotations

import numpy as np

from ..render import RenderPipeline
from .transitions import _LookContext, get_transition

_INACTIVE = {None, "", "none", "None", "off"}


class Compositor:
    """Renders a Composition frame-by-frame on top of RenderPipeline. One pipeline
    is cached per Look (so each look keeps its own color engine + effect stack)."""

    def __init__(self, composition, registry=None, seed: int = 0) -> None:
        self.composition = composition
        if registry is None:
            from ..dithering import registry as _reg
            registry = _reg
        self.registry = registry
        self.seed = int(seed)
        self._pipelines: dict[int, RenderPipeline] = {}

    def _pipeline_for(self, look) -> RenderPipeline:
        pid = id(look)
        p = self._pipelines.get(pid)
        if p is None:
            p = RenderPipeline(self.registry, look.build_color_engine(),
                               look.build_effect_stack())
            self._pipelines[pid] = p
        return p

    def _render_fn(self, look, idx, base_gray, pattern, amplitude):
        pipeline = self._pipeline_for(look)
        h, w = base_gray.shape[:2]
        active = (pattern not in _INACTIVE and float(amplitude) > 0.0)
        seed = self.seed

        def render_fn(settings):
            field = None
            if active:
                from ..animation.temporal import temporal_noise
                factor = max(1, int(settings.scale))
                small = (max(1, h // factor), max(1, w // factor))
                field = temporal_noise(idx, small, pattern, float(amplitude), seed)
            return pipeline.render(base_gray, settings, temporal_field=field)

        return render_fn

    def _context(self, clip, idx, base_gray, pattern, amplitude) -> _LookContext:
        return _LookContext(
            settings=clip.settings_at(idx),
            render_fn=self._render_fn(clip.look, idx, base_gray, pattern, amplitude),
            key=clip.look.color_effect_key(),
        )

    def render_frame(self, idx, base_gray_f32, temporal_pattern=None,
                     temporal_amplitude=0.0) -> np.ndarray:
        g = np.asarray(base_gray_f32, dtype=np.float32)
        r = self.composition.resolve(int(idx))
        if r.kind == "hold":
            ctx = self._context(r.clip, idx, g, temporal_pattern, temporal_amplitude)
            return np.asarray(ctx.frame(), np.uint8)
        a = self._context(r.clip, idx, g, temporal_pattern, temporal_amplitude)
        b = self._context(r.clip_b, idx, g, temporal_pattern, temporal_amplitude)
        tr = get_transition(r.spec.kind)
        return np.asarray(tr.render_frame(r.t, a, b, g, r.spec.params), np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_compositor.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/composition/compositor.py tests/test_composition_compositor.py
git commit -m "feat(composition): Compositor.render_frame + byte-identical hold parity"
```

---

### Task 7: `render_composition` + `export_composition`

**Files:**
- Modify: `ditherzam/composition/compositor.py` (append module-level functions)
- Test: `tests/test_composition_render.py`

**Interfaces:**
- Consumes: `Compositor` (Task 6); `ditherzam.video.ffmpeg.assemble_video(frames_dir, fps, orig_video, out, runner=...)` and `ditherzam.video.ffmpeg.run_command`; `PIL.Image`.
- Produces:
  - `render_composition(compositor, frame_source, length, temporal_pattern=None, temporal_amplitude=0.0) -> Iterator[uint8 HxWx3]` — `frame_source(idx) -> gray_f32`.
  - `still_source(gray_f32) -> callable` — returns a `frame_source` that yields the same gray for every idx.
  - `export_composition(compositor, frame_source, length, out_path, fps=24, temporal_pattern=None, temporal_amplitude=0.0, progress=None, runner=None) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition_render.py
import os
import numpy as np
from ditherzam.composition.look import Look
from ditherzam.composition.clip import LookClip, Composition
from ditherzam.composition.compositor import (
    Compositor, render_composition, export_composition, still_source,
)


def _preset(style="Floyd-Steinberg"):
    return {
        "adjustments": {"contrast": 50, "midtones": 50, "highlights": 50,
                        "luminance_threshold": 50, "blur": 0, "saturation": 50,
                        "invert": False},
        "dither": {"style": style, "scale": 4, "depth": 2,
                   "color_mapping": "match", "preview_disabled": False, "params": {}},
    }


def _one_clip_comp(length):
    comp = Composition(length=length)
    comp.add_clip(LookClip(Look("A", _preset()), 0, length))
    comp.validate()
    return comp


def test_render_composition_yields_length_frames():
    comp = _one_clip_comp(6)
    cz = Compositor(comp)
    g = (np.random.default_rng(1).random((16, 16)) * 255).astype(np.float32)
    frames = list(render_composition(cz, still_source(g), 6))
    assert len(frames) == 6
    assert frames[0].shape == (16, 16, 3) and frames[0].dtype == np.uint8


def test_video_frame_source_receives_idx():
    comp = _one_clip_comp(3)
    cz = Compositor(comp)
    seen = []
    g = np.zeros((8, 8), np.float32)
    def src(idx):
        seen.append(idx)
        return g
    list(render_composition(cz, src, 3))
    assert seen == [0, 1, 2]


def test_export_composition_calls_assemble(tmp_path):
    comp = _one_clip_comp(2)
    cz = Compositor(comp)
    g = np.zeros((8, 8), np.float32)
    calls = {"n": 0}
    def fake_runner(cmd):
        calls["n"] += 1               # never spawns real ffmpeg
    out = tmp_path / "out.mp4"
    # frames get written to a temp dir; assemble is stubbed via runner
    res = export_composition(cz, still_source(g), 2, str(out), fps=12,
                             runner=fake_runner)
    assert res == str(out)
    assert calls["n"] >= 1            # assemble_video invoked the runner
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_render.py -q`
Expected: FAIL — `ImportError: cannot import name 'render_composition'`

- [ ] **Step 3: Write minimal implementation (append to compositor.py)**

```python
# --- Task 7: whole-composition render + export ----------------------------

import os as _os
import tempfile as _tempfile
from typing import Callable, Iterator, Optional


def still_source(gray_f32) -> Callable[[int], np.ndarray]:
    """A frame_source that returns the same grayscale frame for every index."""
    g = np.asarray(gray_f32, dtype=np.float32)
    return lambda idx: g


def render_composition(compositor: "Compositor", frame_source, length: int,
                       temporal_pattern=None, temporal_amplitude: float = 0.0
                       ) -> Iterator[np.ndarray]:
    """Yield `length` uint8 HxWx3 frames. `frame_source(idx) -> gray_f32` decouples
    the source: a held still (see still_source) or a decoded video frame loader."""
    for idx in range(int(length)):
        base = frame_source(idx)
        yield compositor.render_frame(idx, base, temporal_pattern, temporal_amplitude)


def export_composition(compositor: "Compositor", frame_source, length: int,
                       out_path: str, fps: int = 24, temporal_pattern=None,
                       temporal_amplitude: float = 0.0,
                       progress: Optional[Callable[[int, int], None]] = None,
                       runner=None) -> str:
    """Render every frame to PNG then encode to MP4 via ffmpeg.assemble_video.
    `runner` is injectable for tests (defaults to ffmpeg.run_command)."""
    from PIL import Image
    from ..video.ffmpeg import assemble_video, run_command

    if runner is None:
        runner = run_command
    frames_dir = _tempfile.mkdtemp(prefix="ditherzam_comp_")
    n = int(length)
    for i, frame in enumerate(render_composition(
            compositor, frame_source, n, temporal_pattern, temporal_amplitude)):
        Image.fromarray(np.asarray(frame, np.uint8)).save(
            _os.path.join(frames_dir, f"frame{i:06d}.png"))
        if progress is not None:
            progress(i + 1, n)
    assemble_video(frames_dir, fps, None, out_path, runner=runner)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_render.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/composition/compositor.py tests/test_composition_render.py
git commit -m "feat(composition): render_composition + export_composition (source-agnostic)"
```

---

### Task 8: Composition (de)serialization + package exports

**Files:**
- Create: `ditherzam/composition/serialize.py`
- Modify: `ditherzam/composition/__init__.py` (public exports)
- Test: `tests/test_composition_serialize.py`

**Interfaces:**
- Consumes: `Look` (Task 1); `LookClip`, `TransitionSpec`, `Composition` (Task 2); `ditherzam.animation.timeline.Timeline`, `Keyframe`.
- Produces:
  - `composition_to_dict(comp: Composition) -> dict`.
  - `composition_from_dict(d: dict) -> Composition`.
  - `ditherzam.composition` package exports: `Look, LookClip, TransitionSpec, Composition, Compositor, render_composition, export_composition, still_source, composition_to_dict, composition_from_dict, TRANSITIONS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_composition_serialize.py
import numpy as np
import yaml
from ditherzam.composition import (
    Look, LookClip, TransitionSpec, Composition,
    composition_to_dict, composition_from_dict,
)
from ditherzam.animation.timeline import Timeline, Keyframe


def _preset(style="Floyd-Steinberg"):
    return {
        "adjustments": {"contrast": 50, "midtones": 50, "highlights": 50,
                        "luminance_threshold": 50, "blur": 0, "saturation": 50,
                        "invert": False},
        "dither": {"style": style, "scale": 4, "depth": 2,
                   "color_mapping": "match", "preview_disabled": False, "params": {}},
    }


def _comp():
    comp = Composition(length=200)
    tl = Timeline(length=100)
    tl.add(Keyframe(0, "contrast", 10.0))
    tl.add(Keyframe(99, "contrast", 90.0))
    comp.add_clip(LookClip(Look("A", _preset("Floyd-Steinberg")), 0, 100, automation=tl))
    comp.add_clip(LookClip(Look("B", _preset("Atkinson")), 100, 200))
    comp.set_transition(1, TransitionSpec("dither-dissolve", 15, {"mask": "bayer"}))
    return comp


def test_roundtrip_structure():
    comp = _comp()
    d = composition_to_dict(comp)
    # survives a YAML round-trip (must be plain data)
    d = yaml.safe_load(yaml.safe_dump(d))
    back = composition_from_dict(d)
    assert back.length == 200
    assert [c.look.name for c in back.clips] == ["A", "B"]
    assert [(c.start, c.end) for c in back.clips] == [(0, 100), (100, 200)]
    assert back.clips[1].look.settings.style == "Atkinson"
    spec = back.transitions[1]
    assert spec.kind == "dither-dissolve" and spec.duration == 15


def test_roundtrip_automation_preserved():
    back = composition_from_dict(composition_to_dict(_comp()))
    clip = back.clips[0]
    assert clip.automation is not None
    assert clip.settings_at(0).contrast == 10
    assert clip.settings_at(99).contrast == 90


def test_package_exports_compositor():
    from ditherzam.composition import Compositor, render_composition, TRANSITIONS
    assert "crossfade" in TRANSITIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_serialize.py -q`
Expected: FAIL — `ImportError: cannot import name 'composition_to_dict'`

- [ ] **Step 3: Write minimal implementation**

```python
# ditherzam/composition/serialize.py
from __future__ import annotations

from ..animation.timeline import Keyframe, Timeline
from .clip import Composition, LookClip, TransitionSpec
from .look import Look


def _automation_to_list(tl) -> list | None:
    if tl is None:
        return None
    out = []
    for field in tl.fields():
        for kf in tl._keys[field]:
            out.append({"frame": int(kf.frame), "field": str(kf.field),
                        "value": float(kf.value), "kind": str(kf.kind)})
    return {"length": int(tl.length), "keys": out}


def _automation_from_dict(d) -> Timeline | None:
    if not d:
        return None
    tl = Timeline(length=int(d.get("length", 0)))
    for kf in d.get("keys", []):
        tl.add(Keyframe(int(kf["frame"]), str(kf["field"]),
                        float(kf["value"]), str(kf.get("kind", "linear"))))
    return tl


def composition_to_dict(comp: Composition) -> dict:
    return {
        "length": int(comp.length),
        "clips": [
            {"name": c.look.name, "preset": c.look.preset,
             "start": int(c.start), "end": int(c.end),
             "automation": _automation_to_list(c.automation)}
            for c in comp.clips
        ],
        "transitions": {
            str(idx): {"kind": s.kind, "duration": int(s.duration),
                       "params": dict(s.params)}
            for idx, s in comp.transitions.items()
        },
    }


def composition_from_dict(d: dict) -> Composition:
    comp = Composition(length=int(d.get("length", 0)))
    for c in d.get("clips", []):
        comp.add_clip(LookClip(
            Look(c.get("name", "look"), c.get("preset", {})),
            int(c["start"]), int(c["end"]),
            automation=_automation_from_dict(c.get("automation")),
        ))
    for idx, s in (d.get("transitions") or {}).items():
        comp.set_transition(int(idx), TransitionSpec(
            str(s["kind"]), int(s["duration"]), dict(s.get("params", {}))))
    return comp
```

Replace `ditherzam/composition/__init__.py` contents with:

```python
"""ditherzam.composition — timeline of look-clips with transitions/morphs.

Qt-free. Composes on top of RenderPipeline; never modifies it.
"""
from __future__ import annotations

from .look import Look
from .clip import LookClip, TransitionSpec, Composition, Resolution
from .transitions import TRANSITIONS, get_transition
from .compositor import (
    Compositor, render_composition, export_composition, still_source,
)
from .serialize import composition_to_dict, composition_from_dict

__all__ = [
    "Look", "LookClip", "TransitionSpec", "Composition", "Resolution",
    "TRANSITIONS", "get_transition",
    "Compositor", "render_composition", "export_composition", "still_source",
    "composition_to_dict", "composition_from_dict",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_composition_serialize.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the WHOLE suite (regression gate) and commit**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all prior tests still pass + the new composition tests (JIT-off green; the one `QMouseEvent` deprecation warning is pre-existing).

```bash
git add ditherzam/composition/__init__.py ditherzam/composition/serialize.py tests/test_composition_serialize.py
git commit -m "feat(composition): YAML round-trip + package exports"
```

---

## Self-Review

**Spec coverage:**
- Look (preset-backed config) → Task 1. ✔
- LookClip + Composition + resolve (contiguous track, transition window) → Task 2. ✔
- Transition base + registry + crossfade → Task 3. ✔
- dither-dissolve + spatial-wipe (+ mask helpers, temporal_noise reuse) → Task 4. ✔
- param-morph (render-once + crossfade fallback, discrete-field handling) → Task 5. ✔
- Compositor.render_frame, per-look pipeline cache, byte-identical hold parity, temporal field per look at its own scale → Task 6. ✔
- render_composition (frame_source: still vs video) + export_composition (ffmpeg reuse, injectable runner) → Task 7. ✔
- Composition (de)serialization (YAML round-trip incl. automation) + package exports → Task 8. ✔
- Invariants (Qt-free, frozen render.py/RenderSettings, TDD, JIT-off green) → Global Constraints + no edits to render.py in any task. ✔

**Placeholder scan:** none — every code step contains full runnable code; no TODO/TBD.

**Type consistency:** `_LookContext(settings, render_fn, key)`, `Resolution(kind, clip, clip_b, t, spec)`, `TransitionSpec(kind, duration, params)`, `render_frame(t, a, b, base_gray, params)`, `_morph_settings(a_settings, b_settings, t)`, `frame_source(idx)->gray`, `assemble_video(..., runner=)` are used identically across Tasks 1-8. `color_effect_key()` (Task 1) is consumed by Compositor `_context` (Task 6) as `key` and by ParamMorph (Task 5). ✔

**Note for the orchestrator:** per WORKFLOW-new-feature.md, branch in-place off `main` (`git checkout -b feat/composition-engine`) — NOT a worktree (the repo-root `.venv` is absent inside worktrees). Runner is `./.venv/Scripts/python.exe -m pytest`.
```
