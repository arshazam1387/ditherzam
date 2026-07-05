# Phase 4 — Effects Stack & Render Pipeline — Completion Task List

Build the stackable post-processing effects system (Blur, Sharpen, Chromatic
Aberration, JPEG Glitch, Epsilon Glow), the reorderable `EffectStack`, and the
top-level `RenderPipeline` that composes **adjustments → dither → color →
saturation → effects → invert** into one deterministic, Qt-free call — proven by a
dedicated render-order contract test.

## Prereqs: Phases 1 (foundation & dither core) and 3 (color engine) green

This phase consumes, verbatim, the following already-frozen signatures:

```python
# ditherzam/adjustments.py   (float32, 0..255)
apply_contrast(img, value); apply_midtones(img, value); apply_highlights(img, value)
apply_blur(img, value); apply_invert(img, enabled); apply_saturation(rgb, value)

# ditherzam/dithering/pipeline.py   (frozen contract INCLUDES threshold_field)
def apply_dither(gray_f32, *, style, scale, luminance_threshold, params, registry,
                 preview_disabled=False, threshold_field=None) -> np.ndarray

# ditherzam/imaging.py
def clamp_u8(arr) -> np.ndarray                    # uint8, clipped 0..255

# ditherzam/color/engine.py
class ColorEngine:
    palette: Palette; mode: str                    # "off"|"nearest"|"ordered"|"diffused"
    def map(self, gray_or_rgb_f32) -> np.ndarray   # uint8 HxWx3
# ditherzam/color/palette.py
class Palette:  # .from_list(name, rgb_list); .name; .colors  float32[K,3]
```

> **Contract note (threshold_field):** the FROZEN `apply_dither` signature includes
> `threshold_field=None`. Phase 1's completion checklist adds this keyword. If your
> Phase-1 `apply_dither` predates it, add the `threshold_field=None` keyword there
> *before* running Task 4.3 — `RenderPipeline.render` forwards `temporal_field`
> straight into it so Phase 8 (temporal) needs no pipeline change.

## Global constraints (from `00-ROADMAP.md`)

- Effects operate on `uint8[H,W,3]` RGB and return `uint8[H,W,3]`. **Qt-free.**
- Clean-room: only public-domain techniques (Gaussian blur, unsharp mask, JPEG
  round-trip, channel roll). No third-party app code/strings/URLs.
- TDD: red → green → refactor; commit after every green. No placeholders — every
  code step is complete, runnable code.
- Tests run with `NUMBA_DISABLE_JIT=1` (already defaulted in `tests/conftest.py`).

## File structure (this phase)

- Create `ditherzam/effects/__init__.py` (empty), `ditherzam/effects/post.py`,
  `ditherzam/effects/stack.py`
- Create `ditherzam/render.py`
- Tests: `tests/test_effects_post.py`, `tests/test_effect_stack.py`,
  `tests/test_render.py`, `tests/test_render_order.py`

---

### Task 4.0: Prereq sanity check (no new code)

**Files:** none (verification only)
**Interfaces:** Consumes Phase 1 + Phase 3 public API.

- [ ] **Step 1: Verify the consumed contracts import and behave**

Run:

```bash
NUMBA_DISABLE_JIT=1 python -c "import numpy as np; \
from ditherzam.adjustments import apply_contrast, apply_saturation; \
from ditherzam.dithering import registry; \
from ditherzam.dithering.pipeline import apply_dither; \
from ditherzam.imaging import clamp_u8; \
from ditherzam.color.palette import Palette; \
from ditherzam.color.engine import ColorEngine; \
import inspect; \
assert 'threshold_field' in inspect.signature(apply_dither).parameters, 'apply_dither missing threshold_field (see Prereqs note)'; \
print('prereqs OK')"
```

Expected: prints `prereqs OK`. If it raises `AssertionError`, add
`threshold_field=None` to `apply_dither` (Phase 1) first. If an import fails,
Phase 1/3 is not green — stop and finish it.

- [ ] **Step 2: Create the empty package marker**

Create `ditherzam/effects/__init__.py`:

```python
"""ditherzam post-processing effects (stackable, reorderable)."""
```

- [ ] **Step 3: Commit**

```bash
git add ditherzam/effects/__init__.py
git commit -m "chore(effects): create effects package"
```

---

### Task 4.1: Post-effect functions (Blur, Sharpen, Chromatic Aberration, JPEG Glitch, Epsilon Glow)

**Files:** Create `ditherzam/effects/post.py`; Test `tests/test_effects_post.py`
**Interfaces:** Produces `EFFECTS: dict[str, Callable]` and:

```python
def blur(rgb_u8, radius: float) -> np.ndarray            # uint8 HxWx3
def sharpen(rgb_u8, amount: float) -> np.ndarray         # uint8 HxWx3
def chromatic_aberration(rgb_u8, shift: int) -> np.ndarray
def jpeg_glitch(rgb_u8, quality: int) -> np.ndarray
def epsilon_glow(rgb_u8, radius: float, strength: float) -> np.ndarray
EFFECTS = {"Blur","Sharpen","Chromatic Aberration","JPEG Glitch","Epsilon Glow"}
```

- [ ] **Step 1: Write the failing test — `tests/test_effects_post.py`**

```python
import numpy as np
import pytest
from ditherzam.effects.post import (
    EFFECTS, blur, sharpen, chromatic_aberration, jpeg_glitch, epsilon_glow,
)


def rand_img():
    return np.random.RandomState(0).randint(0, 256, (16, 16, 3), np.uint8)


def gray_img(v):
    return np.full((16, 16, 3), v, np.uint8)


def test_all_five_effects_registered():
    for k in ("Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch", "Epsilon Glow"):
        assert k in EFFECTS
    assert set(EFFECTS) == {
        "Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch", "Epsilon Glow"}
    assert all(callable(fn) for fn in EFFECTS.values())


def test_shape_and_dtype_preserved():
    x = rand_img()
    for fn, kw in [
        (blur, {"radius": 2}),
        (sharpen, {"amount": 1.5}),
        (chromatic_aberration, {"shift": 2}),
        (jpeg_glitch, {"quality": 10}),
        (epsilon_glow, {"radius": 3, "strength": 0.5}),
    ]:
        out = fn(x, **kw)
        assert out.shape == x.shape and out.dtype == np.uint8


def test_blur_zero_is_identity():
    x = rand_img()
    np.testing.assert_array_equal(blur(x, 0), x)


def test_blur_uniform_unchanged():
    # Gaussian blur of a flat field is the same flat field.
    x = gray_img(100)
    np.testing.assert_array_equal(blur(x, 3), x)


def test_sharpen_uniform_is_identity():
    # unsharp mask on a flat field: (a - b) == 0, so output == input.
    x = gray_img(100)
    np.testing.assert_array_equal(sharpen(x, 1.5), x)


def test_chromatic_aberration_shifts_red_right():
    x = np.zeros((4, 8, 3), np.uint8)
    x[:, 3, 0] = 255                                  # red column at x=3
    out = chromatic_aberration(x, shift=2)
    assert out[:, 5, 0].max() == 255                  # red moved +2 (right)
    assert out[:, 3, 0].max() == 0                    # vacated by the roll


def test_chromatic_aberration_shifts_blue_left():
    x = np.zeros((4, 8, 3), np.uint8)
    x[:, 5, 2] = 255                                  # blue column at x=5
    out = chromatic_aberration(x, shift=2)
    assert out[:, 3, 2].max() == 255                  # blue moved -2 (left)


def test_chromatic_aberration_leaves_green_untouched():
    x = rand_img()
    out = chromatic_aberration(x, shift=3)
    np.testing.assert_array_equal(out[..., 1], x[..., 1])


def test_jpeg_glitch_preserves_shape_and_degrades():
    x = rand_img()
    out = jpeg_glitch(x, quality=5)
    assert out.shape == x.shape and out.dtype == np.uint8
    assert not np.array_equal(out, x)                 # lossy round-trip changed it


def test_jpeg_glitch_clamps_quality():
    x = rand_img()
    # out-of-range quality must not raise (clamped into 1..100)
    assert jpeg_glitch(x, quality=0).shape == x.shape
    assert jpeg_glitch(x, quality=999).shape == x.shape


def test_epsilon_glow_brightens_uniform_field():
    # glow(blur)==100 on a flat field; out = clip(100 + 100*0.5) = 150
    x = gray_img(100)
    out = epsilon_glow(x, radius=3, strength=0.5)
    assert np.all(out == 150)


def test_epsilon_glow_clips_to_255():
    x = gray_img(200)
    out = epsilon_glow(x, radius=2, strength=1.0)      # 200 + 200 -> clipped 255
    assert out.max() == 255 and out.dtype == np.uint8
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_effects_post.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.effects.post'`

- [ ] **Step 3: Implement `ditherzam/effects/post.py`**

```python
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def blur(rgb_u8: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur; radius <= 0 is the identity."""
    if radius <= 0:
        return rgb_u8
    pil = Image.fromarray(rgb_u8).filter(ImageFilter.GaussianBlur(float(radius)))
    return np.asarray(pil, np.uint8)


def sharpen(rgb_u8: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask: out = a + (a - blur(a)) * amount, clipped to 0..255."""
    pil = Image.fromarray(rgb_u8)
    blurred = pil.filter(ImageFilter.GaussianBlur(2))
    a = np.asarray(pil, np.float32)
    b = np.asarray(blurred, np.float32)
    return np.clip(a + (a - b) * amount, 0, 255).astype(np.uint8)


def chromatic_aberration(rgb_u8: np.ndarray, shift: int) -> np.ndarray:
    """Roll the red channel right by `shift` and the blue channel left; green stays."""
    out = rgb_u8.copy()
    s = int(shift)
    out[..., 0] = np.roll(rgb_u8[..., 0], s, axis=1)      # red -> right
    out[..., 2] = np.roll(rgb_u8[..., 2], -s, axis=1)     # blue -> left
    return out


def jpeg_glitch(rgb_u8: np.ndarray, quality: int) -> np.ndarray:
    """Lossy JPEG round-trip; low quality introduces block/DCT artifacts."""
    q = int(max(1, min(100, quality)))
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), np.uint8)


def epsilon_glow(rgb_u8: np.ndarray, radius: float, strength: float) -> np.ndarray:
    """Additive bloom tuned for dithered art: base + blur(base) * strength."""
    pil = Image.fromarray(rgb_u8)
    glow = np.asarray(pil.filter(ImageFilter.GaussianBlur(float(radius))), np.float32)
    base = np.asarray(pil, np.float32)
    return np.clip(base + glow * strength, 0, 255).astype(np.uint8)


EFFECTS = {
    "Blur": blur,
    "Sharpen": sharpen,
    "Chromatic Aberration": chromatic_aberration,
    "JPEG Glitch": jpeg_glitch,
    "Epsilon Glow": epsilon_glow,
}
```

- [ ] **Step 4: Run — expect PASS**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_effects_post.py -v`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/effects/post.py tests/test_effects_post.py
git commit -m "feat(effects): post-processing effect functions (blur/sharpen/chromatic/jpeg/glow)"
```

---

### Task 4.2: `EffectStack` (add / move / remove / apply)

**Files:** Create `ditherzam/effects/stack.py`; Test `tests/test_effect_stack.py`
**Interfaces:**

```python
class EffectStack:
    items: list[tuple[str, dict]]
    def add(self, name: str, **params) -> None
    def move(self, index: int, new_index: int) -> None
    def remove(self, index: int) -> None
    def apply(self, rgb_u8: np.ndarray) -> np.ndarray    # applies items in order
```

- [ ] **Step 1: Write the failing test — `tests/test_effect_stack.py`**

```python
import numpy as np
import pytest
from ditherzam.effects.stack import EffectStack


def img():
    return np.zeros((8, 8, 3), np.uint8)


def test_new_stack_is_empty():
    s = EffectStack()
    assert s.items == []


def test_add_appends_name_and_params():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=1)
    s.add("Blur", radius=0)
    assert [name for name, _ in s.items] == ["Chromatic Aberration", "Blur"]
    assert s.items[0][1] == {"shift": 1}
    assert s.items[1][1] == {"radius": 0}


def test_apply_runs_in_order_shape_preserved():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=1)
    s.add("Blur", radius=0)                       # identity
    out = s.apply(img())
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_apply_actually_uses_the_effect():
    x = np.zeros((4, 8, 3), np.uint8)
    x[:, 3, 0] = 255
    s = EffectStack()
    s.add("Chromatic Aberration", shift=2)
    out = s.apply(x)
    assert out[:, 5, 0].max() == 255             # red shifted right by the stack


def test_empty_stack_apply_is_identity():
    x = np.random.RandomState(1).randint(0, 256, (8, 8, 3), np.uint8)
    np.testing.assert_array_equal(EffectStack().apply(x), x)


def test_move_reorders():
    s = EffectStack()
    s.add("Blur", radius=0)
    s.add("Sharpen", amount=1)
    s.move(1, 0)
    assert [name for name, _ in s.items] == ["Sharpen", "Blur"]


def test_remove_deletes_index():
    s = EffectStack()
    s.add("Blur", radius=0)
    s.add("Sharpen", amount=1)
    s.remove(0)
    assert [name for name, _ in s.items] == ["Sharpen"]


def test_unknown_effect_raises_keyerror():
    with pytest.raises(KeyError):
        EffectStack().add("Nope")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_effect_stack.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.effects.stack'`

- [ ] **Step 3: Implement `ditherzam/effects/stack.py`**

```python
from __future__ import annotations

import numpy as np

from .post import EFFECTS


class EffectStack:
    """Ordered, mutable list of (effect_name, params) applied left-to-right."""

    def __init__(self) -> None:
        self.items: list[tuple[str, dict]] = []

    def add(self, name: str, **params) -> None:
        if name not in EFFECTS:
            raise KeyError(name)
        self.items.append((name, params))

    def move(self, index: int, new_index: int) -> None:
        item = self.items.pop(index)
        self.items.insert(new_index, item)

    def remove(self, index: int) -> None:
        self.items.pop(index)

    def apply(self, rgb_u8: np.ndarray) -> np.ndarray:
        out = rgb_u8
        for name, params in self.items:
            out = EFFECTS[name](out, **params)
        return out
```

- [ ] **Step 4: Run — expect PASS**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_effect_stack.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/effects/stack.py tests/test_effect_stack.py
git commit -m "feat(effects): reorderable EffectStack (add/move/remove/apply)"
```

---

### Task 4.3: `RenderPipeline` + `RenderSettings` (compose everything)

**Files:** Create `ditherzam/render.py`; Test `tests/test_render.py`
**Interfaces:**

```python
@dataclass
class RenderSettings:
    contrast=50; midtones=50; highlights=50; blur=50; luminance_threshold=50
    invert=False; saturation=50; style="None"; scale=5; preview_disabled=False; params={}

class RenderPipeline:
    STAGE_ORDER: tuple[str, ...]                       # canonical order (added in 4.4)
    def __init__(self, registry, color_engine=None, effect_stack=None)
    def render(self, base_gray_f32, settings, temporal_field=None) -> np.ndarray  # uint8 HxWx3
```

**Render order (spec §8.1 + color/effects insert, FROZEN):**
`contrast → midtones → highlights → blur → dither(downscale→kernel→upscale) →
color(map) → saturation → effects stack → invert(last)`

- [ ] **Step 1: Write the failing test — `tests/test_render.py`**

```python
import numpy as np
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.effects.stack import EffectStack


def test_render_grayscale_none_returns_rgb():
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 100.0, np.float32), RenderSettings())
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_render_none_style_is_gray_broadcast_to_rgb():
    # style "None", no color engine, blur uniform -> flat 100 across all 3 channels
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 100.0, np.float32),
                   RenderSettings(style="None"))
    assert out.shape == (8, 8, 3)
    assert np.all(out[..., 0] == out[..., 1]) and np.all(out[..., 1] == out[..., 2])


def test_render_applies_color_palette():
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)
    out = p.render(np.tile(np.linspace(0, 255, 8, np.float32), (8, 1)),
                   RenderSettings(style="Floyd-Steinberg", scale=1))
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert uniq <= {(0, 0, 0), (255, 255, 255)}


def test_render_effects_applied():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=1)
    p = RenderPipeline(registry, effect_stack=s)
    out = p.render(np.full((8, 8), 128.0, np.float32), RenderSettings())
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_render_invert_is_last():
    # black base, style None, invert True -> pure white output
    p = RenderPipeline(registry)
    out = p.render(np.zeros((8, 8), np.float32), RenderSettings(invert=True))
    assert np.all(out == 255)


def test_render_preview_disabled_skips_dither():
    # preview_disabled makes apply_dither a passthrough; uniform stays uniform gray
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 77.0, np.float32),
                   RenderSettings(style="Floyd-Steinberg", scale=1, preview_disabled=True))
    assert np.all(out == 77)


def test_render_accepts_temporal_field_kwarg():
    # temporal_field defaults to None and is forwarded to apply_dither(threshold_field=...)
    p = RenderPipeline(registry)
    out = p.render(np.full((8, 8), 100.0, np.float32), RenderSettings(), temporal_field=None)
    assert out.shape == (8, 8, 3)
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.render'`

- [ ] **Step 3: Implement `ditherzam/render.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adjustments import (
    apply_contrast, apply_midtones, apply_highlights, apply_blur,
    apply_saturation, apply_invert,
)
from .dithering.pipeline import apply_dither
from .imaging import clamp_u8


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
    """Compose adjustments -> dither -> color -> saturation -> effects -> invert."""

    def __init__(self, registry, color_engine=None, effect_stack=None) -> None:
        self.registry = registry
        self.color_engine = color_engine
        self.effect_stack = effect_stack

    def render(self, base_gray_f32, settings: RenderSettings,
               temporal_field=None) -> np.ndarray:
        g = np.asarray(base_gray_f32, dtype=np.float32)

        # 1-4: tonal adjustments (grayscale float32, 0..255)
        g = apply_contrast(g, settings.contrast)
        g = apply_midtones(g, settings.midtones)
        g = apply_highlights(g, settings.highlights)
        g = apply_blur(g, settings.blur)

        # 5: dither (downscale -> kernel -> upscale); temporal field forwarded
        d = apply_dither(
            g,
            style=settings.style,
            scale=settings.scale,
            luminance_threshold=settings.luminance_threshold,
            params=settings.params,
            registry=self.registry,
            preview_disabled=settings.preview_disabled,
            threshold_field=temporal_field,
        )

        # 6: color — palette map, or broadcast grayscale to RGB
        if self.color_engine is not None:
            rgb = self.color_engine.map(d).astype(np.float32)
        else:
            rgb = np.repeat(np.asarray(d, np.float32)[..., None], 3, axis=2)

        # 7: saturation (RGB float32) then clamp to uint8
        rgb = apply_saturation(rgb, settings.saturation)
        rgb_u8 = clamp_u8(rgb)

        # 8: effects stack (RGB uint8)
        if self.effect_stack is not None:
            rgb_u8 = self.effect_stack.apply(rgb_u8)

        # 9: invert LAST (on RGB)
        if settings.invert:
            rgb_u8 = clamp_u8(apply_invert(rgb_u8.astype(np.float32), True))

        return np.asarray(rgb_u8, np.uint8)
```

- [ ] **Step 4: Run — expect PASS**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_render.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/render.py tests/test_render.py
git commit -m "feat(render): RenderPipeline composing adjustments/dither/color/saturation/effects/invert"
```

---

### Task 4.4: Render-ORDER contract test (lock the fixed stage order)

**Files:** Modify `ditherzam/render.py` (add `STAGE_ORDER`); Test `tests/test_render_order.py`
**Interfaces:** Produces `RenderPipeline.STAGE_ORDER: tuple[str, ...]` — the single
source of truth for the pipeline order. The test spies on every stage (adjustments,
`apply_dither`, `ColorEngine.map`, `apply_saturation`, `EffectStack.apply`,
`apply_invert`) and asserts the recorded call order equals both `STAGE_ORDER` and
the literal frozen sequence.

- [ ] **Step 1: Write the failing test — `tests/test_render_order.py`**

```python
import numpy as np
from unittest import mock

import ditherzam.render as R
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.effects.stack import EffectStack

FROZEN_ORDER = (
    "contrast", "midtones", "highlights", "blur", "dither",
    "color", "saturation", "effects", "invert",
)


def _recorder(calls, tag):
    def _rec(*args, **kwargs):
        calls.append(tag)
        return mock.DEFAULT           # fall through to wraps -> real function result
    return _rec


def test_stage_order_constant_is_frozen():
    # The declared contract must match the spec (§8.1 + color/effects/saturation insert).
    assert RenderPipeline.STAGE_ORDER == FROZEN_ORDER


def test_render_calls_stages_in_frozen_order():
    calls: list[str] = []

    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    stack = EffectStack()
    stack.add("Chromatic Aberration", shift=1)
    p = RenderPipeline(registry, color_engine=eng, effect_stack=stack)

    base = np.tile(np.linspace(0, 255, 8, np.float32), (8, 1))
    settings = RenderSettings(
        style="Floyd-Steinberg", scale=1, invert=True,
        contrast=60, midtones=60, highlights=60, blur=60, saturation=60,
    )

    with mock.patch.object(R, "apply_contrast", side_effect=_recorder(calls, "contrast"), wraps=R.apply_contrast), \
         mock.patch.object(R, "apply_midtones", side_effect=_recorder(calls, "midtones"), wraps=R.apply_midtones), \
         mock.patch.object(R, "apply_highlights", side_effect=_recorder(calls, "highlights"), wraps=R.apply_highlights), \
         mock.patch.object(R, "apply_blur", side_effect=_recorder(calls, "blur"), wraps=R.apply_blur), \
         mock.patch.object(R, "apply_dither", side_effect=_recorder(calls, "dither"), wraps=R.apply_dither), \
         mock.patch.object(R, "apply_saturation", side_effect=_recorder(calls, "saturation"), wraps=R.apply_saturation), \
         mock.patch.object(R, "apply_invert", side_effect=_recorder(calls, "invert"), wraps=R.apply_invert), \
         mock.patch.object(eng, "map", side_effect=_recorder(calls, "color"), wraps=eng.map), \
         mock.patch.object(stack, "apply", side_effect=_recorder(calls, "effects"), wraps=stack.apply):
        out = p.render(base, settings)

    assert calls == list(FROZEN_ORDER)
    assert tuple(calls) == RenderPipeline.STAGE_ORDER
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_invert_is_strictly_last():
    calls: list[str] = []
    p = RenderPipeline(registry)
    settings = RenderSettings(style="None", invert=True)
    with mock.patch.object(R, "apply_invert", side_effect=_recorder(calls, "invert"), wraps=R.apply_invert), \
         mock.patch.object(R, "apply_saturation", side_effect=_recorder(calls, "saturation"), wraps=R.apply_saturation):
        p.render(np.zeros((8, 8), np.float32), settings)
    assert calls[-1] == "invert"                     # invert runs after saturation


def test_color_before_saturation_before_effects():
    calls: list[str] = []
    eng = ColorEngine(Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]]), "nearest")
    stack = EffectStack()
    stack.add("Blur", radius=0)
    p = RenderPipeline(registry, color_engine=eng, effect_stack=stack)
    with mock.patch.object(eng, "map", side_effect=_recorder(calls, "color"), wraps=eng.map), \
         mock.patch.object(R, "apply_saturation", side_effect=_recorder(calls, "saturation"), wraps=R.apply_saturation), \
         mock.patch.object(stack, "apply", side_effect=_recorder(calls, "effects"), wraps=stack.apply):
        p.render(np.full((8, 8), 100.0, np.float32), RenderSettings(style="None"))
    assert calls == ["color", "saturation", "effects"]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_render_order.py -v`
Expected: FAIL — `AttributeError: type object 'RenderPipeline' has no attribute 'STAGE_ORDER'`
(the `test_stage_order_constant_is_frozen` and the two order tests error on the
missing `STAGE_ORDER` constant).

- [ ] **Step 3: Add the `STAGE_ORDER` contract constant to `ditherzam/render.py`**

Insert the class attribute at the top of `class RenderPipeline` (immediately after
the docstring, before `__init__`):

```python
class RenderPipeline:
    """Compose adjustments -> dither -> color -> saturation -> effects -> invert."""

    # FROZEN stage order (spec §8.1 + color/saturation/effects insert). The
    # render() body MUST call stages in exactly this sequence; test_render_order
    # spies on each stage and asserts the recorded call order equals this tuple.
    STAGE_ORDER: tuple[str, ...] = (
        "contrast", "midtones", "highlights", "blur", "dither",
        "color", "saturation", "effects", "invert",
    )

    def __init__(self, registry, color_engine=None, effect_stack=None) -> None:
        self.registry = registry
        self.color_engine = color_engine
        self.effect_stack = effect_stack
```

> No change to `render()` is needed — Task 4.3 already calls the stages in
> `STAGE_ORDER`. This step only publishes the order as an inspectable contract.

- [ ] **Step 4: Run — expect PASS**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_render_order.py -v`
Expected: `4 passed`

- [ ] **Step 5: Run the whole phase suite — expect PASS**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_effects_post.py tests/test_effect_stack.py tests/test_render.py tests/test_render_order.py -v`
Expected: `31 passed`

- [ ] **Step 6: Commit**

```bash
git add ditherzam/render.py tests/test_render_order.py
git commit -m "test(render): lock fixed stage order via STAGE_ORDER contract + spy test"
```

---

## Phase 4 Definition of Done (checklist)

- [ ] `ditherzam/effects/post.py` exports all 5 effects and the `EFFECTS` dict whose
      keys are exactly `{"Blur","Sharpen","Chromatic Aberration","JPEG Glitch","Epsilon Glow"}`.
- [ ] Each effect preserves `uint8[H,W,3]` shape/dtype; `blur(x,0)` and
      `sharpen(uniform,·)` are identity; chromatic aberration rolls R right / B left and
      leaves green untouched; `jpeg_glitch` clamps quality and degrades; `epsilon_glow`
      brightens and clips to 255.
- [ ] `EffectStack` starts empty; `add` appends `(name, params)` and raises `KeyError`
      for unknown effects; `move`/`remove` mutate order; `apply` runs items in order and
      is identity when empty.
- [ ] `RenderSettings` dataclass matches the frozen defaults
      (contrast/midtones/highlights/blur/luminance_threshold=50, invert=False,
      saturation=50, style="None", scale=5, preview_disabled=False, params={}).
- [ ] `RenderPipeline.__init__(registry, color_engine=None, effect_stack=None)` and
      `.render(base_gray_f32, settings, temporal_field=None) -> uint8[H,W,3]`.
- [ ] `RenderPipeline.STAGE_ORDER == ("contrast","midtones","highlights","blur",
      "dither","color","saturation","effects","invert")` and the spy test proves the
      live call order matches it, with **invert strictly last**.
- [ ] `temporal_field` defaults to `None` and is forwarded to
      `apply_dither(..., threshold_field=temporal_field)` (Phase 8 needs no change here).
- [ ] `NUMBA_DISABLE_JIT=1 pytest` for all four test modules is green (31 passed).
- [ ] No PySide6/Qt import anywhere in `effects/` or `render.py`.

## Self-Review

**Spec coverage:**
- §17.4 stackable/reorderable effects (Epsilon Glow, Chromatic Aberration, JPEG
  Glitch, Blur/Sharpen) → Tasks 4.1 (functions) + 4.2 (`EffectStack` add/move/remove).
- §17.4 saturation as the color-aware adjustment → wired at stage 7 of `render()`.
- §8.1 effect order (contrast → midtones → highlights → blur → dither → invert)
  extended with the 6.0 color+saturation+effects inserts → Task 4.3 `render()` body,
  enforced by Task 4.4 `STAGE_ORDER` + spy test.
- §8.2 dither is `downscale→kernel→upscale` → delegated unchanged to Phase 1
  `apply_dither`; `render()` never re-implements pixelation.

**Frozen-contract / type consistency:**
- `EffectStack.items: list[tuple[str, dict]]`, `add(name, **params)`,
  `move(i, j)`, `remove(i)`, `apply(rgb_u8) -> np.ndarray` — verbatim.
- `RenderPipeline.__init__(registry, color_engine=None, effect_stack=None)` and
  `render(base_gray_f32, settings, temporal_field=None) -> uint8 HxWx3` — verbatim.
- Consumes `apply_contrast/midtones/highlights/blur/saturation/invert`,
  `apply_dither(..., threshold_field=...)`, `ColorEngine.map`, `EffectStack.apply`,
  `clamp_u8` at their exact frozen signatures — no re-declaration or drift.

**Placeholder scan:** no `TODO`, no `pass`-only bodies, no `...` — every code step
is complete, runnable, and covered by an assertion.

**Clean-room:** only public-domain techniques (Gaussian blur, unsharp mask, JPEG
round-trip via Pillow, channel roll, additive bloom). No third-party app code,
strings, URLs, or binaries.
