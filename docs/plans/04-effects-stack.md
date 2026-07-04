# Phase 4 — Effects Stack & Render Pipeline — Implementation Plan

> REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Read
> [`00-ROADMAP.md`](00-ROADMAP.md); complete Phases 1 & 3.

**Goal:** A reorderable, stackable post-processing effects system (Epsilon Glow,
Chromatic Aberration, JPEG Glitch, Blur, Sharpen) plus the top-level
`RenderPipeline` that composes adjustments → dither → color → effects → invert
into one deterministic call.

**Architecture:** Each effect is a pure `f(rgb_u8, **params) -> rgb_u8`
registered by name. `EffectStack` holds an ordered list of `(name, params)` and
applies them in order; add/move/remove mutate the order. `RenderPipeline` wires
the config + registry + color engine + stack into `render(base_gray_f32)`.

**Tech Stack:** NumPy · Pillow (JPEG round-trip) · pytest.

## Global Constraints
See roadmap. Effects operate on `uint8[H,W,3]` RGB and return the same. Qt-free.

---

## File structure (this phase)

- Create `ditherzam/effects/__init__.py`, `stack.py`, `post.py`
- Create `ditherzam/render.py`
- Tests: `tests/test_effects_post.py`, `tests/test_effect_stack.py`, `tests/test_render.py`

---

### Task 1: Post-effect functions

**Files:** Create `ditherzam/effects/post.py`; Test `tests/test_effects_post.py`
**Interfaces:** Produces an `EFFECTS: dict[str, Callable]` and functions:
```python
def blur(rgb_u8, radius: float) -> np.ndarray
def sharpen(rgb_u8, amount: float) -> np.ndarray
def chromatic_aberration(rgb_u8, shift: int) -> np.ndarray
def jpeg_glitch(rgb_u8, quality: int) -> np.ndarray
def epsilon_glow(rgb_u8, radius: float, strength: float) -> np.ndarray
```

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.effects.post import (EFFECTS, blur, sharpen,
    chromatic_aberration, jpeg_glitch, epsilon_glow)

def img(): return np.random.RandomState(0).randint(0,256,(16,16,3),np.uint8)

def test_all_effects_registered():
    for k in ("Blur","Sharpen","Chromatic Aberration","JPEG Glitch","Epsilon Glow"):
        assert k in EFFECTS

def test_shape_and_dtype_preserved():
    x = img()
    for fn, kw in [(blur,{"radius":2}),(sharpen,{"amount":1.5}),
                   (chromatic_aberration,{"shift":2}),(jpeg_glitch,{"quality":10}),
                   (epsilon_glow,{"radius":3,"strength":0.5})]:
        out = fn(x, **kw)
        assert out.shape == x.shape and out.dtype == np.uint8

def test_chromatic_aberration_shifts_red_right():
    x = np.zeros((4,8,3), np.uint8); x[:,3,0] = 255      # red column at x=3
    out = chromatic_aberration(x, shift=2)
    assert out[:,5,0].max() == 255                        # red moved +2

def test_blur_zero_identity():
    x = img(); np.testing.assert_array_equal(blur(x,0), x)
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement `post.py`**

```python
from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageFilter


def blur(rgb_u8, radius: float):
    if radius <= 0:
        return rgb_u8
    return np.array(Image.fromarray(rgb_u8).filter(ImageFilter.GaussianBlur(radius)), np.uint8)


def sharpen(rgb_u8, amount: float):
    pil = Image.fromarray(rgb_u8)
    blurred = pil.filter(ImageFilter.GaussianBlur(2))
    a = np.asarray(pil, np.float32); b = np.asarray(blurred, np.float32)
    return np.clip(a + (a - b) * amount, 0, 255).astype(np.uint8)


def chromatic_aberration(rgb_u8, shift: int):
    out = rgb_u8.copy()
    out[..., 0] = np.roll(rgb_u8[..., 0], shift, axis=1)      # R right
    out[..., 2] = np.roll(rgb_u8[..., 2], -shift, axis=1)     # B left
    return out


def jpeg_glitch(rgb_u8, quality: int):
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, format="JPEG", quality=int(max(1, min(100, quality))))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), np.uint8)


def epsilon_glow(rgb_u8, radius: float, strength: float):
    pil = Image.fromarray(rgb_u8)
    glow = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), np.float32)
    base = np.asarray(pil, np.float32)
    return np.clip(base + glow * strength, 0, 255).astype(np.uint8)


EFFECTS = {
    "Blur": blur, "Sharpen": sharpen, "Chromatic Aberration": chromatic_aberration,
    "JPEG Glitch": jpeg_glitch, "Epsilon Glow": epsilon_glow,
}
```

- [ ] **Step 4: Pass. Step 5: Commit** `feat(effects): post-processing effect functions`

---

### Task 2: `EffectStack` (add / move / remove / apply)

**Files:** Create `ditherzam/effects/stack.py`; Test `tests/test_effect_stack.py`
**Interfaces:** per roadmap contract.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.effects.stack import EffectStack

def img(): return np.zeros((8,8,3), np.uint8)

def test_add_and_apply_in_order():
    s = EffectStack()
    s.add("Chromatic Aberration", shift=1)
    s.add("Blur", radius=0)              # identity
    out = s.apply(img())
    assert out.shape == (8,8,3)
    assert [e[0] for e in s.items] == ["Chromatic Aberration", "Blur"]

def test_move_reorders():
    s = EffectStack(); s.add("Blur", radius=0); s.add("Sharpen", amount=1)
    s.move(1, 0)
    assert [e[0] for e in s.items] == ["Sharpen", "Blur"]

def test_remove():
    s = EffectStack(); s.add("Blur", radius=0); s.add("Sharpen", amount=1)
    s.remove(0)
    assert [e[0] for e in s.items] == ["Sharpen"]

def test_unknown_effect_raises():
    import pytest
    with pytest.raises(KeyError):
        EffectStack().add("Nope")
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement**

```python
from __future__ import annotations
import numpy as np
from .post import EFFECTS


class EffectStack:
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

- [ ] **Step 4: Pass. Step 5: Commit** `feat(effects): reorderable EffectStack`

---

### Task 3: `RenderPipeline` (compose everything)

**Files:** Create `ditherzam/render.py`; Test `tests/test_render.py`
**Interfaces:**
```python
@dataclass
class RenderSettings:
    contrast: float = 50; midtones: float = 50; highlights: float = 50
    blur: float = 50; luminance_threshold: float = 50; invert: bool = False
    saturation: float = 50
    style: str = "None"; scale: int = 5; preview_disabled: bool = False
    params: dict = field(default_factory=dict)

class RenderPipeline:
    def __init__(self, registry, color_engine=None, effect_stack=None): ...
    def render(self, base_gray_f32, settings: RenderSettings) -> np.ndarray:  # uint8 HxWx3
```

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.effects.stack import EffectStack

def test_render_grayscale_none_returns_rgb():
    p = RenderPipeline(registry)
    out = p.render(np.full((8,8),100.0,np.float32), RenderSettings())
    assert out.shape == (8,8,3) and out.dtype == np.uint8

def test_render_applies_color_palette():
    eng = ColorEngine(Palette.from_list("duo",[[0,0,0],[255,255,255]]), "nearest")
    p = RenderPipeline(registry, color_engine=eng)
    out = p.render(np.tile(np.linspace(0,255,8,np.float32),(8,1)),
                   RenderSettings(style="Floyd-Steinberg", scale=1))
    uniq = {tuple(c) for c in out.reshape(-1,3).tolist()}
    assert uniq <= {(0,0,0),(255,255,255)}

def test_render_effects_applied():
    s = EffectStack(); s.add("Chromatic Aberration", shift=1)
    p = RenderPipeline(registry, effect_stack=s)
    out = p.render(np.full((8,8),128.0,np.float32), RenderSettings())
    assert out.shape == (8,8,3)
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement `render.py`** — order (spec §8.1 + color/effects insert):
  1. `apply_contrast → apply_midtones → apply_highlights → apply_blur` (grayscale float32)
  2. `apply_dither(...)` (grayscale float32, downscale/upscale)
  3. color: if `color_engine`: `rgb = color_engine.map(dithered)` else `rgb = stack3(dithered)`
  4. `apply_saturation` (on rgb float32) then `clamp_u8`
  5. `effect_stack.apply(rgb_u8)` if present
  6. `apply_invert` last (on rgb) if `settings.invert`
- [ ] **Step 4: Pass. Step 5: Commit** `feat(render): RenderPipeline composing adjustments/dither/color/effects`

---

## Phase 4 Self-Review
- [ ] All 5 effects registered, shape/dtype preserved, identity cases correct.
- [ ] Stack add/move/remove/apply order correct; unknown effect raises.
- [ ] `RenderPipeline.render` returns `uint8[H,W,3]`; palette-constrained output; effect + invert applied last.
- [ ] Pipeline order matches spec §8.1 with color/effects inserted; Qt-free.
