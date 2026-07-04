# Phase 3 — Color Engine — Implementation Plan

> REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Read
> [`00-ROADMAP.md`](00-ROADMAP.md) and complete Phase 1. (Phase 2 optional but recommended.)

**Goal:** A headless color subsystem — `Palette` (built-in libraries, image
extraction, "source/complete" retention), `apply_saturation`, and `ColorEngine`
that maps a grayscale/RGB float32 image to an RGB uint8 image using nearest,
ordered, or error-diffused color quantization. This is the 6.0 "color engine"
(spec §17.2) the 3.0.2 build lacked.

**Architecture:** Palettes are `float32[K,3]` RGB arrays. Extraction uses
median-cut (dependency-free) with optional k-means refine. Color mapping runs on
the *dithered tone bands* or directly on RGB, inserted into the render pipeline
after tone/dither. Fully Qt-free and unit-tested against reference palettes.

**Tech Stack:** NumPy · (Numba for the map loop) · pytest. No sklearn.

## Global Constraints
See roadmap. Palettes are RGB `float32` in `0..255`; `ColorEngine.map` returns
`uint8[H,W,3]`. Palette files are YAML: `{name, colors: [[r,g,b], ...]}` stored in
`user_data_dir/palettes/*.yaml` plus a bundled `ditherzam/color/builtin/*.yaml`.

---

## File structure (this phase)

- Create `ditherzam/color/__init__.py`
- Create `ditherzam/color/palette.py` — `Palette`, load/save, built-ins, `extract_palette`
- Create `ditherzam/color/engine.py` — `ColorEngine`, `nearest_indices`, color-dither modes
- Modify `ditherzam/adjustments.py` — add `apply_saturation`
- Create `ditherzam/color/builtin/{grayscale,gameboy,cga,pico8,sepia}.yaml`
- Tests: `tests/test_palette.py`, `tests/test_color_engine.py`, `tests/test_saturation.py`

---

### Task 1: `apply_saturation`

**Files:** Modify `ditherzam/adjustments.py`; Test `tests/test_saturation.py`
**Interfaces:** Produces `apply_saturation(rgb_f32, value) -> float32[H,W,3]`
(`value` 0..100; 50 = identity; 0 = grayscale; 100 = 2× saturation).

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.adjustments import apply_saturation

def rgb(r,g,b): return np.array([[[r,g,b]]], dtype=np.float32)

def test_saturation_50_identity():
    c = rgb(200, 50, 30)
    np.testing.assert_allclose(apply_saturation(c, 50), c, atol=1e-3)

def test_saturation_0_is_gray():
    c = rgb(200, 50, 30)
    out = apply_saturation(c, 0)
    assert abs(out[0,0,0]-out[0,0,1]) < 1e-3 and abs(out[0,0,1]-out[0,0,2]) < 1e-3
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement**

```python
def apply_saturation(rgb: np.ndarray, value: float) -> np.ndarray:
    factor = value / 50.0                      # 0..2, 50->1
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
    return (lum + (rgb - lum) * factor).astype(np.float32)
```

- [ ] **Step 4: Pass. Step 5: Commit** `feat(color): saturation adjustment`

---

### Task 2: `Palette` — model, load/save, built-ins

**Files:** Create `ditherzam/color/__init__.py`, `palette.py`, `builtin/*.yaml`;
Test `tests/test_palette.py`
**Interfaces:**
```python
@dataclass
class Palette:
    name: str
    colors: np.ndarray            # float32[K,3], 0..255
    @classmethod
    def from_list(cls, name, rgb_list) -> "Palette"
    def to_yaml(self, path) -> None
    @classmethod
    def load(cls, path) -> "Palette"
def builtin_palettes() -> dict[str, Palette]
```

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.color.palette import Palette, builtin_palettes

def test_from_list_and_shape():
    p = Palette.from_list("duo", [[0,0,0],[255,255,255]])
    assert p.colors.shape == (2,3) and p.colors.dtype == np.float32

def test_roundtrip_yaml(tmp_path):
    p = Palette.from_list("t", [[10,20,30],[40,50,60]])
    f = tmp_path/"t.yaml"; p.to_yaml(f)
    q = Palette.load(f)
    assert q.name == "t"
    np.testing.assert_array_equal(q.colors, p.colors)

def test_builtins_present():
    b = builtin_palettes()
    assert "gameboy" in b and b["gameboy"].colors.shape[1] == 3
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement `palette.py`** (dataclass + PyYAML load/save; `builtin_palettes()`
  scans `ditherzam/color/builtin/*.yaml`). Ship at least: `grayscale` (2), `gameboy` (4),
  `cga` (4/16), `pico8` (16), `sepia` (4).
- [ ] **Step 4: Pass. Step 5: Commit** `feat(color): Palette model, YAML IO, built-ins`

---

### Task 3: Palette extraction (median-cut) — the "source/automatic" palette

**Files:** Modify `palette.py`; Test extend `tests/test_palette.py`
**Interfaces:** Produces `extract_palette(rgb_u8, k=16, name="source") -> Palette`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.color.palette import extract_palette

def test_extract_k_colors():
    img = np.random.RandomState(0).randint(0,256,(32,32,3),dtype=np.uint8)
    p = extract_palette(img, k=8)
    assert p.colors.shape == (8,3)

def test_extract_two_color_image():
    img = np.zeros((10,10,3), np.uint8); img[:, :5] = [255,0,0]; img[:, 5:] = [0,0,255]
    p = extract_palette(img, k=2)
    got = {tuple(map(int, c)) for c in np.round(p.colors)}
    # both dominant colors should be represented (median-cut buckets)
    assert any(r>200 for (r,g,b) in got) and any(b>200 for (r,g,b) in got)
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement median-cut** (recursively split the box along its longest
  RGB axis at the median until `k` buckets; palette color = bucket mean). The
  **"complete"/source** option = `extract_palette(img, k=<large>)` to maximize
  color retention; expose `source_palette(img, completeness: float) -> Palette`
  mapping `completeness∈[0,1] → k∈[2, 256]`.
- [ ] **Step 4: Pass. Step 5: Commit** `feat(color): median-cut extraction + source palette`

---

### Task 4: `ColorEngine.map` — nearest / ordered / diffused color

**Files:** Create `ditherzam/color/engine.py`; Test `tests/test_color_engine.py`
**Interfaces:**
```python
class ColorEngine:
    def __init__(self, palette: Palette, mode: str = "nearest"): ...
    palette: Palette
    mode: str            # "off"|"nearest"|"ordered"|"diffused"
    def map(self, img_f32) -> np.ndarray:   # gray HxW or rgb HxWx3 -> uint8 HxWx3
```

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine

DUO = Palette.from_list("duo", [[0,0,0],[255,255,255]])

def test_off_mode_passthrough_to_rgb():
    eng = ColorEngine(DUO, mode="off")
    gray = np.full((4,4), 100.0, np.float32)
    out = eng.map(gray)
    assert out.shape == (4,4,3) and out.dtype == np.uint8

def test_nearest_snaps_to_palette():
    eng = ColorEngine(DUO, mode="nearest")
    gray = np.array([[10.0, 240.0]], np.float32)
    out = eng.map(gray)
    assert out[0,0].tolist() == [0,0,0]
    assert out[0,1].tolist() == [255,255,255]

def test_output_only_contains_palette_colors():
    pal = Palette.from_list("t", [[0,0,0],[128,0,0],[0,128,0],[255,255,255]])
    eng = ColorEngine(pal, mode="nearest")
    img = np.random.RandomState(1).randint(0,256,(8,8,3)).astype(np.float32)
    out = eng.map(img)
    uniq = {tuple(c) for c in out.reshape(-1,3).tolist()}
    allowed = {tuple(map(int,c)) for c in pal.colors}
    assert uniq <= allowed
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement**
  - Promote gray→RGB by stacking 3 channels.
  - `nearest_indices(rgb, palette)`: argmin squared-distance (Numba loop or
    vectorized broadcasting `((rgb[:,:,None,:]-pal)**2).sum(-1).argmin(-1)`).
  - `mode="nearest"`: index → palette color.
  - `mode="ordered"`: add a Bayer threshold offset per channel before snapping.
  - `mode="diffused"`: Floyd–Steinberg in RGB space against palette (carry vector error).
  - `mode="off"`: return `clamp_u8` of the RGB directly.
- [ ] **Step 4: Pass. Step 5: Commit** `feat(color): ColorEngine map (nearest/ordered/diffused)`

---

### Task 5: Lock + shuffle palette editing

**Files:** Modify `palette.py`; Test extend `tests/test_palette.py`
**Interfaces:** `Palette.shuffle(locked: set[int], rng) -> Palette` — randomizes
every swatch whose index is not in `locked`.

- [ ] **Step 1** failing test: locked index unchanged, others changed, shape same.
- [ ] **Step 2** fail. **Step 3** implement (copy colors, replace unlocked rows with
  `rng.integers(0,256,3)`). **Step 4** pass. **Step 5** commit
  `feat(color): swatch lock + shuffle`.

---

## Phase 3 Self-Review
- [ ] `apply_saturation` 50=identity, 0=gray — verified.
- [ ] `ColorEngine.map` output only ever contains palette colors (nearest/ordered/diffused).
- [ ] Extraction returns exactly `k` colors; "source/complete" maps completeness→k.
- [ ] Palettes round-trip through YAML; built-ins load.
- [ ] Lock+shuffle preserves locked swatches.
- [ ] Still Qt-free.
