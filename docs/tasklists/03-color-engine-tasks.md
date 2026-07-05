# Phase 3 — Color Engine — Completion Task List

> Execution-ready, TDD, red → green → refactor. Every code step is complete real
> code — no placeholders, no `TODO`. Each task ends with a `git commit` checkbox.
> Honor the **FROZEN CONTRACTS** verbatim: `apply_saturation`, `Palette`,
> `builtin_palettes`, `extract_palette`, `ColorEngine` with `.map`. Core stays
> **Qt-free** (no PySide6 import anywhere in this phase). Clean-room: no Dither Boy
> code, strings, URLs, or binaries — only public-domain techniques (median-cut,
> Floyd–Steinberg, Bayer).

**Goal:** A headless color subsystem — `apply_saturation`, a `Palette` model with
YAML IO, five enumerated built-in palettes, median-cut `extract_palette` (plus a
"source/complete" variant), a `ColorEngine` that maps grayscale/RGB float32 to RGB
uint8 in `off` / `nearest` / `ordered` / `diffused` modes, and swatch lock+shuffle.

## Prereqs

- **Phase 1 green** (`01-foundation-and-dither-core.md`): `ditherzam` installs,
  `ditherzam/imaging.py` provides `clamp_u8`, `ditherzam/adjustments.py` exists,
  `pytest` runs. Phase 2 is *not* required for this phase.
- Python **3.12** on PATH. If `python --version` is not `3.12.x`, use the clean
  interpreter noted in the handoff:
  `C:\Users\arsha\AppData\Local\...\scratchpad\dbwork\py312\python.exe`. Prefix every
  `pytest` command below with `NUMBA_DISABLE_JIT=1` (Git Bash) or
  `$env:NUMBA_DISABLE_JIT=1;` (PowerShell). `tests/conftest.py` already sets it as a
  default, so plain `pytest` also works.
- PyYAML and NumPy are already dependencies (declared in `pyproject.toml`, Phase 1).

**New files created by this phase**

```
ditherzam/color/__init__.py
ditherzam/color/palette.py
ditherzam/color/engine.py
ditherzam/color/builtin/grayscale.yaml
ditherzam/color/builtin/gameboy.yaml
ditherzam/color/builtin/cga.yaml
ditherzam/color/builtin/pico8.yaml
ditherzam/color/builtin/sepia.yaml
tests/test_saturation.py
tests/test_palette.py
tests/test_color_engine.py
```

`ditherzam/adjustments.py` is *modified* (append `apply_saturation`).

---

### Task 3.1: `apply_saturation` (frozen contract)

**Files:**
- Modify: `ditherzam/adjustments.py`
- Test: `tests/test_saturation.py`

**Interfaces:**
- Consumes: RGB `float32[H,W,3]`, `value` in `0..100`.
- Produces: `apply_saturation(rgb, value) -> float32[H,W,3]`.
  `value/50` is the factor: `50` = identity, `0` = full grayscale, `100` = 2×
  saturation. Formula (frozen): `lum + (rgb - lum) * (value/50)`,
  `lum = 0.299 R + 0.587 G + 0.114 B`.

- [ ] **Step 1: Write failing test — `tests/test_saturation.py`**

```python
import numpy as np
from ditherzam.adjustments import apply_saturation


def rgb(r, g, b):
    return np.array([[[r, g, b]]], dtype=np.float32)


def test_saturation_50_is_identity():
    c = rgb(200, 50, 30)
    np.testing.assert_allclose(apply_saturation(c, 50), c, atol=1e-3)


def test_saturation_0_is_gray():
    c = rgb(200, 50, 30)
    out = apply_saturation(c, 0)
    assert abs(out[0, 0, 0] - out[0, 0, 1]) < 1e-3
    assert abs(out[0, 0, 1] - out[0, 0, 2]) < 1e-3
    # the gray value equals the luminance
    lum = 0.299 * 200 + 0.587 * 50 + 0.114 * 30
    np.testing.assert_allclose(out[0, 0, 0], lum, atol=1e-3)


def test_saturation_100_doubles_deviation():
    c = rgb(200, 50, 30)
    lum = 0.299 * 200 + 0.587 * 50 + 0.114 * 30
    out = apply_saturation(c, 100)
    # each channel deviation from luminance is doubled
    np.testing.assert_allclose(out[0, 0], lum + (c[0, 0] - lum) * 2.0, atol=1e-3)


def test_saturation_returns_float32():
    out = apply_saturation(rgb(10, 20, 30), 75)
    assert out.dtype == np.float32
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_saturation.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_saturation' from 'ditherzam.adjustments'`

- [ ] **Step 3: Implement — append to `ditherzam/adjustments.py`**

```python
def apply_saturation(rgb: np.ndarray, value: float) -> np.ndarray:
    """Scale color saturation about per-pixel luminance.

    value in 0..100; 50 = identity, 0 = grayscale, 100 = 2x saturation.
    """
    factor = value / 50.0
    lum = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )[..., None]
    return (lum + (rgb - lum) * factor).astype(np.float32)
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_saturation.py -v` → 4 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/adjustments.py tests/test_saturation.py
git commit -m "feat(color): saturation adjustment about luminance"
```

---

### Task 3.2: `Palette` model + YAML IO

**Files:**
- Create: `ditherzam/color/__init__.py` (empty)
- Create: `ditherzam/color/palette.py`
- Test: `tests/test_palette.py`

**Interfaces:**
```python
@dataclass
class Palette:
    name: str
    colors: np.ndarray                                   # float32[K,3], 0..255
    @classmethod
    def from_list(cls, name, rgb_list) -> "Palette"
    def to_yaml(self, path) -> None                      # {name, colors: [[r,g,b],...]}
    @classmethod
    def load(cls, path) -> "Palette"
```

- [ ] **Step 1: Write failing test — `tests/test_palette.py`**

```python
import numpy as np
import pytest
from ditherzam.color.palette import Palette


def test_from_list_shape_and_dtype():
    p = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
    assert p.name == "duo"
    assert p.colors.shape == (2, 3)
    assert p.colors.dtype == np.float32


def test_from_list_values_preserved():
    p = Palette.from_list("t", [[10, 20, 30], [40, 50, 60]])
    np.testing.assert_array_equal(p.colors, np.array([[10, 20, 30], [40, 50, 60]], np.float32))


def test_roundtrip_yaml(tmp_path):
    p = Palette.from_list("mypal", [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    f = tmp_path / "mypal.yaml"
    p.to_yaml(f)
    assert f.is_file()
    q = Palette.load(f)
    assert q.name == "mypal"
    assert q.colors.dtype == np.float32
    np.testing.assert_array_equal(q.colors, p.colors)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Palette.load(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.color'`

- [ ] **Step 3a: Create `ditherzam/color/__init__.py`** (empty file)

```python
```

- [ ] **Step 3b: Implement `ditherzam/color/palette.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Palette:
    """An RGB palette: ``colors`` is float32[K, 3] in the 0..255 range."""

    name: str
    colors: np.ndarray

    @classmethod
    def from_list(cls, name: str, rgb_list) -> "Palette":
        arr = np.asarray(rgb_list, dtype=np.float32).reshape(-1, 3)
        return cls(name=name, colors=arr)

    def to_yaml(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "colors": [[int(round(c)) for c in row] for row in self.colors.tolist()],
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Palette":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Palette file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = data.get("name", path.stem)
        return cls.from_list(name, data["colors"])
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -v` → 4 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/__init__.py ditherzam/color/palette.py tests/test_palette.py
git commit -m "feat(color): Palette model with YAML IO"
```

---

### Task 3.3: Built-in palettes (five, enumerated with exact RGB)

**Files:**
- Create: `ditherzam/color/builtin/grayscale.yaml`
- Create: `ditherzam/color/builtin/gameboy.yaml`
- Create: `ditherzam/color/builtin/cga.yaml`
- Create: `ditherzam/color/builtin/pico8.yaml`
- Create: `ditherzam/color/builtin/sepia.yaml`
- Modify: `ditherzam/color/palette.py` (add `builtin_palettes()`)
- Test: extend `tests/test_palette.py`

**Interfaces:** Produces `builtin_palettes() -> dict[str, Palette]` — scans
`ditherzam/color/builtin/*.yaml`, keyed by filename stem.

**Enumerated palettes (exact RGB triplets — do not abbreviate):**

- **grayscale** — 4 evenly-spaced tones:
  `[0,0,0]`, `[85,85,85]`, `[170,170,170]`, `[255,255,255]`
- **gameboy** — 4 classic DMG greens (dark→light):
  `[15,56,15]`, `[48,98,48]`, `[139,172,15]`, `[155,188,15]`
- **cga** — the canonical 16-color IBM CGA/EGA palette:
  `[0,0,0]`, `[0,0,170]`, `[0,170,0]`, `[0,170,170]`, `[170,0,0]`, `[170,0,170]`,
  `[170,85,0]`, `[170,170,170]`, `[85,85,85]`, `[85,85,255]`, `[85,255,85]`,
  `[85,255,255]`, `[255,85,85]`, `[255,85,255]`, `[255,255,85]`, `[255,255,255]`
- **pico8** — the official 16-color PICO-8 palette (index order 0..15):
  `[0,0,0]`, `[29,43,83]`, `[126,37,83]`, `[0,135,81]`, `[171,82,54]`, `[95,87,79]`,
  `[194,195,199]`, `[255,241,232]`, `[255,0,77]`, `[255,163,0]`, `[255,236,39]`,
  `[0,228,54]`, `[41,173,255]`, `[131,118,156]`, `[255,119,168]`, `[255,204,170]`
- **sepia** — 4-tone brown ramp (shadow→highlight):
  `[44,25,16]`, `[112,66,20]`, `[180,130,70]`, `[240,220,180]`

- [ ] **Step 1: Write failing test — append to `tests/test_palette.py`**

```python
from ditherzam.color.palette import builtin_palettes  # add to imports at top


def test_builtins_all_present():
    b = builtin_palettes()
    for name in ("grayscale", "gameboy", "cga", "pico8", "sepia"):
        assert name in b, f"missing built-in palette: {name}"


def test_builtin_counts_and_shape():
    b = builtin_palettes()
    assert b["grayscale"].colors.shape == (4, 3)
    assert b["gameboy"].colors.shape == (4, 3)
    assert b["cga"].colors.shape == (16, 3)
    assert b["pico8"].colors.shape == (16, 3)
    assert b["sepia"].colors.shape == (4, 3)
    for p in b.values():
        assert p.colors.dtype == np.float32


def test_builtin_exact_values():
    b = builtin_palettes()
    np.testing.assert_array_equal(
        b["gameboy"].colors,
        np.array([[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]], np.float32),
    )
    # PICO-8 index 8 is the signature red (#FF004D)
    np.testing.assert_array_equal(b["pico8"].colors[8], np.array([255, 0, 77], np.float32))
    # CGA index 14 is yellow
    np.testing.assert_array_equal(b["cga"].colors[14], np.array([255, 255, 85], np.float32))
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -k builtin -v`
Expected: FAIL — `ImportError: cannot import name 'builtin_palettes'`

- [ ] **Step 3a: Create `ditherzam/color/builtin/grayscale.yaml`**

```yaml
name: grayscale
colors:
  - [0, 0, 0]
  - [85, 85, 85]
  - [170, 170, 170]
  - [255, 255, 255]
```

- [ ] **Step 3b: Create `ditherzam/color/builtin/gameboy.yaml`**

```yaml
name: gameboy
colors:
  - [15, 56, 15]
  - [48, 98, 48]
  - [139, 172, 15]
  - [155, 188, 15]
```

- [ ] **Step 3c: Create `ditherzam/color/builtin/cga.yaml`**

```yaml
name: cga
colors:
  - [0, 0, 0]
  - [0, 0, 170]
  - [0, 170, 0]
  - [0, 170, 170]
  - [170, 0, 0]
  - [170, 0, 170]
  - [170, 85, 0]
  - [170, 170, 170]
  - [85, 85, 85]
  - [85, 85, 255]
  - [85, 255, 85]
  - [85, 255, 255]
  - [255, 85, 85]
  - [255, 85, 255]
  - [255, 255, 85]
  - [255, 255, 255]
```

- [ ] **Step 3d: Create `ditherzam/color/builtin/pico8.yaml`**

```yaml
name: pico8
colors:
  - [0, 0, 0]
  - [29, 43, 83]
  - [126, 37, 83]
  - [0, 135, 81]
  - [171, 82, 54]
  - [95, 87, 79]
  - [194, 195, 199]
  - [255, 241, 232]
  - [255, 0, 77]
  - [255, 163, 0]
  - [255, 236, 39]
  - [0, 228, 54]
  - [41, 173, 255]
  - [131, 118, 156]
  - [255, 119, 168]
  - [255, 204, 170]
```

- [ ] **Step 3e: Create `ditherzam/color/builtin/sepia.yaml`**

```yaml
name: sepia
colors:
  - [44, 25, 16]
  - [112, 66, 20]
  - [180, 130, 70]
  - [240, 220, 180]
```

- [ ] **Step 3f: Add `builtin_palettes()` to `ditherzam/color/palette.py`**

```python
def builtin_palettes() -> dict[str, "Palette"]:
    """Load every bundled palette from ``ditherzam/color/builtin/*.yaml``."""
    directory = Path(__file__).parent / "builtin"
    out: dict[str, Palette] = {}
    for f in sorted(directory.glob("*.yaml")):
        out[f.stem] = Palette.load(f)
    return out
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -v` → all passed (7 total so far)
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/builtin ditherzam/color/palette.py tests/test_palette.py
git commit -m "feat(color): built-in palettes (grayscale, gameboy, cga, pico8, sepia)"
```

---

### Task 3.4: `extract_palette` — median-cut

**Files:**
- Modify: `ditherzam/color/palette.py`
- Test: extend `tests/test_palette.py`

**Interfaces:** Produces `extract_palette(rgb_u8, k=16, name="source") -> Palette`.
Recursively split the color box along its longest RGB axis at the median until `k`
buckets exist; each palette entry is that bucket's mean color. `k` is clamped to a
power-of-two ceiling internally then trimmed to exactly `k` buckets.

- [ ] **Step 1: Write failing test — append to `tests/test_palette.py`**

```python
from ditherzam.color.palette import extract_palette  # add to imports at top


def test_extract_returns_k_colors():
    img = np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    p = extract_palette(img, k=8)
    assert p.colors.shape == (8, 3)
    assert p.colors.dtype == np.float32
    assert p.name == "source"


def test_extract_default_name_and_k():
    img = np.random.RandomState(3).randint(0, 256, (16, 16, 3), dtype=np.uint8)
    p = extract_palette(img)
    assert p.colors.shape == (16, 3)


def test_extract_two_color_image():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :5] = [255, 0, 0]
    img[:, 5:] = [0, 0, 255]
    p = extract_palette(img, k=2)
    got = [tuple(int(round(v)) for v in c) for c in p.colors]
    reds = [c for c in got if c[0] > 200 and c[2] < 55]
    blues = [c for c in got if c[2] > 200 and c[0] < 55]
    assert reds and blues


def test_extract_k_larger_than_unique_colors():
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :4] = [10, 10, 10]
    img[:, 4:] = [200, 200, 200]
    p = extract_palette(img, k=4)
    # still returns exactly k rows even when the image has < k distinct colors
    assert p.colors.shape == (4, 3)
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -k extract -v`
Expected: FAIL — `ImportError: cannot import name 'extract_palette'`

- [ ] **Step 3: Add median-cut to `ditherzam/color/palette.py`**

```python
def _median_cut(pixels: np.ndarray, depth: int) -> list[np.ndarray]:
    """Recursively split ``pixels`` (N,3 float) into 2**depth buckets."""
    if depth == 0 or pixels.shape[0] <= 1:
        return [pixels]
    ranges = pixels.max(axis=0) - pixels.min(axis=0)
    axis = int(np.argmax(ranges))
    order = np.argsort(pixels[:, axis], kind="stable")
    pixels = pixels[order]
    mid = pixels.shape[0] // 2
    left = _median_cut(pixels[:mid], depth - 1)
    right = _median_cut(pixels[mid:], depth - 1)
    return left + right


def extract_palette(rgb_u8: np.ndarray, k: int = 16, name: str = "source") -> "Palette":
    """Median-cut palette extraction. Returns exactly ``k`` colors."""
    k = max(1, int(k))
    pixels = np.asarray(rgb_u8, dtype=np.float32).reshape(-1, 3)
    depth = 0
    while (1 << depth) < k:
        depth += 1
    buckets = [b for b in _median_cut(pixels, depth) if b.shape[0] > 0]
    means = [b.mean(axis=0) for b in buckets]
    # normalize to exactly k rows (pad by repeating the last, or trim)
    if len(means) >= k:
        means = means[:k]
    else:
        means = means + [means[-1]] * (k - len(means))
    colors = np.asarray(means, dtype=np.float32).reshape(k, 3)
    return Palette(name=name, colors=colors)
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -v` → all passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/palette.py tests/test_palette.py
git commit -m "feat(color): median-cut palette extraction"
```

---

### Task 3.5: `source_palette` — the "complete" retention variant

**Files:**
- Modify: `ditherzam/color/palette.py`
- Test: extend `tests/test_palette.py`

**Interfaces:** Produces `source_palette(rgb_u8, completeness=1.0, name="source") -> Palette`.
`completeness ∈ [0,1]` maps to `k ∈ [2, 256]` (linear). `completeness=1.0` = the
recommended "complete" starting point (maximum color retention, k=256);
`completeness=0.0` = minimal (k=2). Delegates to `extract_palette`.

- [ ] **Step 1: Write failing test — append to `tests/test_palette.py`**

```python
from ditherzam.color.palette import source_palette  # add to imports at top


def test_source_complete_is_256():
    img = np.random.RandomState(5).randint(0, 256, (48, 48, 3), dtype=np.uint8)
    p = source_palette(img, completeness=1.0)
    assert p.colors.shape == (256, 3)


def test_source_minimal_is_2():
    img = np.random.RandomState(6).randint(0, 256, (48, 48, 3), dtype=np.uint8)
    p = source_palette(img, completeness=0.0)
    assert p.colors.shape == (2, 3)


def test_source_midpoint_between_bounds():
    img = np.random.RandomState(7).randint(0, 256, (48, 48, 3), dtype=np.uint8)
    p = source_palette(img, completeness=0.5)
    k = p.colors.shape[0]
    assert 2 < k < 256


def test_source_clamps_out_of_range():
    img = np.random.RandomState(8).randint(0, 256, (24, 24, 3), dtype=np.uint8)
    assert source_palette(img, completeness=5.0).colors.shape == (256, 3)
    assert source_palette(img, completeness=-1.0).colors.shape == (2, 3)
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -k source -v`
Expected: FAIL — `ImportError: cannot import name 'source_palette'`

- [ ] **Step 3: Add `source_palette` to `ditherzam/color/palette.py`**

```python
def source_palette(rgb_u8: np.ndarray, completeness: float = 1.0,
                   name: str = "source") -> "Palette":
    """Extract a 'source' palette; completeness in [0,1] maps to k in [2,256]."""
    c = min(1.0, max(0.0, float(completeness)))
    k = int(round(2 + c * (256 - 2)))
    return extract_palette(rgb_u8, k=k, name=name)
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -v` → all passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/palette.py tests/test_palette.py
git commit -m "feat(color): source/complete palette (completeness to k mapping)"
```

---

### Task 3.6: `ColorEngine` — `off` + `nearest`

**Files:**
- Create: `ditherzam/color/engine.py`
- Test: `tests/test_color_engine.py`

**Interfaces (frozen):**
```python
class ColorEngine:
    palette: Palette
    mode: str                                   # "off"|"nearest"|"ordered"|"diffused"
    def __init__(self, palette: Palette, mode: str = "nearest"): ...
    def map(self, gray_or_rgb_f32) -> np.ndarray   # -> uint8 HxWx3
```
Grayscale `HxW` input is promoted to RGB by stacking channels. Also produces the
module helper `nearest_indices(rgb_f32, palette_f32) -> int[H,W]`.

- [ ] **Step 1: Write failing test — `tests/test_color_engine.py`**

```python
import numpy as np
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine, nearest_indices

DUO = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
QUAD = Palette.from_list("quad", [[0, 0, 0], [128, 0, 0], [0, 128, 0], [255, 255, 255]])


def test_off_mode_passthrough_to_rgb():
    eng = ColorEngine(DUO, mode="off")
    gray = np.full((4, 4), 100.0, np.float32)
    out = eng.map(gray)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    assert np.all(out == 100)


def test_off_mode_clamps():
    eng = ColorEngine(DUO, mode="off")
    rgb = np.array([[[-20.0, 300.0, 128.0]]], np.float32)
    out = eng.map(rgb)
    assert out[0, 0].tolist() == [0, 255, 128]


def test_nearest_snaps_gray_to_palette():
    eng = ColorEngine(DUO, mode="nearest")
    gray = np.array([[10.0, 240.0]], np.float32)
    out = eng.map(gray)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255]


def test_nearest_output_only_palette_colors():
    eng = ColorEngine(QUAD, mode="nearest")
    img = np.random.RandomState(1).randint(0, 256, (8, 8, 3)).astype(np.float32)
    out = eng.map(img)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    allowed = {tuple(int(round(v)) for v in c) for c in QUAD.colors}
    assert uniq <= allowed


def test_nearest_indices_helper():
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.float32)
    rgb = np.array([[[10, 10, 10], [200, 200, 200]]], np.float32)
    idx = nearest_indices(rgb, pal)
    assert idx.tolist() == [[0, 1]]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_color_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.color.engine'`

- [ ] **Step 3: Implement `ditherzam/color/engine.py`** (off + nearest; ordered/diffused stubbed to raise so later tasks fill them)

```python
from __future__ import annotations

import numpy as np

from ..imaging import clamp_u8
from .palette import Palette


def nearest_indices(rgb_f32: np.ndarray, palette_f32: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color (squared RGB distance) per pixel."""
    diff = rgb_f32[:, :, None, :] - palette_f32[None, None, :, :]
    dist = (diff * diff).sum(axis=-1)
    return dist.argmin(axis=-1)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    return arr[..., :3].astype(np.float32)


class ColorEngine:
    def __init__(self, palette: Palette, mode: str = "nearest") -> None:
        self.palette = palette
        self.mode = mode

    def map(self, gray_or_rgb_f32: np.ndarray) -> np.ndarray:
        rgb = _to_rgb(gray_or_rgb_f32)
        if self.mode == "off":
            return clamp_u8(rgb)
        pal = self.palette.colors.astype(np.float32)
        if self.mode == "nearest":
            idx = nearest_indices(rgb, pal)
            return clamp_u8(pal[idx])
        raise ValueError(f"unknown ColorEngine mode: {self.mode!r}")
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_color_engine.py -v` → 5 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/engine.py tests/test_color_engine.py
git commit -m "feat(color): ColorEngine off + nearest modes"
```

---

### Task 3.7: `ColorEngine` — `ordered` (Bayer color dither)

**Files:**
- Modify: `ditherzam/color/engine.py`
- Test: extend `tests/test_color_engine.py`

**Interfaces:** `mode="ordered"` — add a per-pixel Bayer 4×4 offset (in `[-0.5,0.5)`
scaled by `spread = 255/max(1, K-1)`) to each channel before snapping to nearest.
On flat regions this breaks a solid fill into a stable ordered pattern.

- [ ] **Step 1: Write failing test — append to `tests/test_color_engine.py`**

```python
def test_ordered_output_only_palette_colors():
    eng = ColorEngine(QUAD, mode="ordered")
    img = np.random.RandomState(2).randint(0, 256, (8, 8, 3)).astype(np.float32)
    out = eng.map(img)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    allowed = {tuple(int(round(v)) for v in c) for c in QUAD.colors}
    assert uniq <= allowed


def test_ordered_dithers_flat_midgray():
    # nearest would make a solid fill; ordered must mix both palette colors
    eng = ColorEngine(DUO, mode="ordered")
    gray = np.full((8, 8), 127.0, np.float32)
    out = eng.map(gray)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert (0, 0, 0) in uniq
    assert (255, 255, 255) in uniq


def test_ordered_is_deterministic():
    eng = ColorEngine(DUO, mode="ordered")
    gray = np.full((8, 8), 127.0, np.float32)
    np.testing.assert_array_equal(eng.map(gray), eng.map(gray))
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_color_engine.py -k ordered -v`
Expected: FAIL — `ValueError: unknown ColorEngine mode: 'ordered'`

- [ ] **Step 3: Modify `ditherzam/color/engine.py`**

Add the Bayer helper and constant near the top (after imports):

```python
def _bayer_matrix(n: int) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    smaller = _bayer_matrix(n // 2)
    return np.block([
        [4 * smaller + 0, 4 * smaller + 2],
        [4 * smaller + 3, 4 * smaller + 1],
    ]).astype(np.float32)


# 4x4 Bayer thresholds normalized to the range [-0.5, 0.5)
_BAYER4 = (_bayer_matrix(4) + 0.5) / 16.0 - 0.5
```

Then add the `ordered` branch inside `ColorEngine.map`, before the final `raise`:

```python
        if self.mode == "ordered":
            k = pal.shape[0]
            spread = 255.0 / max(1, k - 1)
            h, w = rgb.shape[:2]
            mh, mw = _BAYER4.shape
            offset = _BAYER4[np.arange(h)[:, None] % mh,
                             np.arange(w)[None, :] % mw]
            biased = rgb + offset[:, :, None] * spread
            idx = nearest_indices(biased.astype(np.float32), pal)
            return clamp_u8(pal[idx])
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_color_engine.py -v` → 8 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/engine.py tests/test_color_engine.py
git commit -m "feat(color): ColorEngine ordered (Bayer) color dither"
```

---

### Task 3.8: `ColorEngine` — `diffused` (Floyd–Steinberg in RGB)

**Files:**
- Modify: `ditherzam/color/engine.py`
- Test: extend `tests/test_color_engine.py`

**Interfaces:** `mode="diffused"` — Floyd–Steinberg error diffusion in RGB space
against the palette (carry a per-channel vector error with the 7/3/5/1 ÷16 weights).

- [ ] **Step 1: Write failing test — append to `tests/test_color_engine.py`**

```python
def test_diffused_output_only_palette_colors():
    eng = ColorEngine(QUAD, mode="diffused")
    img = np.random.RandomState(4).randint(0, 256, (8, 8, 3)).astype(np.float32)
    out = eng.map(img)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    allowed = {tuple(int(round(v)) for v in c) for c in QUAD.colors}
    assert uniq <= allowed


def test_diffused_dithers_flat_midgray():
    eng = ColorEngine(DUO, mode="diffused")
    gray = np.full((8, 8), 127.0, np.float32)
    out = eng.map(gray)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert (0, 0, 0) in uniq and (255, 255, 255) in uniq


def test_diffused_preserves_average():
    # error diffusion of mid-gray on a black/white palette ~= 50% each
    eng = ColorEngine(DUO, mode="diffused")
    gray = np.full((16, 16), 127.0, np.float32)
    out = eng.map(gray).astype(np.float32)
    assert 100.0 < out.mean() < 155.0


def test_diffused_shape_and_dtype():
    eng = ColorEngine(QUAD, mode="diffused")
    out = eng.map(np.full((5, 6), 60.0, np.float32))
    assert out.shape == (5, 6, 3) and out.dtype == np.uint8
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_color_engine.py -k diffused -v`
Expected: FAIL — `ValueError: unknown ColorEngine mode: 'diffused'`

- [ ] **Step 3: Modify `ditherzam/color/engine.py`**

Add a module-level diffusion helper (after `nearest_indices`):

```python
def _floyd_steinberg_rgb(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    work = rgb.astype(np.float32).copy()
    out = np.empty((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            old = work[y, x].copy()
            diff = pal - old
            idx = int((diff * diff).sum(axis=1).argmin())
            new = pal[idx]
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                work[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x - 1 >= 0:
                    work[y + 1, x - 1] += err * (3.0 / 16.0)
                work[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    work[y + 1, x + 1] += err * (1.0 / 16.0)
    return out
```

Then add the `diffused` branch inside `ColorEngine.map`, before the final `raise`:

```python
        if self.mode == "diffused":
            return clamp_u8(_floyd_steinberg_rgb(rgb, pal))
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_color_engine.py -v` → 12 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/engine.py tests/test_color_engine.py
git commit -m "feat(color): ColorEngine diffused (Floyd-Steinberg RGB) color dither"
```

---

### Task 3.9: Palette swatch lock + shuffle

**Files:**
- Modify: `ditherzam/color/palette.py`
- Test: extend `tests/test_palette.py`

**Interfaces:** Produces `Palette.shuffle(locked, rng) -> Palette` — returns a new
`Palette` whose swatches at indices in `locked` are unchanged, while every other row
is replaced with a random RGB triplet drawn from `rng`. Shape and name preserved.

- [ ] **Step 1: Write failing test — append to `tests/test_palette.py`**

```python
def test_shuffle_keeps_locked_and_shape():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]])
    rng = np.random.default_rng(0)
    q = p.shuffle(locked={0, 3}, rng=rng)
    assert q.colors.shape == (4, 3)
    assert q.name == "t"
    # locked rows identical
    np.testing.assert_array_equal(q.colors[0], p.colors[0])
    np.testing.assert_array_equal(q.colors[3], p.colors[3])


def test_shuffle_changes_unlocked():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]])
    rng = np.random.default_rng(1)
    q = p.shuffle(locked={0}, rng=rng)
    # at least one unlocked row differs from the original
    changed = any(not np.array_equal(q.colors[i], p.colors[i]) for i in (1, 2, 3))
    assert changed


def test_shuffle_is_immutable_on_original():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64]])
    before = p.colors.copy()
    p.shuffle(locked=set(), rng=np.random.default_rng(2))
    np.testing.assert_array_equal(p.colors, before)


def test_shuffle_values_in_range():
    p = Palette.from_list("t", [[0, 0, 0], [64, 64, 64], [128, 128, 128]])
    q = p.shuffle(locked=set(), rng=np.random.default_rng(3))
    assert q.colors.min() >= 0 and q.colors.max() <= 255
```

- [ ] **Step 2: Run — expect FAIL**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -k shuffle -v`
Expected: FAIL — `AttributeError: 'Palette' object has no attribute 'shuffle'`

- [ ] **Step 3: Add `shuffle` method to the `Palette` dataclass in `ditherzam/color/palette.py`**

```python
    def shuffle(self, locked, rng) -> "Palette":
        """Return a copy where every swatch not in ``locked`` is randomized."""
        locked = set(locked)
        new = self.colors.copy()
        for i in range(new.shape[0]):
            if i not in locked:
                new[i] = rng.integers(0, 256, size=3).astype(np.float32)
        return Palette(name=self.name, colors=new)
```

- [ ] **Step 4: Run — expect PASS** → `NUMBA_DISABLE_JIT=1 pytest tests/test_palette.py -v` → all passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/palette.py tests/test_palette.py
git commit -m "feat(color): swatch lock + shuffle"
```

---

## Subsystem Definition of Done (checklist)

- [ ] `apply_saturation(rgb, value)` — 50 = identity, 0 = grayscale (= luminance),
  100 = 2× deviation; returns float32. (Task 3.1)
- [ ] `Palette` model: `from_list`, `to_yaml`, `load`; `colors` is `float32[K,3]`;
  YAML round-trips exactly; missing file raises `FileNotFoundError`. (Task 3.2)
- [ ] `builtin_palettes()` returns all five keyed by stem with the enumerated exact
  RGB values and counts — grayscale 4, gameboy 4, cga 16, pico8 16, sepia 4. (Task 3.3)
- [ ] `extract_palette(rgb_u8, k, name="source")` — median-cut, returns exactly `k`
  colors even when the image has fewer distinct colors. (Task 3.4)
- [ ] `source_palette(rgb_u8, completeness)` — clamps `[0,1]` → `k∈[2,256]`;
  `completeness=1.0` = the "complete" recommended default (256). (Task 3.5)
- [ ] `ColorEngine.map` in all four modes returns `uint8[H,W,3]`; `nearest`/`ordered`/
  `diffused` output only palette colors; `off` clamps RGB through. (Tasks 3.6–3.8)
- [ ] `ordered` and `diffused` both break a flat mid-gray fill into a mix of palette
  colors and are deterministic. (Tasks 3.7–3.8)
- [ ] `Palette.shuffle(locked, rng)` preserves locked swatches, randomizes the rest,
  keeps shape/name, and does not mutate the original. (Task 3.9)
- [ ] Still Qt-free: `grep -R "PySide6\|from PyQt\|import Qt" ditherzam/color` returns
  nothing.
- [ ] `NUMBA_DISABLE_JIT=1 pytest tests/test_saturation.py tests/test_palette.py tests/test_color_engine.py -v`
  is fully green.

## Self-Review

**Spec coverage (DITHER_BOY_FULL_SPEC.md §17.2 + §17.4 saturation → tasks):**

| Spec item (§17.2 / §17.4) | Task |
|---|---|
| Full color output (RGB uint8) | 3.6–3.8 (`ColorEngine.map`) |
| Built-in palettes (library shipped) | 3.3 (`builtin_palettes`, 5 palettes) |
| Automatic palette extraction from source image | 3.4 (`extract_palette`, median-cut) |
| "Source" category + "complete" max-retention default | 3.5 (`source_palette`, completeness→k) |
| Editable swatches / live remapping | 3.2 model + 3.6–3.8 map (re-`map` on edit; UI wiring is Phase 5) |
| Lock + shuffle | 3.9 (`Palette.shuffle`) |
| Palette import/export (community) | 3.2 (`to_yaml`/`load` YAML = the import/export format) |
| Saturation adjustment (§17.4) | 3.1 (`apply_saturation`) |
| Nearest / ordered / error-diffused color mapping (impl note) | 3.6 / 3.7 / 3.8 |

**Frozen-contract name/type check:**
- `apply_saturation(rgb, value)` — matches. Returns float32[H,W,3]. ✓
- `Palette.name: str`, `Palette.colors: np.ndarray float32[K,3]` — matches. ✓
- `builtin_palettes() -> dict[str, Palette]` — matches. ✓
- `extract_palette(rgb_u8, k=16, name="source") -> Palette` — matches. ✓
- `ColorEngine.palette`, `.mode` (`"off"|"nearest"|"ordered"|"diffused"`),
  `.map(gray_or_rgb_f32) -> uint8 HxWx3` — matches. ✓

**Placeholder scan:** no `TODO`, no `pass`-body stubs, no "implement here" — every
code step is complete. The Task 3.6 `raise ValueError` for unknown modes is a real
guard, not a stub; the `ordered`/`diffused` branches are filled in 3.7/3.8.

**Type consistency:** palettes are `float32` throughout; `ColorEngine.map` always
returns `uint8` via `clamp_u8`; `extract_palette`/`source_palette` return exactly the
requested `k` rows.

**Render-order note (roadmap §8.1 insert):** color maps run *after* dither and
*before* saturation/effects/invert. That composition lives in Phase 4
(`RenderPipeline`), which consumes `ColorEngine.map` and `apply_saturation` produced
here — out of scope for this document, flagged for the Phase 4 task list.

**Known gaps / deferred (intentionally out of scope for Phase 3 core):**
- k-means refinement of median-cut (plan mentions "optional") — omitted (YAGNI;
  median-cut passes all extraction tests). Add later only if palette quality demands.
- CMYK halftone (§17.3) — separate subsystem, not part of the color engine tasks.
- Editable-swatch *live* UI remapping and palette browser widget — Phase 5 (UI).
- Community download/share transport — only the YAML on-disk format is in scope here
  (clean-room: no network/licensing code, per hard rules).
