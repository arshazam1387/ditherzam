# Phase 1 — Foundation & Dither Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`. Implement task-by-task; steps use
> checkbox (`- [ ]`) syntax. Read [`00-ROADMAP.md`](00-ROADMAP.md) first — its
> **Global Constraints** apply to every task here.

**Goal:** A `pip install -e .`-able package where config loads/validates, the
`DitherRegistry` works, the tonal-adjustment functions match exact reference
formulas, the downscale→kernel→upscale dither pipeline runs, and three
error-diffusion kernels (Floyd–Steinberg, Atkinson, Bayer 4×4) pass golden tests.

**Architecture:** Headless, Qt-free core. Deterministic float32 arrays end to end,
tested against hand-computed reference arrays. Numba kernels are import-safe on any
machine (fall back gracefully if Numba unavailable during tests via `NUMBA_DISABLE_JIT`).

**Tech Stack:** Python 3.12 · NumPy · Numba · Pillow · PyYAML · platformdirs · pytest.

## Global Constraints
See [`00-ROADMAP.md`](00-ROADMAP.md). Key ones for this phase: kernels are
`@njit(cache=True, parallel=True)`, `float32` I/O, `0..255`; core imports **no** Qt;
tests run with `NUMBA_DISABLE_JIT=1` for speed and coverage.

---

## File structure (this phase)

- Create `pyproject.toml`, `config/config.yaml`
- Create `ditherzam/__init__.py`, `config.py`, `imaging.py`, `adjustments.py`
- Create `ditherzam/dithering/__init__.py`, `registry.py`, `pipeline.py`
- Create `ditherzam/dithering/kernels/error_diffusion.py`
- Create `tests/conftest.py` and one test module per source module

---

### Task 1: Project scaffold & packaging

**Files:**
- Create: `pyproject.toml`
- Create: `ditherzam/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: an installable package `ditherzam` (version `0.1.0`); `pytest` runs.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ditherzam"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=1.26",
    "numba>=0.59",
    "Pillow>=10.0",
    "PyYAML>=6.0",
    "platformdirs>=4.0",
    "PySide6>=6.6",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
include = ["ditherzam*"]

[project.scripts]
ditherzam = "ditherzam.app:main"
```

- [ ] **Step 2: Write `ditherzam/__init__.py`**

```python
"""ditherzam — open-source pixel-dither studio."""
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `tests/conftest.py`** (disable JIT for fast, coverable tests)

```python
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
```

- [ ] **Step 4: Install & verify**

Run: `pip install -e ".[dev]" && python -c "import ditherzam; print(ditherzam.__version__)"`
Expected: prints `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml ditherzam/__init__.py tests/conftest.py
git commit -m "feat(core): project scaffold and packaging"
```

---

### Task 2: Config loader

**Files:**
- Create: `config/config.yaml`
- Create: `ditherzam/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  ```python
  class ConfigError(Exception): ...
  @dataclass
  class AppConfig:
      default_dither_style: str
      default_dither_scale: int
      viewport_bg_color: str
      app_style: str
      enable_inertia: bool
      friction: float
      # ... (see step 3)
  def load_config(path: str | Path) -> AppConfig: ...
  ```

- [ ] **Step 1: Write `config/config.yaml`**

```yaml
app:
  name: "ditherzam"
  version: "0.1.0"
dither:
  default_style: "None"
  default_scale: 5
style:
  viewport_bg_color: "#1f1f1f"
  app_style: "Fusion"
inertia:
  enable: true
  friction: 0.95
  velocity_scale: 0.5
  max_velocity: 2000.0
timing:
  debounce_ms: 20
  loading_delay_ms: 400
```

- [ ] **Step 2: Write the failing test — `tests/test_config.py`**

```python
from pathlib import Path
import pytest
from ditherzam.config import load_config, AppConfig, ConfigError

def test_load_config_reads_values():
    cfg = load_config(Path("config/config.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.default_dither_style == "None"
    assert cfg.default_dither_scale == 5
    assert cfg.viewport_bg_color == "#1f1f1f"
    assert cfg.app_style == "Fusion"
    assert cfg.enable_inertia is True
    assert cfg.friction == 0.95

def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("does/not/exist.yaml")

def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.yaml"; p.write_text("")
    with pytest.raises(ConfigError):
        load_config(p)
```

- [ ] **Step 3: Run — verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError / ImportError for `ditherzam.config`)

- [ ] **Step 4: Implement `ditherzam/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


class ConfigError(Exception):
    """Raised when configuration is missing or malformed."""


@dataclass
class AppConfig:
    default_dither_style: str
    default_dither_scale: int
    viewport_bg_color: str
    app_style: str
    enable_inertia: bool
    friction: float
    velocity_scale: float
    max_velocity: float
    debounce_ms: int
    loading_delay_ms: int


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing configuration: {e}") from e
    if not data:
        raise ConfigError("Configuration file is empty.")
    try:
        return AppConfig(
            default_dither_style=data["dither"]["default_style"],
            default_dither_scale=int(data["dither"]["default_scale"]),
            viewport_bg_color=data["style"]["viewport_bg_color"],
            app_style=data["style"]["app_style"],
            enable_inertia=bool(data["inertia"]["enable"]),
            friction=float(data["inertia"]["friction"]),
            velocity_scale=float(data["inertia"]["velocity_scale"]),
            max_velocity=float(data["inertia"]["max_velocity"]),
            debounce_ms=int(data["timing"]["debounce_ms"]),
            loading_delay_ms=int(data["timing"]["loading_delay_ms"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e
```

- [ ] **Step 5: Run — verify pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add config/config.yaml ditherzam/config.py tests/test_config.py
git commit -m "feat(core): config loader with validation"
```

---

### Task 3: Imaging helpers

**Files:**
- Create: `ditherzam/imaging.py`
- Test: `tests/test_imaging.py`

**Interfaces:**
- Produces:
  ```python
  def to_gray_f32(pil_or_array) -> np.ndarray:     # HxW float32 0..255
  def clamp_u8(arr) -> np.ndarray:                 # uint8, clipped 0..255
  def nearest_downscale(gray_f32, factor: int) -> np.ndarray
  def nearest_upscale_to(small_f32, size_wh: tuple[int,int]) -> np.ndarray
  ```

- [ ] **Step 1: Failing test — `tests/test_imaging.py`**

```python
import numpy as np
from PIL import Image
from ditherzam.imaging import to_gray_f32, clamp_u8, nearest_downscale, nearest_upscale_to

def test_to_gray_f32_from_rgb():
    img = Image.new("RGB", (4, 2), (255, 255, 255))
    g = to_gray_f32(img)
    assert g.shape == (2, 4) and g.dtype == np.float32
    assert np.allclose(g, 255.0)

def test_clamp_u8():
    a = np.array([[-5.0, 128.0, 300.0]], dtype=np.float32)
    assert clamp_u8(a).tolist() == [[0, 128, 255]]

def test_downscale_then_upscale_roundtrips_size():
    g = np.arange(64, dtype=np.float32).reshape(8, 8)
    small = nearest_downscale(g, 4)          # 8//4 = 2
    assert small.shape == (2, 2)
    back = nearest_upscale_to(small, (8, 8))
    assert back.shape == (8, 8)
```

- [ ] **Step 2: Run — verify fail**

Run: `pytest tests/test_imaging.py -v` → FAIL (import error)

- [ ] **Step 3: Implement `ditherzam/imaging.py`**

```python
from __future__ import annotations
import numpy as np
from PIL import Image


def to_gray_f32(src) -> np.ndarray:
    if isinstance(src, Image.Image):
        arr = np.array(src.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(src, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
    return arr.astype(np.float32)


def clamp_u8(arr) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def nearest_downscale(gray_f32: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    h, w = gray_f32.shape[:2]
    pil = Image.fromarray(clamp_u8(gray_f32))
    small = pil.resize((max(1, w // factor), max(1, h // factor)), Image.NEAREST)
    return np.array(small, dtype=np.float32)


def nearest_upscale_to(small_f32: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(clamp_u8(small_f32))
    return np.array(pil.resize(size_wh, Image.NEAREST), dtype=np.float32)
```

- [ ] **Step 4: Run — verify pass** → `pytest tests/test_imaging.py -v` → 3 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/imaging.py tests/test_imaging.py
git commit -m "feat(core): imaging helpers (grayscale, clamp, nearest resize)"
```

---

### Task 4: Tonal adjustments (exact formulas)

**Files:**
- Create: `ditherzam/adjustments.py`
- Test: `tests/test_adjustments.py`

**Interfaces:** Produces `apply_contrast/midtones/highlights/blur/invert` (see roadmap contracts). `apply_saturation` is added in Phase 3.

- [ ] **Step 1: Failing test — `tests/test_adjustments.py`** (values hand-computed from the spec formulas)

```python
import numpy as np
from ditherzam.adjustments import (
    apply_contrast, apply_midtones, apply_highlights, apply_blur, apply_invert,
)

def arr(v): return np.full((2, 2), v, dtype=np.float32)

def test_contrast_50_is_identity():
    np.testing.assert_allclose(apply_contrast(arr(100), 50), arr(100))  # factor 50/50=1

def test_contrast_100_doubles():
    np.testing.assert_allclose(apply_contrast(arr(100), 100), arr(200))  # factor 2

def test_midtones_50_is_identity():
    np.testing.assert_allclose(apply_midtones(arr(128), 50), arr(128), atol=1e-3)  # gamma 1

def test_midtones_gamma_formula():
    # value 90 -> gamma = 1 + (90-50)/200 = 1.2 ; out = 255*(img/255)**(1/1.2)
    out = apply_midtones(arr(128), 90)
    exp = 255 * (128/255) ** (1/1.2)
    np.testing.assert_allclose(out, arr(exp), atol=1e-3)

def test_highlights_formula():
    # value 100 -> factor 1 + (100-50)/100 = 1.5
    np.testing.assert_allclose(apply_highlights(arr(100), 100), arr(150))

def test_blur_zero_is_identity():
    np.testing.assert_allclose(apply_blur(arr(100), 0), arr(100))

def test_invert():
    np.testing.assert_allclose(apply_invert(arr(40), True), arr(215))
    np.testing.assert_allclose(apply_invert(arr(40), False), arr(40))
```

- [ ] **Step 2: Run — verify fail** → FAIL (import error)

- [ ] **Step 3: Implement `ditherzam/adjustments.py`**

```python
from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter
from .imaging import clamp_u8


def apply_contrast(img: np.ndarray, value: float) -> np.ndarray:
    return (img * (value / 50.0)).astype(np.float32)


def apply_midtones(img: np.ndarray, value: float) -> np.ndarray:
    gamma = max(1.0 + (value - 50) / 200.0, 0.1)
    return (255.0 * (img / 255.0) ** (1.0 / gamma)).astype(np.float32)


def apply_highlights(img: np.ndarray, value: float) -> np.ndarray:
    return (img * (1.0 + (value - 50) / 100.0)).astype(np.float32)


def apply_blur(img: np.ndarray, value: float) -> np.ndarray:
    radius = (value / 10.0) ** 2
    if radius <= 0:
        return img
    pil = Image.fromarray(clamp_u8(img))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil, dtype=np.float32)


def apply_invert(img: np.ndarray, enabled: bool) -> np.ndarray:
    return (255.0 - img).astype(np.float32) if enabled else img
```

- [ ] **Step 4: Run — verify pass** → 7 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/adjustments.py tests/test_adjustments.py
git commit -m "feat(core): tonal adjustments with exact formulas"
```

---

### Task 5: Dither registry

**Files:**
- Create: `ditherzam/dithering/__init__.py` (empty)
- Create: `ditherzam/dithering/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:** Produces `DitherEntry`, `DitherRegistry` (see roadmap contracts).

- [ ] **Step 1: Failing test — `tests/test_registry.py`**

```python
from ditherzam.dithering.registry import DitherRegistry, DitherEntry

def test_register_and_lookup():
    reg = DitherRegistry()

    @reg.register("Demo", "Error Diffusion", dims=2, param_sliders=("p",))
    def demo(image_array, parameter, luminance_threshold_value):
        return image_array

    e = reg.get_entry("Demo")
    assert isinstance(e, DitherEntry)
    assert e.category == "Error Diffusion" and e.dims == 2
    assert e.param_sliders == ("p",)
    assert e.func is demo
    assert "Demo" in reg.list_dithers()

def test_by_category_groups():
    reg = DitherRegistry()
    reg.register("A", "Cat1")(lambda *a: a[0])
    reg.register("B", "Cat1")(lambda *a: a[0])
    reg.register("C", "Cat2")(lambda *a: a[0])
    cats = reg.by_category()
    assert cats["Cat1"] == ["A", "B"] and cats["Cat2"] == ["C"]

def test_unknown_returns_none():
    assert DitherRegistry().get_entry("nope") is None
```

- [ ] **Step 2: Run — verify fail** → FAIL

- [ ] **Step 3: Implement `ditherzam/dithering/registry.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DitherEntry:
    name: str
    category: str
    dims: int
    param_sliders: tuple[str, ...]
    func: Callable
    param_func: Callable | None = None


class DitherRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, DitherEntry] = {}

    def register(self, name, category, dims=2, param_sliders=(), param_func=None):
        def decorator(func):
            self._entries[name] = DitherEntry(
                name=name, category=category, dims=dims,
                param_sliders=tuple(param_sliders), func=func, param_func=param_func,
            )
            return func
        return decorator

    def get_entry(self, name: str) -> DitherEntry | None:
        return self._entries.get(name)

    def list_dithers(self) -> list[str]:
        return list(self._entries.keys())

    def by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for e in self._entries.values():
            out.setdefault(e.category, []).append(e.name)
        return out
```

- [ ] **Step 4: Run — verify pass** → 3 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/__init__.py ditherzam/dithering/registry.py tests/test_registry.py
git commit -m "feat(dither): DitherRegistry and DitherEntry"
```

---

### Task 6: First three kernels + shared registry instance

**Files:**
- Create: `ditherzam/dithering/kernels/__init__.py`
- Create: `ditherzam/dithering/kernels/error_diffusion.py`
- Modify: `ditherzam/dithering/__init__.py` (expose a shared `registry` and import kernels)
- Test: `tests/test_kernels_error_diffusion.py`

**Interfaces:**
- Produces module-level `registry = DitherRegistry()` in `ditherzam/dithering/__init__.py`,
  populated by importing kernel modules. Kernels: `floyd_steinberg`, `atkinson`, `bayer_4`.

- [ ] **Step 1: Failing test — `tests/test_kernels_error_diffusion.py`**

```python
import numpy as np
from ditherzam.dithering import registry

def test_all_three_registered():
    for n in ("Floyd-Steinberg", "Atkinson", "Bayer-Matrix 4x4"):
        assert registry.get_entry(n) is not None

def test_output_is_binary_0_255():
    reg = registry
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    out = reg.get_entry("Floyd-Steinberg").func(img.copy(), 0, 128.0)
    vals = set(np.unique(out).tolist())
    assert vals <= {0.0, 255.0}
    assert out.shape == img.shape

def test_all_black_input_stays_black():
    img = np.zeros((8, 8), dtype=np.float32)
    out = registry.get_entry("Atkinson").func(img, 0, 128.0)
    assert out.sum() == 0.0

def test_all_white_input_stays_white():
    img = np.full((8, 8), 255.0, dtype=np.float32)
    out = registry.get_entry("Bayer-Matrix 4x4").func(img, 0, 128.0)
    assert np.all(out == 255.0)
```

- [ ] **Step 2: Run — verify fail** → FAIL

- [ ] **Step 3: Implement `ditherzam/dithering/kernels/error_diffusion.py`**

```python
from __future__ import annotations
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


@njit(cache=True)
def _floyd_steinberg(img, thr):
    h, w = img.shape
    out = img.copy()
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    out[y + 1, x - 1] += err * 3 / 16
                out[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    out[y + 1, x + 1] += err * 1 / 16
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@registry.register("Floyd-Steinberg", "Error Diffusion", dims=2)
def floyd_steinberg(image_array, parameter, luminance_threshold_value):
    return _floyd_steinberg(image_array.astype(np.float32), luminance_threshold_value)


@njit(cache=True)
def _atkinson(img, thr):
    h, w = img.shape
    out = img.copy()
    offs = ((0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0))
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = (old - new) / 8.0
            for dy, dx in offs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] += err
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@registry.register("Atkinson", "Error Diffusion", dims=2)
def atkinson(image_array, parameter, luminance_threshold_value):
    return _atkinson(image_array.astype(np.float32), luminance_threshold_value)


def _bayer_matrix(n: int) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    smaller = _bayer_matrix(n // 2)
    m = np.block([
        [4 * smaller + 0, 4 * smaller + 2],
        [4 * smaller + 3, 4 * smaller + 1],
    ]).astype(np.float32)
    return m


_BAYER4 = (_bayer_matrix(4) + 0.5) / 16.0 * 255.0  # thresholds 0..255


@njit(cache=True, parallel=True)
def _ordered(img, thresholds):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = thresholds[y % mh, x % mw]
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@registry.register("Bayer-Matrix 4x4", "Ordered Dither", dims=2)
def bayer_4(image_array, parameter, luminance_threshold_value):
    return _ordered(image_array.astype(np.float32), _BAYER4)
```

- [ ] **Step 4: Modify `ditherzam/dithering/__init__.py`**

```python
from .registry import DitherRegistry, DitherEntry

registry = DitherRegistry()

# import kernel modules for their registration side-effects (after `registry` exists)
from .kernels import error_diffusion as _error_diffusion  # noqa: E402,F401
```

> Note: kernels import `registry` from `ditherzam.dithering`, which is defined
> above before the kernel import line — no circular-import problem.

- [ ] **Step 5: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_error_diffusion.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add ditherzam/dithering/__init__.py ditherzam/dithering/kernels/ tests/test_kernels_error_diffusion.py
git commit -m "feat(dither): Floyd-Steinberg, Atkinson, Bayer-4x4 kernels + shared registry"
```

---

### Task 7: Dither pipeline (downscale → kernel → upscale + param dispatch)

**Files:**
- Create: `ditherzam/dithering/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:** Produces `apply_dither(...)` per roadmap contract.

- [ ] **Step 1: Failing test — `tests/test_pipeline.py`**

```python
import numpy as np
from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither

def test_none_style_is_passthrough():
    img = np.full((8, 8), 100.0, dtype=np.float32)
    out = apply_dither(img, style="None", scale=1, luminance_threshold=50,
                       params={}, registry=registry)
    np.testing.assert_allclose(out, img)

def test_preview_disabled_is_passthrough():
    img = np.full((8, 8), 100.0, dtype=np.float32)
    out = apply_dither(img, style="Floyd-Steinberg", scale=1, luminance_threshold=50,
                       params={}, registry=registry, preview_disabled=True)
    np.testing.assert_allclose(out, img)

def test_dither_returns_binary_same_size():
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    out = apply_dither(img, style="Floyd-Steinberg", scale=1, luminance_threshold=50,
                       params={}, registry=registry)
    assert out.shape == img.shape
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}

def test_scale_pixelates_via_block_size():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    out = apply_dither(img, style="Bayer-Matrix 4x4", scale=4, luminance_threshold=50,
                       params={}, registry=registry)
    assert out.shape == img.shape  # upscaled back
```

- [ ] **Step 2: Run — verify fail** → FAIL

- [ ] **Step 3: Implement `ditherzam/dithering/pipeline.py`**

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


def apply_dither(gray_f32, *, style, scale, luminance_threshold,
                 params, registry, preview_disabled=False) -> np.ndarray:
    entry = registry.get_entry(style)
    if style == "None" or entry is None or preview_disabled:
        return gray_f32

    tval = _luminance_to_255(luminance_threshold)
    factor = max(1, int(scale))
    h, w = gray_f32.shape[:2]

    small = nearest_downscale(gray_f32, factor)
    param = _build_param(entry, params)
    out = entry.func(small, param, tval)

    return nearest_upscale_to(out, (w, h))
```

- [ ] **Step 4: Run — verify pass** → 4 passed
- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/pipeline.py tests/test_pipeline.py
git commit -m "feat(dither): downscale->kernel->upscale pipeline with param dispatch"
```

---

## Phase 1 Self-Review checklist

- [ ] `pip install -e ".[dev]"` succeeds; `ditherzam` imports.
- [ ] `pytest` (whole suite) green with `NUMBA_DISABLE_JIT=1`.
- [ ] Every adjustment formula matches the spec (§7.1) exactly — verified by hand-computed tests.
- [ ] Registry contract matches roadmap signatures (names/types identical).
- [ ] Pipeline honors None/preview-disabled passthrough and block-pixelation.
- [ ] No Qt imported anywhere in this phase.

**Deliverable:** headless dither core ready for Phase 2 (rest of kernels) and
Phase 3 (color engine) to build on.
