# Color System Core (Depth Ramp) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Dither-Boy-style depth-ramp color: dither luminance into N tonal levels (1–64) and map a palette across those levels as a tone ramp, with six mapping modes.

**Architecture:** Two existing `RenderPipeline` stages get richer without changing stage order. The **dither** stage's shared njit cores gain a `levels` arg (`levels<=2` = byte-identical old path; `levels>=3` = new N-level quantization). A new **`color/ramp.py`** builds a `depth`-length RGB ramp from a palette; `ColorEngine` gains a `ramp` mode that bins the dithered grayscale and recolors via that ramp. A single `depth` value feeds both the kernel `levels` and the ramp length.

**Tech Stack:** Python 3.12, NumPy, Numba (njit), PySide6 (UI only), pytest, PyYAML.

## Global Constraints

- Clean-room: no Studio/Dither-Boy code, strings, or binaries — our own implementation.
- Qt-free core: `color/ramp.py`, `color/engine.py`, `dithering/**` must not import PySide6.
- Frozen `RenderPipeline.render()` stage order: `contrast, midtones, highlights, blur, dither, color, saturation, effects, invert` — unchanged.
- Golden fixtures byte-identical for all existing kernels; the `levels<=2` path must call the exact existing code.
- Python 3.12. TDD per task. Run tests with `NUMBA_DISABLE_JIT=1`; Qt tests with `QT_QPA_PLATFORM=offscreen`. Full suite green (currently 375) in **both** JIT-on and JIT-off.
- Test runner (Windows): `./.venv/Scripts/python.exe -m pytest`. Env for a run:
  `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1`.
- Luminance weights Rec.601 `0.299/0.587/0.114` (consistent with `adjustments.apply_saturation`).
- Depth clamps to `[1, 64]`. Ramp modes (single source of truth = `RAMP_MODES`):
  `match, interpolated, glitch, reverse, hue_cycle, banded`.

---

### Task 1: Ramp builder module

**Files:**
- Create: `ditherzam/color/ramp.py`
- Test: `tests/test_ramp.py`

**Interfaces:**
- Consumes: `ditherzam.color.palette.Palette` (has `.colors` float32[K,3] in 0..255, `.name`).
- Produces:
  - `RAMP_MODES: tuple[str, ...] = ("match", "interpolated", "glitch", "reverse", "hue_cycle", "banded")`
  - `build_ramp(palette: Palette, depth: int, mapping: str, phase: float = 0.0) -> np.ndarray` returning float32[depth,3] in 0..255.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ramp.py
import numpy as np
import pytest
from ditherzam.color.palette import Palette
from ditherzam.color.ramp import build_ramp, RAMP_MODES

DUO = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
# deliberately NOT luminance-ordered, to exercise sorting/glitch:
TRI = Palette.from_list("tri", [[255, 255, 255], [0, 0, 0], [200, 50, 50]])

_LUMW = np.array([0.299, 0.587, 0.114], np.float32)
def _lum(rows):
    return (np.asarray(rows, np.float32) @ _LUMW)


@pytest.mark.parametrize("mode", RAMP_MODES)
@pytest.mark.parametrize("depth", [1, 2, 5, 8, 64])
def test_shape_dtype_range(mode, depth):
    r = build_ramp(TRI, depth, mode)
    assert r.shape == (depth, 3)
    assert r.dtype == np.float32
    assert r.min() >= 0.0 and r.max() <= 255.0


def test_depth_clamped():
    assert build_ramp(TRI, 0, "match").shape[0] == 1
    assert build_ramp(TRI, 999, "match").shape[0] == 64


@pytest.mark.parametrize("mode", ["match", "interpolated"])
def test_luminance_monotonic(mode):
    r = build_ramp(TRI, 16, mode)
    lum = _lum(r)
    assert np.all(np.diff(lum) >= -1e-3)  # non-decreasing


def test_match_uses_only_palette_colors():
    r = build_ramp(TRI, 10, "match")
    allowed = {tuple(np.round(c).astype(int)) for c in TRI.colors}
    got = {tuple(np.round(c).astype(int)) for c in r}
    assert got <= allowed


def test_interpolated_blends_between_anchors():
    # a value strictly between the two DUO endpoints must appear
    r = build_ramp(DUO, 5, "interpolated")
    mids = r[(r[:, 0] > 5) & (r[:, 0] < 250)]
    assert mids.shape[0] >= 1


def test_glitch_is_raw_order_modulo_k():
    r = build_ramp(TRI, 7, "glitch")
    for i in range(7):
        np.testing.assert_array_equal(r[i], TRI.colors[i % 3])


def test_reverse_is_match_flipped():
    m = build_ramp(TRI, 9, "match")
    rev = build_ramp(TRI, 9, "reverse")
    np.testing.assert_array_equal(rev, m[::-1])


def test_hue_cycle_is_palette_independent_and_colorful():
    a = build_ramp(TRI, 12, "hue_cycle")
    b = build_ramp(DUO, 12, "hue_cycle")
    np.testing.assert_array_equal(a, b)          # ignores palette
    assert a.std(axis=0).sum() > 10.0            # actual color variation


def test_banded_repeats_every_k():
    r = build_ramp(TRI, 9, "banded")             # K=3
    np.testing.assert_array_equal(r[0], r[3])
    np.testing.assert_array_equal(r[3], r[6])


def test_deterministic():
    np.testing.assert_array_equal(build_ramp(TRI, 20, "glitch"),
                                  build_ramp(TRI, 20, "glitch"))


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_ramp(TRI, 4, "nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_ramp.py -q`
Expected: FAIL — `ModuleNotFoundError: ditherzam.color.ramp`.

- [ ] **Step 3: Implement `ditherzam/color/ramp.py`**

```python
from __future__ import annotations

import colorsys

import numpy as np

from .palette import Palette

RAMP_MODES: tuple[str, ...] = (
    "match", "interpolated", "glitch", "reverse", "hue_cycle", "banded",
)

_LUM_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _clamp_depth(depth: int) -> int:
    return int(min(64, max(1, int(depth))))


def _lum_sorted(colors: np.ndarray) -> np.ndarray:
    lum = colors @ _LUM_W
    order = np.argsort(lum, kind="stable")   # stable: deterministic on ties
    return colors[order]


def _nearest_sample(sorted_colors: np.ndarray, depth: int) -> np.ndarray:
    k = sorted_colors.shape[0]
    if depth == 1:
        return sorted_colors[:1].copy()
    idx = np.round(np.linspace(0, k - 1, depth)).astype(np.int64)
    return sorted_colors[idx].copy()


def _interp_sample(sorted_colors: np.ndarray, depth: int) -> np.ndarray:
    k = sorted_colors.shape[0]
    if depth == 1 or k == 1:
        return np.repeat(sorted_colors[:1], depth, axis=0).astype(np.float32)
    pos = np.linspace(0.0, k - 1, depth)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, k - 1)
    frac = (pos - lo)[:, None].astype(np.float32)
    return (sorted_colors[lo] * (1.0 - frac) + sorted_colors[hi] * frac).astype(np.float32)


def _hue_cycle(depth: int, phase: float) -> np.ndarray:
    out = np.empty((depth, 3), dtype=np.float32)
    for i in range(depth):
        h = (phase + (i / depth if depth else 0.0)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        out[i] = (r * 255.0, g * 255.0, b * 255.0)
    return out


def _apply_phase(ramp: np.ndarray, phase: float) -> np.ndarray:
    depth = ramp.shape[0]
    if depth <= 1 or not phase:
        return ramp
    shift = int(round((phase % 1.0) * depth)) % depth
    if shift == 0:
        return ramp
    return np.roll(ramp, shift, axis=0)


def build_ramp(palette: Palette, depth: int, mapping: str,
               phase: float = 0.0) -> np.ndarray:
    """Build a float32[depth,3] RGB tone ramp (0..255) from ``palette``.

    ``depth`` is clamped to [1,64]. ``phase`` in [0,1] cyclically rotates the
    ramp (reserved for animation). See RAMP_MODES for ``mapping`` values.
    """
    depth = _clamp_depth(depth)
    colors = np.ascontiguousarray(palette.colors, dtype=np.float32)

    if mapping == "hue_cycle":
        return _apply_phase(_hue_cycle(depth, phase), 0.0)
    if mapping == "glitch":
        idx = np.arange(depth) % colors.shape[0]
        return _apply_phase(colors[idx].astype(np.float32), phase)
    if mapping == "banded":
        s = _lum_sorted(colors)
        idx = np.arange(depth) % s.shape[0]
        return _apply_phase(s[idx].astype(np.float32), phase)

    s = _lum_sorted(colors)
    if mapping == "match":
        ramp = _nearest_sample(s, depth)
    elif mapping == "interpolated":
        ramp = _interp_sample(s, depth)
    elif mapping == "reverse":
        ramp = _nearest_sample(s, depth)[::-1].copy()
    else:
        raise ValueError(f"unknown ramp mapping: {mapping!r}")
    return _apply_phase(ramp.astype(np.float32), phase)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_ramp.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/ramp.py tests/test_ramp.py
git commit -m "feat(color): depth-ramp builder with six mapping modes"
```

---

### Task 2: ColorEngine `ramp` mode

**Files:**
- Modify: `ditherzam/color/engine.py`
- Test: `tests/test_color_engine_ramp.py`

**Interfaces:**
- Consumes: `build_ramp`, `RAMP_MODES` from Task 1; existing `_to_rgb`, `clamp_u8`.
- Produces: `ColorEngine(palette, mode="ramp", depth=2, mapping="match", phase=0.0)`;
  `map()` handles `mode == "ramp"`. Existing modes and signatures unchanged (new args are keyword-only-with-defaults, so old call sites keep working).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_color_engine_ramp.py
import numpy as np
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine
from ditherzam.color.ramp import build_ramp

DUO = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
QUAD = Palette.from_list("quad", [[0, 0, 0], [90, 0, 0], [0, 160, 0], [255, 255, 255]])


def test_ramp_mode_recolors_levels():
    eng = ColorEngine(DUO, mode="ramp", depth=2, mapping="match")
    gray = np.array([[0.0, 255.0]], np.float32)
    out = eng.map(gray)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255]


def test_ramp_depth_limits_distinct_tones():
    eng = ColorEngine(QUAD, mode="ramp", depth=3, mapping="match")
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (4, 1))
    out = eng.map(grad)
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert len(uniq) <= 3


def test_ramp_output_matches_builder():
    eng = ColorEngine(QUAD, mode="ramp", depth=4, mapping="interpolated")
    ramp = build_ramp(QUAD, 4, "interpolated")
    gray = np.array([[0.0, 85.0, 170.0, 255.0]], np.float32)
    out = eng.map(gray).astype(np.float32)
    # each pixel maps to its binned ramp entry
    levels = np.clip(np.round(gray / 255.0 * 3), 0, 3).astype(int)
    expected = np.round(ramp[levels[0]]).astype(np.uint8)
    np.testing.assert_array_equal(out[0], expected)


def test_ramp_depth_one_is_single_color():
    eng = ColorEngine(QUAD, mode="ramp", depth=1, mapping="match")
    out = eng.map(np.tile(np.linspace(0, 255, 20, np.float32), (3, 1)))
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert len(uniq) == 1


def test_ramp_cache_rebuilds_on_param_change():
    eng = ColorEngine(QUAD, mode="ramp", depth=2, mapping="match")
    gray = np.tile(np.linspace(0, 255, 16, np.float32), (2, 1))
    a = eng.map(gray)
    eng.depth = 5
    b = eng.map(gray)
    assert not np.array_equal(a, b)  # changing depth changes output


def test_existing_modes_untouched():
    eng = ColorEngine(DUO, mode="nearest")
    out = eng.map(np.array([[10.0, 240.0]], np.float32))
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_color_engine_ramp.py -q`
Expected: FAIL — `ColorEngine.__init__` rejects `depth`/`mapping` (unexpected kwarg) or ramp branch missing.

- [ ] **Step 3: Modify `ditherzam/color/engine.py`**

Add the import near the top (after `from .palette import Palette`):

```python
from .ramp import build_ramp
```

Replace the `ColorEngine` class body with:

```python
class ColorEngine:
    def __init__(self, palette: Palette, mode: str = "nearest", *,
                 depth: int = 2, mapping: str = "match", phase: float = 0.0) -> None:
        self.palette = palette
        self.mode = mode
        self.depth = depth
        self.mapping = mapping
        self.phase = phase
        self._ramp = None
        self._ramp_key = None

    def _get_ramp(self) -> np.ndarray:
        key = (self.palette.colors.tobytes(), int(self.depth),
               self.mapping, float(self.phase))
        if self._ramp_key != key:
            self._ramp = build_ramp(self.palette, self.depth, self.mapping, self.phase)
            self._ramp_key = key
        return self._ramp

    def map(self, gray_or_rgb_f32: np.ndarray) -> np.ndarray:
        rgb = _to_rgb(gray_or_rgb_f32)
        if self.mode == "off":
            return clamp_u8(rgb)
        if self.mode == "ramp":
            ramp = self._get_ramp()
            depth = ramp.shape[0]
            gray = rgb[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32) \
                if rgb.ndim == 3 else rgb
            if depth == 1:
                level = np.zeros(gray.shape, dtype=np.int64)
            else:
                level = np.clip(np.round(gray / 255.0 * (depth - 1)), 0, depth - 1)
                level = level.astype(np.int64)
            return clamp_u8(ramp[level])
        pal = self.palette.colors.astype(np.float32)
        if self.mode == "nearest":
            idx = nearest_indices(rgb, pal)
            return clamp_u8(pal[idx])
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
        if self.mode == "diffused":
            return clamp_u8(_floyd_steinberg_rgb(rgb, pal))
        raise ValueError(f"unknown ColorEngine mode: {self.mode!r}")
```

Note: the incoming `d` from the pipeline is a dithered **grayscale** (`_to_rgb` broadcasts it to 3 equal channels), so the luminance dot-product recovers the tone exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_color_engine_ramp.py tests/test_color_engine.py -q`
Expected: PASS (new + existing engine tests).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/engine.py tests/test_color_engine_ramp.py
git commit -m "feat(color): ramp mode in ColorEngine (binned tone -> ramp)"
```

---

### Task 3: N-level error-diffusion cores

**Files:**
- Create: `ditherzam/dithering/nlevels.py`
- Modify: `ditherzam/dithering/kernels/error_diffusion.py`
- Test: `tests/test_nlevels_diffusion.py`

**Interfaces:**
- Produces (in `nlevels.py`):
  - `quantize_to_levels(v: float, levels: int) -> float` (njit) — nearest of `levels` evenly-spaced values in 0..255.
- Modifies the shared njit cores `_floyd_steinberg`, `_atkinson`, `_diffuse` in
  `error_diffusion.py` to accept a trailing `levels` arg. `levels <= 2` executes the
  **exact current body**; `levels >= 3` runs the N-level path. Their registered wrapper
  funcs (`floyd_steinberg`, `atkinson`, `jjn`, `stucki`, `burkes`, `sierra`,
  `sierra_lite`, `two_row_sierra`, `stevenson_arce`, `fan`, `shiau_fan`,
  `false_floyd_steinberg`, `atkinson_light`) gain a keyword `levels: int = 2` and forward it.
- The tone bias from the threshold slider in N-level mode: input shifted by
  `(127.5 - thr)` before quantization (so slider 50 ⇒ no shift).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nlevels_diffusion.py
import numpy as np
from ditherzam.dithering.nlevels import quantize_to_levels
from ditherzam.dithering.kernels import error_diffusion as ed
from tests.golden_harness import STD_INPUT


def test_quantize_levels_values():
    assert quantize_to_levels(0.0, 4) == 0.0
    assert quantize_to_levels(255.0, 4) == 255.0
    # 4 levels -> {0, 85, 170, 255}
    assert abs(quantize_to_levels(80.0, 4) - 85.0) < 1e-3
    assert abs(quantize_to_levels(200.0, 4) - 170.0) < 1e-3


def test_levels2_identical_to_current_fs():
    # levels<=2 must reproduce the exact current output at any threshold
    for thr in (60.0, 127.5, 200.0):
        base = ed._floyd_steinberg(STD_INPUT.copy(), thr)          # current 2-arg core is now 3-arg default
        got = ed._floyd_steinberg(STD_INPUT.copy(), thr, 2)
        np.testing.assert_array_equal(base, got)


def test_nlevel_fs_limits_distinct_tones():
    out = ed._floyd_steinberg(STD_INPUT.copy(), 127.5, 4)
    uniq = np.unique(np.round(out))
    assert len(uniq) <= 4


def test_nlevel_fs_preserves_mean():
    flat = np.full((32, 32), 120.0, np.float32)
    out = ed._floyd_steinberg(flat.copy(), 127.5, 5)
    assert abs(out.mean() - 120.0) < 8.0


def test_diffuse_core_levels2_identity():
    # jjn uses _diffuse; verify the shared core keeps identity at levels=2
    from ditherzam.dithering.kernels.error_diffusion import _JJN_OFF, _JJN_W, _JJN_DIV
    base = ed._diffuse(STD_INPUT.copy(), 127.5, _JJN_OFF, _JJN_W, _JJN_DIV)
    got = ed._diffuse(STD_INPUT.copy(), 127.5, _JJN_OFF, _JJN_W, _JJN_DIV, 2)
    np.testing.assert_array_equal(base, got)
```

Note: if the JJN weight constants are inline (not module-level `_JJN_*`), the implementer
introduces `_JJN_OFF`, `_JJN_W`, `_JJN_DIV` module constants for `jjn` and reuses them —
otherwise adapt the test to whatever constant the file already exposes for JJN.

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_nlevels_diffusion.py -q`
Expected: FAIL — `ditherzam.dithering.nlevels` missing; cores reject the 3rd/6th arg.

- [ ] **Step 3a: Create `ditherzam/dithering/nlevels.py`**

```python
from __future__ import annotations

from numba import njit


@njit(cache=True)
def quantize_to_levels(v, levels):
    """Snap ``v`` (0..255) to the nearest of ``levels`` evenly-spaced values."""
    if levels <= 1:
        return 0.0
    x = v
    if x < 0.0:
        x = 0.0
    elif x > 255.0:
        x = 255.0
    step = 255.0 / (levels - 1)
    q = round(x / step)
    return q * step
```

- [ ] **Step 3b: Generalize the cores in `error_diffusion.py`**

Import the helper at the top:

```python
from ditherzam.dithering.nlevels import quantize_to_levels
```

Rewrite `_floyd_steinberg` (keep the old body verbatim under `levels <= 2`):

```python
@njit(cache=True)
def _floyd_steinberg(img, thr, levels=2):
    h, w = img.shape
    out = img.copy()
    if levels <= 2:
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
    bias = 127.5 - thr
    for y in range(h):
        for x in range(w):
            old = out[y, x] + bias
            new = quantize_to_levels(old, levels)
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
    return out
```

Apply the identical pattern to `_atkinson` (wrap old body under `levels <= 2`; else use
`new = quantize_to_levels(out[y,x] + bias, levels)`, `err = (old - new)/8.0`, no final
threshold pass), and to `_diffuse`:

```python
@njit(cache=True)
def _diffuse(img, thr, offsets, weights, divisor, levels=2):
    h, w = img.shape
    out = img.copy()
    n = offsets.shape[0]
    if levels <= 2:
        for y in range(h):
            for x in range(w):
                old = out[y, x]
                new = 255.0 if old >= thr else 0.0
                out[y, x] = new
                err = old - new
                for k in range(n):
                    ny = y + offsets[k, 0]
                    nx = x + offsets[k, 1]
                    if 0 <= ny < h and 0 <= nx < w:
                        out[ny, nx] += err * weights[k] / divisor
        for y in range(h):
            for x in range(w):
                out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
        return out
    bias = 127.5 - thr
    for y in range(h):
        for x in range(w):
            old = out[y, x] + bias
            new = quantize_to_levels(old, levels)
            out[y, x] = new
            err = old - new
            for k in range(n):
                ny = y + offsets[k, 0]
                nx = x + offsets[k, 1]
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] += err * weights[k] / divisor
    return out
```

Then update each registered wrapper to accept and forward `levels`, e.g.:

```python
@registry.register("Floyd-Steinberg", "Error Diffusion", dims=2)
def floyd_steinberg(image_array, parameter, luminance_threshold_value, levels=2):
    return _floyd_steinberg(image_array.astype(np.float32),
                            luminance_threshold_value, levels)
```

Do the same for `atkinson` and every `_diffuse`-based wrapper listed in Interfaces
(`jjn, stucki, burkes, sierra, sierra_lite, two_row_sierra, stevenson_arce, fan,
shiau_fan, false_floyd_steinberg, atkinson_light`), passing `levels` as the final arg.
Leave `bayer_4`, `no_dither`, `ostromukhov`, and `gaussian` unchanged in this task
(bayer_4 is handled by Task 4's ordered path is separate; the other two stay 2-tone and
are simply not marked `supports_levels` in Task 5).

- [ ] **Step 4: Run tests + full error-diffusion + golden suites**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_nlevels_diffusion.py tests/test_kernels_error_diffusion.py tests/test_kernels_diffusion.py tests/test_kernels_all.py -q`
Expected: PASS (new tests + all existing golden/kernel tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/nlevels.py ditherzam/dithering/kernels/error_diffusion.py tests/test_nlevels_diffusion.py
git commit -m "feat(dither): N-level error diffusion (levels<=2 byte-identical)"
```

---

### Task 4: N-level ordered core

**Files:**
- Modify: `ditherzam/dithering/kernels/ordered.py`
- Test: `tests/test_nlevels_ordered.py`

**Interfaces:**
- Modifies the shared njit core `_ordered(img, thresholds, levels=2)` in `ordered.py`:
  `levels <= 2` = exact current body; `levels >= 3` = add centered screen offset then
  `quantize_to_levels`. Wrappers that call `_ordered` and should gain depth
  (`bayer_2, bayer_8, bayer_16, bayer_ordered, halftone_ordered, cluster_dot`) get a
  keyword `levels: int = 2` and forward it. Other ordered kernels with bespoke cores
  (`bayer_void, random_ordered, bit_tone, mosaic, modulated_bayer, dot_screen,
  line_screen`) stay 2-tone (unchanged).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nlevels_ordered.py
import numpy as np
from ditherzam.dithering.kernels import ordered as od
from tests.golden_harness import STD_INPUT


def test_ordered_levels2_identity():
    base = od._ordered(STD_INPUT.copy(), od._BAYER4)
    got = od._ordered(STD_INPUT.copy(), od._BAYER4, 2)
    np.testing.assert_array_equal(base, got)


def test_ordered_nlevel_tone_count():
    out = od._ordered(STD_INPUT.copy(), od._BAYER4, 4)
    uniq = np.unique(np.round(out))
    assert len(uniq) <= 4


def test_ordered_nlevel_preserves_mean_region():
    flat = np.full((32, 32), 120.0, np.float32)
    out = od._ordered(flat.copy(), od._BAYER4, 5)
    assert abs(out.mean() - 120.0) < 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_nlevels_ordered.py -q`
Expected: FAIL — `_ordered` rejects the 3rd arg.

- [ ] **Step 3: Generalize `_ordered` in `ordered.py`**

Add the import:

```python
from ditherzam.dithering.nlevels import quantize_to_levels
```

Rewrite `_ordered`:

```python
@njit(cache=True, parallel=True)
def _ordered(img, thresholds, levels=2):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    if levels <= 2:
        for y in prange(h):
            for x in range(w):
                out[y, x] = 255.0 if img[y, x] >= thresholds[y % mh, x % mw] else 0.0
        return out
    step = 255.0 / (levels - 1)
    for y in prange(h):
        for x in range(w):
            # thresholds are 0..255; recenter to [-0.5,0.5]*step as a sub-step offset
            off = (thresholds[y % mh, x % mw] / 255.0 - 0.5) * step
            out[y, x] = quantize_to_levels(img[y, x] + off, levels)
    return out
```

Update the wrappers listed in Interfaces to forward `levels`, e.g.:

```python
@registry.register("Bayer-Matrix 2x2", "Ordered Dither", dims=2)
def bayer_2(image_array, parameter, luminance_threshold_value, levels=2):
    return _ordered(image_array.astype(np.float32), _BAYER2, levels)
```

Do the same for `bayer_8, bayer_16, bayer_ordered, halftone_ordered, cluster_dot`
(whichever call `_ordered` with a threshold matrix). Also update `error_diffusion.py`'s
`bayer_4` (it calls the *local* `_ordered` in that module) the same way — add
`levels=2` there and generalize that module's `_ordered` too with the identical body.

- [ ] **Step 4: Run tests + ordered golden suite**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_nlevels_ordered.py tests/test_kernels_ordered.py tests/test_kernels_all.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/kernels/ordered.py ditherzam/dithering/kernels/error_diffusion.py tests/test_nlevels_ordered.py
git commit -m "feat(dither): N-level ordered dithering (levels<=2 byte-identical)"
```

---

### Task 5: Registry `supports_levels` + `apply_dither` plumbing

**Files:**
- Modify: `ditherzam/dithering/registry.py`
- Modify: `ditherzam/dithering/pipeline.py`
- Test: `tests/test_pipeline_levels.py`

**Interfaces:**
- Consumes: level-capable wrappers from Tasks 3–4.
- Produces:
  - `DitherEntry.supports_levels: bool = False`; `register(..., supports_levels=False)`.
  - `apply_dither(..., levels: int = 2)`: when `entry.supports_levels` is True calls
    `entry.func(small, param, tval, levels)`; else `entry.func(small, param, tval)`.
- The level-capable wrappers (Tasks 3–4) must set `supports_levels=True` in their
  `@registry.register(...)` decorators.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_levels.py
import numpy as np
from ditherzam.dithering import registry as reg
from ditherzam.dithering.pipeline import apply_dither
from ditherzam.dithering.kernels import error_diffusion  # noqa: F401 (registers kernels)

R = error_diffusion.registry  # the module-level shared registry instance


def test_entry_has_supports_levels_flag():
    e = R.get_entry("Floyd-Steinberg")
    assert e.supports_levels is True
    e2 = R.get_entry("Ostromukhov")
    assert e2.supports_levels is False


def test_apply_dither_passes_levels_to_capable_kernel():
    g = np.tile(np.linspace(0, 255, 64, np.float32), (32, 1))
    out2 = apply_dither(g, style="Floyd-Steinberg", scale=1,
                        luminance_threshold=50, params={}, registry=R, levels=2)
    out4 = apply_dither(g, style="Floyd-Steinberg", scale=1,
                        luminance_threshold=50, params={}, registry=R, levels=4)
    assert len(np.unique(np.round(out2))) <= 2
    assert 2 < len(np.unique(np.round(out4))) <= 4


def test_apply_dither_levels_ignored_for_incapable_kernel():
    g = np.tile(np.linspace(0, 255, 64, np.float32), (32, 1))
    a = apply_dither(g, style="Ostromukhov", scale=1, luminance_threshold=50,
                     params={}, registry=R, levels=6)
    b = apply_dither(g, style="Ostromukhov", scale=1, luminance_threshold=50,
                     params={}, registry=R, levels=2)
    np.testing.assert_array_equal(a, b)  # levels had no effect
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_pipeline_levels.py -q`
Expected: FAIL — `supports_levels` attribute / `apply_dither(levels=...)` missing.

- [ ] **Step 3a: Modify `registry.py`**

```python
@dataclass(frozen=True)
class DitherEntry:
    name: str
    category: str
    dims: int
    param_sliders: tuple[str, ...]
    func: Callable
    param_func: Callable | None = None
    supports_levels: bool = False
```

```python
    def register(self, name, category, dims=2, param_sliders=(), param_func=None,
                 supports_levels=False):
        def decorator(func):
            self._entries[name] = DitherEntry(
                name=name, category=category, dims=dims,
                param_sliders=tuple(param_sliders), func=func, param_func=param_func,
                supports_levels=supports_levels,
            )
            return func
        return decorator
```

- [ ] **Step 3b: Modify `apply_dither` in `pipeline.py`**

Change the signature and the call site:

```python
def apply_dither(gray_f32, *, style, scale, luminance_threshold,
                 params, registry, preview_disabled=False,
                 threshold_field=None, levels=2) -> np.ndarray:
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
        small = (small - fld).astype(np.float32)

    param = _build_param(entry, params)
    if entry.supports_levels:
        out = entry.func(small, param, tval, int(levels))
    else:
        out = entry.func(small, param, tval)

    return nearest_upscale_to(out, (w, h))
```

- [ ] **Step 3c: Set `supports_levels=True`** on every level-capable wrapper from Tasks
3–4. In `error_diffusion.py` and `ordered.py`, change each relevant decorator to e.g.
`@registry.register("Floyd-Steinberg", "Error Diffusion", dims=2, supports_levels=True)`.
The exact set: `Floyd-Steinberg, Atkinson, JJN, Stucki, Burkes, Sierra, Sierra-Lite,
Two-Row Sierra, Stevenson-Arce, Fan, Shiau-Fan, False Floyd-Steinberg, Atkinson-Light`
(error diffusion) and `Bayer-Matrix 2x2, Bayer-Matrix 4x4, Bayer-Matrix 8x8,
Bayer-Matrix 16x16, Bayer-Ordered, Halftone Ordered, Cluster-Dot` (ordered) — use the
kernels' **registered display names** exactly as they appear in the source decorators.

- [ ] **Step 4: Run tests + registry + pipeline + all kernels**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_pipeline_levels.py tests/test_registry.py tests/test_pipeline.py tests/test_kernels_all.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/registry.py ditherzam/dithering/pipeline.py ditherzam/dithering/kernels/error_diffusion.py ditherzam/dithering/kernels/ordered.py tests/test_pipeline_levels.py
git commit -m "feat(dither): plumb levels through registry + apply_dither"
```

---

### Task 6: RenderSettings + pipeline wiring

**Files:**
- Modify: `ditherzam/render.py`
- Test: `tests/test_render_ramp.py`

**Interfaces:**
- Consumes: `apply_dither(levels=...)` (Task 5), `ColorEngine(mode="ramp", depth, mapping)`
  (Task 2).
- Produces: `RenderSettings.depth: int = 2`, `RenderSettings.color_mapping: str = "match"`;
  `render()` and `render_cached()` forward `settings.depth` as `levels` to `apply_dither`
  and (when the color engine is a ramp engine) keep the ramp in sync; cache signatures
  include `depth` and color-engine ramp params.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_render_ramp.py
import numpy as np
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.dithering.kernels import error_diffusion as ed
from ditherzam.color.palette import Palette
from ditherzam.color.engine import ColorEngine

R = ed.registry
GRAD = np.tile(np.linspace(0, 255, 128, np.float32), (128, 1))
QUAD = Palette.from_list("quad", [[0, 0, 0], [90, 0, 0], [0, 160, 0], [255, 255, 255]])


def _settings(**kw):
    base = dict(style="Floyd-Steinberg", scale=1, depth=4, color_mapping="match")
    base.update(kw)
    return RenderSettings(**base)


def test_depth_field_default():
    s = RenderSettings()
    assert s.depth == 2 and s.color_mapping == "match"


def test_render_depth_produces_multi_tone_color():
    eng = ColorEngine(QUAD, mode="ramp", depth=4, mapping="match")
    p = RenderPipeline(R, color_engine=eng)
    out = p.render(GRAD, _settings(depth=4))
    uniq = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert 2 < len(uniq) <= 4


def test_render_cached_matches_render_with_depth():
    eng = ColorEngine(QUAD, mode="ramp", depth=5, mapping="interpolated")
    p = RenderPipeline(R, color_engine=eng)
    s = _settings(depth=5, color_mapping="interpolated")
    a = p.render(GRAD, s)
    b = p.render_cached(GRAD, s)
    np.testing.assert_array_equal(a, b)


def test_render_cached_invalidates_on_depth_change():
    eng = ColorEngine(QUAD, mode="ramp", depth=3, mapping="match")
    p = RenderPipeline(R, color_engine=eng)
    a = p.render_cached(GRAD, _settings(depth=3))
    eng.depth = 6
    b = p.render_cached(GRAD, _settings(depth=6))
    assert not np.array_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_render_ramp.py -q`
Expected: FAIL — `RenderSettings` has no `depth`/`color_mapping`; renders ignore depth.

- [ ] **Step 3: Modify `ditherzam/render.py`**

Add fields to `RenderSettings` (after `scale`):

```python
    depth: int = 2
    color_mapping: str = "match"
```

In `render()`, pass `levels` to the dither call and sync the ramp engine before coloring:

```python
        d = apply_dither(
            g,
            style=settings.style,
            scale=settings.scale,
            luminance_threshold=settings.luminance_threshold,
            params=settings.params,
            registry=self.registry,
            preview_disabled=settings.preview_disabled,
            threshold_field=temporal_field,
            levels=settings.depth,
        )

        if self.color_engine is not None:
            if getattr(self.color_engine, "mode", None) == "ramp":
                self.color_engine.depth = settings.depth
                self.color_engine.mapping = settings.color_mapping
            rgb = self.color_engine.map(d).astype(np.float32)
        else:
            rgb = np.repeat(np.asarray(d, np.float32)[..., None], 3, axis=2)
```

In `render_cached()`, add `levels=settings.depth` to **both** `apply_dither` calls, add
`settings.depth` to `dith_sig`, and extend the color layer:

```python
                dith_sig = (settings.style, settings.scale,
                            settings.luminance_threshold,
                            _params_sig(settings.params), settings.preview_disabled,
                            settings.depth)
```

```python
            col_sig = _color_sig(self.color_engine) + (settings.depth, settings.color_mapping)
            if dirty or c.get("col_sig") != col_sig or "colored" not in c:
                if self.color_engine is not None:
                    if getattr(self.color_engine, "mode", None) == "ramp":
                        self.color_engine.depth = settings.depth
                        self.color_engine.mapping = settings.color_mapping
                    colored = self.color_engine.map(d).astype(np.float32)
                else:
                    colored = np.repeat(np.asarray(d, np.float32)[..., None], 3, axis=2)
                c["col_sig"] = col_sig
                c["colored"] = colored
                dirty = True
```

Also update the temporal-field branch's `apply_dither` call in `render_cached` to include
`levels=settings.depth` (mirror the change). Note `_color_sig` returns `None` when the
engine is None — guard by making `col_sig` a tuple even then:

```python
def _color_sig(engine):
    if engine is None:
        return (None,)
    return (engine.mode, engine.palette.name, engine.palette.colors.tobytes())
```

- [ ] **Step 4: Run tests + render suites**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_render_ramp.py tests/test_render.py tests/test_render_order.py tests/test_render_cache.py -q`
Expected: PASS (including the frozen stage-order test).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/render.py tests/test_render_ramp.py
git commit -m "feat(render): depth+mapping in RenderSettings, wired to dither+ramp"
```

---

### Task 7: settings_map + preset round-trip

**Files:**
- Modify: `ditherzam/ui/settings_map.py`
- Modify: `ditherzam/presets.py`
- Test: `tests/test_settings_map_ramp.py`

**Interfaces:**
- Consumes: `RenderSettings.depth`, `.color_mapping` (Task 6); `RAMP_MODES` (Task 1).
- Produces: `settings_from_controls` reads `depth` (default 2) and `color_mapping`
  (default "match"); presets save/load both.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_settings_map_ramp.py
from ditherzam.ui.settings_map import settings_from_controls
from ditherzam.render import RenderSettings
from ditherzam.presets import settings_to_preset, preset_to_settings


def test_settings_map_reads_depth_and_mapping():
    s = settings_from_controls({"depth": 12, "color_mapping": "glitch"})
    assert s.depth == 12 and s.color_mapping == "glitch"


def test_settings_map_defaults():
    s = settings_from_controls({})
    assert s.depth == 2 and s.color_mapping == "match"


def test_preset_round_trip_depth_mapping():
    s = RenderSettings(depth=9, color_mapping="hue_cycle")
    back = preset_to_settings(settings_to_preset(s))
    assert back.depth == 9 and back.color_mapping == "hue_cycle"
```

Note: use whatever the preset module's actual public function names are — check
`ditherzam/presets.py` and adapt `settings_to_preset` / `preset_to_settings` to the real
names (e.g. `to_preset` / `from_preset`). Keep the assertions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_settings_map_ramp.py -q`
Expected: FAIL — depth/mapping not mapped; preset drops them.

- [ ] **Step 3a: `settings_map.py`** — add to the `RenderSettings(...)` construction:

```python
        depth=int(state.get("depth", 2)),
        color_mapping=state.get("color_mapping", "match"),
```

- [ ] **Step 3b: `presets.py`** — add `depth` (int) and `color_mapping` (str) to the
serialized dict and to the reconstruction, mirroring how `scale`/`style` are handled.
Read the file first; add `depth` to the numeric ranges map if it validates numeric
fields, and pass `color_mapping` through as a plain string (validate against
`ditherzam.color.ramp.RAMP_MODES`, falling back to `"match"` if unknown).

- [ ] **Step 4: Run tests + preset + settings-map suites**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_settings_map_ramp.py tests/test_settings_map.py tests/test_presets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/settings_map.py ditherzam/presets.py tests/test_settings_map_ramp.py
git commit -m "feat(ui): depth+color_mapping in settings map and presets"
```

---

### Task 8: UI controls — Depth slider + Mapping dropdown

**Files:**
- Modify: `ditherzam/ui/controls.py`
- Test: `tests/test_controls_ramp.py`

**Interfaces:**
- Consumes: `RAMP_MODES` (Task 1); the control state keys `depth`, `color_mapping` read by
  `settings_from_controls` (Task 7).
- Produces: a Depth slider (range 1–64, default 2) and a Mapping dropdown whose current
  values appear in the control panel's exported state dict under `depth` and
  `color_mapping`.

- [ ] **Step 1: Read `ui/controls.py`** to match the existing slider/spin and combo
patterns (how `luminance_threshold` slider and the dither-style combo are built and how
state is collected). Follow them exactly.

- [ ] **Step 2: Write the failing test** (adapt widget/getter names to the file's actual
API discovered in Step 1):

```python
# tests/test_controls_ramp.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from ditherzam.color.ramp import RAMP_MODES

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from ditherzam.ui.controls import ControlPanel  # adapt to real class name

app = QApplication.instance() or QApplication([])


def test_panel_exposes_depth_and_mapping_state():
    panel = ControlPanel()
    state = panel.state()  # adapt: whatever collects the control dict
    assert "depth" in state and "color_mapping" in state
    assert 1 <= int(state["depth"]) <= 64
    assert state["color_mapping"] in RAMP_MODES


def test_depth_slider_default_is_two():
    panel = ControlPanel()
    assert int(panel.state()["depth"]) == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_controls_ramp.py -q`
Expected: FAIL — no depth/mapping controls.

- [ ] **Step 4: Implement** the Depth slider (1–64, default 2, with its number display
wired via `slider.valueChanged.connect(spin.setValue)` — see the slider-number-display
fix, memory 016) and a Mapping `QComboBox` populated from `RAMP_MODES`, and include their
values in the exported state dict. Match the file's existing construction helpers.

- [ ] **Step 5: Run test to verify it passes + full controls suite**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest tests/test_controls_ramp.py tests/test_controls.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ditherzam/ui/controls.py tests/test_controls_ramp.py
git commit -m "feat(ui): Depth slider (1-64) and ramp Mapping dropdown"
```

---

### Task 9: Full-suite gate + live smoke

**Files:** none (verification only).

- [ ] **Step 1: Full suite, JIT disabled**

Run: `QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 ./.venv/Scripts/python.exe -m pytest -q`
Expected: all green (≥ 375 + new tests).

- [ ] **Step 2: Full suite, JIT enabled** (compiles the new njit N-level paths)

Run: `QT_QPA_PLATFORM=offscreen ./.venv/Scripts/python.exe -m pytest -q`
Expected: all green.

- [ ] **Step 3: Live smoke** — launch the app, load an image, pick Floyd-Steinberg, set a
ramp palette, sweep Depth 2→32 and cycle Mapping modes; confirm the preview shows
multi-tone color and glitch modes visibly differ. (Manual; report result.)

- [ ] **Step 4: Update project memory** via the `zam-memory` skill: record a `progress`
entry (sub-project A shipped: depth ramp, six mapping modes, N-level dither) and a
`gotcha` (threshold slider becomes a tone bias in N-level mode). Commit
`docs/memory`.

- [ ] **Step 5: Final commit** if any memory files changed.

```bash
git add docs/memory && git commit -m "chore(memory): color system core (A) shipped"
```

---

## Self-Review

**Spec coverage:** ramp module + 6 modes (T1) ✓; ColorEngine ramp mode (T2) ✓; N-level
error diffusion (T3) + ordered (T4) with `levels<=2` byte-identity ✓; registry/pipeline
plumbing (T5) ✓; RenderSettings depth/mapping + render + render_cached cache sigs (T6) ✓;
settings_map + presets round-trip (T7) ✓; UI Depth slider + Mapping dropdown (T8) ✓;
both-JIT + live smoke + memory (T9) ✓. Threshold-becomes-bias gotcha documented (T3, T9).
Depth clamp [1,64] (T1). Rec.601 luminance (T1, T2).

**Placeholder scan:** two intentional "adapt to the real names" notes (T3 JJN constants,
T7 preset fn names, T8 panel API) point the implementer at concrete files to read first —
each keeps full assertions/behavior; not open-ended TODOs.

**Type consistency:** `build_ramp(palette, depth, mapping, phase)` and `RAMP_MODES` used
consistently T1→T8; `ColorEngine(..., depth, mapping, phase)` T2/T6; `apply_dither(...,
levels)` T5/T6; `DitherEntry.supports_levels` T5; `RenderSettings.depth`/`.color_mapping`
T6/T7/T8. Consistent.
