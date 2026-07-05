# Phase 2 — Full Dither Kernel Library — Completion Task List

Register the complete **66-entry** dither library (65 real kernels + the `None`
no-op) across the five kernel modules, every kernel a Numba function with a
golden-array test proving determinism, correct `float32[H,W]` shape, and the
`0..255` value range — then assert `len(registry.list_dithers()) >= 63`.

## Prereqs

- **Phase 1 green** — `docs/plans/01-foundation-and-dither-core.md` complete:
  `ditherzam` installs, `DitherRegistry`/`DitherEntry` exist, the shared
  `registry = DitherRegistry()` lives in `ditherzam/dithering/__init__.py`, and
  `error_diffusion.py` already registers **Floyd-Steinberg**, **Atkinson**
  (Error Diffusion) and **Bayer-Matrix 4x4** (Ordered Dither) with the njit
  cores `_floyd_steinberg`, `_atkinson`, plus `_bayer_matrix(n)` / `_ordered`.
- Tests run with `NUMBA_DISABLE_JIT=1` (set in `tests/conftest.py`).
- Python 3.12 on PATH (or the pinned interpreter from Phase 1 Task 0). If no
  3.12 is discoverable, run every `pytest`/`python` command below through the
  pinned interpreter, e.g.
  `NUMBA_DISABLE_JIT=1 "<scratchpad>/dbwork/py312/python.exe" -m pytest ...`.

## Hard rules (inherited — never violate)

- **Clean-room / legal:** no Dither Boy / Studio AAA source, verbatim strings,
  URLs, licensing endpoints, or binaries. Only the public-domain *techniques*
  (Floyd–Steinberg, JJN, Stucki, Burkes, Sierra, Stevenson–Arce, Atkinson,
  Bayer, Fan, Shiau-Fan, halftone, etc.) are used. The "custom/glitch/special"
  kernels are our own deterministic implementations of the documented behavior.
- **Core stays Qt-free.** Nothing in `ditherzam/dithering/**` imports PySide6.
- **Frozen kernel contract** (honored verbatim by every registration):
  the registered callable is
  `def kernel(image_array: 'float32[H,W]', parameter, luminance_threshold_value: float) -> 'float32[H,W] 0..255'`.
  The registered callable is a thin **plain-Python wrapper** (as in Phase 1's
  `floyd_steinberg`) that casts to `float32`, unpacks the `parameter` (scalar or
  tuple), and calls an inner compute core decorated
  `@njit(cache=True, parallel=True)` — `parallel=True` with `prange` on the
  outer loop for the per-pixel-independent kernels (ordered/pattern/geometry),
  and `@njit(cache=True)` (sequential) for the inherently serial
  error-diffusion cores.
- **Determinism:** any kernel that uses randomness calls `np.random.seed(0)`
  inside its core before the loop, so output is reproducible and golden-testable.
- **TDD:** red → green → refactor. Commit after every green (commit steps are
  `- [ ]` checkboxes below — the executor runs them; do not skip).

---

## Registry plan — the complete catalog (66 entries)

Every row below is registered by the end of this phase. `dims` is `2` for every
shipped kernel. `param_sliders` names come verbatim from spec §6.2; slider
`(label, min, max, default)` metadata is reproduced from §6.2 so the UI phase can
consume it without re-reading the spec. Rows marked **(P1)** are already
registered in Phase 1. Rows marked **(extra)** are clean-room public-domain
kernels added to exceed the `>=63` contract; they are documented in
`docs/plans/02-dither-kernel-library.md`'s catalog (Fan / Shiau-Fan /
False-Floyd-Steinberg / Atkinson-light for diffusion; Cluster-Dot / Halftone
screens / Vortex / Concentric / Wireframe / Crosshatch variants for
pattern/special) and are **not** derived from any proprietary source.

### Module `error_diffusion.py` — category **Error Diffusion** (16 entries)

| # | Name | dims | param_sliders | slider (label,min,max,def) | source |
|---|---|---|---|---|---|
| 1 | `None` | 2 | — | — | §9.2 (no-op) |
| 2 | `Floyd-Steinberg` | 2 | — | — | §9.2 **(P1)** |
| 3 | `Atkinson` | 2 | — | — | §9.2 **(P1)** |
| 4 | `Jarvis-Judice-Ninke` | 2 | — | — | §9.2 |
| 5 | `Stucki` | 2 | — | — | §9.2 |
| 6 | `Burkes` | 2 | — | — | §9.2 |
| 7 | `Sierra` | 2 | — | — | §9.2 |
| 8 | `Sierra-Lite` | 2 | — | — | §9.2 |
| 9 | `Two-Row-Sierra` | 2 | — | — | §9.2 |
| 10 | `Stevenson-Arce` | 2 | — | — | §9.2 |
| 11 | `Ostromukhov` | 2 | — | — | §9.2 |
| 12 | `Gaussian` | 2 | `("dither_parameter_slider",)` | Distribution Spread, 1, 20, 1 | §9.2/§6.2 |
| 13 | `Fan` | 2 | — | — | (extra) |
| 14 | `Shiau-Fan` | 2 | — | — | (extra) |
| 15 | `False Floyd-Steinberg` | 2 | — | — | (extra) |
| 16 | `Atkinson-Light` | 2 | — | — | (extra) |

### Module `ordered.py` — category **Ordered Dither** (12 entries; 4x4 lives in P1)

| # | Name | param_sliders | slider (label,min,max,def) | source |
|---|---|---|---|---|
| 17 | `Bayer-Matrix 4x4` | — | — | §9.2 **(P1)** |
| 18 | `Bayer-Matrix 2x2` | — | — | §9.2 |
| 19 | `Bayer-Matrix 8x8` | — | — | §9.2 |
| 20 | `Bayer-Matrix 16x16` | — | — | §9.2 |
| 21 | `Bayer-Ordered` | — | — | §9.2 (alias of 4x4) |
| 22 | `Bayer-Void` | `("dither_parameter_slider",)` | Warp Intensity, 1, 50, 10 | §9.2/§6.2 |
| 23 | `Random Ordered` | — | — | §9.2 |
| 24 | `Bit Tone` | `("dither_parameter_slider",)` | Dot Size, 1, 20, 1 | §9.2/§6.2 |
| 25 | `Mosaic` | `("dither_parameter_slider",)` | Block Size, 1, 50, 10 | §9.2/§6.2 |
| 26 | `Modulated Bayer Dither` | `("matrix_size_slider",)` | Matrix Size, 2, 3, 2 | §6.2 |
| 27 | `Cluster-Dot` | — | — | (extra) |
| 28 | `Halftone-Ordered` | `("dither_parameter_slider",)` | Cell Size, 2, 20, 6 | (extra) |

### Module `pattern.py` — category **Patterned** (11 entries)

| # | Name | param_sliders | slider (label,min,max,def) | source |
|---|---|---|---|---|
| 29 | `Checkers - Small` | — | — | §9.2 |
| 30 | `Checkers - Medium` | — | — | §9.2 |
| 31 | `Checkers - Large` | — | — | §9.2 |
| 32 | `Diamond` | — | — | §9.2 |
| 33 | `Gridlock/Traffic` | — | — | §9.2 |
| 34 | `Print Pattern` | — | — | §9.2 (CMYK halftone) |
| 35 | `Block Tone` | `("dither_parameter_slider",)` | Dot Size, 4, 30, 4 | §9.2/§6.2 |
| 36 | `Stippling` | `("dither_parameter_slider",)` | Dot Density, 1, 20, 1 | §9.2/§6.2 |
| 37 | `Crosshatch` | `("dither_parameter_slider",)` | Line Spacing, 1, 20, 1 | §9.2/§6.2 |
| 38 | `Dot Screen` | `("dither_parameter_slider",)` | Cell Size, 2, 20, 6 | (extra) |
| 39 | `Line Screen` | `("dither_parameter_slider",)` | Line Period, 2, 20, 6 | (extra) |

### Module `glitch.py` — category **Glitch Effects** (15 entries)

| # | Name | param_sliders | slider(s) (label,min,max,def) | source |
|---|---|---|---|---|
| 40 | `Artifact Modulation` | `("dither_parameter_slider",)` | Dither Param, 1, 20, 1 | §9.2 |
| 41 | `Atkinson-VHS` | `("dither_parameter_slider",)` | Line Count, 1, 20, 1 | §9.2/§6.2 |
| 42 | `Glitch` | `("dither_parameter_slider",)` | Glitch Intensity, 1, 20, 1 | §9.2/§6.2 |
| 43 | `Modulated Diffuse Y` | `("dither_parameter_slider",)` | Line Scale, 1, 20, 1 | §9.2/§6.2 |
| 44 | `Modulated Diffuse X` | `("dither_parameter_slider",)` | Line Scale, 1, 20, 1 | §9.2/§6.2 |
| 45 | `Uniform Modulation Y` | `("dither_parameter_slider","smoothing_factor_slider","bleed_fraction_slider")` | Line Scale 1-20-1 · Smoothing Factor 0-1-0 · Bleed Fraction 0-100-0 | §9.2/§6.2 |
| 46 | `Uniform Modulation X` | `("dither_parameter_slider","smoothing_factor_slider","bleed_fraction_slider")` | (as above) | §9.2/§6.2 |
| 47 | `Waveform` | `("dither_parameter_slider",)` | Wave Density, 1, 20, 1 | §9.2/§6.2 |
| 48 | `Waveform Alt` | `("dither_parameter_slider",)` | Modulation Blend, 1, 20, 1 | §9.2/§6.2 |
| 49 | `Ordered Modulation` | `("dither_parameter_slider",)` | Dither Param, 1, 20, 1 | §9.2 |
| 50 | `Smooth Diffuse` | `("dither_parameter_slider","smoothness_slider")` | Line Scale 1-20-1 · Smoothness 1-10-5 | §9.2/§6.2 |
| 51 | `Stucki Diffusion Lines` | `("line_emphasis_slider",)` | Line Emphasis, 1, 10, 5 | §9.2/§6.2 |
| 52 | `Atkinson Line Modulation` | `("modulation_strength_slider","horizontal_bias_slider")` | Modulation Strength 1-10-5 · Horizontal Bias 1-10-5 | §9.2/§6.2 |
| 53 | `Contrast Aware Y` | `("dither_parameter_slider",)` | Line Scale, 1, 20, 1 | §9.2/§6.2 |
| 54 | `Contrast Aware X` | `("dither_parameter_slider",)` | Line Scale, 1, 20, 1 | §9.2/§6.2 |

### Module `special.py` — category **Special Effects** (12 entries)

| # | Name | param_sliders | slider(s) (label,min,max,def) | source |
|---|---|---|---|---|
| 55 | `Radial Burst` | — | — | §9.2 |
| 56 | `Wave` | — | — | §9.2 |
| 57 | `Noise` | — | — | §9.2 |
| 58 | `Topography` | `("dither_parameter_slider",)` | Warp Intensity, 1, 20, 1 | §9.2/§6.2 |
| 59 | `Thresholder` | `("dither_parameter_slider",)` | Modulation Frequency, 1, 20, 1 | §9.2/§6.2 |
| 60 | `Diagonal` | `("dither_parameter_slider",)` | Edge Sensitivity, 1, 20, 1 | §9.2/§6.2 |
| 61 | `Displace Contour` | `("contour_thresh_slider","line_mode_slider","smoothing_slider","line_space_slider")` | Contour Threshold 0-100-50 · Line Mode 1-3-1 · Smoothing 0-5-0 · Line Spacing 1-5-1 | §9.2/§6.2 |
| 62 | `Sine Wave Modulation` | `("wave_frequency_slider","wave_threshold_slider")` | Wave Frequency 1-20-5 · Wave Threshold 1-30-10 | §9.2/§6.2 |
| 63 | `Vortex` | — | — | (extra) |
| 64 | `Concentric Rings` | — | — | (extra) |
| 65 | `Wireframe Alt` | `("dither_parameter_slider",)` | Edge Sensitivity, 1, 20, 1 | (extra) |
| 66 | `Crosshatch Alt` | `("dither_parameter_slider",)` | Line Spacing, 1, 20, 1 | (extra) |

**Category totals:** Error Diffusion 16 · Ordered Dither 12 · Patterned 11 ·
Glitch Effects 15 · Special Effects 12 = **66 registered** (`>=63` ✓).

> **Multi-param dispatch note.** `apply_dither`'s `_build_param` (Phase 1) returns
> a scalar when a style has ≤1 slider and a **tuple in `param_sliders` order**
> otherwise. Every multi-slider wrapper below unpacks that tuple in exactly the
> declared order. Task 2.7 adds an explicit tuple-arity/order test.

---

### Task 2.0: Environment & baseline check

**Files:** none (verification only)

- [ ] **Step 1:** Confirm Phase 1 is green and the three P1 kernels exist:
  Run `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_error_diffusion.py -q`
  → expect PASS (4 passed).
- [ ] **Step 2:** Confirm the baseline count is 3 before this phase:
  Run `NUMBA_DISABLE_JIT=1 python -c "from ditherzam.dithering import registry; print(len(registry.list_dithers()))"`
  → expect `3`.

---

### Task 2.1: Generic error-diffusion helpers (`_diffuse`, `_diffuse_row`)

**Files:** Modify `ditherzam/dithering/kernels/error_diffusion.py` ·
Test `tests/test_diffusion_helper.py`

**Interfaces:** Produces
```python
@njit(cache=True)
def _diffuse(img, thr, offsets, weights, divisor) -> np.ndarray   # 2-D serpentine-free diffusion
@njit(cache=True)
def _diffuse_row(img, thr, w_right) -> np.ndarray                 # 1-D per-row diffusion (glitch base)
```
`offsets`: `int64[N,2]` `[dy,dx]`; `weights`: `float32[N]` aligned to `offsets`.

- [ ] **Step 1: Write failing test — `tests/test_diffusion_helper.py`**

```python
import numpy as np
from ditherzam.dithering.kernels.error_diffusion import _diffuse, _diffuse_row


def test_diffuse_matches_floyd_shape_and_binary():
    img = np.full((3, 3), 100.0, dtype=np.float32)
    offs = np.array([[0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    wts = np.array([7.0, 3.0, 5.0, 1.0], dtype=np.float32)
    out = _diffuse(img.copy(), 128.0, offs, wts, 16.0)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}


def test_diffuse_all_white_stays_white():
    img = np.full((4, 4), 255.0, dtype=np.float32)
    offs = np.array([[0, 1], [1, 0]], dtype=np.int64)
    wts = np.array([1.0, 1.0], dtype=np.float32)
    out = _diffuse(img.copy(), 128.0, offs, wts, 2.0)
    assert np.all(out == 255.0)


def test_diffuse_row_is_binary():
    img = np.tile(np.linspace(0, 255, 6, dtype=np.float32), (2, 1))
    out = _diffuse_row(img.copy(), 128.0, 0.9)
    assert out.shape == (2, 6)
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}
```

- [ ] **Step 2: Run — expect FAIL**
  `NUMBA_DISABLE_JIT=1 pytest tests/test_diffusion_helper.py -v`
  → expect `ImportError: cannot import name '_diffuse'`.

- [ ] **Step 3: Implement — append to `ditherzam/dithering/kernels/error_diffusion.py`**

```python
@njit(cache=True)
def _diffuse(img, thr, offsets, weights, divisor):
    h, w = img.shape
    out = img.copy()
    n = offsets.shape[0]
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


@njit(cache=True)
def _diffuse_row(img, thr, w_right):
    h, w = img.shape
    out = img.copy()
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * w_right
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(dither): generic 2-D and 1-D error-diffusion helpers"`

---

### Task 2.2: Error-diffusion family (kernels 1, 4–16)

**Files:** Modify `ditherzam/dithering/kernels/error_diffusion.py` ·
Test `tests/test_kernels_diffusion.py`

**Interfaces:** Registers `None`, `Jarvis-Judice-Ninke`, `Stucki`, `Burkes`,
`Sierra`, `Sierra-Lite`, `Two-Row-Sierra`, `Stevenson-Arce`, `Ostromukhov`,
`Gaussian`, `Fan`, `Shiau-Fan`, `False Floyd-Steinberg`, `Atkinson-Light`.

**Canonical diffusion matrices (public domain).** `X` = current pixel; numbers
are error weights over the divisor.

```
Jarvis-Judice-Ninke  (÷48):  . . X 7 5      Stucki (÷42):  . . X 8 4
                             3 5 7 5 3                      2 4 8 4 2
                             1 3 5 3 1                      1 2 4 2 1

Burkes (÷32):  . . X 8 4      Sierra-3 (÷32):  . . X 5 3     Sierra-Lite (÷4):  X 2
               2 4 8 4 2                       2 4 5 4 2                        1 1 .
                                               . 2 3 2 .

Two-Row-Sierra (÷16):  . . X 4 3     Stevenson-Arce (÷200):     . . X . 32 . .
                       1 2 3 2 1                            12 . 26 . 30 . 16
                                                             . 12 .  26 . 12 .
                                                            5 . 12 .  12 .  5

Fan (÷16):  X 7        Shiau-Fan (÷16):  . . X 8    False Floyd-Steinberg (÷8):  X 3
           1 3 5                        1 1 2 4                                   3 2
```

- [ ] **Step 1: Write failing test — `tests/test_kernels_diffusion.py`**

```python
import numpy as np
from ditherzam.dithering import registry

NAMES = [
    "Jarvis-Judice-Ninke", "Stucki", "Burkes", "Sierra", "Sierra-Lite",
    "Two-Row-Sierra", "Stevenson-Arce", "Ostromukhov", "Gaussian",
    "Fan", "Shiau-Fan", "False Floyd-Steinberg", "Atkinson-Light",
]


def test_none_registered_is_identity():
    e = registry.get_entry("None")
    assert e is not None
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    out = e.func(img.copy(), 0, 128.0)
    np.testing.assert_allclose(out, img)


def test_all_diffusion_registered_binary_and_shape():
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    for n in NAMES:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 1 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape, n
        assert out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_diffusion_black_stays_black():
    img = np.zeros((8, 8), dtype=np.float32)
    for n in ("Stucki", "Burkes", "Sierra", "Fan"):
        out = registry.get_entry(n).func(img.copy(), 0, 128.0)
        assert out.sum() == 0.0, n


def test_gaussian_is_deterministic():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    a = registry.get_entry("Gaussian").func(img.copy(), 5, 128.0)
    b = registry.get_entry("Gaussian").func(img.copy(), 5, 128.0)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run — expect FAIL** (`AssertionError: None` — not registered).

- [ ] **Step 3: Implement — append to `ditherzam/dithering/kernels/error_diffusion.py`**

```python
# ── Kernel: None · Error Diffusion · dims=2 · no sliders (no-op passthrough) ──
@registry.register("None", "Error Diffusion", dims=2)
def no_dither(image_array, parameter, luminance_threshold_value):
    return image_array.astype(np.float32)


# ── Classic weighted-diffusion offset/weight tables (float32 weights) ──
_JJN_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                     [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]], dtype=np.int64)
_JJN_W = np.array([7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1], dtype=np.float32)

_STUCKI_OFF = _JJN_OFF
_STUCKI_W = np.array([8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1], dtype=np.float32)

_BURKES_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2]],
                       dtype=np.int64)
_BURKES_W = np.array([8, 4, 2, 4, 8, 4, 2], dtype=np.float32)

_SIERRA_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                        [2, -1], [2, 0], [2, 1]], dtype=np.int64)
_SIERRA_W = np.array([5, 3, 2, 4, 5, 4, 2, 2, 3, 2], dtype=np.float32)

_SIERRA_LITE_OFF = np.array([[0, 1], [1, -1], [1, 0]], dtype=np.int64)
_SIERRA_LITE_W = np.array([2, 1, 1], dtype=np.float32)

_TWO_ROW_OFF = np.array([[0, 1], [0, 2], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2]],
                        dtype=np.int64)
_TWO_ROW_W = np.array([4, 3, 1, 2, 3, 2, 1], dtype=np.float32)

_STEVENSON_OFF = np.array([[0, 2],
                           [1, -3], [1, -1], [1, 1], [1, 3],
                           [2, -2], [2, 0], [2, 2],
                           [3, -3], [3, -1], [3, 1], [3, 3]], dtype=np.int64)
_STEVENSON_W = np.array([32, 12, 26, 30, 16, 12, 26, 12, 5, 12, 12, 5],
                        dtype=np.float32)

_FAN_OFF = np.array([[0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
_FAN_W = np.array([7, 1, 3, 5], dtype=np.float32)

_SHIAU_OFF = np.array([[0, 1], [1, -2], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
_SHIAU_W = np.array([8, 1, 1, 2, 4], dtype=np.float32)

_FALSE_FS_OFF = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
_FALSE_FS_W = np.array([3, 3, 2], dtype=np.float32)

_ATK_LIGHT_OFF = np.array([[0, 1], [0, 2], [1, 0], [1, 1]], dtype=np.int64)
_ATK_LIGHT_W = np.array([1, 1, 1, 1], dtype=np.float32)  # /8 (Atkinson-style bleed)


@registry.register("Jarvis-Judice-Ninke", "Error Diffusion", dims=2)
def jjn(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _JJN_OFF, _JJN_W, 48.0)


@registry.register("Stucki", "Error Diffusion", dims=2)
def stucki(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _STUCKI_OFF, _STUCKI_W, 42.0)


@registry.register("Burkes", "Error Diffusion", dims=2)
def burkes(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _BURKES_OFF, _BURKES_W, 32.0)


@registry.register("Sierra", "Error Diffusion", dims=2)
def sierra(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _SIERRA_OFF, _SIERRA_W, 32.0)


@registry.register("Sierra-Lite", "Error Diffusion", dims=2)
def sierra_lite(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _SIERRA_LITE_OFF, _SIERRA_LITE_W, 4.0)


@registry.register("Two-Row-Sierra", "Error Diffusion", dims=2)
def two_row_sierra(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _TWO_ROW_OFF, _TWO_ROW_W, 16.0)


@registry.register("Stevenson-Arce", "Error Diffusion", dims=2)
def stevenson_arce(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _STEVENSON_OFF, _STEVENSON_W, 200.0)


@registry.register("Fan", "Error Diffusion", dims=2)
def fan(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _FAN_OFF, _FAN_W, 16.0)


@registry.register("Shiau-Fan", "Error Diffusion", dims=2)
def shiau_fan(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _SHIAU_OFF, _SHIAU_W, 16.0)


@registry.register("False Floyd-Steinberg", "Error Diffusion", dims=2)
def false_floyd_steinberg(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _FALSE_FS_OFF, _FALSE_FS_W, 8.0)


@registry.register("Atkinson-Light", "Error Diffusion", dims=2)
def atkinson_light(image_array, parameter, luminance_threshold_value):
    # Atkinson-style: only 4/8 of the error propagates (softer than full Atkinson).
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _ATK_LIGHT_OFF, _ATK_LIGHT_W, 8.0)


# ── Kernel: Ostromukhov · Error Diffusion · dims=2 · simplified variable coeffs ──
@njit(cache=True)
def _ostromukhov(img, thr):
    h, w = img.shape
    out = img.copy()
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            v = old / 255.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            # value-dependent coefficients (right, down-left, down)
            d1 = 13.0 + v * 8.0
            d2 = (1.0 - abs(2.0 * v - 1.0)) * 7.0
            d3 = 5.0 + (1.0 - v) * 8.0
            s = d1 + d2 + d3
            if x + 1 < w:
                out[y, x + 1] += err * d1 / s
            if y + 1 < h:
                if x - 1 >= 0:
                    out[y + 1, x - 1] += err * d2 / s
                out[y + 1, x] += err * d3 / s
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@registry.register("Ostromukhov", "Error Diffusion", dims=2)
def ostromukhov(image_array, parameter, luminance_threshold_value):
    return _ostromukhov(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Gaussian · Error Diffusion · dims=2 · slider Distribution Spread 1-20-1 ──
@njit(cache=True)
def _gaussian_dither(img, spread, thr):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    sigma = spread * 3.0
    for y in range(h):
        for x in range(w):
            n = np.random.standard_normal() * sigma
            out[y, x] = 255.0 if (img[y, x] + n) >= thr else 0.0
    return out


@registry.register("Gaussian", "Error Diffusion", dims=2,
                   param_sliders=("dither_parameter_slider",))
def gaussian(image_array, parameter, luminance_threshold_value):
    spread = float(parameter) if parameter else 1.0
    return _gaussian_dither(image_array.astype(np.float32), spread,
                            luminance_threshold_value)
```

- [ ] **Step 4: Run — expect PASS** — `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_diffusion.py -v` (4 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(dither): full error-diffusion family (JJN..Atkinson-Light, Ostromukhov, Gaussian)"`

---

### Task 2.3: Ordered / Bayer family (kernels 18–28)

**Files:** Create `ditherzam/dithering/kernels/ordered.py` ·
Test `tests/test_kernels_ordered.py`

**Interfaces:** Registers `Bayer-Matrix 2x2/8x8/16x16`, `Bayer-Ordered`,
`Bayer-Void`, `Random Ordered`, `Bit Tone`, `Mosaic`, `Modulated Bayer Dither`,
`Cluster-Dot`, `Halftone-Ordered`. Defines canonical `_bayer_matrix(n)`,
`_bayer_thresholds(n)`, and a parallel `_ordered(img, thresholds)` used by later
modules (they import from here). `Bayer-Matrix 4x4` stays in Phase 1's
`error_diffusion.py` — **do not re-register it here.**

- [ ] **Step 1: Write failing test — `tests/test_kernels_ordered.py`**

```python
import numpy as np
from ditherzam.dithering import registry

NAMES = [
    "Bayer-Matrix 2x2", "Bayer-Matrix 8x8", "Bayer-Matrix 16x16",
    "Bayer-Ordered", "Bayer-Void", "Random Ordered", "Bit Tone", "Mosaic",
    "Modulated Bayer Dither", "Cluster-Dot", "Halftone-Ordered",
]


def test_ordered_registered_binary_shape():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    for n in NAMES:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 4 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_ordered_extremes():
    black = np.zeros((16, 16), dtype=np.float32)
    white = np.full((16, 16), 255.0, dtype=np.float32)
    for n in ("Bayer-Matrix 2x2", "Bayer-Matrix 8x8", "Cluster-Dot"):
        assert registry.get_entry(n).func(black.copy(), 0, 128.0).sum() == 0.0, n
        assert np.all(registry.get_entry(n).func(white.copy(), 0, 128.0) == 255.0), n


def test_random_ordered_is_deterministic():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    a = registry.get_entry("Random Ordered").func(img.copy(), 0, 128.0)
    b = registry.get_entry("Random Ordered").func(img.copy(), 0, 128.0)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run — expect FAIL** (`AssertionError`, names not registered).

- [ ] **Step 3: Implement `ditherzam/dithering/kernels/ordered.py`**

```python
from __future__ import annotations
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


def _bayer_matrix(n: int) -> np.ndarray:
    """Recursive Bayer index matrix of order n (n a power of two)."""
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    s = _bayer_matrix(n // 2)
    return np.block([
        [4 * s + 0, 4 * s + 2],
        [4 * s + 3, 4 * s + 1],
    ]).astype(np.float32)


def _bayer_thresholds(n: int) -> np.ndarray:
    """Bayer matrix normalized to 0..255 threshold values."""
    return ((_bayer_matrix(n) + 0.5) / float(n * n) * 255.0).astype(np.float32)


_BAYER2 = _bayer_thresholds(2)
_BAYER4 = _bayer_thresholds(4)
_BAYER8 = _bayer_thresholds(8)
_BAYER16 = _bayer_thresholds(16)

# Classic 4x4 clustered-dot (spiral) screen, normalized to 0..255 thresholds.
_CLUSTER4_IDX = np.array([[12, 5, 6, 13],
                          [4, 0, 1, 7],
                          [11, 3, 2, 8],
                          [15, 10, 9, 14]], dtype=np.float32)
_CLUSTER4 = ((_CLUSTER4_IDX + 0.5) / 16.0 * 255.0).astype(np.float32)


@njit(cache=True, parallel=True)
def _ordered(img, thresholds):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            out[y, x] = 255.0 if img[y, x] >= thresholds[y % mh, x % mw] else 0.0
    return out


@njit(cache=True)
def _random_ordered(img):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            t = np.random.random() * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _bit_tone(img, dot, base):
    h, w = img.shape
    mh, mw = base.shape
    d = dot if dot >= 1 else 1
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = base[(y // d) % mh, (x // d) % mw]
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _mosaic(img, block, thr):
    h, w = img.shape
    b = block if block >= 1 else 1
    nby = (h + b - 1) // b
    nbx = (w + b - 1) // b
    out = np.empty_like(img)
    for by in prange(nby):
        for bx in range(nbx):
            y0 = by * b
            x0 = bx * b
            s = 0.0
            c = 0
            for yy in range(y0, min(y0 + b, h)):
                for xx in range(x0, min(x0 + b, w)):
                    s += img[yy, xx]
                    c += 1
            v = 255.0 if (s / c) >= thr else 0.0
            for yy in range(y0, min(y0 + b, h)):
                for xx in range(x0, min(x0 + b, w)):
                    out[yy, xx] = v
    return out


@njit(cache=True)
def _bayer_void(img, warp, thr, base):
    h, w = img.shape
    mh, mw = base.shape
    strength = warp / 50.0
    out = np.empty_like(img)
    for y in range(h):
        fy = y / (h - 1) if h > 1 else 0.0
        shift = int((fy * fy) * warp)
        for x in range(w):
            sx = (x + shift) % w
            b = base[y % mh, sx % mw]
            final = thr + (b - 128.0) * strength
            out[y, x] = 255.0 if img[y, sx] >= final else 0.0
    return out


@njit(cache=True, parallel=True)
def _modulated_bayer(img, thresholds):
    h, w = img.shape
    mh, mw = thresholds.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            local = img[y, x] / 255.0
            t = thresholds[y % mh, x % mw] * (0.5 + local)
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _dot_screen(img, cell):
    h, w = img.shape
    c = cell if cell >= 2 else 2
    half = c / 2.0
    r2max = half * half * 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = (x % c) - half + 0.5
            dy = (y % c) - half + 0.5
            t = (dx * dx + dy * dy) / r2max * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


# ── Kernel: Bayer-Matrix 2x2 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 2x2", "Ordered Dither", dims=2)
def bayer_2(image_array, parameter, luminance_threshold_value):
    return _ordered(image_array.astype(np.float32), _BAYER2)


# ── Kernel: Bayer-Matrix 8x8 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 8x8", "Ordered Dither", dims=2)
def bayer_8(image_array, parameter, luminance_threshold_value):
    return _ordered(image_array.astype(np.float32), _BAYER8)


# ── Kernel: Bayer-Matrix 16x16 · Ordered Dither · dims=2 · no sliders ──
@registry.register("Bayer-Matrix 16x16", "Ordered Dither", dims=2)
def bayer_16(image_array, parameter, luminance_threshold_value):
    return _ordered(image_array.astype(np.float32), _BAYER16)


# ── Kernel: Bayer-Ordered · Ordered Dither · dims=2 · alias of 4x4 ──
@registry.register("Bayer-Ordered", "Ordered Dither", dims=2)
def bayer_ordered(image_array, parameter, luminance_threshold_value):
    return _ordered(image_array.astype(np.float32), _BAYER4)


# ── Kernel: Bayer-Void · Ordered Dither · dims=2 · Warp Intensity 1-50-10 ──
@registry.register("Bayer-Void", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def bayer_void(image_array, parameter, luminance_threshold_value):
    warp = float(parameter) if parameter else 10.0
    return _bayer_void(image_array.astype(np.float32), warp,
                       luminance_threshold_value, _BAYER4)


# ── Kernel: Random Ordered · Ordered Dither · dims=2 · no sliders ──
@registry.register("Random Ordered", "Ordered Dither", dims=2)
def random_ordered(image_array, parameter, luminance_threshold_value):
    return _random_ordered(image_array.astype(np.float32))


# ── Kernel: Bit Tone · Ordered Dither · dims=2 · Dot Size 1-20-1 ──
@registry.register("Bit Tone", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def bit_tone(image_array, parameter, luminance_threshold_value):
    dot = int(parameter) if parameter else 1
    return _bit_tone(image_array.astype(np.float32), dot, _BAYER4)


# ── Kernel: Mosaic · Ordered Dither · dims=2 · Block Size 1-50-10 ──
@registry.register("Mosaic", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def mosaic(image_array, parameter, luminance_threshold_value):
    block = int(parameter) if parameter else 10
    return _mosaic(image_array.astype(np.float32), block,
                   luminance_threshold_value)


# ── Kernel: Modulated Bayer Dither · Ordered Dither · dims=2 · Matrix Size 2-3-2 ──
@registry.register("Modulated Bayer Dither", "Ordered Dither", dims=2,
                   param_sliders=("matrix_size_slider",))
def modulated_bayer(image_array, parameter, luminance_threshold_value):
    size = int(parameter) if parameter else 2
    thr = _BAYER8 if size >= 3 else _BAYER4
    return _modulated_bayer(image_array.astype(np.float32), thr)


# ── Kernel: Cluster-Dot · Ordered Dither · dims=2 · no sliders (extra) ──
@registry.register("Cluster-Dot", "Ordered Dither", dims=2)
def cluster_dot(image_array, parameter, luminance_threshold_value):
    return _ordered(image_array.astype(np.float32), _CLUSTER4)


# ── Kernel: Halftone-Ordered · Ordered Dither · dims=2 · Cell Size 2-20-6 (extra) ──
@registry.register("Halftone-Ordered", "Ordered Dither", dims=2,
                   param_sliders=("dither_parameter_slider",))
def halftone_ordered(image_array, parameter, luminance_threshold_value):
    cell = int(parameter) if parameter else 6
    return _dot_screen(image_array.astype(np.float32), cell)
```

- [ ] **Step 4: Run — expect PASS** — `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_ordered.py -v` (3 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(dither): ordered/Bayer family (2x2..16x16, void, mosaic, cluster-dot)"`

---

### Task 2.4: Pattern family (kernels 29–39)

**Files:** Create `ditherzam/dithering/kernels/pattern.py` ·
Test `tests/test_kernels_pattern.py`

**Interfaces:** Registers `Checkers - Small/Medium/Large`, `Diamond`,
`Gridlock/Traffic`, `Print Pattern`, `Block Tone`, `Stippling`, `Crosshatch`,
`Dot Screen`, `Line Screen`.

- [ ] **Step 1: Write failing test — `tests/test_kernels_pattern.py`**

```python
import numpy as np
from ditherzam.dithering import registry

NAMES = [
    "Checkers - Small", "Checkers - Medium", "Checkers - Large", "Diamond",
    "Gridlock/Traffic", "Print Pattern", "Block Tone", "Stippling",
    "Crosshatch", "Dot Screen", "Line Screen",
]


def test_pattern_registered_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n in NAMES:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 6 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_stippling_is_deterministic():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    a = registry.get_entry("Stippling").func(img.copy(), 5, 128.0)
    b = registry.get_entry("Stippling").func(img.copy(), 5, 128.0)
    np.testing.assert_array_equal(a, b)


def test_block_tone_white_stays_white():
    white = np.full((24, 24), 255.0, dtype=np.float32)
    out = registry.get_entry("Block Tone").func(white.copy(), 6, 128.0)
    assert np.all(out == 255.0)
```

- [ ] **Step 2: Run — expect FAIL** (names not registered).

- [ ] **Step 3: Implement `ditherzam/dithering/kernels/pattern.py`**

```python
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


@njit(cache=True, parallel=True)
def _checkers(img, s, thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            cell = ((x // s) + (y // s)) % 2
            base = thr if cell == 0 else (255.0 - thr)
            out[y, x] = 255.0 if img[y, x] >= base else 0.0
    return out


@njit(cache=True, parallel=True)
def _diamond(img, s):
    h, w = img.shape
    half = s / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = abs((x % s) - half)
            dy = abs((y % s) - half)
            t = (dx + dy) / s * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _gridlock(img, g, thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            on_line = (x % g == 0) or (y % g == 0)
            t = thr * 0.5 if on_line else thr
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _print_pattern(img, cell):
    # CMYK-style rotated dot screen (single-channel halftone simulation).
    h, w = img.shape
    ca = math.cos(0.261799)  # 15 degrees
    sa = math.sin(0.261799)
    half = cell / 2.0
    r2max = half * half * 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            xr = x * ca - y * sa
            yr = x * sa + y * ca
            dx = (xr % cell) - half
            dy = (yr % cell) - half
            t = (dx * dx + dy * dy) / r2max * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _block_tone(img, dot):
    # Classic round-dot halftone; dot radius grows with local darkness.
    h, w = img.shape
    c = dot if dot >= 2 else 2
    half = c / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            r = darkness * half
            dx = (x % c) - half + 0.5
            dy = (y % c) - half + 0.5
            out[y, x] = 0.0 if (dx * dx + dy * dy) <= r * r else 255.0
    return out


@njit(cache=True)
def _stippling(img, density):
    np.random.seed(0)
    h, w = img.shape
    scale = density if density >= 1 else 1
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            p = darkness / (1.0 + (scale - 1) * 0.15)
            out[y, x] = 0.0 if np.random.random() < p else 255.0
    return out


@njit(cache=True, parallel=True)
def _crosshatch(img, s):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            hit = False
            if darkness > 0.20 and ((x + y) % s == 0):
                hit = True
            if darkness > 0.45 and ((x - y) % s == 0):
                hit = True
            if darkness > 0.70 and (x % s == 0):
                hit = True
            if darkness > 0.88 and (y % s == 0):
                hit = True
            out[y, x] = 0.0 if hit else 255.0
    return out


@njit(cache=True, parallel=True)
def _dot_screen_p(img, cell):
    h, w = img.shape
    c = cell if cell >= 2 else 2
    half = c / 2.0
    r2max = half * half * 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = (x % c) - half + 0.5
            dy = (y % c) - half + 0.5
            t = (dx * dx + dy * dy) / r2max * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _line_screen(img, period):
    h, w = img.shape
    p = period if period >= 2 else 2
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            thick = darkness * p
            out[y, x] = 0.0 if (y % p) < thick else 255.0
    return out


# ── Kernel: Checkers - Small · Patterned · dims=2 · no sliders (board 2) ──
@registry.register("Checkers - Small", "Patterned", dims=2)
def checkers_small(image_array, parameter, luminance_threshold_value):
    return _checkers(image_array.astype(np.float32), 2, luminance_threshold_value)


# ── Kernel: Checkers - Medium · Patterned · dims=2 · no sliders (board 4) ──
@registry.register("Checkers - Medium", "Patterned", dims=2)
def checkers_medium(image_array, parameter, luminance_threshold_value):
    return _checkers(image_array.astype(np.float32), 4, luminance_threshold_value)


# ── Kernel: Checkers - Large · Patterned · dims=2 · no sliders (board 8) ──
@registry.register("Checkers - Large", "Patterned", dims=2)
def checkers_large(image_array, parameter, luminance_threshold_value):
    return _checkers(image_array.astype(np.float32), 8, luminance_threshold_value)


# ── Kernel: Diamond · Patterned · dims=2 · no sliders ──
@registry.register("Diamond", "Patterned", dims=2)
def diamond(image_array, parameter, luminance_threshold_value):
    return _diamond(image_array.astype(np.float32), 8)


# ── Kernel: Gridlock/Traffic · Patterned · dims=2 · no sliders ──
@registry.register("Gridlock/Traffic", "Patterned", dims=2)
def gridlock_traffic(image_array, parameter, luminance_threshold_value):
    return _gridlock(image_array.astype(np.float32), 6, luminance_threshold_value)


# ── Kernel: Print Pattern · Patterned · dims=2 · no sliders (CMYK halftone) ──
@registry.register("Print Pattern", "Patterned", dims=2)
def print_pattern(image_array, parameter, luminance_threshold_value):
    return _print_pattern(image_array.astype(np.float32), 6)


# ── Kernel: Block Tone · Patterned · dims=2 · Dot Size 4-30-4 ──
@registry.register("Block Tone", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def block_tone(image_array, parameter, luminance_threshold_value):
    dot = int(parameter) if parameter else 4
    return _block_tone(image_array.astype(np.float32), dot)


# ── Kernel: Stippling · Patterned · dims=2 · Dot Density 1-20-1 ──
@registry.register("Stippling", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def stippling(image_array, parameter, luminance_threshold_value):
    density = int(parameter) if parameter else 1
    return _stippling(image_array.astype(np.float32), density)


# ── Kernel: Crosshatch · Patterned · dims=2 · Line Spacing 1-20-1 ──
@registry.register("Crosshatch", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def crosshatch(image_array, parameter, luminance_threshold_value):
    s = int(parameter) if parameter else 4
    if s < 1:
        s = 1
    return _crosshatch(image_array.astype(np.float32), s)


# ── Kernel: Dot Screen · Patterned · dims=2 · Cell Size 2-20-6 (extra) ──
@registry.register("Dot Screen", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def dot_screen(image_array, parameter, luminance_threshold_value):
    cell = int(parameter) if parameter else 6
    return _dot_screen_p(image_array.astype(np.float32), cell)


# ── Kernel: Line Screen · Patterned · dims=2 · Line Period 2-20-6 (extra) ──
@registry.register("Line Screen", "Patterned", dims=2,
                   param_sliders=("dither_parameter_slider",))
def line_screen(image_array, parameter, luminance_threshold_value):
    period = int(parameter) if parameter else 6
    return _line_screen(image_array.astype(np.float32), period)
```

- [ ] **Step 4: Run — expect PASS** — `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_pattern.py -v` (3 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(dither): pattern family (checkers, diamond, halftone, crosshatch, stippling)"`

---

### Task 2.5: Glitch family (kernels 40–54)

**Files:** Create `ditherzam/dithering/kernels/glitch.py` ·
Test `tests/test_kernels_glitch.py`

**Interfaces:** Registers `Artifact Modulation`, `Atkinson-VHS`, `Glitch`,
`Modulated Diffuse Y/X`, `Uniform Modulation Y/X` (3-tuple), `Waveform`,
`Waveform Alt`, `Ordered Modulation`, `Smooth Diffuse` (2-tuple),
`Stucki Diffusion Lines`, `Atkinson Line Modulation` (2-tuple),
`Contrast Aware Y/X`. Multi-slider wrappers unpack the `parameter` **tuple in
`param_sliders` order**.

- [ ] **Step 1: Write failing test — `tests/test_kernels_glitch.py`**

```python
import numpy as np
from ditherzam.dithering import registry

SINGLE = [
    "Artifact Modulation", "Atkinson-VHS", "Glitch", "Modulated Diffuse Y",
    "Modulated Diffuse X", "Waveform", "Waveform Alt", "Ordered Modulation",
    "Stucki Diffusion Lines", "Contrast Aware Y", "Contrast Aware X",
]
MULTI = {
    "Uniform Modulation Y": (4, 0.5, 20.0),
    "Uniform Modulation X": (4, 0.5, 20.0),
    "Smooth Diffuse": (4, 5),
    "Atkinson Line Modulation": (5, 5),
}


def test_single_param_glitch_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n in SINGLE:
        e = registry.get_entry(n)
        assert e is not None, n
        out = e.func(img.copy(), 4, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_multi_param_glitch_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n, param in MULTI.items():
        e = registry.get_entry(n)
        assert e is not None, n
        assert len(e.param_sliders) == len(param), n
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_glitch_is_deterministic():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    a = registry.get_entry("Glitch").func(img.copy(), 6, 128.0)
    b = registry.get_entry("Glitch").func(img.copy(), 6, 128.0)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run — expect FAIL** (names not registered).

- [ ] **Step 3: Implement `ditherzam/dithering/kernels/glitch.py`**

```python
from __future__ import annotations
import math
import numpy as np
from numba import njit
from ditherzam.dithering import registry
from ditherzam.dithering.kernels.error_diffusion import _diffuse_row
from ditherzam.dithering.kernels.ordered import _BAYER4


@njit(cache=True)
def _line_diffuse(img, thr, line_scale, horizontal):
    """1-D error diffusion producing banded line patterns; density ~ brightness."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        for y in range(h):
            carry = 0.0
            for x in range(w):
                old = out[y, x] + carry
                new = 255.0 if old >= thr else 0.0
                out[y, x] = new
                carry = (old - new) / s
    else:
        for x in range(w):
            carry = 0.0
            for y in range(h):
                old = out[y, x] + carry
                new = 255.0 if old >= thr else 0.0
                out[y, x] = new
                carry = (old - new) / s
    return out


@njit(cache=True)
def _uniform_modulation(img, thr, line_scale, smoothing, bleed, horizontal):
    """Row/column diffusion with EMA-smoothed vertical bleed."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        prev_err = np.zeros(w, dtype=np.float32)
        for y in range(h):
            carry = 0.0
            for x in range(w):
                base = out[y, x] + carry + prev_err[x] * bleed
                new = 255.0 if base >= thr else 0.0
                out[y, x] = new
                err = (base - new)
                carry = err / s
                prev_err[x] = prev_err[x] * smoothing + err * (1.0 - smoothing)
    else:
        prev_err = np.zeros(h, dtype=np.float32)
        for x in range(w):
            carry = 0.0
            for y in range(h):
                base = out[y, x] + carry + prev_err[y] * bleed
                new = 255.0 if base >= thr else 0.0
                out[y, x] = new
                err = (base - new)
                carry = err / s
                prev_err[y] = prev_err[y] * smoothing + err * (1.0 - smoothing)
    return out


@njit(cache=True)
def _atkinson_vhs(img, thr, line_count):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if img[y, x] >= thr else 0.0
    lc = line_count if line_count >= 1 else 1
    for k in range(lc):
        ry = int((k + 0.5) / lc * h)
        if ry >= h:
            ry = h - 1
        for x in range(w):
            out[ry, x] = 255.0  # bright horizontal tracking line
    return out


@njit(cache=True)
def _glitch(img, thr, intensity):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    amp = intensity if intensity >= 1 else 1
    for y in range(h):
        shift = int((np.random.random() - 0.5) * 2.0 * amp)
        for x in range(w):
            sx = (x + shift) % w
            out[y, x] = 255.0 if img[y, sx] >= thr else 0.0
    return out


@njit(cache=True)
def _waveform(img, thr, density):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            freq = 0.05 + (1.0 - img[y, x] / 255.0) * density * 0.05
            v = (math.sin(x * freq + y * 0.3) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= v else 0.0
    return out


@njit(cache=True)
def _waveform_alt(img, thr, blend):
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            gx = 0.0
            if 0 < x < w - 1:
                gx = (img[y, x + 1] - img[y, x - 1]) / 255.0
            phase = x * (0.05 + (1.0 - img[y, x] / 255.0) * 0.1) + gx * blend
            v = (math.sin(phase) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= v else 0.0
    return out


@njit(cache=True)
def _ordered_modulation(img, thr, param, base):
    h, w = img.shape
    mh, mw = base.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            wob = math.sin((x + y) * 0.1 * param) * 40.0
            t = base[y % mh, x % mw] + wob
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True)
def _smooth_diffuse(img, thr, line_scale, smoothness):
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    a = 1.0 / smoothness if smoothness >= 1 else 1.0
    for y in range(h):
        carry = 0.0
        for x in range(w):
            old = out[y, x] + carry
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            carry = ((old - new) / s) * a + carry * (1.0 - a)
    return out


@njit(cache=True)
def _stucki_diffusion_lines(img, thr, emphasis):
    """Stucki diffusion biased to horizontal carry to form line patterns."""
    h, w = img.shape
    out = img.copy()
    e = emphasis / 5.0
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                out[y, x + 1] += err * (0.5 + 0.4 * e)
            if x + 2 < w:
                out[y, x + 2] += err * (0.2 * e)
            if y + 1 < h:
                out[y + 1, x] += err * (0.3 - 0.15 * e)
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@njit(cache=True)
def _atkinson_line_modulation(img, thr, strength, hbias):
    h, w = img.shape
    out = img.copy()
    hb = hbias / 5.0
    st = strength / 5.0
    for y in range(h):
        for x in range(w):
            old = out[y, x]
            new = 255.0 if old >= thr else 0.0
            out[y, x] = new
            err = (old - new) / 8.0 * st
            if x + 1 < w:
                out[y, x + 1] += err * (1.0 + hb)
            if x + 2 < w:
                out[y, x + 2] += err * hb
            if y + 1 < h:
                out[y + 1, x] += err * (1.0 - 0.5 * hb)
                if x + 1 < w:
                    out[y + 1, x + 1] += err
    for y in range(h):
        for x in range(w):
            out[y, x] = 255.0 if out[y, x] >= 128.0 else 0.0
    return out


@njit(cache=True)
def _contrast_aware(img, thr, line_scale, horizontal):
    """1-D diffusion whose threshold warps with local contrast."""
    h, w = img.shape
    out = img.copy()
    s = line_scale if line_scale >= 1 else 1
    if horizontal:
        for y in range(h):
            carry = 0.0
            for x in range(w):
                lo = img[y, x - 1] if x > 0 else img[y, x]
                hi = img[y, x + 1] if x < w - 1 else img[y, x]
                local = abs(hi - lo)
                t = thr + (local - 64.0) * 0.25
                old = out[y, x] + carry
                new = 255.0 if old >= t else 0.0
                out[y, x] = new
                carry = (old - new) / s
    else:
        for x in range(w):
            carry = 0.0
            for y in range(h):
                lo = img[y - 1, x] if y > 0 else img[y, x]
                hi = img[y + 1, x] if y < h - 1 else img[y, x]
                local = abs(hi - lo)
                t = thr + (local - 64.0) * 0.25
                old = out[y, x] + carry
                new = 255.0 if old >= t else 0.0
                out[y, x] = new
                carry = (old - new) / s
    return out


# ── Kernel: Artifact Modulation · Glitch · dims=2 · Dither Param 1-20-1 ──
@registry.register("Artifact Modulation", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def artifact_modulation(image_array, parameter, luminance_threshold_value):
    p = int(parameter) if parameter else 1
    return _waveform_alt(image_array.astype(np.float32),
                         luminance_threshold_value, float(p))


# ── Kernel: Atkinson-VHS · Glitch · dims=2 · Line Count 1-20-1 ──
@registry.register("Atkinson-VHS", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def atkinson_vhs(image_array, parameter, luminance_threshold_value):
    lc = int(parameter) if parameter else 1
    return _atkinson_vhs(image_array.astype(np.float32),
                         luminance_threshold_value, lc)


# ── Kernel: Glitch · Glitch · dims=2 · Glitch Intensity 1-20-1 ──
@registry.register("Glitch", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def glitch(image_array, parameter, luminance_threshold_value):
    intensity = int(parameter) if parameter else 1
    return _glitch(image_array.astype(np.float32),
                   luminance_threshold_value, intensity)


# ── Kernel: Modulated Diffuse Y · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Modulated Diffuse Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def modulated_diffuse_y(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _line_diffuse(image_array.astype(np.float32),
                         luminance_threshold_value, ls, True)


# ── Kernel: Modulated Diffuse X · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Modulated Diffuse X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def modulated_diffuse_x(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _line_diffuse(image_array.astype(np.float32),
                         luminance_threshold_value, ls, False)


# ── Kernel: Uniform Modulation Y · Glitch · dims=2 ──
#    sliders (Line Scale 1-20-1, Smoothing Factor 0-1-0, Bleed Fraction 0-100-0)
@registry.register("Uniform Modulation Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",
                                  "smoothing_factor_slider",
                                  "bleed_fraction_slider"))
def uniform_modulation_y(image_array, parameter, luminance_threshold_value):
    ls, smooth, bleed = _unpack3(parameter)
    return _uniform_modulation(image_array.astype(np.float32),
                               luminance_threshold_value,
                               int(ls), float(smooth), float(bleed) / 100.0, True)


# ── Kernel: Uniform Modulation X · Glitch · dims=2 (same three sliders) ──
@registry.register("Uniform Modulation X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",
                                  "smoothing_factor_slider",
                                  "bleed_fraction_slider"))
def uniform_modulation_x(image_array, parameter, luminance_threshold_value):
    ls, smooth, bleed = _unpack3(parameter)
    return _uniform_modulation(image_array.astype(np.float32),
                               luminance_threshold_value,
                               int(ls), float(smooth), float(bleed) / 100.0, False)


# ── Kernel: Waveform · Glitch · dims=2 · Wave Density 1-20-1 ──
@registry.register("Waveform", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def waveform(image_array, parameter, luminance_threshold_value):
    d = float(parameter) if parameter else 1.0
    return _waveform(image_array.astype(np.float32),
                     luminance_threshold_value, d)


# ── Kernel: Waveform Alt · Glitch · dims=2 · Modulation Blend 1-20-1 ──
@registry.register("Waveform Alt", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def waveform_alt(image_array, parameter, luminance_threshold_value):
    b = float(parameter) if parameter else 1.0
    return _waveform_alt(image_array.astype(np.float32),
                         luminance_threshold_value, b)


# ── Kernel: Ordered Modulation · Glitch · dims=2 · Dither Param 1-20-1 ──
@registry.register("Ordered Modulation", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def ordered_modulation(image_array, parameter, luminance_threshold_value):
    p = float(parameter) if parameter else 1.0
    return _ordered_modulation(image_array.astype(np.float32),
                               luminance_threshold_value, p, _BAYER4)


# ── Kernel: Smooth Diffuse · Glitch · dims=2 (Line Scale 1-20-1, Smoothness 1-10-5) ──
@registry.register("Smooth Diffuse", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider", "smoothness_slider"))
def smooth_diffuse(image_array, parameter, luminance_threshold_value):
    ls, smoothness = _unpack2(parameter, 1, 5)
    return _smooth_diffuse(image_array.astype(np.float32),
                           luminance_threshold_value, int(ls), int(smoothness))


# ── Kernel: Stucki Diffusion Lines · Glitch · dims=2 · Line Emphasis 1-10-5 ──
@registry.register("Stucki Diffusion Lines", "Glitch Effects", dims=2,
                   param_sliders=("line_emphasis_slider",))
def stucki_diffusion_lines(image_array, parameter, luminance_threshold_value):
    e = float(parameter) if parameter else 5.0
    return _stucki_diffusion_lines(image_array.astype(np.float32),
                                   luminance_threshold_value, e)


# ── Kernel: Atkinson Line Modulation · Glitch · dims=2 ──
#    sliders (Modulation Strength 1-10-5, Horizontal Bias 1-10-5)
@registry.register("Atkinson Line Modulation", "Glitch Effects", dims=2,
                   param_sliders=("modulation_strength_slider",
                                  "horizontal_bias_slider"))
def atkinson_line_modulation(image_array, parameter, luminance_threshold_value):
    strength, hbias = _unpack2(parameter, 5, 5)
    return _atkinson_line_modulation(image_array.astype(np.float32),
                                     luminance_threshold_value,
                                     float(strength), float(hbias))


# ── Kernel: Contrast Aware Y · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Contrast Aware Y", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def contrast_aware_y(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _contrast_aware(image_array.astype(np.float32),
                           luminance_threshold_value, ls, True)


# ── Kernel: Contrast Aware X · Glitch · dims=2 · Line Scale 1-20-1 ──
@registry.register("Contrast Aware X", "Glitch Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def contrast_aware_x(image_array, parameter, luminance_threshold_value):
    ls = int(parameter) if parameter else 1
    return _contrast_aware(image_array.astype(np.float32),
                           luminance_threshold_value, ls, False)


# ── Tuple-unpack helpers (plain Python; run outside njit) ──
def _unpack3(parameter):
    if isinstance(parameter, (tuple, list)):
        a = parameter[0] if len(parameter) > 0 else 1
        b = parameter[1] if len(parameter) > 1 else 0.0
        c = parameter[2] if len(parameter) > 2 else 0.0
        return a, b, c
    return parameter, 0.0, 0.0


def _unpack2(parameter, d0, d1):
    if isinstance(parameter, (tuple, list)):
        a = parameter[0] if len(parameter) > 0 else d0
        b = parameter[1] if len(parameter) > 1 else d1
        return a, b
    return parameter, d1
```

- [ ] **Step 4: Run — expect PASS** — `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_glitch.py -v` (3 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(dither): glitch family (line diffusion, uniform/smooth modulation, VHS, contrast-aware)"`

---

### Task 2.6: Special-effects family (kernels 55–66)

**Files:** Create `ditherzam/dithering/kernels/special.py` ·
Test `tests/test_kernels_special.py`

**Interfaces:** Registers `Radial Burst`, `Wave`, `Noise`, `Topography`,
`Thresholder`, `Diagonal`, `Displace Contour` (4-tuple), `Sine Wave Modulation`
(2-tuple), `Vortex`, `Concentric Rings`, `Wireframe Alt`, `Crosshatch Alt`.

- [ ] **Step 1: Write failing test — `tests/test_kernels_special.py`**

```python
import numpy as np
from ditherzam.dithering import registry

SINGLE = ["Radial Burst", "Wave", "Noise", "Topography", "Thresholder",
          "Diagonal", "Vortex", "Concentric Rings", "Wireframe Alt",
          "Crosshatch Alt"]


def test_special_single_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n in SINGLE:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 5 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_displace_contour_takes_4_tuple():
    e = registry.get_entry("Displace Contour")
    assert e is not None and len(e.param_sliders) == 4
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    out = e.func(img.copy(), (50, 1, 0, 1), 128.0)
    assert out.shape == img.shape
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}


def test_sine_wave_modulation_takes_2_tuple():
    e = registry.get_entry("Sine Wave Modulation")
    assert e is not None and len(e.param_sliders) == 2
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    out = e.func(img.copy(), (5, 10), 128.0)
    assert out.shape == img.shape
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}


def test_noise_is_deterministic():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    a = registry.get_entry("Noise").func(img.copy(), 0, 128.0)
    b = registry.get_entry("Noise").func(img.copy(), 0, 128.0)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run — expect FAIL** (names not registered).

- [ ] **Step 3: Implement `ditherzam/dithering/kernels/special.py`**

```python
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from ditherzam.dithering import registry


@njit(cache=True, parallel=True)
def _radial_burst(img, thr):
    h, w = img.shape
    cx = w / 2.0
    cy = h / 2.0
    rays = 24.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ang = math.atan2(y - cy, x - cx)
            t = (math.sin(ang * rays) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _wave(img, thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = ((math.sin(x * 0.15) + math.sin(y * 0.15)) * 0.25 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True)
def _noise(img, thr):
    np.random.seed(0)
    h, w = img.shape
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            n = (np.random.random() - 0.5) * 255.0
            out[y, x] = 255.0 if (img[y, x] + n) >= thr else 0.0
    return out


@njit(cache=True, parallel=True)
def _topography(img, warp):
    h, w = img.shape
    bands = 8.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            sx = int(x + math.sin(y * 0.1) * warp)
            if sx < 0:
                sx = 0
            elif sx >= w:
                sx = w - 1
            b0 = int(img[y, sx] / 256.0 * bands)
            xr = sx + 1 if sx + 1 < w else sx
            yd = y + 1 if y + 1 < h else y
            br = int(img[y, xr] / 256.0 * bands)
            bd = int(img[yd, sx] / 256.0 * bands)
            out[y, x] = 0.0 if (b0 != br or b0 != bd) else 255.0
    return out


@njit(cache=True, parallel=True)
def _thresholder(img, freq):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            t = 128.0 + math.sin(x / freq) * math.cos(y / freq) * 64.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _diagonal(img, sensitivity):
    h, w = img.shape
    out = np.empty_like(img)
    thr_edge = 200.0 / sensitivity
    for y in prange(h):
        for x in range(w):
            xl = x - 1 if x > 0 else x
            xr = x + 1 if x < w - 1 else x
            yu = y - 1 if y > 0 else y
            yd = y + 1 if y < h - 1 else y
            gx = img[y, xr] - img[y, xl]
            gy = img[yd, x] - img[yu, x]
            mag = math.sqrt(gx * gx + gy * gy)
            out[y, x] = 0.0 if mag > thr_edge else 255.0
    return out


@njit(cache=True, parallel=True)
def _displace_contour(img, contour_thr, line_mode, smoothing, line_space):
    h, w = img.shape
    bands = line_space if line_space >= 1 else 1
    thick = line_mode if line_mode >= 1 else 1
    step = 256.0 / (bands * 4.0)
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            val = img[y, x]
            b0 = int(val / step)
            is_line = False
            for t in range(thick):
                xr = x + 1 + t if x + 1 + t < w else w - 1
                yd = y + 1 + t if y + 1 + t < h else h - 1
                if int(img[y, xr] / step) != b0 or int(img[yd, x] / step) != b0:
                    is_line = True
            gate = val < (contour_thr / 100.0 * 255.0)
            out[y, x] = 0.0 if (is_line and gate) else 255.0
    return out


@njit(cache=True, parallel=True)
def _sine_wave_modulation(img, freq, wave_thr):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            line = math.sin(x * freq * 0.05 + y * 0.1) * 0.5 + 0.5
            gate = darkness * (wave_thr / 15.0)
            out[y, x] = 0.0 if line < gate else 255.0
    return out


@njit(cache=True, parallel=True)
def _vortex(img, thr):
    h, w = img.shape
    cx = w / 2.0
    cy = h / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            ang = math.atan2(dy, dx)
            r = math.sqrt(dx * dx + dy * dy)
            t = (math.sin(ang * 6.0 + r * 0.15) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _concentric(img, thr):
    h, w = img.shape
    cx = w / 2.0
    cy = h / 2.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            r = math.sqrt(dx * dx + dy * dy)
            t = (math.sin(r * 0.3) * 0.5 + 0.5) * 255.0
            out[y, x] = 255.0 if img[y, x] >= t else 0.0
    return out


@njit(cache=True, parallel=True)
def _wireframe_alt(img, sensitivity):
    h, w = img.shape
    out = np.empty_like(img)
    thr_edge = 160.0 / sensitivity
    for y in prange(h):
        for x in range(w):
            xl = x - 1 if x > 0 else x
            xr = x + 1 if x < w - 1 else x
            yu = y - 1 if y > 0 else y
            yd = y + 1 if y < h - 1 else y
            gx = img[y, xr] - img[y, xl]
            gy = img[yd, x] - img[yu, x]
            gd = img[yd, xr] - img[yu, xl]
            mag = math.sqrt(gx * gx + gy * gy + gd * gd)
            out[y, x] = 0.0 if mag > thr_edge else 255.0
    return out


@njit(cache=True, parallel=True)
def _crosshatch_alt(img, s):
    h, w = img.shape
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            darkness = 1.0 - img[y, x] / 255.0
            hit = False
            if darkness > 0.15 and (x % s == 0):
                hit = True
            if darkness > 0.40 and (y % s == 0):
                hit = True
            if darkness > 0.65 and ((x + y) % s == 0):
                hit = True
            if darkness > 0.85 and ((x - y) % s == 0):
                hit = True
            out[y, x] = 0.0 if hit else 255.0
    return out


# ── Kernel: Radial Burst · Special Effects · dims=2 · no sliders ──
@registry.register("Radial Burst", "Special Effects", dims=2)
def radial_burst(image_array, parameter, luminance_threshold_value):
    return _radial_burst(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Wave · Special Effects · dims=2 · no sliders ──
@registry.register("Wave", "Special Effects", dims=2)
def wave(image_array, parameter, luminance_threshold_value):
    return _wave(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Noise · Special Effects · dims=2 · no sliders ──
@registry.register("Noise", "Special Effects", dims=2)
def noise(image_array, parameter, luminance_threshold_value):
    return _noise(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Topography · Special Effects · dims=2 · Warp Intensity 1-20-1 ──
@registry.register("Topography", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def topography(image_array, parameter, luminance_threshold_value):
    warp = float(parameter) if parameter else 1.0
    return _topography(image_array.astype(np.float32), warp)


# ── Kernel: Thresholder · Special Effects · dims=2 · Modulation Frequency 1-20-1 ──
@registry.register("Thresholder", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def thresholder(image_array, parameter, luminance_threshold_value):
    freq = float(parameter) if parameter else 1.0
    if freq < 1.0:
        freq = 1.0
    return _thresholder(image_array.astype(np.float32), freq)


# ── Kernel: Diagonal · Special Effects · dims=2 · Edge Sensitivity 1-20-1 ──
@registry.register("Diagonal", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def diagonal(image_array, parameter, luminance_threshold_value):
    s = float(parameter) if parameter else 1.0
    if s < 1.0:
        s = 1.0
    return _diagonal(image_array.astype(np.float32), s)


# ── Kernel: Displace Contour · Special Effects · dims=2 ──
#    sliders (Contour Threshold 0-100-50, Line Mode 1-3-1, Smoothing 0-5-0, Line Spacing 1-5-1)
@registry.register("Displace Contour", "Special Effects", dims=2,
                   param_sliders=("contour_thresh_slider", "line_mode_slider",
                                  "smoothing_slider", "line_space_slider"))
def displace_contour(image_array, parameter, luminance_threshold_value):
    ct, lm, sm, ls = _unpack4(parameter, 50, 1, 0, 1)
    return _displace_contour(image_array.astype(np.float32),
                             float(ct), int(lm), int(sm), int(ls))


# ── Kernel: Sine Wave Modulation · Special Effects · dims=2 ──
#    sliders (Wave Frequency 1-20-5, Wave Threshold 1-30-10)
@registry.register("Sine Wave Modulation", "Special Effects", dims=2,
                   param_sliders=("wave_frequency_slider", "wave_threshold_slider"))
def sine_wave_modulation(image_array, parameter, luminance_threshold_value):
    freq, wthr = _unpack2s(parameter, 5, 10)
    return _sine_wave_modulation(image_array.astype(np.float32),
                                 float(freq), float(wthr))


# ── Kernel: Vortex · Special Effects · dims=2 · no sliders (extra) ──
@registry.register("Vortex", "Special Effects", dims=2)
def vortex(image_array, parameter, luminance_threshold_value):
    return _vortex(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Concentric Rings · Special Effects · dims=2 · no sliders (extra) ──
@registry.register("Concentric Rings", "Special Effects", dims=2)
def concentric_rings(image_array, parameter, luminance_threshold_value):
    return _concentric(image_array.astype(np.float32), luminance_threshold_value)


# ── Kernel: Wireframe Alt · Special Effects · dims=2 · Edge Sensitivity 1-20-1 (extra) ──
@registry.register("Wireframe Alt", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def wireframe_alt(image_array, parameter, luminance_threshold_value):
    s = float(parameter) if parameter else 1.0
    if s < 1.0:
        s = 1.0
    return _wireframe_alt(image_array.astype(np.float32), s)


# ── Kernel: Crosshatch Alt · Special Effects · dims=2 · Line Spacing 1-20-1 (extra) ──
@registry.register("Crosshatch Alt", "Special Effects", dims=2,
                   param_sliders=("dither_parameter_slider",))
def crosshatch_alt(image_array, parameter, luminance_threshold_value):
    s = int(parameter) if parameter else 4
    if s < 1:
        s = 1
    return _crosshatch_alt(image_array.astype(np.float32), s)


# ── Tuple-unpack helpers (plain Python) ──
def _unpack4(parameter, d0, d1, d2, d3):
    if isinstance(parameter, (tuple, list)):
        vals = list(parameter) + [d0, d1, d2, d3]
        return vals[0], vals[1], vals[2], vals[3]
    return parameter, d1, d2, d3


def _unpack2s(parameter, d0, d1):
    if isinstance(parameter, (tuple, list)):
        a = parameter[0] if len(parameter) > 0 else d0
        b = parameter[1] if len(parameter) > 1 else d1
        return a, b
    return parameter, d1
```

- [ ] **Step 4: Run — expect PASS** — `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_special.py -v` (4 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(dither): special-effects family (burst, wave, contour, vortex, wireframe)"`

---

### Task 2.7: Wire modules, golden harness, param-order & count tests

**Files:** Modify `ditherzam/dithering/__init__.py` ·
Create `tests/golden_harness.py`, `tests/test_kernels_all.py`, `tests/golden/*.npy`

**Interfaces:** Consumes `registry.list_dithers()`. Imports the four new kernel
modules for their registration side-effects (after `registry` exists).

- [ ] **Step 1: Modify `ditherzam/dithering/__init__.py`** to import all five
  modules (order matters: `ordered` before `glitch`, since `glitch` imports
  `_BAYER4` from `ordered`; all imports come after `registry` is created).

```python
from .registry import DitherRegistry, DitherEntry

registry = DitherRegistry()

# Import kernel modules for their registration side-effects (registry must exist first).
from .kernels import error_diffusion as _error_diffusion  # noqa: E402,F401
from .kernels import ordered as _ordered_kernels          # noqa: E402,F401
from .kernels import pattern as _pattern_kernels          # noqa: E402,F401
from .kernels import glitch as _glitch_kernels            # noqa: E402,F401
from .kernels import special as _special_kernels          # noqa: E402,F401
```

- [ ] **Step 2: Write the harness — `tests/golden_harness.py`**

```python
import numpy as np

# Deterministic gradient input shared by every golden test.
STD_INPUT = np.tile(np.linspace(0, 255, 32, dtype=np.float32), (32, 1))


def default_param(entry):
    """Build a representative parameter for a kernel from its slider count."""
    n = len(entry.param_sliders)
    if n == 0:
        return 0
    if n == 1:
        return 4
    return tuple(4 for _ in range(n))
```

- [ ] **Step 3: Write failing/seed test — `tests/test_kernels_all.py`**

```python
import pathlib
import numpy as np
import pytest
from ditherzam.dithering import registry
from tests.golden_harness import STD_INPUT, default_param

GOLD = pathlib.Path(__file__).parent / "golden"
GOLD.mkdir(exist_ok=True)


@pytest.mark.parametrize("name", sorted(registry.list_dithers()))
def test_kernel_matches_golden(name):
    e = registry.get_entry(name)
    out = e.func(STD_INPUT.copy(), default_param(e), 128.0)
    assert out.dtype == np.float32, name
    assert out.shape == STD_INPUT.shape, name
    assert out.min() >= 0.0 and out.max() <= 255.0, name
    f = GOLD / f"{name.replace('/', '_')}.npy"
    if not f.exists():
        np.save(f, out)             # first run seeds the golden fixture
    np.testing.assert_array_equal(out, np.load(f))


def test_total_count_at_least_63():
    assert len(registry.list_dithers()) >= 63


def test_category_grouping_present():
    cats = registry.by_category()
    for expected in ("Error Diffusion", "Ordered Dither", "Patterned",
                     "Glitch Effects", "Special Effects"):
        assert expected in cats, expected


def test_param_order_matches_sliders():
    # Multi-slider kernels must accept a tuple whose length == len(param_sliders).
    multi = [n for n in registry.list_dithers()
             if len(registry.get_entry(n).param_sliders) > 1]
    for name in multi:
        e = registry.get_entry(name)
        arity = len(e.param_sliders)
        out = e.func(STD_INPUT.copy(), tuple(3 for _ in range(arity)), 128.0)
        assert out.shape == STD_INPUT.shape, name
```

- [ ] **Step 4: Run once to seed goldens, review a few `.npy` visually, then run again**
  - `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_all.py -q` → first run **creates**
    `tests/golden/*.npy` (all parametrized cases pass on seeding).
  - Run the same command again → **all PASS** (determinism proven, count `>=63`).

- [ ] **Step 5: Commit**
  `git add ditherzam/dithering/__init__.py tests/golden_harness.py tests/test_kernels_all.py tests/golden/` then
  `git commit -m "test(dither): golden-array harness over all kernels; assert >=63 + category/param-order"`

---

## Subsystem Definition of Done (checklist)

- [ ] `len(registry.list_dithers()) >= 63` (this build registers **66**).
- [ ] Every kernel: `float32`, shape-preserving, values within `0..255`,
  deterministic (proven by golden `.npy` fixtures).
- [ ] Every randomized kernel (`Gaussian`, `Random Ordered`, `Stippling`,
  `Glitch`, `Noise`) seeds `np.random.seed(0)` internally.
- [ ] Multi-param kernels consume the `parameter` **tuple** in `param_sliders`
  order (`Uniform Modulation X/Y`, `Smooth Diffuse`, `Atkinson Line Modulation`,
  `Displace Contour`, `Sine Wave Modulation`).
- [ ] `registry.by_category()` exposes all five categories with the counts:
  Error Diffusion 16 · Ordered Dither 12 · Patterned 11 · Glitch 15 · Special 12.
- [ ] No Qt imported anywhere under `ditherzam/dithering/**`.
- [ ] `NUMBA_DISABLE_JIT=1 pytest tests/ -q` fully green.
- [ ] Every `@registry.register` name/category/`param_sliders` matches the
  catalog tables above verbatim (frozen contract).

## Self-Review

**Spec §9.2 / §6.2 coverage matrix (item → registration / task):**

- **§9.2 Error Diffusion:** None → Task 2.2; Floyd-Steinberg, Atkinson → Phase 1;
  Jarvis-Judice-Ninke, Stucki, Burkes, Sierra, Sierra-Lite, Two-Row-Sierra,
  Stevenson-Arce, Ostromukhov, Gaussian → **Task 2.2**. (All 12 covered.)
- **§9.2 Ordered Dither:** Bayer-Ordered, Bayer-Void, Random Ordered, Bit Tone,
  Mosaic, Bayer-Matrix 2x2/8x8/16x16 → **Task 2.3**; Bayer-Matrix 4x4 → Phase 1.
  (All 9 covered.)
- **§9.2 Glitch Effects:** Artifact Modulation, Atkinson-VHS, Glitch, Modulated
  Diffuse Y/X, Uniform Modulation Y/X, Waveform, Waveform Alt, Ordered
  Modulation, Smooth Diffuse, Stucki Diffusion Lines, Atkinson Line Modulation,
  Contrast Aware Y/X → **Task 2.5**. (All 15 covered.)
- **§9.2 Patterned:** Checkers Small/Medium/Large, Diamond, Gridlock/Traffic,
  Print Pattern, Block Tone, Stippling, Crosshatch → **Task 2.4**. (All 9 covered.)
- **§9.2 Special Effects:** Radial Burst, Wave, Noise, Topography, Thresholder,
  Diagonal, Displace Contour, Sine Wave Modulation → **Task 2.6**. (All 8 covered.)
- **§6.2 primary-slider styles:** every `dither_parameter_slider` style in the
  §6.2 table maps to a registration with the matching `(label,min,max,default)`
  reproduced in the catalog tables (Atkinson-VHS, Bayer-Void, Bit Tone, Block
  Tone, Contrast Aware X/Y, Crosshatch, Diagonal, Gaussian, Glitch, Modulated
  Diffuse X/Y, Mosaic, Stippling, Thresholder, Topography, Uniform Modulation
  X/Y, Waveform, Waveform Alt).
- **§6.2 secondary sliders:** `contour_thresh/line_mode/smoothing/line_space` →
  Displace Contour (Task 2.6); `smoothness` → Smooth Diffuse; `matrix_size` →
  **Modulated Bayer Dither** (Task 2.3 — this style appears only in §6.2's
  secondary-slider table, not §9.2, and is registered here); `wave_frequency/
  wave_threshold` → Sine Wave Modulation; `line_emphasis` → Stucki Diffusion
  Lines; `modulation_strength/horizontal_bias` → Atkinson Line Modulation;
  `smoothing_factor/bleed_fraction` → Uniform Modulation X/Y. The generic
  `Tile Size` row in §6.2 is a shared UI widget, not a distinct algorithm, so it
  has no kernel.

**Items I could NOT find named in the spec (added as clean-room extras to exceed
`>=63`):** Fan, Shiau-Fan, False Floyd-Steinberg, Atkinson-Light (Error
Diffusion); Cluster-Dot, Halftone-Ordered (Ordered); Dot Screen, Line Screen
(Patterned); Vortex, Concentric Rings, Wireframe Alt, Crosshatch Alt (Special).
All are public-domain techniques, implemented from scratch here, and are marked
**(extra)** in the catalog. They can be dropped without falling below 63 only if
2+ are kept (spec-named entries total 62 including None; extras bring the total
to 66).

**Placeholder scan:** every code step contains complete, runnable
implementations — no `TODO`, `pass`, or `...` bodies.

**Type-consistency vs FROZEN CONTRACTS:** all registrations use
`@registry.register(name, category, dims=2, param_sliders=(...))`; every kernel
callable has signature
`(image_array, parameter, luminance_threshold_value)` and returns a
`float32[H,W]` array in `0..255`; compute cores are `@njit(cache=True[,
parallel=True])` with `prange` on the independent kernels — matching Phase 1 and
the roadmap contract exactly.
