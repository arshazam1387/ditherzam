# Phase 2 — Full Dither Kernel Library — Implementation Plan

> REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Read
> [`00-ROADMAP.md`](00-ROADMAP.md) (Global Constraints) and complete Phase 1 first.

**Goal:** Register the full ~63-algorithm library across five kernel modules,
each kernel a Numba function with a golden test proving determinism, correct
shape, and value range.

**Architecture:** Each family lives in its own module; every module imports the
shared `registry` from `ditherzam.dithering` and registers via decorator. A
single parametrized golden-test harness covers all kernels; hand-picked kernels
get extra behavioral tests.

**Tech Stack:** NumPy · Numba · pytest.

## Global Constraints
See roadmap. Additionally: **golden arrays** are stored as `.npy` fixtures under
`tests/golden/<kernel>.npy`, generated once with a fixed seed and reviewed, then
frozen. Any kernel using randomness must seed with `np.random.seed(0)` internally
(or accept a seed) so output is reproducible.

---

## File structure (this phase)

- Modify: `ditherzam/dithering/__init__.py` (import the four new kernel modules)
- Create: `ditherzam/dithering/kernels/ordered.py`
- Create: `ditherzam/dithering/kernels/pattern.py`
- Create: `ditherzam/dithering/kernels/glitch.py`
- Create: `ditherzam/dithering/kernels/special.py`
- Extend: `ditherzam/dithering/kernels/error_diffusion.py` (remaining diffusion kernels)
- Create: `tests/golden_harness.py`, `tests/test_kernels_all.py`, `tests/golden/*.npy`

---

## Kernel catalog to implement (name · category · dims · param sliders)

**Error Diffusion (15):** Floyd-Steinberg✓, Atkinson✓, Jarvis-Judice-Ninke,
Stucki, Burkes, Sierra, Sierra-Lite, Two-Row-Sierra, Stevenson-Arce, Ostromukhov,
Gaussian(`dither_parameter_slider`), + 4 more diffusion variants to reach 15
(e.g. Fan, Shiau-Fan, Atkinson-light, False-Floyd-Steinberg — pick documented public kernels).

**Ordered Dither (5+4 matrices):** Bayer-Ordered, Bayer-Void, Random Ordered,
Bit Tone(`dither_parameter_slider`), Mosaic(`dither_parameter_slider`),
Bayer-Matrix 2x2/4x4✓/8x8/16x16.

**Pattern (8):** Checkers-Small/Medium/Large, Diamond, Gridlock/Traffic,
Print Pattern (CMYK), Block Tone(`dither_parameter_slider`),
Stippling(`dither_parameter_slider`), Crosshatch(`dither_parameter_slider`).

**Glitch (17):** Artifact Modulation, Atkinson-VHS(`dither_parameter_slider`),
Glitch(`…`), Modulated Diffuse Y/X(`…`), Uniform Modulation Y/X
(`dither_parameter_slider,smoothing_factor_slider,bleed_fraction_slider`),
Waveform(`…`), Waveform Alt(`…`), Ordered Modulation, Smooth Diffuse
(`dither_parameter_slider,smoothness_slider`), Stucki Diffusion Lines
(`line_emphasis_slider`), Atkinson Line Modulation
(`modulation_strength_slider,horizontal_bias_slider`), Contrast Aware Y/X.

**Special (16):** Radial Burst, Wave, Noise, Topography(`…`),
Thresholder(`…`), Diagonal(`…`), Displace Contour
(`contour_thresh_slider,line_mode_slider,smoothing_slider,line_space_slider`),
Sine Wave Modulation(`wave_frequency_slider,wave_threshold_slider`), + 8 more
special/geometric patterns (Vortex, Halftone-CMYK, Classic-Halftone, Crosshatch-alt,
Wireframe, etc. — public techniques).

> Kernel math: see `DITHER_BOY_FULL_SPEC.md` §9 for behavior; use the canonical
> public diffusion matrices below.

### Canonical diffusion matrices (public domain)

| Kernel | divisor | weights (right, then next rows) |
|---|---|---|
| Jarvis-Judice-Ninke | 48 | `X 7 5` / `3 5 7 5 3` / `1 3 5 3 1` |
| Stucki | 42 | `X 8 4` / `2 4 8 4 2` / `1 2 4 2 1` |
| Burkes | 32 | `X 8 4` / `2 4 8 4 2` |
| Sierra (3-row) | 32 | `X 5 3` / `2 4 5 4 2` / `. 2 3 2 .` |
| Sierra-Lite | 4 | `X 2` / `1 1 .` |
| Two-Row-Sierra | 16 | `X 4 3` / `1 2 3 2 1` |
| Stevenson-Arce | 200 | hex lattice, weights 32,12,26,30,16,5 |

---

### Task 1: Generic weighted error-diffusion helper

**Files:**
- Modify: `ditherzam/dithering/kernels/error_diffusion.py`
- Test: `tests/test_diffusion_helper.py`

**Interfaces:** Produces
```python
@njit(cache=True)
def _diffuse(img, thr, offsets, weights, divisor) -> np.ndarray
```
`offsets`: int array `[[dy,dx],...]`; `weights`: float array aligned to offsets.

- [ ] **Step 1: Failing test — `tests/test_diffusion_helper.py`**

```python
import numpy as np
from ditherzam.dithering.kernels.error_diffusion import _diffuse

def test_diffuse_matches_floyd_manual():
    img = np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)
    offs = np.array([[0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    wts  = np.array([7.0, 3.0, 5.0, 1.0], dtype=np.float32)
    out = _diffuse(img.copy(), 128.0, offs, wts, 16.0)
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}
    assert out.shape == (2, 2)
```

- [ ] **Step 2: Run — verify fail**
- [ ] **Step 3: Implement `_diffuse` (append to error_diffusion.py)**

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
```

- [ ] **Step 4: Run — verify pass**
- [ ] **Step 5: Commit** — `git commit -m "feat(dither): generic weighted error-diffusion helper"`

---

### Task 2: Register the classic diffusion kernels via the helper

**Files:** Modify `error_diffusion.py`; Test `tests/test_kernels_diffusion.py`

**Interfaces:** Registers `Jarvis-Judice-Ninke`, `Stucki`, `Burkes`, `Sierra`,
`Sierra-Lite`, `Two-Row-Sierra`, `Stevenson-Arce`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.dithering import registry

NAMES = ["Jarvis-Judice-Ninke","Stucki","Burkes","Sierra","Sierra-Lite",
         "Two-Row-Sierra","Stevenson-Arce"]

def test_registered_and_binary():
    img = np.tile(np.linspace(0,255,8,dtype=np.float32),(8,1))
    for n in NAMES:
        e = registry.get_entry(n)
        assert e is not None, n
        out = e.func(img.copy(), 0, 128.0)
        assert set(np.unique(out).tolist()) <= {0.0,255.0}, n
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement** — module-level constant offset/weight arrays + a small
  registration block per kernel, e.g.:

```python
_JJN_OFF = np.array([[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],
                     [2,-2],[2,-1],[2,0],[2,1],[2,2]], dtype=np.int64)
_JJN_W = np.array([7,5,3,5,7,5,3,1,3,5,3,1], dtype=np.float32)

@registry.register("Jarvis-Judice-Ninke", "Error Diffusion", dims=2)
def jjn(image_array, parameter, luminance_threshold_value):
    return _diffuse(image_array.astype(np.float32), luminance_threshold_value,
                    _JJN_OFF, _JJN_W, 48.0)
# ...repeat for Stucki(42), Burkes(32), Sierra(32), Sierra-Lite(4),
#    Two-Row-Sierra(16), Stevenson-Arce(200) with the matrices in the table above.
```

- [ ] **Step 4: Run — pass**
- [ ] **Step 5: Commit** — `feat(dither): 7 classic error-diffusion kernels`

---

### Task 3: Ordered / Bayer family

**Files:** Create `ordered.py`; Test `tests/test_kernels_ordered.py`
**Interfaces:** Registers Bayer-Ordered, Bayer 2x2/8x8/16x16, Random Ordered,
Bit Tone, Mosaic, Bayer-Void. Reuse `_ordered`, `_bayer_matrix` from Phase 1
(move them into `ordered.py` and import into `error_diffusion.py` if shared).

- [ ] **Step 1** failing test asserting each name registered + white→white, black→black.
- [ ] **Step 2** run/fail.
- [ ] **Step 3** implement:
  - `bayer_2/8/16` via `_bayer_matrix(n)` normalized to 0..255 thresholds.
  - `Bayer-Ordered` = alias of 4×4.
  - `Random Ordered`: per-pixel `np.random.seed(0)` threshold matrix.
  - `Bit Tone`: threshold matrix scaled by `dither_parameter_slider`.
  - `Mosaic`: average pooling into blocks of size `parameter`, then threshold.
  - `Bayer-Void`: CRT-warp row shift `∝ (y/H)**2 * parameter` + Bayer threshold `thr + (b-128)*strength`.
- [ ] **Step 4** pass. **Step 5** commit `feat(dither): ordered/Bayer family`.

---

### Task 4: Pattern family

**Files:** Create `pattern.py`; Test `tests/test_kernels_pattern.py`
**Interfaces:** Checkers-Small/Medium/Large, Diamond, Gridlock/Traffic,
Print Pattern, Block Tone, Stippling, Crosshatch.

- [ ] Steps mirror Task 3 (failing test → implement → pass → commit). Each kernel is
  a documented pattern generator (checkerboard modulo, diamond distance field,
  round-dot halftone whose dot radius ∝ darkness, hatch lines by intensity).
  Commit `feat(dither): pattern family`.

---

### Task 5: Glitch family (per-scanline & multi-param)

**Files:** Create `glitch.py`; Test `tests/test_kernels_glitch.py`
**Interfaces:** 17 glitch kernels. Multi-param ones consume tuples:
- `Uniform Modulation X/Y(line_scale, thr, smoothing_factor, bleed_fraction)`
- `Smooth Diffuse(line_scale, thr, smoothness)`
- `Atkinson Line Modulation(modulation_strength, horizontal_bias)`
- `Stucki Diffusion Lines(line_emphasis, thr)`

- [ ] Steps: failing test (registered + binary + shape) → implement 1-D diffusion
  helpers (`_diffuse_row`) + waveform/modulation kernels → pass → commit
  `feat(dither): glitch family`.

> The pipeline's per-style dispatch (spec §8.2) must be mirrored: the `params`
> dict → tuple ordering matches each kernel's signature. Add a
> `PARAM_ORDER: dict[str, tuple[str,...]]` map in `pipeline.py` if needed and a
> test that the tuple is built in the right order.

---

### Task 6: Special-effects family

**Files:** Create `special.py`; Test `tests/test_kernels_special.py`
**Interfaces:** Radial Burst, Wave, Noise, Topography, Thresholder, Diagonal,
Displace Contour(4-tuple), Sine Wave Modulation(2-tuple), + remaining specials.

- [ ] Steps mirror above. Displace Contour and Sine Wave are the special-dispatch
  cases — assert their tuple param arity in a test. Commit `feat(dither): special family`.

---

### Task 7: Golden-array harness over ALL kernels

**Files:** Create `tests/golden_harness.py`, `tests/test_kernels_all.py`, `tests/golden/*.npy`
**Interfaces:** Consumes `registry.list_dithers()`.

- [ ] **Step 1: Write the harness**

```python
# tests/golden_harness.py
import numpy as np
STD_INPUT = np.tile(np.linspace(0, 255, 32, dtype=np.float32), (32, 1))
```

- [ ] **Step 2: Failing parametrized test — `tests/test_kernels_all.py`**

```python
import numpy as np, pathlib, pytest
from ditherzam.dithering import registry
from tests.golden_harness import STD_INPUT

GOLD = pathlib.Path(__file__).parent / "golden"

@pytest.mark.parametrize("name", sorted(registry.list_dithers()))
def test_kernel_matches_golden(name):
    e = registry.get_entry(name)
    param = e.param_sliders and tuple(1 for _ in e.param_sliders) or 0
    out = e.func(STD_INPUT.copy(), param if e.param_sliders else 0, 128.0)
    assert out.dtype == np.float32 and out.shape == STD_INPUT.shape
    assert out.min() >= 0.0 and out.max() <= 255.0
    f = GOLD / f"{name.replace('/','_')}.npy"
    if not f.exists():
        np.save(f, out)          # first run seeds the golden fixture
    np.testing.assert_array_equal(out, np.load(f))
```

- [ ] **Step 3: Run once to seed goldens, review a few visually, then run again**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_kernels_all.py -q`
Expected: first run creates `tests/golden/*.npy`; second run all PASS (determinism proven).

- [ ] **Step 4: Assert count**

Add: `def test_total_count(): assert len(registry.list_dithers()) >= 63`

- [ ] **Step 5: Commit**

```bash
git add tests/golden_harness.py tests/test_kernels_all.py tests/golden/
git commit -m "test(dither): golden-array harness over all kernels; assert >=63"
```

---

## Phase 2 Self-Review
- [ ] `len(registry.list_dithers()) >= 63`, grouped per roadmap category counts.
- [ ] Every kernel: float32, shape-preserving, 0..255, deterministic (golden).
- [ ] Randomized kernels seed internally.
- [ ] Multi-param kernels consume tuples in the documented order.
