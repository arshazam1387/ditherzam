# Echo Smear Dither Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One new binary dither style, "Echo Smear", that recreates a TouchDesigner smear/echo reference: wavy silhouette echo outlines, full-height streaks, and speckle dissolution, with a single Breath slider sweeping solid ↔ dissolved.

**Architecture:** Single fused `@njit(parallel=True)` gather kernel in the existing Special Effects family (spec: `docs/superpowers/specs/2026-07-13-echo-smear-style-design.md`). Per-pixel deterministic integer hashing (no RNG state), a per-column peak pre-pass for streaks, Topography-style neighbor tests for echo outlines. UI, presets, palette depth-ramp, and animation come free from registration + parameter metadata.

**Tech Stack:** Python 3.12 (`.venv/Scripts/python.exe`), NumPy, Numba (`cache=True, parallel=True`), pytest.

## Global Constraints

- Clean-room: no Dither Boy / Studio AAA code or strings.
- Core stays Qt-free; this plan touches only `ditherzam/dithering/` and `tests/`.
- Kernels are float32 in/out, values 0..255, binary output (0.0 ink / 255.0 background).
- Run tests with JIT off: `NUMBA_DISABLE_JIT=1` env var (PowerShell: `$env:NUMBA_DISABLE_JIT='1'`).
- Test command base: `.venv/Scripts/python.exe -m pytest` from repo root `C:\Users\arsha\Desktop\custom dither`.
- TDD: red → green → commit per task. Never edit existing kernels' behavior.
- Branch: `feat/echo-smear-style` (based on `feat/smart-subject-masking`; do not touch that branch).

## Established codebase facts (read once, trust throughout)

- Registration: `@registry.register(name, "Special Effects", dims=2, param_sliders=(...))` decorating a plain wrapper that unpacks the param tuple and calls a private `@njit` kernel (see `ditherzam/dithering/kernels/special.py:242` Topography).
- The app sends kernel params as a tuple ordered exactly like `param_sliders`; slider metadata (label, min, max, default) lives in `_KEY_SPECS` in `ditherzam/dithering/parameters.py`. A missing key raises `KeyError` in `parameter_specs()`.
- UI sliders are generated automatically from `parameter_specs(entry)` (`ditherzam/ui/controls.py:314`) — no UI code needed.
- `tests/test_kernels_all.py` auto-covers every registered style: golden fixture (auto-seeded on first run into `tests/golden/<name>.npy`), dtype/shape/range checks, and a tuple-arity check that calls every multi-slider kernel with `tuple(3 for _ in sliders)`.
- Binary kernels get multi-tone palette support automatically via `_binary_to_levels` (`ditherzam/dithering/pipeline.py:94`) — the kernel may be called with a synthetic "in-band fraction" image, so it must not assume natural image statistics.
- `luminance_threshold_value` (`tval`) arrives in 0..255 space; darker-than-threshold = subject, matching `out = 0.0 if dark else 255.0` ink convention.

---

### Task 1: Parameter metadata, `_unpack7`, and body-only kernel skeleton

**Files:**
- Modify: `ditherzam/dithering/parameters.py` (add 7 rows to `_KEY_SPECS`, after the `"wave_line_spacing_slider"` row at ~line 167)
- Modify: `ditherzam/dithering/kernels/special.py` (kernel + wrapper + `_unpack7`)
- Create: `tests/test_echo_smear.py`

**Interfaces:**
- Produces: registered style `"Echo Smear"` whose wrapper is `echo_smear(image_array, parameter, luminance_threshold_value)`; private kernel `_echo_smear(img, thr, count, spacing, wave, phase, streak, dissolve, breath)` returning float32 0/255; helper `_unpack7(parameter, d0..d6)`; slider keys `echo_count_slider, echo_spacing_slider, echo_wave_amount_slider, echo_wave_phase_slider, echo_streak_slider, echo_dissolve_slider, echo_breath_slider` with defaults `(6, 10, 8, 0, 20, 30, 50)`.
- Consumes: existing `registry`, `parameters.parameter_specs`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_echo_smear.py`:

```python
import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs

# param tuple order: (count, spacing, wave, phase, streak, dissolve, breath)
DEFAULTS = (6, 10, 8, 0, 20, 30, 50)
THR = np.float32(128.0)


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(7).integers(0, 256, (48, 48)).astype(np.float32)


@pytest.fixture(scope="module")
def gradient():
    return np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))


def entry():
    e = registry.get_entry("Echo Smear")
    assert e is not None
    return e


def test_registered_in_special_effects():
    e = entry()
    assert e.category == "Special Effects"
    assert e.dims == 2
    assert e.param_sliders == (
        "echo_count_slider", "echo_spacing_slider", "echo_wave_amount_slider",
        "echo_wave_phase_slider", "echo_streak_slider", "echo_dissolve_slider",
        "echo_breath_slider",
    )


def test_parameter_metadata_exact():
    specs = {s.key: (s.label, s.minimum, s.maximum, s.default)
             for s in parameter_specs(entry())
             if not s.key.startswith("creative_")}
    assert specs == {
        "echo_count_slider": ("Echo Count", 0, 16, 6),
        "echo_spacing_slider": ("Echo Spacing", 2, 40, 10),
        "echo_wave_amount_slider": ("Wave Amount", 0, 32, 8),
        "echo_wave_phase_slider": ("Wave Phase", 0, 360, 0),
        "echo_streak_slider": ("Streak Amount", 0, 100, 20),
        "echo_dissolve_slider": ("Dissolve Amount", 0, 100, 30),
        "echo_breath_slider": ("Breath", 0, 100, 50),
    }


def test_output_contract(image):
    out = entry().func(image.copy(), DEFAULTS, THR)
    assert out.dtype == np.float32
    assert out.shape == image.shape
    assert set(np.unique(out)) <= {0.0, 255.0}


def test_ingredients_off_equals_plain_threshold(gradient):
    # count=0, wave irrelevant, streak=0, dissolve=0, breath=0 -> pure threshold body
    out = entry().func(gradient.copy(), (0, 10, 8, 0, 0, 0, 0), THR)
    expected = np.where(gradient < 128.0, 0.0, 255.0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_deterministic(image):
    a = entry().func(image.copy(), DEFAULTS, THR)
    b = entry().func(image.copy(), DEFAULTS, THR)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: all FAIL/ERROR (`assert e is not None` fails — style not registered).

- [ ] **Step 3: Implement metadata + skeleton**

In `ditherzam/dithering/parameters.py`, add to `_KEY_SPECS` (after the `"diffusion_line_spacing_slider"` line at the end of the dict):

```python
    "echo_count_slider": ("Echo Count", 0, 16, 6),
    "echo_spacing_slider": ("Echo Spacing", 2, 40, 10),
    "echo_wave_amount_slider": ("Wave Amount", 0, 32, 8),
    "echo_wave_phase_slider": ("Wave Phase", 0, 360, 0),
    "echo_streak_slider": ("Streak Amount", 0, 100, 20),
    "echo_dissolve_slider": ("Dissolve Amount", 0, 100, 30),
    "echo_breath_slider": ("Breath", 0, 100, 50),
```

In `ditherzam/dithering/kernels/special.py`, add below `_crosshatch_alt`'s private kernel block (before the wrappers is fine — keep private kernels together, wrappers together, matching file layout):

```python
@njit(cache=True)
def _hash01(x, y, salt):
    # Same integer-hash family as pipeline jitter; pure function of (x, y, salt),
    # identical under JIT on/off because it is plain masked int arithmetic.
    h = (x * 374761393 + y * 668265263 + salt * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h >> 8) & 0xFFFF) / 65535.0


@njit(cache=True, parallel=True)
def _echo_smear(img, thr, count, spacing, wave, phase, streak, dissolve, breath):
    h, w = img.shape
    b = breath / 100.0
    dissolve_gate = b * dissolve / 100.0 * 2.0
    if dissolve_gate > 1.0:
        dissolve_gate = 1.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ink = False
            if img[y, x] < thr:
                if _hash01(x, y, 101) >= dissolve_gate:
                    ink = True
            out[y, x] = 0.0 if ink else 255.0
    return out
```

And the wrapper + unpack helper (wrapper next to the other `@registry.register` wrappers; `_unpack7` beside `_unpack6`):

```python
# ── Kernel: Echo Smear · Special Effects · dims=2 ──
#    sliders (Echo Count 0-16-6, Echo Spacing 2-40-10, Wave Amount 0-32-8,
#             Wave Phase 0-360-0, Streak 0-100-20, Dissolve 0-100-30, Breath 0-100-50)
@registry.register("Echo Smear", "Special Effects", dims=2,
                   param_sliders=("echo_count_slider", "echo_spacing_slider",
                                  "echo_wave_amount_slider", "echo_wave_phase_slider",
                                  "echo_streak_slider", "echo_dissolve_slider",
                                  "echo_breath_slider"))
def echo_smear(image_array, parameter, luminance_threshold_value):
    count, spacing, wave, phase, streak, dissolve, breath = _unpack7(
        parameter, 6, 10, 8, 0, 20, 30, 50)
    return _echo_smear(image_array.astype(np.float32),
                       float(luminance_threshold_value), max(0, int(count)),
                       float(spacing), float(wave), float(phase),
                       float(streak), float(dissolve), float(breath))
```

```python
def _unpack7(parameter, d0, d1, d2, d3, d4, d5, d6):
    if isinstance(parameter, (tuple, list)):
        defaults = (d0, d1, d2, d3, d4, d5, d6)
        return tuple(parameter[i] if i < len(parameter) else defaults[i] for i in range(7))
    return (parameter if parameter not in (None, 0) else d0), d1, d2, d3, d4, d5, d6
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py tests/test_kernels_all.py -v
```

Expected: `test_echo_smear.py` all PASS. `test_kernels_all.py` PASS — including the auto-seeded new golden `tests/golden/Echo Smear.npy` (first run writes it) and `test_param_order_matches_sliders` accepting the 7-tuple.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/parameters.py ditherzam/dithering/kernels/special.py tests/test_echo_smear.py "tests/golden/Echo Smear.npy"
git commit -m "feat(dither): Echo Smear skeleton - registration, params, body threshold"
```

Note: the golden seeded here will be re-baked in Task 4 when the remaining ingredients change default output; that is expected.

---

### Task 2: Contour echo trails

**Files:**
- Modify: `ditherzam/dithering/kernels/special.py` (`_echo_smear` only)
- Modify: `tests/test_echo_smear.py`

**Interfaces:**
- Consumes: `_echo_smear`, `_hash01` from Task 1 (exact signatures above).
- Produces: echo behavior — for echo n in 1..count, a pixel becomes ink when the source sample at `sx = x - n*spacing - sin(y*0.1 + phase_rad + n*0.7)*wave` sits on the thresholded silhouette boundary; per-echo hash decay; whole term scaled by Breath (`b=0` → no echoes).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_echo_smear.py`:

```python
def _subject_square(size=64):
    # Dark 20px-wide square on white: crisp silhouette with a right edge at x=30.
    img = np.full((size, size), 230.0, dtype=np.float32)
    img[22:42, 10:31] = 20.0
    return img


def test_echoes_add_ink_right_of_subject():
    img = _subject_square()
    none = entry().func(img.copy(), (0, 10, 8, 0, 0, 0, 100), THR)
    some = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 100), THR)
    right = slice(None), slice(32, None)          # strictly right of the square
    assert (some[right] == 0.0).sum() > (none[right] == 0.0).sum()
    # body region unchanged by echoes
    np.testing.assert_array_equal(some[22:42, 10:31], none[22:42, 10:31])


def test_echoes_vanish_at_breath_zero():
    img = _subject_square()
    out = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 0), THR)
    expected = np.where(img < 128.0, 0.0, 255.0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_wave_phase_moves_the_waves():
    img = _subject_square()
    a = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 100), THR)
    b = entry().func(img.copy(), (6, 10, 8, 180, 0, 0, 100), THR)
    assert np.any(a != b)


def test_echo_spacing_changes_pixels():
    img = _subject_square()
    a = entry().func(img.copy(), (6, 4, 8, 0, 0, 0, 100), THR)
    b = entry().func(img.copy(), (6, 20, 8, 0, 0, 0, 100), THR)
    assert np.any(a != b)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: the four new tests FAIL (`test_echoes_add_ink_right_of_subject` finds no extra ink); Task 1 tests still PASS.

- [ ] **Step 3: Implement echoes**

Replace `_echo_smear`'s pixel loop with:

```python
@njit(cache=True, parallel=True)
def _echo_smear(img, thr, count, spacing, wave, phase, streak, dissolve, breath):
    h, w = img.shape
    b = breath / 100.0
    dissolve_gate = b * dissolve / 100.0 * 2.0
    if dissolve_gate > 1.0:
        dissolve_gate = 1.0
    phase_rad = phase * math.pi / 180.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ink = False
            if img[y, x] < thr:
                if _hash01(x, y, 101) >= dissolve_gate:
                    ink = True
            if not ink and count > 0 and b > 0.0:
                for n in range(1, count + 1):
                    off = math.sin(y * 0.1 + phase_rad + n * 0.7) * wave
                    sx = int(x - n * spacing - off)
                    if sx < 0 or sx >= w:
                        continue
                    s0 = img[y, sx] < thr
                    xr = sx + 2 if sx + 2 < w else w - 1
                    yd = y + 2 if y + 2 < h else h - 1
                    if s0 != (img[y, xr] < thr) or s0 != (img[yd, sx] < thr):
                        decay = 1.0 - (n - 1.0) / count      # 1.0 .. 1/count
                        if _hash01(x, y, 202 + n) < b * (0.35 + 0.65 * decay):
                            ink = True
                            break
            out[y, x] = 0.0 if ink else 255.0
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: all PASS. Defaults changed output, so re-bake the golden to keep the whole suite green at this commit:

```powershell
Remove-Item "tests/golden/Echo Smear.npy"
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest "tests/test_kernels_all.py" -v -k "Echo"
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest "tests/test_kernels_all.py" -v -k "Echo"
```

Expected: PASS both times (first run seeds, second verifies).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/kernels/special.py tests/test_echo_smear.py "tests/golden/Echo Smear.npy"
git commit -m "feat(dither): Echo Smear contour echo trails"
```

---

### Task 3: Full-height streaks

**Files:**
- Modify: `ditherzam/dithering/kernels/special.py` (`_echo_smear` only)
- Modify: `tests/test_echo_smear.py`

**Interfaces:**
- Consumes: Task 2 `_echo_smear`.
- Produces: streak behavior — columns whose peak darkness passes the subject gate become 1-px full-height ink lines, selected by `_hash01(x, 0, 303)` with probability `streak/100 * 0.08`, independent of Breath.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_echo_smear.py`:

```python
def _streak_columns(out):
    # Columns that are ink for their entire height.
    return {x for x in range(out.shape[1]) if np.all(out[:, x] == 0.0)}


def test_streaks_are_full_height_and_scale_with_slider():
    img = _subject_square(128)
    img[50:80, 40:90] = 20.0                     # widen subject so many columns qualify
    off = entry().func(img.copy(), (0, 10, 0, 0, 0, 0, 0), THR)
    lo = entry().func(img.copy(), (0, 10, 0, 0, 30, 0, 0), THR)
    hi = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 0), THR)
    assert _streak_columns(off) == set()
    assert len(_streak_columns(hi)) >= len(_streak_columns(lo))
    assert len(_streak_columns(hi)) >= 1


def test_streaks_only_from_subject_columns():
    img = np.full((64, 64), 230.0, dtype=np.float32)   # no subject anywhere
    out = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 0), THR)
    assert _streak_columns(out) == set()


def test_streaks_survive_breath_zero():
    img = _subject_square(128)
    img[50:80, 40:90] = 20.0
    b0 = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 0), THR)
    b100 = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 100), THR)
    assert _streak_columns(b0) == _streak_columns(b100)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: `test_streaks_are_full_height_and_scale_with_slider` FAILS (no full-height columns exist). The no-subject and breath tests may trivially pass — that is fine; they pin behavior.

- [ ] **Step 3: Implement streaks**

Add a column pre-pass at the top of `_echo_smear` (after `phase_rad`), and a streak clause in the pixel loop:

```python
    subject_gate = thr            # column qualifies if it contains any subject pixel
    streak_prob = streak / 100.0 * 0.08
    col_streak = np.zeros(w, dtype=np.uint8)
    for x in prange(w):
        has_subject = False
        for y in range(h):
            if img[y, x] < subject_gate:
                has_subject = True
                break
        if has_subject and _hash01(x, 0, 303) < streak_prob:
            col_streak[x] = 1
```

In the pixel loop, before the body clause:

```python
            if col_streak[x] == 1:
                out[y, x] = 0.0
                continue
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: all PASS. Streaks change default output (Streak default 20), so re-bake the golden exactly as in Task 2 step 4 (delete `tests/golden/Echo Smear.npy`, run the Echo golden test twice).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/kernels/special.py tests/test_echo_smear.py "tests/golden/Echo Smear.npy"
git commit -m "feat(dither): Echo Smear full-height streaks"
```

---

### Task 4: Speckle dust, breath endpoints, golden re-bake

**Files:**
- Modify: `ditherzam/dithering/kernels/special.py` (`_echo_smear` only)
- Modify: `tests/test_echo_smear.py`
- Re-bake: `tests/golden/Echo Smear.npy`

**Interfaces:**
- Consumes: Task 3 `_echo_smear`.
- Produces: dust behavior — a background pixel becomes ink when a hash-jittered sample toward the subject (`xs = x + 1 + hash*reach`, `reach = spacing*count`) lands inside the subject, gated by `b * dissolve/100 * 0.5`. Final endpoint semantics: Breath 0 → exact threshold body (+streaks); Breath 100 + Dissolve 100 → no interior body ink.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_echo_smear.py`:

```python
def test_dust_appears_left_of_subject():
    img = _subject_square()
    no_dust = entry().func(img.copy(), (0, 10, 0, 0, 0, 0, 100), THR)
    dust = entry().func(img.copy(), (6, 10, 0, 0, 0, 100, 100), THR)
    left = slice(None), slice(0, 10)             # strictly left of the square
    assert (dust[left] == 0.0).sum() > (no_dust[left] == 0.0).sum()


def test_full_breath_full_dissolve_erases_body():
    img = _subject_square()
    out = entry().func(img.copy(), (0, 10, 0, 0, 0, 100, 100), THR)
    interior = out[24:40, 12:29]                 # deep inside the square
    assert np.all(interior == 255.0)


def test_partial_dissolve_erodes_partially():
    img = _subject_square()
    out = entry().func(img.copy(), (0, 10, 0, 0, 0, 50, 50), THR)
    interior = out[24:40, 12:29]
    frac = (interior == 0.0).mean()
    assert 0.2 < frac < 0.9                      # eroded but present


def test_each_native_control_changes_pixels(image):
    e = entry()
    base = e.func(image.copy(), DEFAULTS, THR)
    alternatives = (12, 20, 20, 180, 80, 90, 100)
    assert len(e.param_sliders) == 7
    for index, value in enumerate(alternatives):
        params = list(DEFAULTS)
        params[index] = value
        changed = e.func(image.copy(), tuple(params), THR)
        assert np.any(changed != base), e.param_sliders[index]
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: `test_dust_appears_left_of_subject` FAILS (no dust clause yet). `test_full_breath_full_dissolve_erases_body` PASSES already (gate clamps to 1.0) — keep it as a pin. Others may fail on the dust-free kernel; note which.

- [ ] **Step 3: Implement dust + edge-biased body erosion**

Replace the body clause (spec: erosion is densest near the silhouette edge — near-edge pixels erode at 1.5x the interior rate; both multipliers exceed 1.0 when Breath and Dissolve are maxed, preserving the full-erase endpoint):

```python
            if img[y, x] < thr:
                xl = x - 3 if x >= 3 else 0
                xr2 = x + 3 if x + 3 < w else w - 1
                near_edge = img[y, xl] >= thr or img[y, xr2] >= thr
                local = dissolve_gate * (1.5 if near_edge else 0.75)
                if _hash01(x, y, 101) >= local:
                    ink = True
```

(`dissolve_gate` keeps its Task 1 definition `b * dissolve / 100.0 * 2.0` but drop the clamp there — clamping now happens implicitly via the hash comparison, and at Breath 100 + Dissolve 100 both `2.0 * 0.75` and `2.0 * 1.5` are ≥ 1.0 so every body pixel erodes.)

Add the dust clause to the pixel loop after the echo clause:

```python
            if not ink and img[y, x] >= thr:
                dust_gate = b * dissolve / 100.0 * 0.5
                if dust_gate > 0.0:
                    reach = spacing * (count if count > 0 else 4)
                    xs = x + 1 + int(_hash01(x, y, 404) * reach)
                    if xs < w and img[y, xs] < thr:
                        if _hash01(x, y, 505) < dust_gate:
                            ink = True
```

- [ ] **Step 4: Run tests, then re-bake the golden**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: all PASS. Then re-bake (defaults' output changed since Task 1):

```powershell
Remove-Item "tests/golden/Echo Smear.npy"
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest "tests/test_kernels_all.py" -v -k "Echo"
```

Expected: PASS (first run re-seeds the fixture). Run it twice to confirm the seeded golden verifies.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/kernels/special.py tests/test_echo_smear.py "tests/golden/Echo Smear.npy"
git commit -m "feat(dither): Echo Smear speckle dust + breath endpoints, bake golden"
```

---

### Task 5: Quality gates, JIT-on parity, full suite, visual proof

**Files:**
- Modify: `tests/test_echo_smear.py` (audit gates)
- No production code changes expected (fix kernel only if a gate fails).

**Interfaces:**
- Consumes: everything above.
- Produces: merged-quality certification — non-collapse, non-duplicate, JIT-on parity, green suite, and a visual sample for user approval.

- [ ] **Step 1: Write the audit-gate tests**

Append to `tests/test_echo_smear.py`:

```python
def test_default_output_not_collapsed(gradient):
    out = entry().func(gradient.copy(), DEFAULTS, THR)
    ink = (out == 0.0).mean()
    assert 0.02 < ink < 0.98


def test_not_a_duplicate_of_contour_family(image):
    from tests.golden_harness import default_param
    ours = entry().func(image.copy(), DEFAULTS, THR)
    for name in ("Topography", "Topography Alt", "Displace Contour"):
        other = registry.get_entry(name)
        theirs = other.func(image.copy(), default_param(other), THR)
        assert np.any(ours != theirs), name
```

- [ ] **Step 2: Run the audit gates**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_echo_smear.py -v
```

Expected: PASS. If `test_default_output_not_collapsed` fails, the defaults are producing near-empty output on gradients — revisit the echo decay constants (`0.35 + 0.65 * decay`) rather than the test.

- [ ] **Step 3: JIT-on parity check**

```powershell
Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
.venv/Scripts/python.exe -m pytest tests/test_echo_smear.py tests/test_kernels_all.py -v
```

Expected: PASS (compiled kernel matches the JIT-off golden byte-for-byte). Known trap from memory: under JIT-on parfor, any float-valued index must be wrapped `int()` at the getitem site — `sx`, `xr`, `yd`, `xs` are already int-typed above; if numba still complains, apply `int()` at the indexing site, not the assignment.

- [ ] **Step 4: Full suite JIT-off**

```powershell
$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest -q
```

Expected: same green/skip counts as branch base plus the new tests (base was 1642 passed / 310 asset-gated skips; `test_mask_editor_lifecycle.py::test_blocking_inference_pool_...` is a known load-dependent flake — rerun it in isolation if it fails).

- [ ] **Step 5: Visual proof for user approval**

Write `scratchpad` script (do not commit) rendering a real photo through the app pipeline at defaults plus a Breath sweep (0/25/50/75/100), red-on-black palette, and send the PNGs to the user:

```python
# save as <scratchpad>/echo_smear_preview.py; run with .venv python
import numpy as np
from PIL import Image
from ditherzam.dithering import registry
from tests.golden_harness import default_param

src = Image.open(r"purple harrow.png").convert("L").resize((360, 640))
img = np.asarray(src, dtype=np.float32)
e = registry.get_entry("Echo Smear")
panels = []
for breath in (0, 25, 50, 75, 100):
    p = list(default_param(e)); p[6] = breath
    out = e.func(img.copy(), tuple(p), np.float32(128.0))
    rgb = np.zeros((*out.shape, 3), np.uint8)
    rgb[out == 0.0] = (255, 40, 30)              # red ink on black
    panels.append(rgb)
strip = np.concatenate(panels, axis=1)
Image.fromarray(strip).save("echo_smear_breath_sweep.png")
```

Send `echo_smear_breath_sweep.png` to the user and ask whether the look matches the reference before finishing the branch. Tuning iterations (wave frequency `0.1`, echo decay constants, dust reach) happen here against the user's eye; after any kernel change, re-run Task 5 steps 2-4 and re-bake the golden as in Task 4 step 4.

- [ ] **Step 6: Commit + record memory**

```bash
git add tests/test_echo_smear.py
git commit -m "test(dither): Echo Smear audit gates + JIT parity certified"
```

Then record a `progress` entry via the zam-memory skill (new style shipped, branch, HEAD, test counts) and update `docs/memory/INDEX.md`.
