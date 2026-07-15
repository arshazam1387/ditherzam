# Feedback Smear Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new "Feedback Smear" dither style that simulates a TouchDesigner-style feedback loop (backward displacement walk + decay + erosion) so stills — and a scrubbed Time slider — look like the reference video's organic smear.

**Architecture:** One `@njit(parallel=True)` gather kernel + a hash-lattice value-noise helper in the existing Special Effects family. Spec: `docs/superpowers/specs/2026-07-14-feedback-smear-design.md`. Echo Smear is untouched.

**Tech Stack:** Python 3.12 (`.venv/Scripts/python.exe`), NumPy, Numba, pytest.

## Global Constraints

- Clean-room: no Dither Boy / Studio AAA code or strings.
- Touch only `ditherzam/dithering/` and `tests/` (+ scratchpad renders).
- Kernels float32 in/out, 0..255, binary output (0.0 ink / 255.0 background).
- JIT-off test env: `NUMBA_DISABLE_JIT=1`; repo root; `.venv/Scripts/python.exe -m pytest`.
- TDD per task: red → green → commit. Echo Smear and all other kernels must not change behavior.
- Sliders exact (key, label, min, max, default): `fs_length_slider` "Trail Length" 4-64-32; `fs_drift_slider` "Drift" 1-8-2; `fs_noise_amount_slider` "Noise Amount" 0-24-6; `fs_noise_scale_slider` "Noise Scale" 1-100-20; `fs_decay_slider` "Decay" 50-100-88; `fs_time_slider` "Time" 0-360-0; `fs_erode_slider` "Erode" 0-100-30; `fs_density_slider` "Density" 0-200-100.
- Param tuple order = slider order above: `(K, drift, namount, nscale, decay, time, erode, density)`; wrapper defaults `(32, 2, 6, 20, 88, 0, 30, 100)`; unpack via the existing `_unpack8`.
- The A/B acceptance loop (Task 4) is the ship gate: constants may be retuned there; every retune re-runs Task 3's gates and re-bakes the golden.

## Established codebase facts

- Registration/wrapper/param-row pattern, `_unpack8`, `_hash01`, golden auto-seed harness, `test_param_order_matches_sliders` arity check: all as used by Echo Smear in `ditherzam/dithering/kernels/special.py` and `ditherzam/dithering/parameters.py` (see the Echo Smear wrapper and `_KEY_SPECS` tail as the template).
- `tests/golden_harness.default_param` probes real app defaults; `tests/test_kernels_all.py` auto-covers new styles.

---

### Task 1: Value-noise helper

**Files:**
- Modify: `ditherzam/dithering/kernels/special.py`
- Create: `tests/test_feedback_smear.py`

**Interfaces:**
- Produces: `_vnoise(x, y, k, salt) -> float in [0, 1)`, `@njit(cache=True)`, smooth 2D value noise: bilinear smoothstep interpolation over an integer hash lattice, iteration `k` folded into the lattice x with stride 8191. Deterministic, no RNG.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_feedback_smear.py`:

```python
import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs

DEFAULTS = (32, 2, 6, 20, 88, 0, 30, 100)
THR = np.float32(128.0)


def test_vnoise_range_smoothness_determinism():
    from ditherzam.dithering.kernels.special import _vnoise
    vals = [_vnoise(x * 0.13, 7.7, 3, 606) for x in range(200)]
    assert all(0.0 <= v < 1.0 for v in vals)
    # deterministic
    assert vals == [_vnoise(x * 0.13, 7.7, 3, 606) for x in range(200)]
    # smooth: neighboring samples move less than lattice-uncorrelated ones would
    steps = [abs(vals[i + 1] - vals[i]) for i in range(199)]
    assert max(steps) < 0.35
    # k separates fields
    assert any(_vnoise(x * 0.13, 7.7, 4, 606) != vals[x] for x in range(200))
```

- [ ] **Step 2: Run to verify failure**

`$env:NUMBA_DISABLE_JIT='1'; .venv/Scripts/python.exe -m pytest tests/test_feedback_smear.py -v` → ImportError (`_vnoise` missing).

- [ ] **Step 3: Implement**

In `special.py`, below `_hash01`:

```python
@njit(cache=True)
def _vnoise(x, y, k, salt):
    # Smooth 2D value noise on an integer hash lattice; iteration index k is
    # folded into the lattice so every feedback step gets its own field.
    xi = int(math.floor(x)) + k * 8191
    yi = int(math.floor(y))
    fx = x - math.floor(x)
    fy = y - math.floor(y)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    n00 = _hash01(xi, yi, salt)
    n10 = _hash01(xi + 1, yi, salt)
    n01 = _hash01(xi, yi + 1, salt)
    n11 = _hash01(xi + 1, yi + 1, salt)
    return (n00 * (1.0 - ux) + n10 * ux) * (1.0 - uy) + (n01 * (1.0 - ux) + n11 * ux) * uy
```

- [ ] **Step 4: Verify green** (same command → PASS)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/kernels/special.py tests/test_feedback_smear.py
git commit -m "feat(dither): hash-lattice value noise helper for Feedback Smear"
```

---

### Task 2: Feedback Smear kernel + registration + params

**Files:**
- Modify: `ditherzam/dithering/kernels/special.py`, `ditherzam/dithering/parameters.py`
- Modify: `tests/test_feedback_smear.py`

**Interfaces:**
- Consumes: `_vnoise`, `_hash01`, `_unpack8`.
- Produces: registered style `"Feedback Smear"` (Special Effects, dims=2) with `param_sliders = ("fs_length_slider", "fs_drift_slider", "fs_noise_amount_slider", "fs_noise_scale_slider", "fs_decay_slider", "fs_time_slider", "fs_erode_slider", "fs_density_slider")`; private kernel `_feedback_smear(img, thr, k_iters, drift, namount, nscale, decay, time_v, erode, density)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feedback_smear.py`:

```python
def entry():
    e = registry.get_entry("Feedback Smear")
    assert e is not None
    return e


def _pylon(size=128):
    # Dark vertical mast with arms on white: crude reference-like subject.
    img = np.full((size, size), 235.0, dtype=np.float32)
    img[20:118, 58:70] = 20.0
    img[34:42, 34:94] = 20.0
    img[64:72, 40:88] = 20.0
    return img


def test_registered_with_exact_sliders():
    e = entry()
    assert e.category == "Special Effects" and e.dims == 2
    specs = {s.key: (s.label, s.minimum, s.maximum, s.default)
             for s in parameter_specs(e) if not s.key.startswith("creative_")}
    assert specs == {
        "fs_length_slider": ("Trail Length", 4, 64, 32),
        "fs_drift_slider": ("Drift", 1, 8, 2),
        "fs_noise_amount_slider": ("Noise Amount", 0, 24, 6),
        "fs_noise_scale_slider": ("Noise Scale", 1, 100, 20),
        "fs_decay_slider": ("Decay", 50, 100, 88),
        "fs_time_slider": ("Time", 0, 360, 0),
        "fs_erode_slider": ("Erode", 0, 100, 30),
        "fs_density_slider": ("Density", 0, 200, 100),
    }


def test_output_contract_and_determinism():
    img = _pylon()
    a = entry().func(img.copy(), DEFAULTS, THR)
    b = entry().func(img.copy(), DEFAULTS, THR)
    assert a.dtype == np.float32 and a.shape == img.shape
    assert set(np.unique(a)) <= {0.0, 255.0}
    np.testing.assert_array_equal(a, b)


def test_trails_accumulate_on_smear_side_and_fade():
    img = _pylon()
    out = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100), THR)
    near = out[20:118, 71:95]                    # just right of the mast
    far = out[20:118, 110:128]
    assert (near == 0.0).mean() > (far == 0.0).mean() > 0.0


def test_no_trails_at_zero_density():
    img = _pylon()
    out = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 0), THR)
    expected = np.where(img < 128.0, 0.0, 255.0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_time_scrubs_the_field():
    img = _pylon()
    a = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100), THR)
    b = entry().func(img.copy(), (32, 2, 6, 20, 88, 180, 0, 100), THR)
    assert np.any(a != b)


def test_erode_eats_subject_organically():
    img = _pylon()
    solid = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 0, 100), THR)
    eaten = entry().func(img.copy(), (32, 2, 6, 20, 88, 0, 70, 100), THR)
    body = slice(20, 118), slice(58, 70)
    assert (eaten[body] == 255.0).sum() > (solid[body] == 255.0).sum()
```

- [ ] **Step 2: Verify failure** (entry() is None) — same pytest command.

- [ ] **Step 3: Implement**

`parameters.py` `_KEY_SPECS` tail:

```python
    "fs_length_slider": ("Trail Length", 4, 64, 32),
    "fs_drift_slider": ("Drift", 1, 8, 2),
    "fs_noise_amount_slider": ("Noise Amount", 0, 24, 6),
    "fs_noise_scale_slider": ("Noise Scale", 1, 100, 20),
    "fs_decay_slider": ("Decay", 50, 100, 88),
    "fs_time_slider": ("Time", 0, 360, 0),
    "fs_erode_slider": ("Erode", 0, 100, 30),
    "fs_density_slider": ("Density", 0, 200, 100),
```

`special.py` private kernel (below `_echo_smear`) and wrapper (below the Echo Smear wrapper):

```python
@njit(cache=True, parallel=True)
def _feedback_smear(img, thr, k_iters, drift, namount, nscale, decay, time_v, erode, density):
    h, w = img.shape
    s = nscale / 1000.0
    tshift = time_v / 360.0 * 37.0
    dgain = density / 100.0
    surv_rate = decay / 100.0
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            ink = False
            subject = img[y, x] < thr
            if subject:
                field = _vnoise(x * s * 2.0 + tshift, y * s * 2.0, 0, 909)
                gate = erode / 100.0 * (0.35 + 0.65 * field)
                if _hash01(x, y, 111) >= gate:
                    ink = True
            if not ink and dgain > 0.0:
                surv = 1.0
                px = float(x)
                py = float(y)
                prev_in = subject
                for k in range(1, k_iters + 1):
                    surv *= surv_rate
                    if surv * dgain < 0.02:
                        break
                    nx = _vnoise(x * s + tshift + k * 0.618, y * s, k, 606)
                    ny = _vnoise(x * s + tshift + k * 0.618, y * s, k, 707)
                    px -= drift + (nx - 0.5) * 2.0 * namount
                    py -= (ny - 0.5) * 2.0 * namount * 0.6
                    xi = int(px)
                    yi = int(py)
                    if xi < 0 or xi >= w or yi < 0 or yi >= h:
                        break
                    cur_in = img[yi, xi] < thr
                    if cur_in and not prev_in:
                        if _hash01(x, y, 202 + k) < surv * dgain:
                            ink = True
                            break
                    prev_in = cur_in
            out[y, x] = 0.0 if ink else 255.0
    return out
```

```python
# ── Kernel: Feedback Smear · Special Effects · dims=2 ──
#    sliders (Trail Length 4-64-32, Drift 1-8-2, Noise Amount 0-24-6, Noise Scale 1-100-20,
#             Decay 50-100-88, Time 0-360-0, Erode 0-100-30, Density 0-200-100)
@registry.register("Feedback Smear", "Special Effects", dims=2,
                   param_sliders=("fs_length_slider", "fs_drift_slider",
                                  "fs_noise_amount_slider", "fs_noise_scale_slider",
                                  "fs_decay_slider", "fs_time_slider",
                                  "fs_erode_slider", "fs_density_slider"))
def feedback_smear(image_array, parameter, luminance_threshold_value):
    k_iters, drift, namount, nscale, decay, time_v, erode, density = _unpack8(
        parameter, 32, 2, 6, 20, 88, 0, 30, 100)
    return _feedback_smear(image_array.astype(np.float32),
                           float(luminance_threshold_value),
                           max(1, int(k_iters)), float(drift), float(namount),
                           max(1.0, float(nscale)), float(decay), float(time_v),
                           float(erode), float(density))
```

- [ ] **Step 4: Verify green** — `tests/test_feedback_smear.py` + `tests/test_kernels_all.py` (golden auto-seeds `tests/golden/Feedback Smear.npy`; arity check exercises the 8-tuple).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/dithering/kernels/special.py ditherzam/dithering/parameters.py tests/test_feedback_smear.py "tests/golden/Feedback Smear.npy"
git commit -m "feat(dither): Feedback Smear style - simulated feedback walk"
```

---

### Task 3: Quality gates + JIT-on parity

**Files:**
- Modify: `tests/test_feedback_smear.py`

- [ ] **Step 1: Write gate tests**

```python
def test_default_output_not_collapsed():
    gradient = np.tile(np.linspace(0, 255, 96, dtype=np.float32), (96, 1))
    out = entry().func(gradient.copy(), DEFAULTS, THR)
    ink = (out == 0.0).mean()
    assert 0.02 < ink < 0.98


def test_not_a_duplicate_of_echo_smear():
    from tests.golden_harness import default_param
    img = _pylon()
    ours = entry().func(img.copy(), DEFAULTS, THR)
    other = registry.get_entry("Echo Smear")
    theirs = other.func(img.copy(), default_param(other), THR)
    assert np.any(ours != theirs)


def test_each_native_control_changes_pixels():
    img = _pylon()
    e = entry()
    base = e.func(img.copy(), DEFAULTS, THR)
    alternatives = (8, 6, 18, 70, 65, 180, 80, 30)
    assert len(e.param_sliders) == 8
    for index, value in enumerate(alternatives):
        params = list(DEFAULTS)
        params[index] = value
        changed = e.func(img.copy(), tuple(params), THR)
        assert np.any(changed != base), e.param_sliders[index]
```

- [ ] **Step 2: JIT-off green**, then **JIT-on parity**: `Remove-Item Env:NUMBA_DISABLE_JIT ...; .venv/Scripts/python.exe -m pytest tests/test_feedback_smear.py tests/test_kernels_all.py -v` (10-min timeout; golden must match byte-for-byte).

- [ ] **Step 3: Full suite JIT-off** (`-q`, 10-min timeout; known skips/flake per branch norm).

- [ ] **Step 4: Commit**

```bash
git add tests/test_feedback_smear.py
git commit -m "test(dither): Feedback Smear gates + parity"
```

---

### Task 4: A/B acceptance loop vs the reference (USER GATE — controller-run)

- [ ] Render the binarized reference pylon frame (scratchpad `refvid/frame_07.png`, crop+binarize as in `echo_smear_tower_demo.py`) through Feedback Smear at defaults plus a Time sweep; also render the clean cat silhouette.
- [ ] Send side-by-side with the actual reference frames to the user.
- [ ] Iterate constants (noise scale mapping `s`, wobble ratio `0.6`, survival floor `0.02`, erosion band `0.35 + 0.65*field`, drift/namount defaults) per the user's eye. After ANY kernel constant change: re-run Task 3 gates, re-bake the golden (delete + Echo golden run twice pattern), commit with a descriptive message.
- [ ] Repeat until the user says it matches. Only then proceed to Task 5.

### Task 5: Close out

- [ ] Final whole-branch review (superpowers:requesting-code-review) covering both styles' work on this branch.
- [ ] zam-memory progress entry + INDEX line; then superpowers:finishing-a-development-branch.
