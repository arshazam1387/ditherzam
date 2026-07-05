# Phase 1 — Foundation & Dither Core — Completion Task List

Verify every task in [`../plans/01-foundation-and-dither-core.md`](../plans/01-foundation-and-dither-core.md) is actually done to its Definition-of-Done, bootstrap the Python 3.12 environment, and close the two gaps the plan does not cover (a `NUMBA_DISABLE_JIT` JIT-on-vs-off equivalence test and a `param_func` dispatch path in the registry+pipeline).

## Prereqs

- None (this is the first subsystem). Nothing downstream is required.
- This document is a **completion checklist**, not a re-expansion. Tasks 1.1–1.7 below map 1:1 onto Tasks 1–7 of plan 01: each restates the Definition-of-Done as checkboxes and gives the exact verification command + expected output. Tasks 1.0, 1.8, and 1.9 are **new** (env bootstrap + the two gaps) and are full red→green→commit TDD blocks.
- Hard rules carried from the roadmap: core stays **Qt-free**; kernels are `@njit(cache=True, parallel=True)`, `float32` I/O in `0..255`; TDD with a commit after every green; **no placeholders**; honor the FROZEN CONTRACTS (`DitherEntry`, `DitherRegistry`, `apply_dither`, adjustment signatures) verbatim.

---

### Task 1.0: Environment bootstrap (Python 3.12) — NEW

**Files:** none created (environment only). Verifies the interpreter that every later `pip`/`pytest` command uses.

**Interfaces:** Produces a working Python **3.12** interpreter + an editable install of `ditherzam`. Consumed by every other task.

**Context — the interpreter situation on this machine:**
- There is **no `python` / `py` on `PATH`** (`python --version` → `command not found`; `py` → `command not found`). Do **not** assume a global interpreter exists.
- System interpreters that *are* reachable elsewhere are **3.11 and 3.13**, which are **NOT usable** — Numba/PySide6 wheels target 3.12, and `pyproject.toml` pins `requires-python = ">=3.12,<3.13"`.
- A clean, verified **Python 3.12.7** lives in the scratchpad at:
  `C:\Users\arsha\AppData\Local\Temp\claude\C--Users-arsha-Desktop-custom-dither\8cf6d0f0-124e-4a95-9a87-2cf3349c7a26\scratchpad\dbwork\py312\python.exe`
  (Scratchpad paths are session-scoped; if that exact folder is gone, re-provision a 3.12 with `uv python install 3.12` or the python.org 3.12 installer, then repoint `PY312` below.)

- [ ] **Step 1: Confirm the 3.12 interpreter is alive and is really 3.12**

Run (Git Bash):
```bash
PY312="C:/Users/arsha/AppData/Local/Temp/claude/C--Users-arsha-Desktop-custom-dither/8cf6d0f0-124e-4a95-9a87-2cf3349c7a26/scratchpad/dbwork/py312/python.exe"
"$PY312" --version
```
Expected: `Python 3.12.7` (any `3.12.x` is acceptable; a `3.11.x` or `3.13.x` here is a STOP — repoint `PY312`).

- [ ] **Step 2: Create a project-local venv from that 3.12 so later commands don't depend on the scratchpad**

Run:
```bash
"$PY312" -m venv .venv
.venv/Scripts/python.exe --version
```
Expected: `Python 3.12.7`. From here on, `python`/`pip`/`pytest` mean `.venv/Scripts/…`. Add `.venv/` to `.gitignore` if not already ignored.

- [ ] **Step 3: Verify the venv interpreter satisfies the packaging pin** (this is the guard that would reject 3.11/3.13)

Run:
```bash
.venv/Scripts/python.exe -c "import sys; assert (3,12) <= sys.version_info < (3,13), sys.version; print('OK 3.12')"
```
Expected: `OK 3.12`.

- [ ] **Step 4: Confirm `pip` works in the venv** (needed by Task 1.1's editable install)

Run: `.venv/Scripts/python.exe -m pip --version`
Expected: a `pip … from …\.venv\…` line.

- [ ] **DoD for Task 1.0**
  - [ ] A 3.12 interpreter is available and identified (scratchpad `python.exe` and/or `.venv`).
  - [ ] No later step relies on a bare `python` on `PATH`.
  - [ ] Recorded in the working notes: if the scratchpad 3.12 disappears, re-provision before continuing.

---

### Task 1.1: Project scaffold & packaging  (verifies plan 01 Task 1)

**Files:** Verify exist: `pyproject.toml`, `ditherzam/__init__.py`, `tests/conftest.py`.

**Interfaces:** Produces an installable package `ditherzam==0.1.0`; `pytest` runs; JIT disabled by default in tests.

- [ ] **DoD checkboxes**
  - [ ] `pyproject.toml` exists, `name = "ditherzam"`, `version = "0.1.0"`, `requires-python = ">=3.12,<3.13"`, and lists deps `numpy>=1.26, numba>=0.59, Pillow>=10.0, PyYAML>=6.0, platformdirs>=4.0, PySide6>=6.6` with `dev = ["pytest>=8.0"]` and `[project.scripts] ditherzam = "ditherzam.app:main"`.
  - [ ] `ditherzam/__init__.py` defines `__version__ = "0.1.0"`.
  - [ ] `tests/conftest.py` does `os.environ.setdefault("NUMBA_DISABLE_JIT", "1")`.
- [ ] **Verify — editable install**

  Run: `.venv/Scripts/python.exe -m pip install -e ".[dev]"`
  Expected: ends with `Successfully installed … ditherzam-0.1.0 …` (plus deps). No `requires-python` conflict (proves the 3.12 pin is satisfied).
- [ ] **Verify — import + version**

  Run: `.venv/Scripts/python.exe -c "import ditherzam; print(ditherzam.__version__)"`
  Expected: `0.1.0`
- [ ] **Verify — conftest disables JIT**

  Run: `.venv/Scripts/python.exe -c "import tests.conftest, os; print(os.environ['NUMBA_DISABLE_JIT'])"`
  Expected: `1`
- [ ] **Verify — pytest is wired**

  Run: `.venv/Scripts/python.exe -m pytest --collect-only -q`
  Expected: collection succeeds (no import errors); prints the collected test count.

---

### Task 1.2: Config loader  (verifies plan 01 Task 2)

**Files:** Verify exist: `config/config.yaml`, `ditherzam/config.py`, `tests/test_config.py`.

**Interfaces:** Produces `ConfigError`, `@dataclass AppConfig`, `load_config(path) -> AppConfig`.

- [ ] **DoD checkboxes**
  - [ ] `config/config.yaml` has `dither.default_style: "None"`, `dither.default_scale: 5`, `style.viewport_bg_color: "#1f1f1f"`, `style.app_style: "Fusion"`, and `inertia`/`timing` blocks.
  - [ ] `AppConfig` fields: `default_dither_style, default_dither_scale, viewport_bg_color, app_style, enable_inertia, friction, velocity_scale, max_velocity, debounce_ms, loading_delay_ms`.
  - [ ] `load_config` raises `ConfigError` on missing file, empty file, and missing/mistyped keys.
  - [ ] No Qt import in `config.py`.
- [ ] **Verify — tests pass**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
  Expected: `3 passed` (`test_load_config_reads_values`, `test_missing_file_raises`, `test_empty_file_raises`).
- [ ] **Verify — real config parses**

  Run: `.venv/Scripts/python.exe -c "from ditherzam.config import load_config; c=load_config('config/config.yaml'); print(c.app_style, c.default_dither_scale, c.enable_inertia)"`
  Expected: `Fusion 5 True`

---

### Task 1.3: Imaging helpers  (verifies plan 01 Task 3)

**Files:** Verify exist: `ditherzam/imaging.py`, `tests/test_imaging.py`.

**Interfaces:** Produces `to_gray_f32`, `clamp_u8`, `nearest_downscale(gray_f32, factor)`, `nearest_upscale_to(small_f32, size_wh)`.

- [ ] **DoD checkboxes**
  - [ ] `to_gray_f32` returns `HxW float32` in `0..255` from a PIL image or an ndarray (RGB averaged to gray).
  - [ ] `clamp_u8` clips to `0..255` and returns `uint8`.
  - [ ] `nearest_downscale(g, factor)` uses NEAREST, floors size to `w//factor, h//factor` (min 1), `factor` clamped `>=1`.
  - [ ] `nearest_upscale_to(s, (w,h))` NEAREST-resizes back to `(w,h)`, returns `float32`.
  - [ ] No Qt import.
- [ ] **Verify — tests pass**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_imaging.py -v`
  Expected: `3 passed` (`test_to_gray_f32_from_rgb`, `test_clamp_u8`, `test_downscale_then_upscale_roundtrips_size`).
- [ ] **Verify — dtype/range invariants hold**

  Run: `.venv/Scripts/python.exe -c "import numpy as np; from ditherzam.imaging import to_gray_f32, clamp_u8; a=np.array([[-5.,128.,300.]],dtype=np.float32); print(clamp_u8(a).dtype, clamp_u8(a).tolist())"`
  Expected: `uint8 [[0, 128, 255]]`

---

### Task 1.4: Tonal adjustments (exact formulas)  (verifies plan 01 Task 4)

**Files:** Verify exist: `ditherzam/adjustments.py`, `tests/test_adjustments.py`.

**Interfaces (FROZEN CONTRACT — signatures must match verbatim):**
- `apply_contrast(img, value)` → `img * (value/50)`
- `apply_midtones(img, value)` → `gamma=max(1+(value-50)/200,0.1); 255*(img/255)**(1/gamma)`
- `apply_highlights(img, value)` → `img * (1+(value-50)/100)`
- `apply_blur(img, value)` → PIL `GaussianBlur radius=(value/10)**2`; `radius<=0` → identity
- `apply_invert(img, enabled)` → `255-img` when enabled else `img`
- (`apply_saturation(rgb, value)` is Phase 3 — must NOT be in this module yet.)

- [ ] **DoD checkboxes**
  - [ ] All five functions present with the exact names/signatures above, returning `float32`.
  - [ ] `apply_contrast(x,50)` is identity; `apply_contrast(x,100)` doubles.
  - [ ] `apply_midtones(x,50)` is identity (gamma 1); `value=90` → gamma 1.2.
  - [ ] `apply_highlights(x,100)` = `x*1.5`.
  - [ ] `apply_blur(x,0)` is identity (no PIL round-trip).
  - [ ] `apply_invert(x,True)=255-x`, `apply_invert(x,False)=x`.
  - [ ] No Qt import; formulas match spec §7.1 exactly.
- [ ] **Verify — tests pass**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_adjustments.py -v`
  Expected: `7 passed`.
- [ ] **Verify — signatures match the frozen contract (no drift, no premature `apply_saturation`)**

  Run:
  ```bash
  .venv/Scripts/python.exe -c "import inspect, ditherzam.adjustments as a; print([ (n, str(inspect.signature(getattr(a,n)))) for n in ('apply_contrast','apply_midtones','apply_highlights','apply_blur','apply_invert') ]); print('has_saturation', hasattr(a,'apply_saturation'))"
  ```
  Expected: each signature is `(img, value)` except `apply_invert` = `(img, enabled)`, and `has_saturation False`.

---

### Task 1.5: Dither registry  (verifies plan 01 Task 5)

**Files:** Verify exist: `ditherzam/dithering/__init__.py`, `ditherzam/dithering/registry.py`, `tests/test_registry.py`.

**Interfaces (FROZEN CONTRACT):**
```python
@dataclass(frozen=True)
class DitherEntry:
    name: str; category: str; dims: int
    param_sliders: tuple[str, ...]; func: Callable; param_func: Callable | None = None
class DitherRegistry:
    def register(self, name, category, dims=2, param_sliders=(), param_func=None): ...  # decorator
    def get_entry(self, name) -> DitherEntry | None: ...
    def list_dithers(self) -> list[str]: ...
    def by_category(self) -> dict[str, list[str]]: ...
```

- [ ] **DoD checkboxes**
  - [ ] `DitherEntry` is `@dataclass(frozen=True)` with fields in the exact order/names above, `param_func` defaulting to `None`.
  - [ ] `register(...)` is a decorator returning the undecorated func; stores a `DitherEntry`.
  - [ ] `get_entry` returns the entry or `None`; `list_dithers` preserves insertion order; `by_category` groups names per category preserving order.
  - [ ] No Qt import.
- [ ] **Verify — tests pass**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_registry.py -v`
  Expected: `3 passed` (`test_register_and_lookup`, `test_by_category_groups`, `test_unknown_returns_none`).
- [ ] **Verify — frozen fields + immutability**

  Run:
  ```bash
  .venv/Scripts/python.exe -c "import dataclasses as dc; from ditherzam.dithering.registry import DitherEntry; print([f.name for f in dc.fields(DitherEntry)]); e=DitherEntry('n','c',2,(),lambda *a:a); \
  import traceback
  try:
      object.__setattr__  # noqa
      e.name='x'
  except Exception as ex:
      print('frozen', type(ex).__name__)"
  ```
  Expected: field list `['name', 'category', 'dims', 'param_sliders', 'func', 'param_func']` and `frozen FrozenInstanceError`.

---

### Task 1.6: First three kernels + shared registry instance  (verifies plan 01 Task 6)

**Files:** Verify exist: `ditherzam/dithering/kernels/__init__.py`, `ditherzam/dithering/kernels/error_diffusion.py`; `ditherzam/dithering/__init__.py` exposes module-level `registry`; `tests/test_kernels_error_diffusion.py`.

**Interfaces:** `ditherzam.dithering.registry` is a populated `DitherRegistry` with `Floyd-Steinberg`, `Atkinson`, `Bayer-Matrix 4x4`. Kernel signature: `func(image_array, parameter, luminance_threshold_value) -> float32[H,W] in {0,255}`.

- [ ] **DoD checkboxes**
  - [ ] `ditherzam/dithering/__init__.py` defines `registry = DitherRegistry()` **then** imports `kernels.error_diffusion` for its registration side effects (no circular import).
  - [ ] All three names registered; categories are `Floyd-Steinberg`/`Atkinson` → "Error Diffusion", `Bayer-Matrix 4x4` → "Ordered Dither".
  - [ ] Output of each kernel is binary `{0.0, 255.0}`, same shape as input.
  - [ ] All-black input → all `0`; all-white input → all `255`.
  - [ ] Numba decorators present (`@njit(cache=True[, parallel=True])`); import-safe under `NUMBA_DISABLE_JIT=1`.
- [ ] **Verify — tests pass (JIT disabled path)**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_kernels_error_diffusion.py -v`
  Expected: `4 passed` (`test_all_three_registered`, `test_output_is_binary_0_255`, `test_all_black_input_stays_black`, `test_all_white_input_stays_white`).
- [ ] **Verify — the shared registry actually holds the three by name**

  Run: `.venv/Scripts/python.exe -c "from ditherzam.dithering import registry as r; print(sorted(r.list_dithers())); print(r.by_category())"`
  Expected: list contains `Atkinson`, `Bayer-Matrix 4x4`, `Floyd-Steinberg`; `by_category()` shows `Error Diffusion` → `['Floyd-Steinberg','Atkinson']`, `Ordered Dither` → `['Bayer-Matrix 4x4']`.

---

### Task 1.7: Dither pipeline (downscale → kernel → upscale + param dispatch)  (verifies plan 01 Task 7)

**Files:** Verify exist: `ditherzam/dithering/pipeline.py`, `tests/test_pipeline.py`.

**Interfaces (FROZEN CONTRACT — full signature):**
```python
def apply_dither(gray_f32, *, style, scale, luminance_threshold, params, registry,
                 preview_disabled=False, threshold_field=None) -> np.ndarray
```

- [ ] **DoD checkboxes**
  - [ ] `style == "None"`, unknown style, or `preview_disabled=True` → passthrough (returns the input array unchanged).
  - [ ] Otherwise: `tval = luminance_threshold/100*255`; `factor = max(1, int(scale))`; NEAREST downscale → kernel → NEAREST upscale back to original `(w,h)`.
  - [ ] Output shape equals input shape; dithered output is binary `{0,255}`.
  - [ ] No Qt import.
  - [ ] **Signature-drift note:** plan 01's original `apply_dither` omits `threshold_field` — it is aligned to the frozen contract in **Task 1.9** below (do not mark this box until 1.9 is green).
- [ ] **Verify — tests pass**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
  Expected: `4 passed` (`test_none_style_is_passthrough`, `test_preview_disabled_is_passthrough`, `test_dither_returns_binary_same_size`, `test_scale_pixelates_via_block_size`).
- [ ] **Verify — whole Phase-1 suite green together**

  Run: `.venv/Scripts/python.exe -m pytest -q`
  Expected: all Phase-1 tests pass, `0 failed`.

---

### Task 1.8: GAP — `NUMBA_DISABLE_JIT` fallback equivalence test — NEW (full TDD)

The whole suite runs with `NUMBA_DISABLE_JIT=1` (pure-Python interpretation of the kernels). Nothing yet proves the **compiled** kernels (JIT on) produce **identical** results. This closes that gap by running one kernel in two subprocesses — JIT off and JIT on — and asserting byte-identical output. It must be a subprocess test because `NUMBA_DISABLE_JIT` is read by Numba at import time and cannot be toggled inside a single already-imported process.

**Files:** Create/Test: `tests/test_numba_fallback.py`. Modify: none.

**Interfaces:** Consumes `ditherzam.dithering.registry` (Floyd-Steinberg + Bayer). Produces no new API — a behavioral guarantee.

- [ ] **Step 1: Write the failing test — `tests/test_numba_fallback.py`**

```python
"""JIT-on vs JIT-off equivalence: the compiled kernels must match the pure-Python
fallback bit-for-bit. Runs in subprocesses because NUMBA_DISABLE_JIT is read once,
at numba import time, and cannot be flipped inside a running interpreter."""
import os
import sys
import subprocess

import numpy as np

# Script prints the raveled kernel output as a comma-separated list of ints.
_SCRIPT = r"""
import numpy as np
from ditherzam.dithering import registry
img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
name = __import__("os").environ["DZ_KERNEL"]
out = registry.get_entry(name).func(img.copy(), 0, 128.0)
import sys
sys.stdout.write(",".join(str(int(v)) for v in out.ravel().tolist()))
"""


def _run(kernel_name: str, disable_jit: str) -> np.ndarray:
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = disable_jit
    env["DZ_KERNEL"] = kernel_name
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return np.array([int(x) for x in proc.stdout.strip().split(",")], dtype=np.int64)


def test_floyd_steinberg_jit_on_equals_jit_off():
    off = _run("Floyd-Steinberg", "1")   # pure-Python fallback
    on = _run("Floyd-Steinberg", "0")    # JIT-compiled
    assert off.shape == on.shape and off.size == 256
    np.testing.assert_array_equal(on, off)
    assert set(np.unique(off).tolist()) <= {0, 255}


def test_bayer4_jit_on_equals_jit_off():
    off = _run("Bayer-Matrix 4x4", "1")
    on = _run("Bayer-Matrix 4x4", "0")   # exercises the parallel=True prange path
    np.testing.assert_array_equal(on, off)
    assert set(np.unique(off).tolist()) <= {0, 255}
```

- [ ] **Step 2: Run — verify it fails FIRST for the right reason**

  To prove the test is real before the kernels are trusted, temporarily force a mismatch: run only the JIT-on leg and confirm the compile path works at all.
  Run: `.venv/Scripts/python.exe -m pytest tests/test_numba_fallback.py -v`
  Expected on a *fresh* checkout where the kernels are correct: this passes immediately (there is no product code to write — the test is the deliverable). If it FAILS, the failure message will be either `subprocess failed:` + a Numba compile traceback (a real JIT bug in a kernel) or `Arrays are not equal` (a genuine fallback divergence) — both are actionable bugs to fix in `error_diffusion.py`, not in the test.

  > Note: this is a **characterization/guard test**, so its natural state is green. The "red" here is the diagnostic: if either subprocess errors or the arrays differ, you have found a kernel that behaves differently compiled vs. interpreted — fix the kernel until this is green.

- [ ] **Step 3: (Only if red) Fix the offending kernel** — no new file. Typical causes and fixes:
  - Integer/float division differing between Python and Numba: force float math (`err * 7 / 16` → keep operands float, they already are).
  - `parallel=True` race on shared `out` in an error-diffusion kernel (order-dependent) — error diffusion must **not** be `parallel`; only the order-independent ordered/Bayer kernel may be `parallel=True`. Confirm `_floyd_steinberg`/`_atkinson` are `@njit(cache=True)` (no `parallel`), and `_ordered` is `@njit(cache=True, parallel=True)`.

- [ ] **Step 4: Run — verify pass**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_numba_fallback.py -v`
  Expected: `2 passed`. (First run is slow — real Numba compilation happens in the JIT-on subprocess.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_numba_fallback.py
git commit -m "test(dither): NUMBA_DISABLE_JIT on/off kernel equivalence guard"
```

---

### Task 1.9: GAP — `param_func` dispatch path (registry + pipeline) — NEW (full TDD)

The FROZEN CONTRACT gives `DitherEntry` a `param_func: Callable | None`, and spec §8.2 step 3 says *"if entry has `param_func`: `param = param_func(...)` else pull values from `param_sliders`."* Plan 01's `pipeline._build_param` **never consults `param_func`** — so a dither that transforms its slider params before the kernel runs is unreachable. This task adds that path and, at the same time, aligns `apply_dither` to the frozen signature by adding the (Phase-1-inert) `threshold_field=None` parameter so Phase 8 can wire it without a signature change.

**Files:** Modify: `ditherzam/dithering/pipeline.py`. Create/Test: `tests/test_param_func.py`.

**Interfaces (FROZEN CONTRACT, honored verbatim):**
```python
def apply_dither(gray_f32, *, style, scale, luminance_threshold, params, registry,
                 preview_disabled=False, threshold_field=None) -> np.ndarray
# param resolution: entry.param_func(params) if param_func is not None,
#                   else value(s) from entry.param_sliders (single value or tuple).
```

- [ ] **Step 1: Write the failing test — `tests/test_param_func.py`**

```python
import numpy as np
from ditherzam.dithering.registry import DitherRegistry
from ditherzam.dithering.pipeline import apply_dither


def test_param_func_transforms_slider_params_before_kernel():
    reg = DitherRegistry()
    captured = {}

    def double_param(params):
        return params["strength"] * 2

    @reg.register("PF", "Error Diffusion", dims=2,
                  param_sliders=("strength",), param_func=double_param)
    def pf(image_array, parameter, luminance_threshold_value):
        captured["param"] = parameter
        return np.where(image_array >= parameter, 255.0, 0.0).astype(np.float32)

    img = np.full((4, 4), 100.0, dtype=np.float32)
    apply_dither(img, style="PF", scale=1, luminance_threshold=50,
                 params={"strength": 30}, registry=reg)
    # param_func doubled 30 -> 60 and that value reached the kernel
    assert captured["param"] == 60


def test_param_func_overrides_multi_slider_extraction():
    reg = DitherRegistry()
    seen = {}

    def constant_seven(params):
        return 7

    @reg.register("PF2", "Error Diffusion", dims=2,
                  param_sliders=("a", "b"), param_func=constant_seven)
    def pf2(image_array, parameter, luminance_threshold_value):
        seen["param"] = parameter
        return image_array

    img = np.zeros((3, 3), dtype=np.float32)
    apply_dither(img, style="PF2", scale=1, luminance_threshold=50,
                 params={"a": 1, "b": 2}, registry=reg)
    # param_func wins over the (a, b) tuple that param_sliders would have built
    assert seen["param"] == 7


def test_no_param_func_falls_back_to_slider_values():
    reg = DitherRegistry()
    seen = {}

    @reg.register("PF3", "Error Diffusion", dims=2, param_sliders=("k",))
    def pf3(image_array, parameter, luminance_threshold_value):
        seen["param"] = parameter
        return image_array

    img = np.zeros((3, 3), dtype=np.float32)
    apply_dither(img, style="PF3", scale=1, luminance_threshold=50,
                 params={"k": 42}, registry=reg)
    assert seen["param"] == 42  # single slider -> scalar, unchanged


def test_threshold_field_kwarg_is_accepted_and_inert():
    # Frozen-contract signature must accept threshold_field even though Phase 1 ignores it.
    reg = DitherRegistry()

    @reg.register("PF4", "Error Diffusion", dims=2)
    def pf4(image_array, parameter, luminance_threshold_value):
        return image_array

    img = np.full((3, 3), 10.0, dtype=np.float32)
    field = np.zeros((3, 3), dtype=np.float32)
    out = apply_dither(img, style="PF4", scale=1, luminance_threshold=50,
                       params={}, registry=reg, threshold_field=field)
    np.testing.assert_allclose(out, img)  # inert in Phase 1
```

- [ ] **Step 2: Run — verify it fails**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_param_func.py -v`
  Expected: FAIL. First two tests fail with `assert 30 == 60` / `assert (1, 2) == 7` (param_func ignored); `test_threshold_field_kwarg_is_accepted_and_inert` fails with `TypeError: apply_dither() got an unexpected keyword argument 'threshold_field'`.

- [ ] **Step 3: Implement — replace `ditherzam/dithering/pipeline.py` in full**

```python
from __future__ import annotations
import numpy as np
from ..imaging import nearest_downscale, nearest_upscale_to


def _luminance_to_255(luminance_threshold: float) -> float:
    return float(luminance_threshold / 100.0 * 255.0)


def _build_param(entry, params: dict):
    # Spec §8.2 step 3: param_func takes precedence over param_sliders extraction.
    if entry.param_func is not None:
        return entry.param_func(params)
    vals = [params[name] for name in entry.param_sliders if name in params]
    if len(vals) <= 1:
        return vals[0] if vals else 0
    return tuple(vals)


def apply_dither(gray_f32, *, style, scale, luminance_threshold,
                 params, registry, preview_disabled=False,
                 threshold_field=None) -> np.ndarray:
    entry = registry.get_entry(style)
    if style == "None" or entry is None or preview_disabled:
        return gray_f32

    tval = _luminance_to_255(luminance_threshold)
    factor = max(1, int(scale))
    h, w = gray_f32.shape[:2]

    small = nearest_downscale(gray_f32, factor)
    param = _build_param(entry, params)
    out = entry.func(small, param, tval)

    # threshold_field is accepted for the frozen contract; Phase 8 (temporal) wires it.
    return nearest_upscale_to(out, (w, h))
```

- [ ] **Step 4: Run — verify pass**

  Run: `.venv/Scripts/python.exe -m pytest tests/test_param_func.py -v`
  Expected: `4 passed`.

- [ ] **Step 5: Re-run the existing pipeline suite to prove no regression** (the `_build_param` change must not break plan 01 Task 7)

  Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py tests/test_param_func.py -v`
  Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add ditherzam/dithering/pipeline.py tests/test_param_func.py
git commit -m "feat(dither): param_func dispatch + threshold_field kwarg to match frozen contract"
```

---

## Subsystem Definition of Done  (checklist)

- [ ] **Env:** a Python **3.12** interpreter is available (scratchpad `dbwork/py312/python.exe` and/or `.venv`); no step depends on a bare `python`/`py` on PATH (there is none on this box); `.venv` built from 3.12 satisfies the `>=3.12,<3.13` pin.
- [ ] **Install:** `.venv/Scripts/python.exe -m pip install -e ".[dev]"` succeeds; `import ditherzam` prints `0.1.0`.
- [ ] **Plan 01 Task 1–7 verified:** config loader (3 tests), imaging (3), adjustments (7), registry (3), three kernels (4), pipeline (4) — all green.
- [ ] **Gap 1 (NUMBA fallback):** `tests/test_numba_fallback.py` green — compiled kernels equal the pure-Python fallback for Floyd-Steinberg and Bayer-4x4.
- [ ] **Gap 2 (param_func):** `tests/test_param_func.py` green — `param_func` takes precedence over `param_sliders`, slider fallback intact, and `apply_dither` accepts the frozen `threshold_field=None` kwarg.
- [ ] **Full suite:** `.venv/Scripts/python.exe -m pytest -q` → `0 failed` (with `NUMBA_DISABLE_JIT=1` from conftest).
- [ ] **Frozen contracts honored verbatim:** `DitherEntry`, `DitherRegistry.register/get_entry/list_dithers/by_category`, `apply_dither(..., preview_disabled=False, threshold_field=None)`, and the five adjustment signatures.
- [ ] **Clean-room / Qt-free:** no PySide6 import anywhere under `ditherzam/` in this phase; no licensing/telemetry/network/ffmpeg-binary code introduced.
- [ ] Every green step above was committed with a conventional message.

## Self-Review

- [ ] **Spec coverage** — §7.1 tonal formulas → Task 1.4; §8.2 dither pipeline (downscale→kernel→upscale, `tval`, `factor`, param extraction incl. **param_func**) → Tasks 1.7 + 1.9; §9.1 registry mechanics (`register`, `dims`, `param_sliders`, `param_func`) → Tasks 1.5 + 1.9; §9.x kernels sample (FS/Atkinson/Bayer) → Task 1.6. Video/color/effects/UI belong to later subsystems and are intentionally out of scope here.
- [ ] **Placeholder scan** — every code step (Tasks 1.8, 1.9) contains complete, runnable code; no `TODO`/`...`/"implement here"; verification steps for Tasks 1.1–1.7 reference code already present in plan 01.
- [ ] **Type/name consistency** — dataclass field order `(name, category, dims, param_sliders, func, param_func)`; kernel signature `(image_array, parameter, luminance_threshold_value) -> float32[H,W] 0..255`; `apply_dither` keyword-only args match the frozen contract exactly (including `threshold_field=None` added in 1.9).
- [ ] **JIT correctness** — error-diffusion kernels are `@njit(cache=True)` (serial; order-dependent), only the ordered/Bayer kernel is `parallel=True`; Task 1.8 guards that compiled == interpreted.
- [ ] **No regression** — Task 1.9 Step 5 re-runs the plan-01 pipeline tests after modifying `_build_param`.
