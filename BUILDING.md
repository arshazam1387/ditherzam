# Building ditherzam on Windows

The supported release environment is 64-bit CPython 3.12.13 on Windows. The
native extensions are built from Cython sources with NumPy 2.4.6 headers and
MSVC OpenMP support. Fast-math flags are intentionally not used.

The native sources are part of `main`. A source build requires the Microsoft
C++ build tools with the Desktop development with C++ workload.

## Editable development build

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ditherzam.app --native-smoke
.\.venv\Scripts\python.exe -m ditherzam.app
```

The smoke command must report `native_available: true` and execute compositor,
transition, selection, and Round-brush paths. The public wrappers retain exact
Python fallbacks. Square, Diamond, and Texture brushes intentionally stay on the
Python implementation because the native brush contract is Round-only.

## Standard Windows release

Create a Python 3.12 virtual environment, then run:

```powershell
.\tools\build_windows_release.ps1 -PythonPath .\.venv\Scripts\python.exe
```

This installs the pinned build requirements, performs an editable native build,
runs the native smoke test, and creates the ordinary no-model PyInstaller
directory at `dist\ditherzam`. The standard build does not contain ONNX Runtime
or model weights; application startup therefore leaves Smart Mask unavailable,
while every non-model feature remains usable.

Run the frozen native smoke with:

```powershell
.\dist\ditherzam\ditherzam.exe --native-smoke
```

Exit code zero proves the frozen process imported and executed compositor,
transition, selection, and brush native paths. `_composite`, `_selection`,
`_brush`, and `_smoke` `.pyd` files must all be present beneath
`dist\ditherzam\ditherzam\_native`.

## Verification

Run the normal coverage suite with JIT disabled, then run the native integration
checks with JIT enabled:

```powershell
$env:NUMBA_DISABLE_JIT = "1"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q

Remove-Item Env:NUMBA_DISABLE_JIT -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_native_integration_contract.py `
  tests/test_threading_policy.py `
  tests/test_native_pixels_exactness.py
```

Set `DITHERZAM_DISABLE_NATIVE=1` before starting a fresh process to verify public
fallback behavior. Do not use that setting for native-only differential seams,
which are expected to reject an unavailable backend.

## Smart Mask release

An approved Smart Mask release is a distinct build:

```powershell
.\tools\build_windows_release.ps1 -SmartMask -PythonPath .\.venv\Scripts\python.exe
```

It intentionally fails closed unless `packaging\smart-mask-release.lock.json`
and every locally staged, hash-verified licensed asset exist. The build scripts
never download or invent those assets.

## Native threads and benchmarks

Compositor and transitions share an explicit process-wide OpenMP
configuration setter exposed by `ditherzam.threading_policy`; the measured
default is two threads. Each operation snapshots the setting before releasing
the GIL, so a concurrent setter affects the next operation rather than an active
loop. There is intentionally no temporary save/restore context, which would race
between overlapping callers. Selection and brush remain single-threaded. The
native budget is independent of Numba's process-global setting.

Reproduce the 1/2/4/8 scaling and digest matrix with:

```powershell
python -m benchmarks.native_thread_scaling --samples 5
```
