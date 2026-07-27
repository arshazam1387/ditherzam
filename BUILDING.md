# Building ditherzam on Windows

The supported release environment is 64-bit CPython 3.12.13 on Windows. The
native extensions are built from Cython sources with NumPy 2.4.6 headers and
MSVC OpenMP support. Fast-math flags are intentionally not used.

Create a Python 3.12 virtual environment, then run:

```powershell
.\tools\build_windows_release.ps1 -PythonPath .\.venv\Scripts\python.exe
```

This installs the pinned build requirements, performs an editable native build,
runs the native foundation test, and creates the ordinary no-model PyInstaller
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

An approved Smart Mask release is a distinct build:

```powershell
.\tools\build_windows_release.ps1 -SmartMask -PythonPath .\.venv\Scripts\python.exe
```

It intentionally fails closed unless `packaging\smart-mask-release.lock.json`
and every locally staged, hash-verified licensed asset exist. The build scripts
never download or invent those assets.

## Native fallback and threads

Set `DITHERZAM_DISABLE_NATIVE=1` before process startup to force the Python
fallbacks. Compositor and transitions share an explicit process-wide OpenMP
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
