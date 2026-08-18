# ditherzam

ditherzam is an offline-first desktop image editor for creating, combining, and
exporting dither artwork. It is built with Python 3.12, PySide6, NumPy, Numba,
Pillow, and exact Cython acceleration for selected layer and masking operations.

The project is an original clean-room implementation. It does not include or
depend on proprietary third-party application code, licensing services, or
telemetry.

## Current status

The current package version is **0.3.0**. The application is implemented and
usable from source or as an unsigned Windows x64 release. GitHub `main`
contains the tested native integration and its exact Python fallbacks.

The 2026-08-16 promotion gate passed 2,519 tests with 310 asset- or
environment-gated skips. Native-enabled, forced-fallback, JIT-enabled, and
application-startup checks also passed.

## Highlights

- 77 registered dither styles across error diffusion, ordered/Bayer, pattern,
  glitch, modulation, and generative families.
- Editable palettes, image-derived colors, depth ramps, multiple color mappings,
  swatch locking, reordering, and shuffle.
- Tonal controls and a reorderable effects pipeline, including Epsilon Glow,
  Chromatic Aberration, JPEG Glitch, Blur, and Sharpen.
- Same-canvas spatial layers with independent source pixels, transforms, opacity,
  five blend modes, Looks, thumbnails, and exact flattened export.
- Editable post-Look raster masks with Reveal/Hide painting, four brush tips,
  gradients, patterns, luminance masks, mask combinations, inspection views,
  import/export, and temporary selections including Color Range.
- Look Composer transitions, temporal animation preview, exact animation export,
  batch processing, and ffmpeg-based video import/export. Video has post-export
  playback, not a live editing preview.
- Bounded diagnostics and semantic action logs for startup, rendering, layers,
  masks, media, and export operations.

See [Masking in ditherzam](docs/MASKING_GUIDE.md) for the complete masking
workflow and [CHANGELOG.md](CHANGELOG.md) for release history.

## Native acceleration

The Cython backend preserves the Python API and exact pixels while accelerating:

- RGBA layer compositing for Normal, Multiply, Screen, Overlay, and Difference;
- alpha-aware Look Composer transitions;
- document-selection mapping into transformed layer sources; and
- Round mask-brush stamping.

Square, Diamond, and Texture brush tips deliberately use the exact Python path.
Set `DITHERZAM_DISABLE_NATIVE=1` before startup to test all public fallbacks.

## Run from source on Windows

Install 64-bit Python 3.12 and the Microsoft C++ build tools, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ditherzam.app
```

Video features require `ffmpeg` and `ffprobe` either under `assets/ffmpeg/` or
on `PATH`. Smart Mask remains unavailable unless a locally staged, licensed,
hash-verified model bundle is present; the application never downloads one.

For the pinned native toolchain, PyInstaller release process, frozen smoke test,
and Smart Mask release gate, see [BUILDING.md](BUILDING.md).

## Test

The normal coverage run disables Numba JIT and uses Qt's offscreen platform:

```powershell
$env:NUMBA_DISABLE_JIT = "1"
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q
```

Verify the compiled native paths separately:

```powershell
.\.venv\Scripts\python.exe -m ditherzam.app --native-smoke
```

## Repository layout

```text
ditherzam/          application and Qt-free core packages
ditherzam/ui/       PySide6 interface and controllers
ditherzam/_native/  private Cython extensions with Python fallbacks
tests/              core, UI, exactness, lifecycle, and integration tests
benchmarks/         reproducible performance harnesses and dated evidence
packaging/          Windows PyInstaller specifications and pinned build inputs
docs/               user guides, system map, plans, and project memory
```

The fixed render order, Qt-free-core boundary, clean-room requirements, and exact
export contracts are part of the tested architecture, not aspirational design.
