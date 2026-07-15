# Building ditherzam 0.1.0 for Windows

These steps reproduce the unsigned Windows x64 portable and installer builds.
Run them from a clean checkout on 64-bit Windows 10 or newer. No Smart Mask
model is downloaded or included.

## Toolchain

- CPython 3.12.13 x64 (exact)
- packages pinned in `packaging/requirements-windows-build.txt`
- PyInstaller 6.21.0
- Inno Setup 6 (ISCC compiler)
- PowerShell 5.1 or newer
- FFmpeg 8.1.2 essentials x64 static build from Gyan Doshi

Create a clean environment and install only the pinned build inputs:

```powershell
uv venv --python 3.12.13 .venv-release
uv pip install --python .venv-release\Scripts\python.exe `
  -r packaging\requirements-windows-build.txt
uv pip install --python .venv-release\Scripts\python.exe --no-deps -e .
```

Install Inno Setup 6 from its official distribution. Then run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows_release.ps1
```

The script verifies Python architecture/version, downloads and hash-checks the
fixed FFmpeg archive when absent, runs the test suite with JIT disabled, creates
the windowed PyInstaller distribution, builds the portable ZIP and per-user
installer, and writes `release/SHA256SUMS.txt`. Build/cache/output directories
are ignored by Git. To point at a nonstandard ISCC location, pass
`-Iscc C:\path\to\ISCC.exe`.

## FFmpeg provenance

- URL: `https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip`
- Version: `8.1.2-essentials_build-www.gyan.dev`
- Archive SHA-256:
  `db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec`
- FFmpeg source: `https://github.com/FFmpeg/FFmpeg/commit/38b88335f9`
- License: GPLv3
- Key configuration: `--enable-gpl --enable-version3 --enable-static`; no
  `--enable-nonfree`

The upstream GPL text and README (including the complete build configuration
and component inventory) are copied into `assets/ffmpeg` in every build. Exact
binary hashes and corresponding-source information are recorded in
`THIRD_PARTY_NOTICES.md`.

## Packaging checks

Before publishing, verify all of the following on the frozen build:

1. Launch `dist\ditherzam\ditherzam.exe` with no development environment active.
   The repeatable automated workflow is
   `dist\ditherzam\ditherzam.exe --release-smoke release\smoke`.
2. Open a PNG, choose a built-in style and palette, export PNG and SVG, and
   confirm the outputs open correctly.
3. Confirm Smart Mask reports that its model is unavailable and cannot be enabled.
4. Confirm the bundled `ffmpeg.exe` and `ffprobe.exe` are resolved and runnable.
5. Inspect `_internal\PySide6\plugins` for the platform and image-format plugins,
   including SVG support, and run at least one Numba-compiled render with JIT on.
6. Install silently and interactively, launch the installed executable, reinstall
   the same version, uninstall, and verify user-created files outside the install
   directory remain untouched.
7. Inspect archive/installer inventories, PE version fields, Git history, release
   notes, and final checksums for private or development-only material.

The two distributable artifacts are:

```text
release\ditherzam-0.1.0-windows-x64-portable.zip
release\ditherzam-0.1.0-windows-x64-setup.exe
```

They are intentionally unsigned. Do not bypass SmartScreen; publish SHA-256
checksums so users can authenticate their downloads.
