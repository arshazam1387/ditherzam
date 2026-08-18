# Third-Party Notices

This file records third-party code, models, and assets bundled with or staged
by ditherzam, per component license and attribution requirements.

## FFmpeg 8.1.2 essentials build (bundled in Windows downloads)

- **Distributor:** Gyan Doshi, CODEX FFMPEG Windows builds
- **Binary source:**
  `https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip`
- **Archive SHA-256:**
  `db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec`
- **Version:** `8.1.2-essentials_build-www.gyan.dev`, Windows x64 static
- **FFmpeg source revision:**
  `https://github.com/FFmpeg/FFmpeg/commit/38b88335f9`
- **License:** GPL version 3. The upstream `LICENSE` and `README.txt` are
  installed as `assets/ffmpeg/FFMPEG_LICENSE.txt` and
  `assets/ffmpeg/FFMPEG_README.txt`. The README contains the complete component
  inventory and build configuration.
- **Configuration:** `--enable-gpl --enable-version3 --enable-static` with no
  `--enable-nonfree`. Run `assets/ffmpeg/ffmpeg.exe -buildconf` for the complete
  configuration.
- **Files:** `ffmpeg.exe` SHA-256
  `1326dde4c84ff1f96fe6b8916c5bed29e163e9b5dccf995f6f3db069d143ec5e`;
  `ffprobe.exe` SHA-256
  `b49ccc7c6547b141ad5a2f6ec69cc04323d7133d7704d70b331b904c63eecb07`.

The corresponding ditherzam source is available from the matching release tag.
FFmpeg corresponding source is available at the exact source revision above;
the redistributed upstream README preserves its source notice and build details.

## Native build and frozen Python distribution

The Windows distribution contains Cython-generated extension modules, the NumPy
runtime needed by those modules, the Microsoft Visual C++/OpenMP runtime selected
by the supported toolchain, and is assembled with PyInstaller. Cython and
PyInstaller are build tools and their source is not incorporated into
ditherzam's source tree.

The frozen application contains the following runtime distributions. Their
installed `.dist-info` metadata and license files are retained in the bundle:

- NumPy 2.4.6 (BSD-3-Clause and bundled third-party notices)
- Numba 0.66.0 and llvmlite 0.48.0 (BSD-2-Clause and bundled notices)
- PySide6/Shiboken6 6.11.1 (LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only)
- Pillow 12.3.0 (HPND)
- PyYAML 6.0.3 (MIT)
- platformdirs 4.10.0 (MIT)

ditherzam is distributed under GPL-3.0-only; the complete project license is
installed as `LICENSE`.

## Smart Mask — U-2-Net (provisional, weights not yet shipped)

- **Project:** U-2-Net
- **Upstream repository state pinned to commit:**
  `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`
- **License:** Apache-2.0 (repository code)
- **Status:** Provisional. No U-2-Net weights are committed to this
  repository or shipped in any build yet. Pretrained-weight redistribution
  under compatible terms requires written confirmation or equivalent
  authoritative evidence before any weight is staged for release (this is a
  hard release gate). If and when an approved, converted asset ships, its
  exact manifest (source hash, conversion revision/opset, output hash,
  tensors, license/attribution, modification notice) is recorded alongside it
  under `assets/models/smart_mask/`, per
  `ditherzam/masking/model_assets.py`.
- **Modifications:** None shipped. Any future conversion to ONNX is performed
  by the developer-only `tools/stage_smart_mask_model.py` staging step and a
  separate, not-yet-built reproducible conversion step; the original
  checkpoint is never modified in place.
