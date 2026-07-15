# Third-Party Notices

This file records third-party code, models, and assets bundled with or staged
by ditherzam, per component license and attribution requirements.

## FFmpeg 8.1.2 essentials build (bundled in Windows downloads)

- **Distributor:** Gyan Doshi, CODEX FFMPEG Windows builds
- **Binary source:**
  `https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip`
- **Archive SHA-256:**
  `db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec`
- **Version:** `8.1.2-essentials_build-www.gyan.dev`, Windows x64 static
- **FFmpeg source revision:**
  `https://github.com/FFmpeg/FFmpeg/commit/38b88335f9`
- **License:** GPL version 3. The upstream `LICENSE` and `README.txt` are
  installed as `assets/ffmpeg/FFMPEG_LICENSE.txt` and
  `assets/ffmpeg/FFMPEG_README.txt`. The README contains the full component
  inventory and build configuration.
- **Configuration:** `--enable-gpl --enable-version3 --enable-static` with no
  `--enable-nonfree`. Run `assets/ffmpeg/ffmpeg.exe -buildconf` for the complete
  configuration.
- **Files:** `ffmpeg.exe` SHA-256
  `1326dde4c84ff1f96fe6b8916c5bed29e163e9b5dccf995f6f3db069d143ec5e`;
  `ffprobe.exe` SHA-256
  `b49ccc7c6547b141ad5a2f6ec69cc04323d7133d7704d70b331b904c63eecb07`.

The corresponding ditherzam source is available from the release tag in the
public repository. FFmpeg corresponding source is available at the exact source
revision above; the redistributed upstream README preserves its source notice
and build-component details.

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
