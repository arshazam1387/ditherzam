# ditherzam

An open-source PySide6 image/video **pixel-dither studio** — target feature
parity with the current commercial landscape (**~63 dithering algorithms**
across error-diffusion, ordered/Bayer, pattern, glitch, and special-effect
families), a **full color engine** (built-in + image-extracted + custom
palettes, editable swatches, lock+shuffle), a **stackable/reorderable effects
pipeline** (Epsilon Glow, Chromatic Aberration, JPEG Glitch, Blur/Sharpen), CMYK
halftone, tonal adjustments (brightness/contrast/saturation/midtones/highlights),
**temporal animation** (animated noise + timeline with easing), live preview,
presets, batch processing, and video (ffmpeg) with live playback.

All original, clean code — no third-party application code or binaries.

## Status
Design phase. The complete build specification lives in
**[`DITHER_BOY_FULL_SPEC.md`](DITHER_BOY_FULL_SPEC.md)** — every control, slider
range, adjustment formula, algorithm, pipeline stage, ffmpeg command, keyboard
shortcut, theme, and config key needed to build a 1:1 clone.

## Planned stack
- **UI:** PySide6 (Qt), "Fusion" style
- **Image math:** NumPy + Pillow
- **Dither kernels:** Numba `@njit(parallel=True)`
- **Video:** bundled ffmpeg / ffprobe
- **Config:** external `config.yaml` + per-folder YAML themes

## Architecture (target)
```
main.py              # window, viewport, controls, workers, export, video
dither_registry.py   # DitherRegistry + 53 @njit dither kernels
config/config.yaml   # app defaults
themes/*/theme.yaml  # QSS + glow color + idle gif
assets/              # icons, gifs, ffmpeg
```

## Note
This is a clean-room open-source project. It does not include, redistribute, or
depend on any proprietary third-party application code or binaries.
