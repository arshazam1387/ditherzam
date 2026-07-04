# ditherzam

An open-source reimplementation project of a PySide6 image/video **pixel-dither
studio** — 53 dithering algorithms (error-diffusion, ordered/Bayer, halftone,
glitch, patterned, special-effect), tonal adjustments, live preview, presets,
batch processing, and frame-by-frame video dithering via ffmpeg.

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
