---
type: constraint
phase: "-"
status: n/a
date: 2026-07-04
---

The core is headless and Qt-free. Only `ditherzam/ui/`, `ditherzam/app.py`, and
`ditherzam/video/workers.py` may import PySide6. Anything testable without a GUI
(conversion, theme parsing, hotkey tables, viewport math, settings mapping, ffmpeg
command building) must be a pure function with its own unit test. Render order is
fixed: contrast → midtones → highlights → blur → dither(downscale→kernel→upscale)
→ color(palette map) → saturation → effects stack → invert (last).
