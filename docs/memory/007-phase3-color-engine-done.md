---
type: progress
phase: 3
status: done
date: 2026-07-05
---

Phase 3 (color engine) complete — Tasks 3.1–3.9 all committed to `main`, TDD.
Delivered:
- `ditherzam/adjustments.py` → appended `apply_saturation(rgb, value)` (lum-based).
- `ditherzam/color/palette.py` → `Palette` (from_list/to_yaml/load/shuffle),
  `builtin_palettes()`, `extract_palette` (median-cut), `source_palette`
  (completeness∈[0,1]→k∈[2,256]).
- `ditherzam/color/engine.py` → `ColorEngine` (off/nearest/ordered-Bayer/diffused-FS)
  + `nearest_indices` helper.
- `ditherzam/color/builtin/*.yaml` → grayscale(4), gameboy(4), cga(16), pico8(16), sepia(4).
Tests: `tests/test_saturation.py` (4), `tests/test_palette.py` (19),
`tests/test_color_engine.py` (12) = 35, all green. Core stays Qt-free.
Next: Phase 4 (RenderPipeline) consumes ColorEngine.map + apply_saturation.
