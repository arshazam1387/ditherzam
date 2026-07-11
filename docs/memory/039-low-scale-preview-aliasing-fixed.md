---
type: gotcha
phase: 8
status: done
date: 2026-07-10
---

Low-scale/fine dither previews showed rectangular blank patches and uneven-density
bands even though repeated kernel renders were complete and deterministic.

**Why:** `QGraphicsView` always used nearest-neighbour pixmap transforms. When a
fine repeating dither raster was reduced to screen size, Qt dropped rows and
columns, producing large screen-space moire/dropout regions.

**Fix/Apply:** `CustomGraphicsView` now enables `SmoothPixmapTransform` only when
the combined item/view/device transform is below 1:1, and switches back to crisp
nearest-neighbour display at 1:1 or while zoomed in. This changes screen display
only; render and export pixels remain exact.
