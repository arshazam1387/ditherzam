---
type: gotcha
phase: 8
status: done
date: 2026-07-11
---

Checkers Small/Medium/Large disappeared at the default luminance threshold.

**Why:** alternating cell thresholds were `thr` and `255-thr`; at the default
127.5 midpoint both values are identical, so every cell uses the same threshold.

**Fix/Apply:** alternate at `thr-64` and `thr+64`, clamped to 0..255. Midtones
now form a visible checkerboard while black/white endpoints remain stable. Test
all checker sizes on a flat 127.5 fixture so both 0 and 255 cells are required.
