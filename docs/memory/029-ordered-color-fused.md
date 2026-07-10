---
type: progress
phase: 8
status: done
date: 2026-07-09
---

Ordered Bayer color mapping now uses one parallel Numba pass that applies the
float32 Bayer bias, performs strict first-minimum nearest-palette selection, and
writes final uint8 RGB directly. Differential tests cover thin/random shapes,
palette sizes 1/2/4/16/64, non-contiguous inputs, fractional palette colors, and
ties in both JIT modes. At 640x480 with Pico-8, the warmed median improved from
21.2 ms to 5.6 ms (~3.8x), while eliminating the source-sized offset, biased RGB,
and int64 index frames (~7.0 MiB at 480p; ~47.5 MiB at 1080p).

Related: [[026-high-res-performance-program-approved]],
[[028-exact-rgb-diffusion-compiled]].
