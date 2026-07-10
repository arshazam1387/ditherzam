---
type: progress
phase: 8
status: done
date: 2026-07-09
---

RGB Floyd-Steinberg palette diffusion now uses a sequential scalar Numba kernel
in `color/engine.py`, with a frozen Python differential oracle covering thin and
random images, out-of-range float32 values, non-contiguous inputs, palette sizes
1/2/4/16/64, built-ins, boundaries, and strict first-minimum ties. Raw float32
output was byte-identical in the 480p smoke; the warmed K=4 path measured 6.1 ms
versus 4.78 s for the prior Python implementation (~779x). Keep this kernel
sequential and preserve its scan, neighbor, float32, and tie order.

Related: [[026-high-res-performance-program-approved]].
