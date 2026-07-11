---
type: progress
phase: 8
status: done
date: 2026-07-10
---

Eight visually distinct binary kernels were implemented on
`feat/generative-dithers` in `ditherzam/dithering/kernels/generative.py` and sorted
into existing categories: Hilbert (Riemersma) and Spiral Path (Error Diffusion),
Flow Hatch (Patterned), Hex Bayer and Triangular (Ordered Dither), and Spiral
Engrave, Reaction-Diffusion, and Quasicrystal (Special Effects). They use the
existing universal parameter slider and gain palette/depth support through the
binary-to-levels promoter. Deterministic sequential/per-pixel math gives exact
JIT-off/JIT-on golden parity: the focused generative plus catalog suite is 80/80
green in both modes; the complete suite is 934/934 green JIT-off. Design:
`docs/superpowers/specs/2026-07-10-new-dither-styles-design.md`.
