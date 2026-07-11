---
type: decision
phase: 8
status: done
date: 2026-07-09
---

Ramp-mode luminance has two implementations. **Option A (default)** is the fused
position-independent scalar Rec.601 kernel (`_ramp_gray_njit` treats 2D input as
luminance directly; `_ramp_rgb_njit` does `0.299r+0.587g+0.114b` per pixel),
committed `5eb0ee6`. **Option B** is the legacy pre-`5eb0ee6` path that computes
luminance via a BLAS matmul (`rgb @ [0.299,0.587,0.114]`), kept behind a
developer/regression setting only (`3574bf3`).

**Why:** the legacy BLAS matmul is *position-dependent* — the same (r,g,b) yields
different float32 bits depending on SIMD block vs remainder, so flat-color regions
could dither to different ramp levels by pixel position (a latent tone-seam bug,
in exports too). It is therefore not a well-defined per-pixel function, so no
compiled per-pixel kernel can be byte-identical to it. Option A removes the seam
and is the largest-allocation win; the ≤1-level change vs legacy bits is a bug fix,
not a regression.

**Apply:** default = A. Enable B by setting env var `DITHERZAM_RAMP_EXACT_BLAS`
(1/true/yes/on) at process start. The flag is a GLOBAL module-level bool in
`color/engine.py` (`RAMP_EXACT_BLAS_LUMINANCE`) read at call time, so it applies
identically to preview AND export — never make it per-surface or preview/export
diverge. No user-facing UI control. Flipping it mid-session with a warm render
cache may show stale output. Related: [[028-exact-rgb-diffusion-compiled]],
[[029-ordered-color-fused]], [[026-high-res-performance-program-approved]].
