---
type: progress
phase: 8
status: done
date: 2026-07-09
---

High-resolution performance program **Wave 1 is fully green (667 tests passed,
JIT-off)**. Delivered: deterministic capped preview policy returning capped RGB
directly (no source-sized proxy upscaling), source-logical viewport geometry
(capped pixmap scaled into source-relative scene bounds; ordinary preview
replacement does not refit/reset zoom, source replacement may), persistent
QSettings-compatible preview preferences (Auto default, rerender-on-zoom Off,
outside presets), and the byte-exact compiled sequential RGB Floyd–Steinberg
kernel (~779x at 480p). 27 JIT-on exactness tests also pass; no new JIT-on
regression. Preference/policy normalization consolidated to one source of truth.

**Next:** Wave 2 — Task 2.1 visible View-menu preview-resolution controls +
`Ctrl+Enter` Full Quality Preview, Task 2.2 immutable prioritized render requests,
Task 2.3 unified capped lifecycle. Task 2.5 (fused ordered palette mapping) is
already done [[029-ordered-color-fused]]; 2.6 fused ramp and 2.7 content-keyed
color context remain.

Related: [[026-high-res-performance-program-approved]],
[[027-preview-preferences-persistence]], [[028-exact-rgb-diffusion-compiled]],
[[020-jit-on-kernel-failures-preexisting]].
