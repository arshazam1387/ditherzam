---
type: gotcha
phase: 8
status: done
date: 2026-07-08
---

**In N-level (depth≥3) dithering the `luminance_threshold` slider changes meaning:
it becomes a global tone BIAS, not a binary threshold.**

The 2-tone kernels compare each pixel to `tval` (the slider mapped to 0..255). The
N-level error-diffusion path can't use a single threshold, so it does
`old = pixel + (127.5 - tval)` then `quantize_to_levels(old, levels)` — i.e. the
slider shifts overall brightness of the banding. Slider=50 (tval=127.5) ⇒ bias 0
⇒ unbiased default. The N-level **ordered** path applies no `tval` bias at all,
consistent with the original ordered kernels having always ignored the slider.

**Why it's safe:** the `levels<=2` branch is the verbatim original code, so every
existing output (any slider value) is byte-identical — the new semantics only apply
at depth≥3. See [[017-color-depth-ramp-shipped]].

**How to apply:** don't "fix" the N-level branch to re-threshold at 128 — that would
break the tone-bias behavior. If a UI later needs a separate depth-mode threshold
control, add it explicitly rather than overloading `luminance_threshold`.
