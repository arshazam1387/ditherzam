---
type: gotcha
phase: 8
status: done
date: 2026-07-08
---

"Most dither styles only show one/two colours" (user's real diagnosis, after the
color-shadowing red herring): **46 of 66 kernels ignore `depth`/`levels`** and only
binary-threshold, so after colorizing they collapse to 1-2 colours regardless of depth.
The ~20 that "work" (all Bayer/ordered, Floyd-Steinberg, Atkinson, Sierra, Stucki, JJN,
Burkes, Fan, Shiau-Fan, Stevenson-Arce, Cluster-Dot…) are registered `supports_levels=True`
and call the generic `_ordered(img, matrix, levels)` / error-diffusion multi-level path.
The pattern/screen/threshold kernels (checkers, crosshatch, waves, screens, modulation,
glitch, special) hardcode `255 if img>=t else 0`. Three even output a SINGLE tone
(Diagonal, Displace Contour, Wireframe Alt) → literally one colour.

**Fix (TDD, ditherzam/dithering/pipeline.py `_binary_to_levels`):** a generic black-box
promoter. When a `supports_levels=False` kernel is asked for `levels>2`, split the tonal
range into `levels-1` bands and let the kernel dither the in-band fraction:
`out = (floor(v/step) + kernel_bit(frac)) * step`, `step=255/(levels-1)`, `frac` fed to the
kernel as `frac*255`, output ≥128 → the "up" bit. Each kernel keeps its own texture but
builds a smooth multi-tone gradient a palette can span. No per-kernel edits; the 46 all fixed
at once (probe: every one 2→8 tones @depth8; end-to-end ramp render 2→6 colours @depth6).
`levels<=2` path is byte-identical to the raw kernel, so **all goldens (default depth=2) are
untouched** — 585 green JIT-off. Tests: tests/test_pipeline_levels.py (promote + levels=2
unchanged, using the stateless "Dot Screen"; Ostromukhov/Thresholder are stateful/non-repeatable
so bad byte-equality oracles).

NOT the fix that was first attempted: nearest/ordered RGB-nearest shadows saturated colours
(reverted — the user said "isn't the colours, it's the styles"). Depth still defaults to 2, so
these styles need Depth>2 to show tones (same as Bayer always did). Related:
[[017-color-depth-ramp-shipped]], [[018-nlevel-threshold-is-tone-bias]], [[020-jit-on-kernel-failures-preexisting]].
