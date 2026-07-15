---
type: progress
phase: 8
status: done
date: 2026-07-15
---

Smart Mask "Bake fill into dither" shipped (f19bc2f, on `feat/echo-smear-style`).
When Outside = White/Black, the new checkbox paints the fill into the base image
BEFORE the pipeline so dither/effects texture the background; feather = soft
pre-dither blend; the outer composite is skipped. Mechanism: `render_with_mask`
calls the renderer closure with one `bake(base)` callable (all 4 closures accept
`bake=None`); baked results cache under `CompositeIdentity(baked=True)`; baked
branch renders UNCACHED through `pipeline.render` because the staged-cache
"mask-proxy" key does not encode the bake (reusing it would serve stale pixels).
Preset key `smart_mask.bake_fill` (legacy loads False). Tests:
tests/test_mask_bake_fill.py (10) + mask-preset pin updated to eight fields.
Related: [[048-smart-mask-sdd-execution]], [[053-feedback-smear-rebuild]].
