---
type: progress
phase: 4
status: done
date: 2026-07-05
---

Phase 4 effects stack + render pipeline done: `ditherzam/effects/post.py` (5 effects
Blur/Sharpen/Chromatic Aberration/JPEG Glitch/Epsilon Glow + `EFFECTS` dict),
`ditherzam/effects/stack.py` (`EffectStack` add/move/remove/apply), `ditherzam/render.py`
(`RenderSettings` dataclass + `RenderPipeline` with frozen `STAGE_ORDER`). Render order is
contrast→midtones→highlights→blur→dither→color→saturation→effects→invert (invert strictly
last); `temporal_field` forwarded to `apply_dither(threshold_field=...)` for Phase 8.
31 new tests, full suite 185 passed. Qt-free confirmed. Next: Phase 5. [[008-phase2-kernel-library-done]]
