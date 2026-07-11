---
type: progress
phase: 8
status: done
date: 2026-07-10
---

Task 4.1 shipped (commit `f5406f1`, HEAD of `feat/high-res-perf`): animation
scrub/playback now renders async/capped/latest-wins via a new
`_AnimRenderWorker`/`RenderScheduler`-reuse in `ditherzam/ui/timeline_panel.py`
(mirrors main_window's `_RenderWorker` pattern exactly, memory 022
discipline), and video display frames (`VideoController._show_frame`, both
the imported-preview frame and `FramePlayer` playback ticks) are downscaled
via `preview_target_size` before `numpy_to_qimage`. Both controllers take an
injected Qt-free `cap_provider: () -> int` wired to `ImageEditor._policy_cap`
in `_wire_animation`/`_wire_video`; export paths (`export_animation`,
`export_video`'s dither/assemble workers) never read the cap, proven by
poisoned-callable tests. `render_preview()` gained a `temporal_field`
passthrough param — hazard was resolved by discovering `apply_dither`
already nearest-resizes any field it's given to its own internal downscale
shape, so no separate capped-shape field computation was needed. 900 green
JIT-off (baseline 877; +23, ~18 directly attributable to new tests). Next:
whatever Wave 4 task follows 4.1 (not yet identified this session).
