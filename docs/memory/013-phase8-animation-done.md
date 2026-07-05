---
type: progress
phase: 8
status: done
date: 2026-07-05
---

Phase 8 (animation & temporal) done — the final subsystem. **330 tests green.**

- 9 deterministic seeded noise `PATTERNS` in `ditherzam/animation/temporal.py`:
  static, scanline-drift, interlace, rolling-bar, vhs-jitter, blue-noise,
  bayer-cycle, plasma, film-grain. `temporal_noise(frame, shape, pattern, amplitude,
  seed=0) -> float32 HxW ~[-amp,amp]`; determinism + amplitude bounds tested per pattern.
- Threshold-field hook is backward-compatible: `apply_dither(..., threshold_field=None)`
  and `RenderPipeline.render(..., temporal_field=None)` are byte-identical to the
  no-field path (tested via `assert_array_equal`). Field is subtracted from the
  downscaled grid before the kernel.
- `ease(t, kind)`, `Keyframe(frame, field, value, kind="linear")`, `Timeline(length)`
  with `.add/.value_at/.settings_at` in `timeline.py`.
- `render_animation(pipeline, base, base_settings, timeline, temporal_pattern,
  temporal_amplitude, seed=0)` generator + `export_animation(...)` (reuses Phase-7
  `ffmpeg.assemble_video`) live in `ditherzam/animation/__init__.py`.
- UI: `ditherzam/ui/timeline_panel.py` (`TimelinePanel`/`AnimationController`), wired
  into `ImageEditor` following the Phase 6/7 `_wire_*` menu pattern.
- `animation/` core is Qt-free (verified). See [[012-phase7-video-done]], [[005-project-state]].
