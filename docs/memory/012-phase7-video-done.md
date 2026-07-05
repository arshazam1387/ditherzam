---
type: progress
phase: 7
status: done
date: 2026-07-05
---

Phase 7 video done: `ditherzam/video/` (ffmpeg.py builders/limits/runner/probes/assemble,
frames.py per-frame dither, workers.py Qt QRunnables) + `ui/video_controller.py` wired
into `ImageEditor` via `_wire_video()` (Video menu on menubar). 298 tests green headless
+ offscreen. ffmpeg IS installed here (winget Gyan.FFmpeg), so the guarded integration
test runs (not skipped). Command builders tested purely (arg-list asserts, no spawn);
runner injectable in unit tests. 60fps/60s cap via `check_video_limits(fps,dur,expert)` —
strictly-greater boundary, `expert=True` bypasses. Core stays Qt-free (only workers.py
imports PySide6). No ffmpeg binary committed (`git ls-files assets/ffmpeg` empty).

Gotcha: plan's `test_ffmpeg_integration.py::test_assemble_with_audio_muxes` fake_runner
used `"codec_type" in cmd` where cmd is a list whose element is `stream=codec_type` —
list membership can't substring-match, so the probe branch never fired. Fixed the test to
match on `" ".join(cmd)`; builder unchanged (cmds test requires `stream=codec_type`).

Deviation: main_window wiring adapted to real attrs (`self.pipeline`, `_collect_settings`,
added `self.expert_mode=False`) instead of the plan's assumed `render_pipeline`/
`_current_render_settings`/`file_menu`. `_show_frame` uses `numpy_to_qimage`+`set_pixmap`
(viewport has no `set_image`). Next: Phase 8.
