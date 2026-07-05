# ditherzam memory — INDEX

> One line per entry. Load this first on any session (`/zam-memory`). Entry bodies
> live in the `NNN-slug.md` files, never here.

- [constraint|phase -|n/a] Clean-room — no Studio AAA code/strings/binaries (001-clean-room.md)
- [constraint|phase -|n/a] Core is Qt-free; only ui/app/video-workers import PySide6 (002-qt-free-core.md)
- [constraint|phase -|n/a] Python 3.12 only; tests use NUMBA_DISABLE_JIT=1 (003-python-and-tests.md)
- [reference|phase -|n/a] Key files: spec, plans, roadmap, handoff, repo URL (004-key-references.md)
- [progress|phase 6|in-progress] MASTER STATUS + resume: Phases 1-6 built & green (258 passed), Phase 7+8 remain; .venv at repo root; paused on Opus limit til 6:10am (005-project-state.md)
- [reference|phase -|n/a] Fallback when stuck: may consult decompiled Dither Boy for inspiration, then implement our own (006-fallback-inspiration.md)
- [progress|phase 3|done] Phase 3 color engine done: saturation, Palette+YAML, 5 builtins, median-cut, ColorEngine 4 modes, shuffle (007-phase3-color-engine-done.md)
- [progress|phase 2|done] Phase 2 kernel library done: 66 kernels across 5 modules, golden fixtures, 154 tests green (008-phase2-kernel-library-done.md)
- [progress|phase 4|done] Phase 4 effects+render done: 5 effects, EffectStack, RenderPipeline STAGE_ORDER, 185 tests green (009-phase4-effects-render-done.md)
- [progress|phase 5|done] Phase 5 UI shell done: PySide6 ImageEditor + pure helpers, 227 tests green, Qt-isolation holds (010-phase5-ui-shell-done.md)
- [progress|phase 6|done] Phase 6 presets & export done: presets/export/batch Qt-free, SVG run-merge, UI menu wired, 258 tests green (011-phase6-presets-export-done.md)
- [progress|phase 7|done] Phase 7 video done: ffmpeg builders/limits/runner/assemble, per-frame dither, Qt workers + video_controller, 298 tests green (012-phase7-video-done.md)
