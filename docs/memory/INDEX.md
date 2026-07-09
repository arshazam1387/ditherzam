# ditherzam memory — INDEX

> One line per entry. Load this first on any session (`/zam-memory`). Entry bodies
> live in the `NNN-slug.md` files, never here.

- [constraint|phase -|n/a] Clean-room — no Studio AAA code/strings/binaries (001-clean-room.md)
- [constraint|phase -|n/a] Core is Qt-free; only ui/app/video-workers import PySide6 (002-qt-free-core.md)
- [constraint|phase -|n/a] Python 3.12 only; tests use NUMBA_DISABLE_JIT=1 (003-python-and-tests.md)
- [reference|phase -|n/a] Key files: spec, plans, roadmap, handoff, repo URL (004-key-references.md)
- [progress|phase 8|done] MASTER STATUS: ALL 8 phases built + 2 live-app fixes, green (333 passed), pushed; .venv at repo root; next=optimize (005-project-state.md)
- [reference|phase -|n/a] Fallback when stuck: may consult decompiled Dither Boy for inspiration, then implement our own (006-fallback-inspiration.md)
- [progress|phase 3|done] Phase 3 color engine done: saturation, Palette+YAML, 5 builtins, median-cut, ColorEngine 4 modes, shuffle (007-phase3-color-engine-done.md)
- [progress|phase 2|done] Phase 2 kernel library done: 66 kernels across 5 modules, golden fixtures, 154 tests green (008-phase2-kernel-library-done.md)
- [progress|phase 4|done] Phase 4 effects+render done: 5 effects, EffectStack, RenderPipeline STAGE_ORDER, 185 tests green (009-phase4-effects-render-done.md)
- [progress|phase 5|done] Phase 5 UI shell done: PySide6 ImageEditor + pure helpers, 227 tests green, Qt-isolation holds (010-phase5-ui-shell-done.md)
- [progress|phase 6|done] Phase 6 presets & export done: presets/export/batch Qt-free, SVG run-merge, UI menu wired, 258 tests green (011-phase6-presets-export-done.md)
- [progress|phase 7|done] Phase 7 video done: ffmpeg builders/limits/runner/assemble, per-frame dither, Qt workers + video_controller, 298 tests green (012-phase7-video-done.md)
- [progress|phase 8|done] Phase 8 animation done: 9 temporal patterns, threshold-field backward-compat, Timeline+ease, render_animation, 330 tests green (013-phase8-animation-done.md)
- [gotcha|phase 5|done] Live-app bugs fixed: blur-by-default (neutral is 0 not 50) + Palette/Effects not wired to pipeline; 333 tests (014-live-app-bugs-fixed.md)
- [progress|phase 8|done] Optimization pass MERGED+pushed: 5 wins (color njit, coalescing, staged cache, preview proxy, JIT warmup); 374 green both JIT modes (015-optimization-progress.md)
- [gotcha|phase 5|done] Slider number displays weren't wired to sliders (frozen); fixed HEAD c95ad00, 375 green — display widgets need explicit signal connect (016-slider-number-display-fix.md)
- [progress|phase 8|done] Depth-ramp color system (Dither Boy 6.0 style) shipped on branch feat/color-depth-ramp: N-level dither + palette tone ramp, 6 mappings, 511 green both JIT (017-color-depth-ramp-shipped.md)
- [gotcha|phase 8|done] N-level (depth≥3) dither: luminance_threshold slider becomes a tone BIAS not a binary threshold; levels<=2 path is byte-identical (018-nlevel-threshold-is-tone-bias.md)
- [progress|phase 8|done] Sub-project B (palette editing UX) shipped+MERGED to main (e406a7d): PaletteStore fork, SwatchStrip, From-Image, settings; 545 green JIT-off; next=C (019-color-palette-editing-b-shipped.md)
- [gotcha|phase 8|in-progress] 7 kernel tests fail under JIT-ON (special.py float array index); PRE-EXISTING on main, not from color work; JIT-off is green (020-jit-on-kernel-failures-preexisting.md)
- [progress|phase 8|done] Sub-project C (palette library) shipped+MERGED to main (007f84d): Palette.category, PaletteStore.list_by_category, PalettePicker tree (replaces palette_combo), hover/scroll preview, drag-reorder; 578 green JIT-off; import/share now the only deferred color item (021-color-palette-library-c-shipped.md)
- [gotcha|phase 8|done] "App gets stuck the more you use it" ROOT-CAUSED+FIXED: (1) unguarded _RenderWorker.run() exception wedged coalescer _busy=True forever (8240941, failed signal); (2) the throw itself = TOCTOU race reading color_engine/effect_stack multiple times while _sync_pipeline reassigns them → fix = snapshot each once in render()/render_cached(); 584 green JIT-off (022-render-worker-wedge-fix.md)
- [gotcha|phase 8|done] "Most styles only show 1-2 colours": 46/66 kernels ignore depth (binary-only, supports_levels=False) so palette collapses; fix = generic _binary_to_levels promoter in pipeline.py (dither in-band fraction) — all 46 fixed, levels=2 byte-identical, 585 green JIT-off (023-binary-kernels-ignore-depth-fix.md)
