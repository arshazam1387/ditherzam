---
type: progress
phase: 8
status: done
date: 2026-07-05
---

**ALL 8 PHASES BUILT, green, committed, and pushed to origin/main.** The full
ditherzam app exists and works end-to-end.

- Full suite: **333 passed** (`QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 pytest -q`).
- Ran the real GUI; fixed two live-only bugs (blur-by-default; Palette/Effects not
  wired) — see [[014-live-app-bugs-fixed]]. App launches, previews sharp, dithers,
  color+effects apply, exports the shown result.
- 66 dither kernels; color engine (5 palettes, 4 modes); 5 stackable effects;
  `RenderPipeline` with frozen stage order; PySide6 `ImageEditor` UI; presets +
  PNG/JPG/SVG/batch export; video (ffmpeg builders, per-frame dither, mux, Qt
  workers); animation (9 temporal patterns, keyframe `Timeline`, `render_animation`,
  MP4 export).
- Verified live: still + animated demos in `demo_output/` (gitignored) — contact
  sheet + `temporal_anim.gif` render through the real pipeline.
- Qt-isolation holds: only `ui/`, `app.py`, `video/workers.py` import PySide6.
- ffmpeg IS installed on this box, so the guarded video integration test runs (not
  skipped).

**Build env:** full CPython 3.12.13 venv at `.venv/` (gitignored) with
numpy/numba/pillow/PySide6/pytest/pyyaml. Use `.venv/Scripts/python.exe`; Qt tests
need `QT_QPA_PLATFORM=offscreen`; all tests `NUMBA_DISABLE_JIT=1`.

**Optimization pass DONE + merged to main + pushed** (HEAD c95ad00): 5 perf wins
(see [[015-optimization-progress]]) + a slider-number display fix
([[016-slider-number-display-fix]]). **375 green**, JIT-on AND JIT-off. App verified
running live (color/effects/drag/proxy→full all confirmed). Desktop launcher exists
at `Desktop\ditherzam.lnk` (venv pythonw -m ditherzam.app).

**Next:** possible feature — masking + layering system (rough est: ~1wk import/auto
masks, ~2-3wk with brush painting; main cost = UI state refactor from one global
`state`/pipeline to per-layer). Still open: packaging/dist,
real-photo QA, deferred spec items (CMYK halftone §17.3, Photoshop/clipboard export
§11.4–11.6). See per-phase entries
[[007-phase3-color-engine-done]]..[[013-phase8-animation-done]], [[014-live-app-bugs-fixed]],
[[002-qt-free-core]], [[003-python-and-tests]].
