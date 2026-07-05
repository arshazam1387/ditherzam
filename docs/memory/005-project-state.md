---
type: progress
phase: 6
status: in-progress
date: 2026-07-05
---

**Phases 1–6 BUILT, green, and committed on `main`.** Full app code now exists.
Suite: **258 passed** (`QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 pytest -q`).

- Phase 1 foundation, Phase 2 (66 kernels), Phase 3 color engine, Phase 4
  effects+`RenderPipeline`, Phase 5 UI shell (`ImageEditor(QMainWindow)`), Phase 6
  presets+export — all done via TDD, per-task commits.
- Verified live: `RenderPipeline` renders `uint8 HxWx3`; demo contact sheet in
  `demo_output/` (gitignored).

**Remaining: Phase 7 (video) + Phase 8 (animation).** Run them SEQUENTIALLY (both
wire into `ImageEditor` main_window — no parallel edits to `ui/main_window.py`).
Phase 7 wrote nothing yet (no `ditherzam/video/` dir). Paused because Opus hit a
session rate limit (resets 6:10am America/Los_Angeles, 2026-07-05); user chose to
WAIT for Opus rather than fall back to Sonnet. Resume by dispatching a Phase 7
Opus agent, then Phase 8.

**Build env (critical — the memory's old `dbwork/py312` embeddable had NO pip):**
a full CPython 3.12.13 (python-build-standalone) venv is at **`.venv/`** (gitignored)
with numpy/numba/pillow/PySide6/pytest/pyyaml. Use `.venv/Scripts/python.exe` for
everything; Qt tests need `QT_QPA_PLATFORM=offscreen`; all tests `NUMBA_DISABLE_JIT=1`.

Contracts honored across built code: `apply_dither(..., threshold_field=None)`,
`RenderPipeline.render(base_gray_f32, settings, temporal_field=None)`,
`ColorEngine(palette, mode).map`, `EffectStack.items`. Only `ui/`, `app.py`,
`video/workers.py` may import PySide6 (verified clean). See [[002-qt-free-core]],
[[003-python-and-tests]].
