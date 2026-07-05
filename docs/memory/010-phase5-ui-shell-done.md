---
type: progress
phase: 5
status: done
date: 2026-07-05
---

Phase 5 UI shell done: 227 tests green (185 baseline + 42 new), offscreen suite also green.

Pure helpers TDD'd without QApplication: `ui/viewport_math.py`, `ui/settings_map.py`,
`ui/theme.py`, `ui/hotkeys.py` (`ui/convert.py` needs only QImage, no QApplication).
Qt-only code confined to `ui/{convert,widgets,delegates,viewport,controls,main_window}.py`
and `app.py` — invariant [[002-qt-free-core]] holds (grep confirmed pure helpers absent).

Main window class is `ImageEditor(QMainWindow)` in [ditherzam/ui/main_window.py]. Attach
points for later phases (export/video/animation):
- **Menu/action**: extend `_install_shortcuts()` (builds `QAction`s from
  `get_hotkeys(sys.platform)` into `self._actions`); add a `QMenuBar` via
  `self.menuBar()` and wire actions to new handler methods.
- **Panel**: `self.panel` (`ControlPanel`) sits in a `QScrollArea` inside a `QSplitter`;
  add sections by extending `ControlPanel._build_*` and reading `self.panel.state`.
- **Render**: `self.pipeline` (`RenderPipeline`) + `render_now()` (sync) / `schedule_render()`
  (debounced `QTimer` -> `_RenderWorker(QRunnable)` on `QThreadPool`). Video/animation
  should reuse `settings_from_controls(self.panel.state)` and forward a `temporal_field`.

Default theme QSS lives at repo-root [themes/default/theme.yaml] (NOT under ditherzam/;
tests use `Path("themes")` and app.py uses `parent.parent/"themes"`). glow `#5e89ed`.

Gotcha: the debounced smoke test needs real wall-clock to elapse (single-shot QTimer +
threadpool worker), so its poll loop sleeps 5ms/tick — a tight processEvents loop alone
never fires the 20ms timer.
