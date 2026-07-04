# Phase 5 — UI Shell — Implementation Plan

> REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Read
> [`00-ROADMAP.md`](00-ROADMAP.md); complete Phases 1–4.

**Goal:** A runnable PySide6 app: load an image, choose a dither (grouped combo),
tune adjustments with live debounced preview, pan/zoom/inertia in the viewport,
switch themes. UI is a thin wrapper over the headless core from Phases 1–4.

**Architecture:** `ImageEditor(QMainWindow)` owns a `RenderPipeline` and a
`RenderSettings`. Slider signals mutate settings and schedule a debounced
re-render on a `QThreadPool` worker; the worker emits a `QImage` back to the main
thread. This is the **only** phase (plus `app.py`, `video/workers.py`) allowed to
import Qt.

**Tech Stack:** PySide6 · NumPy · pytest-qt (optional) · pytest.

## Global Constraints
See roadmap. Widgets never do image math — they call core functions.
UI logic that *can* be tested headlessly (settings mapping, theme loading, hotkey
tables, numpy↔QImage) MUST be extracted into pure functions and unit-tested.

---

## File structure (this phase)

- Create `ditherzam/ui/__init__.py`
- Create `ditherzam/ui/widgets.py` — GlowSlider, InvisibleSpinBox, ClickableLabel, NoScrollComboBox, Resettable*
- Create `ditherzam/ui/delegates.py` — DitherStyleDelegate (category headers)
- Create `ditherzam/ui/theme.py` — find_themes, load_theme_qss, ThemeData
- Create `ditherzam/ui/viewport.py` — CustomGraphicsView (zoom/pan/inertia)
- Create `ditherzam/ui/controls.py` — build dither/adjustments/color/effects panels
- Create `ditherzam/ui/main_window.py` — ImageEditor
- Create `ditherzam/ui/convert.py` — numpy_to_qimage, qimage_to_numpy (pure, tested)
- Create `ditherzam/app.py` — main()
- Create `themes/default/theme.yaml` (QSS from spec Appendix A)
- Tests: `tests/test_convert.py`, `tests/test_theme.py`, `tests/test_hotkeys.py`, `tests/test_settings_map.py`

---

### Task 1: numpy ↔ QImage conversion (pure, tested)

**Files:** Create `ditherzam/ui/convert.py`; Test `tests/test_convert.py`
**Interfaces:** `numpy_to_qimage(rgb_u8) -> QImage`, `qimage_to_numpy(qimg) -> rgb_u8`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.ui.convert import numpy_to_qimage, qimage_to_numpy

def test_roundtrip():
    a = np.random.RandomState(0).randint(0,256,(5,7,3),np.uint8)
    q = numpy_to_qimage(a)
    assert q.width() == 7 and q.height() == 5
    b = qimage_to_numpy(q)
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement** using `QImage(data, w, h, 3*w, QImage.Format_RGB888)` with a
  contiguous copy; convert back via `constBits`/`memoryview`, reshape `(h,w,3)`.
- [ ] **Step 4: Pass. Step 5: Commit** `feat(ui): numpy<->QImage conversion`

---

### Task 2: Theme discovery & QSS loading (pure parts tested)

**Files:** Create `ditherzam/ui/theme.py`, `themes/default/theme.yaml`;
Test `tests/test_theme.py`
**Interfaces:**
```python
@dataclass
class ThemeData: name: str; stylesheet: str; glow_color: str; idle_gif: str | None
def find_themes(root) -> list[str]
def load_theme(root, name) -> ThemeData
```

- [ ] **Step 1: Failing test** — `find_themes` returns `["default"]`; `load_theme`
  returns QSS containing `#1f1f1f` and glow `#5e89ed`.
- [ ] **Step 2: fail. Step 3: implement** (scan subdirs with `theme.yaml`; parse YAML).
  Paste the full default QSS from `DITHER_BOY_FULL_SPEC.md` Appendix A into
  `themes/default/theme.yaml`.
- [ ] **Step 4: pass. Step 5: commit** `feat(ui): theme discovery + default theme`

---

### Task 3: Hotkey table (pure, tested)

**Files:** Modify `ditherzam/ui/main_window.py` (or `hotkeys.py`); Test `tests/test_hotkeys.py`
**Interfaces:** `get_hotkeys(platform: str) -> dict[str,str]`.

- [ ] **Step 1: Failing test** — Windows maps `import_image → "Ctrl+I"`,
  `zoom_in → "Ctrl+="`; macOS maps `import_image → "Meta+I"` (Qt uses `Ctrl`==⌘, so
  provide the same logical keys; test asserts the dict keys exist and differ per platform).
- [ ] **Step 2–5:** implement dict, pass, commit `feat(ui): hotkey table`.

---

### Task 4: Custom widgets

**Files:** Create `ditherzam/ui/widgets.py`; Test (smoke) `tests/test_widgets.py` (needs `QApplication`)
**Interfaces:** `GlowSlider`, `ResettableGlowSlider(default)`, `InvisibleSpinBox(max_display)`,
`ClickableLabel(text, font_size)`, `NoScrollComboBox`.

- [ ] **Step 1: Failing smoke test** (guard with `pytest.importorskip("PySide6")` and a
  session `QApplication` fixture): construct each widget; `ResettableGlowSlider`
  double-click resets to default; `NoScrollComboBox.wheelEvent` does not change index.
- [ ] **Step 2–5:** implement widgets (subclass `QSlider`/`QSpinBox`/`QLabel`/`QComboBox`;
  glow via `QGraphicsDropShadowEffect`), pass, commit `feat(ui): custom widgets`.

---

### Task 5: Viewport (zoom/pan/inertia)

**Files:** Create `ditherzam/ui/viewport.py`; Test `tests/test_viewport_math.py`
**Interfaces:** Extract inertia math to a pure helper:
```python
def inertia_step(pos, velocity, maximum, friction, dt=0.016) -> tuple[float,float]
def next_zoom(current, direction, factor_in=1.2, factor_out=0.8,
              zmax=100.0, zmin=0.01) -> float | None   # None = clamp reached
```

- [ ] **Step 1: Failing test**

```python
from ditherzam.ui.viewport import inertia_step, next_zoom

def test_zoom_in_multiplies():
    assert abs(next_zoom(1.0, +1) - 1.2) < 1e-9

def test_zoom_in_capped():
    assert next_zoom(90.0, +1) is None       # 90*1.2 > 100

def test_zoom_out_floor():
    assert next_zoom(0.012, -1) is None       # 0.012*0.8 < 0.01

def test_inertia_decays_and_clamps():
    pos, vel = inertia_step(50.0, 1000.0, 100.0, 0.95)
    assert 0.0 <= pos <= 100.0
    assert abs(vel) < 1000.0
```

- [ ] **Step 2–3:** implement helpers + `CustomGraphicsView` using them
  (`Shift+wheel` → `next_zoom`, drag pan, release → `QTimer` inertia loop).
- [ ] **Step 4: pass. Step 5: commit** `feat(ui): viewport zoom/pan/inertia`

---

### Task 6: Settings mapping (pure, tested)

**Files:** Create `ditherzam/ui/settings_map.py`; Test `tests/test_settings_map.py`
**Interfaces:** `settings_from_controls(state: dict) -> RenderSettings` — maps a
plain dict of widget values to a `RenderSettings` (so the render path is testable
without Qt).

- [ ] Failing test: a dict with `contrast=70, style="Atkinson", scale=3` produces a
  `RenderSettings` with those fields. Implement, pass, commit `feat(ui): settings mapping`.

---

### Task 7: Main window + debounced render worker

**Files:** Create `ditherzam/ui/main_window.py`, `ditherzam/ui/controls.py`,
`ditherzam/ui/delegates.py`, `ditherzam/app.py`; Test `tests/test_app_smoke.py`
**Interfaces:** `ImageEditor(QMainWindow)`; `main() -> int`.

- [ ] **Step 1: Failing smoke test** (importorskip PySide6, offscreen platform):

```python
import os, numpy as np, pytest
pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def test_window_constructs_and_renders(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    w = ImageEditor()
    w.load_array(np.tile(np.linspace(0,255,64,np.float32),(64,1)))
    w.set_style("Floyd-Steinberg")
    img = w.render_now()                # synchronous render for test
    assert img.width() == 64 and img.height() == 64
```

- [ ] **Step 2–3:** implement:
  - `controls.py` builds the dither combo (grouped via `DitherStyleDelegate`),
    adjustment sliders (Contrast/Midtones/Highlights/Luminance Threshold/Blur +
    Saturation), color panel (palette picker + mode + lock/shuffle), effects panel
    (add/reorder list). Each control writes into `self.state` and calls
    `schedule_render()`.
  - `main_window.py`: `RenderPipeline`; `schedule_render` debounces via `QTimer`
    (`debounce_ms`), runs a `QRunnable` that calls `pipeline.render`, emits `QImage`;
    `render_now()` runs it synchronously for tests. Zoom/pan via viewport. Menus &
    shortcuts from Task 3.
  - `app.py`: `QApplication`, set style "Fusion", load config + theme, show window.
- [ ] **Step 4: pass. Step 5: commit** `feat(ui): main window, controls, debounced render, app entry`

---

## Phase 5 Self-Review
- [ ] `python -m ditherzam` (or `ditherzam`) launches a window (manual check).
- [ ] Pure helpers (convert, theme, hotkeys, viewport math, settings map) unit-tested green.
- [ ] Smoke test constructs window offscreen and renders an array.
- [ ] Only `ui/`, `app.py` import Qt.
- [ ] Live preview updates on slider change (manual check); Shift-wheel zoom + inertia work (manual).
