# Phase 5 — UI Shell — Completion Task List

Ship a runnable PySide6 studio: load an image, pick a dither (grouped combo), tune
adjustments with a debounced live preview, pan/zoom/fling in the viewport, switch
themes — all as a thin wrapper over the headless Phase 1–4 core.

## Prereqs (must be green before starting)

- Phase 1 (`ditherzam.dithering.registry`, `ditherzam.dithering` shared `registry`,
  `ditherzam.imaging`, `ditherzam.adjustments`, `ditherzam.config`) — green.
- Phase 2 (all kernels registered) — green (combo needs a populated registry).
- Phase 3 (`ditherzam.color.engine.ColorEngine`, `ditherzam.color.palette`) — green.
- Phase 4 (`ditherzam.render.RenderSettings`, `ditherzam.render.RenderPipeline`,
  `ditherzam.effects.stack.EffectStack`) — green. **`ditherzam/render.py` and the
  whole core stay Qt-free.**

## Environment notes (this machine)

- Python **3.12** is required for the Numba/PySide6 wheels. If no 3.12 is on `PATH`,
  a clean interpreter is at
  `C:\Users\arsha\AppData\Local\Temp\claude\...\scratchpad\dbwork\py312\python.exe`.
  All `pytest` / `pip` commands below assume that 3.12 is the active interpreter.
- `pip install -e ".[dev]"` already pulled in `PySide6>=6.6` (declared in Phase 1's
  `pyproject.toml`).
- Qt tests run headless with `QT_QPA_PLATFORM=offscreen` (set in `tests/conftest.py`
  in Task 5.6 so it takes effect before any `QApplication` is created).

## Qt-isolation rule (verify at the end)

Only `ditherzam/ui/*`, `ditherzam/app.py`, and `ditherzam/video/workers.py` may
`import PySide6`. Every pure helper below lives under `ditherzam/ui/` **but imports
no Qt** (except `convert.py`, which needs only `QImage` — a non-widget class that
requires no `QApplication`), so it is unit-tested headlessly. `ditherzam/ui/__init__.py`
stays **empty** so importing a pure submodule never drags in a widget module.

---

## PART A — Pure, headless-testable helpers (TDD first, no `QApplication`)

### Task 5.1: numpy ↔ QImage conversion

**Files:** Create `ditherzam/ui/__init__.py` (empty), `ditherzam/ui/convert.py`;
Test `tests/test_convert.py`
**Interfaces:**
- Produces: `numpy_to_qimage(rgb_u8: np.ndarray) -> QImage`,
  `qimage_to_numpy(qimg: QImage) -> np.ndarray` (uint8 HxWx3).
- Consumes: `PySide6.QtGui.QImage` (no `QApplication` needed — `QImage` is a
  non-GUI paint device).

- [ ] **Step 1: Create empty `ditherzam/ui/__init__.py`**

```python
```

- [ ] **Step 2: Write failing test — `tests/test_convert.py`**

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")
from ditherzam.ui.convert import numpy_to_qimage, qimage_to_numpy


def test_roundtrip_rgb():
    a = np.random.RandomState(0).randint(0, 256, (5, 7, 3), np.uint8)
    q = numpy_to_qimage(a)
    assert q.width() == 7 and q.height() == 5
    b = qimage_to_numpy(q)
    np.testing.assert_array_equal(a, b)


def test_grayscale_2d_is_broadcast_to_rgb():
    g = np.array([[0, 128, 255]], np.uint8)
    q = numpy_to_qimage(g)
    assert q.width() == 3 and q.height() == 1
    b = qimage_to_numpy(q)
    assert b.shape == (1, 3, 3)
    np.testing.assert_array_equal(b[0, :, 0], b[0, :, 1])
    np.testing.assert_array_equal(b[0, :, 0], [0, 128, 255])


def test_non_contiguous_input_is_handled():
    a = np.random.RandomState(1).randint(0, 256, (4, 6, 3), np.uint8)
    view = a[::1, ::1, :]  # keep, then force a non-contiguous slice below
    sliced = np.ascontiguousarray(a)[:, ::2, :]  # width becomes 3, non-standard stride
    q = numpy_to_qimage(sliced)
    b = qimage_to_numpy(q)
    np.testing.assert_array_equal(sliced, b)
```

- [ ] **Step 3: Run — expect FAIL**

Run: `pytest tests/test_convert.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.convert'`

- [ ] **Step 4: Implement `ditherzam/ui/convert.py`**

```python
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def numpy_to_qimage(rgb_u8: np.ndarray) -> QImage:
    """Convert an HxWx3 (or HxW) uint8 array into a standalone RGB888 QImage.

    The returned image owns its pixels (``.copy()``) so it is safe after the
    numpy source is garbage-collected.
    """
    arr = np.asarray(rgb_u8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    arr = np.ascontiguousarray(arr[:, :, :3])
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return qimg.copy()


def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    """Convert any QImage into a contiguous HxWx3 uint8 array (RGB order)."""
    img = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    buf = np.frombuffer(memoryview(img.constBits()), dtype=np.uint8, count=h * bpl)
    return buf.reshape(h, bpl)[:, : w * 3].reshape(h, w, 3).copy()
```

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/test_convert.py -v`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add ditherzam/ui/__init__.py ditherzam/ui/convert.py tests/test_convert.py
git commit -m "feat(ui): numpy<->QImage conversion helpers"
```

---

### Task 5.2: Theme discovery + QSS loading, + default theme

**Files:** Create `ditherzam/ui/theme.py`, `themes/default/theme.yaml`;
Test `tests/test_theme.py`
**Interfaces:**
```python
@dataclass
class ThemeData:
    name: str; stylesheet: str; glow_color: str
    idle_gif: str | None = None; labels: dict = {}
def find_themes(root) -> list[str]
def load_theme(root, name) -> ThemeData
```
- Consumes: `PyYAML`, the `theme.yaml` files. **No Qt.**

- [ ] **Step 1: Write failing test — `tests/test_theme.py`**

```python
from pathlib import Path
import pytest
from ditherzam.ui.theme import find_themes, load_theme, ThemeData

ROOT = Path("themes")


def test_find_themes_lists_default():
    themes = find_themes(ROOT)
    assert "default" in themes


def test_find_themes_missing_root_is_empty():
    assert find_themes(Path("no/such/dir")) == []


def test_load_default_theme_qss_and_glow():
    td = load_theme(ROOT, "default")
    assert isinstance(td, ThemeData)
    assert td.name == "default"
    assert "#1f1f1f" in td.stylesheet          # panel background
    assert "QSlider::sub-page:horizontal" in td.stylesheet
    assert td.glow_color == "#5e89ed"
    assert td.labels.get("Contrast") is True


def test_load_missing_theme_raises():
    with pytest.raises(FileNotFoundError):
        load_theme(ROOT, "does_not_exist")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_theme.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.theme'`

- [ ] **Step 3: Implement `ditherzam/ui/theme.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ThemeData:
    name: str
    stylesheet: str
    glow_color: str
    idle_gif: str | None = None
    labels: dict = field(default_factory=dict)


def find_themes(root) -> list[str]:
    """Return the names of every subfolder of ``root`` that has a theme.yaml."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [
        sub.name
        for sub in sorted(root.iterdir())
        if sub.is_dir() and (sub / "theme.yaml").is_file()
    ]


def load_theme(root, name) -> ThemeData:
    """Load ``<root>/<name>/theme.yaml`` into a ThemeData.

    Resolves the idle GIF to an absolute-ish path (theme-local override, else a
    sibling ``idle.gif``, else None).
    """
    root = Path(root)
    path = root / name / "theme.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Theme not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    idle = data.get("idle_gif")
    gif_path: str | None = None
    if idle:
        gif_path = str(root / name / idle)
    elif (root / name / "idle.gif").is_file():
        gif_path = str(root / name / "idle.gif")
    return ThemeData(
        name=name,
        stylesheet=data.get("app_stylesheet", ""),
        glow_color=data.get("glow_color", "#5e89ed"),
        idle_gif=gif_path,
        labels=data.get("labels", {}) or {},
    )
```

- [ ] **Step 4: Create `themes/default/theme.yaml`** (full clean-room QSS — the exact
  values from the spec's default-theme appendix; every rule verbatim, none abbreviated)

```yaml
app_stylesheet: |
    QWidget#control_panel { background-color: #1f1f1f !important; border: none; }
    QWidget#central_widget { background-color: #1f1f1f !important; border: none; }

    QScrollBar:vertical, QScrollBar:horizontal { background: transparent; margin: 0px; }
    QScrollBar:vertical { width: 12px; }
    QScrollBar:horizontal { height: 12px; }
    QScrollBar::handle:vertical {
        background-color: #444444; border: 1px solid transparent;
        border-radius: 6px; min-height: 20px; margin: 2px;
    }
    QScrollBar::handle:horizontal {
        background-color: #444444; border: 1px solid transparent;
        border-radius: 6px; min-width: 20px; margin: 2px;
    }
    QScrollBar::handle:hover { background-color: #888888; }
    QScrollBar::handle:pressed { background-color: #555555; }
    QScrollBar::add-line, QScrollBar::sub-line {
        background: none; border: none; height: 0px; width: 0px; subcontrol-position: none;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

    QSlider { min-height: 18px; margin: 2px 0; }
    QSlider::groove:horizontal { background-color: #2e2e2e; height: 4px; border-radius: 2px; }
    QSlider::groove:horizontal:disabled { background-color: #222222; height: 4px; border-radius: 2px; }
    QSlider::sub-page:horizontal { background-color: #5e89ed; border-radius: 2px; }
    QSlider::sub-page:horizontal:disabled { background-color: #555555; border-radius: 2px; }
    QSlider::handle:horizontal {
        background: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5,
            stop:0 #a0a0a0, stop:0.2 #c0c0c0, stop:0.5 #f0f0f0, stop:1 #ffffff);
        border: 1px solid #555555; height: 14px; width: 14px; margin: -5px 0; border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5,
            stop:0 #b0b0b0, stop:0.2 #d0d0d0, stop:0.5 #f8f8f8, stop:1 #ffffff);
        border: 1px solid #777777; height: 14px; width: 14px; margin: -5px 0; border-radius: 7px;
    }
    QSlider::handle:horizontal:pressed {
        background: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5,
            stop:0 #909090, stop:0.2 #b0b0b0, stop:0.5 #d0d0d0, stop:1 #e0e0e0);
        border: 1px solid #555555; height: 14px; width: 14px; margin: -5px 0; border-radius: 7px;
    }
    QSlider::handle:horizontal:disabled {
        background: #888888; border: 1px solid #777777; height: 14px; width: 14px;
        margin: -5px 0; border-radius: 7px;
    }

    QPushButton { background-color: #292929; border: none; padding: 5px 10px; border-radius: 10px; color: white; }
    QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); color: white; }
    QPushButton:pressed { background-color: #555555; }
    QPushButton:disabled { background-color: #292929; color: #888888; }

    QSpinBox {
        background-color: transparent; border: none; color: white; font-size: 12px;
        padding: 0px; margin: 0px; min-width: 35px; max-width: 35px;
    }
    QSpinBox:disabled { color: #888888; background-color: transparent; }
    QSpinBox::up-button, QSpinBox::down-button {
        width: 0px; height: 0px; border: none; background: transparent; subcontrol-origin: none;
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow { width: 0px; height: 0px; background: transparent; }
    QHBoxLayout[objectName="slider_layout"] { spacing: 0; }

    QComboBox {
        background-color: #292929; border: none; padding: 5px; border-radius: 6px;
        color: white; width: 125px; font-size: 10.5px;
    }
    QComboBox:disabled { background-color: #444444; color: #888888; }
    QComboBox::drop-down { border: none; width: 20px; border-radius: 8px; }
    QComboBox QAbstractItemView {
        background-color: #292929; border: none; selection-background-color: #777777;
        selection-color: white; color: white; border-radius: 6px; padding: 5px;
    }
    QComboBox:on { border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; }
    QComboBox QAbstractItemView {
        border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;
        border-top-left-radius: 0px; border-top-right-radius: 0px;
    }

    QLabel { font-weight: bold; font-size: 11px; }

labels:
    Disable Preview: true
    Dither Style: true
    Dither Scale: true
    Tile Size: true
    Invert Output: true
    Contrast: true
    Midtones: true
    Highlights: true
    Blur: true
    Luminance Threshold: true

glow_color: "#5e89ed"
```

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/test_theme.py -v`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add ditherzam/ui/theme.py themes/default/theme.yaml tests/test_theme.py
git commit -m "feat(ui): theme discovery + QSS loading + default theme"
```

---

### Task 5.3: Hotkey table

**Files:** Create `ditherzam/ui/hotkeys.py`; Test `tests/test_hotkeys.py`
**Interfaces:** `get_hotkeys(platform: str) -> dict[str, str]`. **No Qt.**
`platform` is `sys.platform` (`"win32"`, `"darwin"`, `"linux"`).

- [ ] **Step 1: Write failing test — `tests/test_hotkeys.py`**

```python
from ditherzam.ui.hotkeys import get_hotkeys

ACTIONS = {
    "change_theme", "export_image", "copy_to_clipboard", "import_image",
    "restart_application", "zoom_in", "zoom_out", "zoom_reset", "show_help",
    "cycle_theme", "open_image", "save", "help",
}


def test_windows_bindings():
    hk = get_hotkeys("win32")
    assert hk["import_image"] == "Ctrl+I"
    assert hk["zoom_in"] == "Ctrl+="
    assert hk["zoom_out"] == "Ctrl+-"
    assert hk["zoom_reset"] == "Ctrl+0"
    assert hk["export_image"] == "Ctrl+Shift+S"


def test_macos_uses_meta():
    hk = get_hotkeys("darwin")
    assert hk["import_image"] == "Meta+I"
    assert hk["zoom_in"] == "Meta+="
    assert hk["change_theme"] == "Meta+Shift+T"
    assert hk["restart_application"] == "Meta+Alt+R"


def test_all_actions_present_and_differ_per_platform():
    win = get_hotkeys("win32")
    mac = get_hotkeys("darwin")
    assert set(win) == ACTIONS and set(mac) == ACTIONS
    assert win != mac


def test_linux_defaults_to_ctrl():
    assert get_hotkeys("linux")["import_image"] == "Ctrl+I"
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_hotkeys.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.hotkeys'`

- [ ] **Step 3: Implement `ditherzam/ui/hotkeys.py`**

```python
from __future__ import annotations

# Logical bindings using the primary modifier ("mod"). On Windows/Linux mod=Ctrl;
# on macOS mod=Meta (Qt maps QKeySequence "Meta" to the Command key).
_BINDINGS = {
    "change_theme": "{mod}+Shift+T",
    "export_image": "{mod}+Shift+S",
    "copy_to_clipboard": "{mod}+Shift+C",
    "import_image": "{mod}+I",
    "restart_application": "{mod}+{alt}+R",
    "zoom_in": "{mod}+=",
    "zoom_out": "{mod}+-",
    "zoom_reset": "{mod}+0",
    "show_help": "{mod}+Shift+/",
    "cycle_theme": "{mod}+T",
    "open_image": "{mod}+O",
    "save": "{mod}+S",
    "help": "{mod}+H",
}


def get_hotkeys(platform: str) -> dict[str, str]:
    is_mac = platform == "darwin"
    mod = "Meta" if is_mac else "Ctrl"
    alt = "Alt"  # Qt names the Option key "Alt" on macOS too
    return {action: tpl.format(mod=mod, alt=alt) for action, tpl in _BINDINGS.items()}
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_hotkeys.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/hotkeys.py tests/test_hotkeys.py
git commit -m "feat(ui): platform hotkey table"
```

---

### Task 5.4: Viewport / zoom / inertia math

**Files:** Create `ditherzam/ui/viewport_math.py`; Test `tests/test_viewport_math.py`
**Interfaces:** **No Qt.**
```python
def next_zoom(current, direction, factor_in=1.2, factor_out=0.8,
              zmax=100.0, zmin=0.01) -> float | None    # None = clamp reached
def inertia_step(pos, velocity, maximum, friction, dt=0.016) -> tuple[float, float]
def clamp_velocity(v, scale=0.5, maximum=2000.0) -> float
def zoom_percent(m11) -> int
```

- [ ] **Step 1: Write failing test — `tests/test_viewport_math.py`**

```python
from ditherzam.ui.viewport_math import (
    next_zoom, inertia_step, clamp_velocity, zoom_percent,
)


def test_zoom_in_multiplies_by_1_2():
    assert abs(next_zoom(1.0, +1) - 1.2) < 1e-9


def test_zoom_out_multiplies_by_0_8():
    assert abs(next_zoom(1.0, -1) - 0.8) < 1e-9


def test_zoom_in_capped_returns_none():
    assert next_zoom(90.0, +1) is None          # 90 * 1.2 = 108 > 100


def test_zoom_out_floor_returns_none():
    assert next_zoom(0.012, -1) is None          # 0.012 * 0.8 = 0.0096 < 0.01


def test_inertia_decays_and_clamps_range():
    pos, vel = inertia_step(50.0, 1000.0, 100.0, 0.95)
    assert 0.0 <= pos <= 100.0
    assert abs(vel) < 1000.0                      # friction shrank it


def test_inertia_clamps_low_bound():
    pos, _ = inertia_step(5.0, 1000.0, 100.0, 0.95)   # 5 - 16 = -11 -> 0
    assert pos == 0.0


def test_inertia_clamps_high_bound():
    pos, _ = inertia_step(95.0, -1000.0, 100.0, 0.95)  # 95 + 16 = 111 -> 100
    assert pos == 100.0


def test_clamp_velocity_scales_and_bounds():
    assert clamp_velocity(1000.0, scale=0.5, maximum=2000.0) == 500.0
    assert clamp_velocity(10000.0, scale=0.5, maximum=2000.0) == 2000.0
    assert clamp_velocity(-10000.0, scale=0.5, maximum=2000.0) == -2000.0


def test_zoom_percent_truncates():
    assert zoom_percent(1.239) == 123
    assert zoom_percent(0.01) == 1
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_viewport_math.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.viewport_math'`

- [ ] **Step 3: Implement `ditherzam/ui/viewport_math.py`**

```python
from __future__ import annotations


def next_zoom(current, direction, factor_in=1.2, factor_out=0.8,
              zmax=100.0, zmin=0.01):
    """Proposed new zoom scale, or None if it would breach the [zmin, zmax] clamp."""
    factor = factor_in if direction > 0 else factor_out
    proposed = current * factor
    if proposed > zmax or proposed < zmin:
        return None
    return proposed


def inertia_step(pos, velocity, maximum, friction, dt=0.016):
    """One fling tick: advance a scrollbar position and decay the velocity.

    Returns (clamped_position, decayed_velocity).
    """
    new_pos = pos - velocity * dt
    if new_pos < 0.0:
        new_pos = 0.0
    elif new_pos > maximum:
        new_pos = maximum
    return new_pos, velocity * friction


def clamp_velocity(v, scale=0.5, maximum=2000.0):
    """Scale a raw pan velocity and clamp its magnitude to +/- maximum."""
    v = v * scale
    if v > maximum:
        return maximum
    if v < -maximum:
        return -maximum
    return v


def zoom_percent(m11) -> int:
    """Integer zoom percent from a view transform's horizontal scale (m11)."""
    return int(m11 * 100)
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_viewport_math.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/viewport_math.py tests/test_viewport_math.py
git commit -m "feat(ui): pure viewport zoom/inertia math"
```

---

### Task 5.5: Settings mapping + grouped-combo row model

**Files:** Create `ditherzam/ui/settings_map.py`; Test `tests/test_settings_map.py`
**Interfaces:** **No Qt.**
```python
def settings_from_controls(state: dict) -> RenderSettings
def build_dither_rows(by_category: dict[str, list[str]]
                     ) -> list[tuple[bool, str, str | None]]   # (is_header, label, style|None)
```
- Consumes: `ditherzam.render.RenderSettings` (Phase 4). This keeps the whole render
  path drivable from a plain dict, no Qt.

- [ ] **Step 1: Write failing test — `tests/test_settings_map.py`**

```python
from ditherzam.render import RenderSettings
from ditherzam.ui.settings_map import settings_from_controls, build_dither_rows


def test_maps_dict_to_render_settings():
    state = {
        "contrast": 70, "midtones": 40, "highlights": 55,
        "blur": 10, "luminance_threshold": 60, "saturation": 80,
        "invert": True, "style": "Atkinson", "scale": 3,
        "preview_disabled": False, "params": {"Line Count": 4},
    }
    s = settings_from_controls(state)
    assert isinstance(s, RenderSettings)
    assert s.contrast == 70 and s.midtones == 40 and s.highlights == 55
    assert s.blur == 10 and s.luminance_threshold == 60 and s.saturation == 80
    assert s.invert is True
    assert s.style == "Atkinson" and s.scale == 3
    assert s.preview_disabled is False
    assert s.params == {"Line Count": 4}


def test_defaults_fill_missing_keys():
    s = settings_from_controls({})
    assert s.contrast == 50 and s.midtones == 50 and s.highlights == 50
    assert s.blur == 50 and s.luminance_threshold == 50 and s.saturation == 50
    assert s.style == "None" and s.scale == 5
    assert s.invert is False and s.preview_disabled is False
    assert s.params == {}


def test_params_is_copied_not_aliased():
    src = {"style": "Glitch", "params": {"Glitch Intensity": 5}}
    s = settings_from_controls(src)
    s.params["Glitch Intensity"] = 999
    assert src["params"]["Glitch Intensity"] == 5


def test_build_dither_rows_headers_and_items():
    by_cat = {
        "Default": ["None"],
        "Error Diffusion": ["Floyd-Steinberg", "Atkinson"],
    }
    rows = build_dither_rows(by_cat)
    assert rows == [
        (True, "Default", None),
        (False, "None", "None"),
        (True, "Error Diffusion", None),
        (False, "Floyd-Steinberg", "Floyd-Steinberg"),
        (False, "Atkinson", "Atkinson"),
    ]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_settings_map.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.settings_map'`

- [ ] **Step 3: Implement `ditherzam/ui/settings_map.py`**

```python
from __future__ import annotations

from ditherzam.render import RenderSettings


def settings_from_controls(state: dict) -> RenderSettings:
    """Build a RenderSettings from a plain dict of control values.

    Every field falls back to its spec default so a partial dict is valid; params
    is deep-copied so later mutation of the RenderSettings never leaks back.
    """
    return RenderSettings(
        contrast=int(state.get("contrast", 50)),
        midtones=int(state.get("midtones", 50)),
        highlights=int(state.get("highlights", 50)),
        blur=int(state.get("blur", 50)),
        luminance_threshold=int(state.get("luminance_threshold", 50)),
        invert=bool(state.get("invert", False)),
        saturation=int(state.get("saturation", 50)),
        style=state.get("style", "None"),
        scale=int(state.get("scale", 5)),
        preview_disabled=bool(state.get("preview_disabled", False)),
        params=dict(state.get("params", {}) or {}),
    )


def build_dither_rows(by_category):
    """Flatten a {category: [style, ...]} map into ordered combo rows.

    Each row is (is_header, label, style_or_None). Header rows carry None style and
    are rendered non-selectable by the delegate.
    """
    rows: list[tuple[bool, str, str | None]] = []
    for category, styles in by_category.items():
        rows.append((True, category, None))
        for style in styles:
            rows.append((False, style, style))
    return rows
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_settings_map.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/settings_map.py tests/test_settings_map.py
git commit -m "feat(ui): settings mapping + grouped-combo row model"
```

---

## PART B — Qt widgets & app (smoke-tested, offscreen)

### Task 5.6: Test fixtures + custom widgets

**Files:** Modify `tests/conftest.py`; Create `ditherzam/ui/widgets.py`;
Test `tests/test_widgets.py`
**Interfaces:** `GlowSlider`, `ResettableGlowSlider(default)`, `InvisibleSpinBox(max_display)`,
`ClickableLabel(text, font_size)`, `ResettableLabel`, `NoScrollComboBox`.

- [ ] **Step 1: Modify `tests/conftest.py`** (add offscreen env + a session QApplication
  fixture; keep the existing Numba line)

```python
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp_fixture():
    """A single offscreen QApplication for all Qt smoke tests."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
```

- [ ] **Step 2: Write failing test — `tests/test_widgets.py`**

```python
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent


def _wheel(widget, dy=120):
    ev = QWheelEvent(
        QPointF(1, 1), QPointF(1, 1), QPoint(0, 0), QPoint(0, dy),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    widget.wheelEvent(ev)


def test_resettable_slider_double_click_resets(qapp_fixture):
    from ditherzam.ui.widgets import ResettableGlowSlider
    s = ResettableGlowSlider(default=5)
    s.setRange(1, 20)
    s.setValue(17)
    assert s.value() == 17
    s.reset()
    assert s.value() == 5
    assert s.default == 5


def test_invisible_spinbox_friendly_display(qapp_fixture):
    from ditherzam.ui.widgets import InvisibleSpinBox
    sb = InvisibleSpinBox(max_display=250)
    sb.setValue(50)
    assert sb.textFromValue(50) == "125"     # 50/100 * 250
    assert sb.textFromValue(100) == "250"


def test_noscroll_combo_ignores_wheel(qapp_fixture):
    from ditherzam.ui.widgets import NoScrollComboBox
    cb = NoScrollComboBox()
    cb.addItems(["a", "b", "c"])
    cb.setCurrentIndex(1)
    _wheel(cb)
    assert cb.currentIndex() == 1            # unchanged


def test_glow_slider_ignores_wheel(qapp_fixture):
    from ditherzam.ui.widgets import GlowSlider
    s = GlowSlider()
    s.setRange(0, 100)
    s.setValue(40)
    _wheel(s)
    assert s.value() == 40


def test_clickable_label_emits(qapp_fixture):
    from PySide6.QtGui import QMouseEvent
    from ditherzam.ui.widgets import ClickableLabel
    lbl = ClickableLabel("hi", font_size=17)
    seen = []
    lbl.clicked.connect(lambda: seen.append(True))
    ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(1, 1),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    lbl.mousePressEvent(ev)
    assert seen == [True]
    assert lbl.text() == "hi"
```

- [ ] **Step 3: Run — expect FAIL**

Run: `pytest tests/test_widgets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.widgets'`

- [ ] **Step 4: Implement `ditherzam/ui/widgets.py`**

```python
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsDropShadowEffect,
    QLabel,
    QSlider,
    QSpinBox,
)


class GlowSlider(QSlider):
    """Horizontal slider with a colored glow and no accidental scroll-wheel edits."""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None,
                 glow_color="#5e89ed"):
        super().__init__(orientation, parent)
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setBlurRadius(15)
        self._glow.setOffset(0, 0)
        self._glow.setColor(QColor(glow_color))
        self.setGraphicsEffect(self._glow)

    def set_glow_color(self, color: str) -> None:
        self._glow.setColor(QColor(color))

    def wheelEvent(self, event):  # swallow — never change value on scroll
        event.accept()


class ResettableGlowSlider(GlowSlider):
    """GlowSlider that remembers a default and restores it on double-click."""

    def __init__(self, default: int = 0, orientation=Qt.Orientation.Horizontal,
                 parent=None, glow_color="#5e89ed"):
        super().__init__(orientation, parent, glow_color)
        self._default = int(default)
        self.setValue(self._default)

    @property
    def default(self) -> int:
        return self._default

    def reset(self) -> None:
        self.setValue(self._default)

    def mouseDoubleClickEvent(self, event):
        self.reset()
        event.accept()


class InvisibleSpinBox(QSpinBox):
    """0..100 spinbox with hidden arrows that displays a friendly scaled number."""

    def __init__(self, max_display: int = 100, parent=None):
        super().__init__(parent)
        self._max_display = int(max_display)
        self.setRange(0, 100)
        self.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

    def textFromValue(self, value: int) -> str:
        return str(round(value / 100.0 * self._max_display))

    def wheelEvent(self, event):  # swallow
        event.accept()


class ClickableLabel(QLabel):
    """Bold label that emits ``clicked`` on left press (used for the ↔ shuffle icon)."""

    clicked = Signal()

    def __init__(self, text: str = "", font_size: int = 11, parent=None):
        super().__init__(text, parent)
        f = self.font()
        f.setPointSize(int(font_size))
        self.setFont(f)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ResettableLabel(ClickableLabel):
    """ClickableLabel that also emits ``double_clicked`` (reset-to-default affordance)."""

    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class NoScrollComboBox(QComboBox):
    """Combo box that ignores the scroll wheel so hovering never changes selection."""

    def wheelEvent(self, event):  # swallow
        event.accept()
```

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/test_widgets.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py ditherzam/ui/widgets.py tests/test_widgets.py
git commit -m "feat(ui): custom widgets (glow slider, invisible spinbox, no-scroll combo)"
```

---

### Task 5.7: Grouped dither combo delegate

**Files:** Create `ditherzam/ui/delegates.py`; Test `tests/test_delegates.py`
**Interfaces:**
```python
HEADER_ROLE  # Qt.ItemDataRole
def populate_dither_combo(combo: QComboBox, by_category: dict) -> None
class DitherStyleDelegate(QStyledItemDelegate)   # bold, non-selectable category headers
```
- Consumes: `build_dither_rows` from Task 5.5.

- [ ] **Step 1: Write failing test — `tests/test_delegates.py`**

```python
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt


def test_populate_marks_headers_nonselectable(qapp_fixture):
    from ditherzam.ui.widgets import NoScrollComboBox
    from ditherzam.ui.delegates import populate_dither_combo, HEADER_ROLE

    combo = NoScrollComboBox()
    by_cat = {"Default": ["None"], "Error Diffusion": ["Floyd-Steinberg", "Atkinson"]}
    populate_dither_combo(combo, by_cat)

    model = combo.model()
    assert combo.count() == 5                       # 2 headers + 3 styles

    header_idx = model.index(0, 0)                  # "Default"
    assert header_idx.data(HEADER_ROLE) is True
    assert not (header_idx.flags() & Qt.ItemFlag.ItemIsSelectable)

    style_idx = model.index(1, 0)                   # "None"
    assert style_idx.data(Qt.ItemDataRole.UserRole) == "None"
    assert style_idx.flags() & Qt.ItemFlag.ItemIsSelectable


def test_delegate_constructs(qapp_fixture):
    from ditherzam.ui.delegates import DitherStyleDelegate
    assert DitherStyleDelegate() is not None
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_delegates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.delegates'`

- [ ] **Step 3: Implement `ditherzam/ui/delegates.py`**

```python
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QComboBox, QStyle, QStyledItemDelegate

from .settings_map import build_dither_rows

HEADER_ROLE = Qt.ItemDataRole.UserRole + 1


def populate_dither_combo(combo: QComboBox, by_category) -> None:
    """Fill a combo with grouped rows; category headers are bold + non-selectable."""
    model = QStandardItemModel(combo)
    for is_header, label, style in build_dither_rows(by_category):
        item = QStandardItem(label)
        if is_header:
            item.setData(True, HEADER_ROLE)
            item.setData(None, Qt.ItemDataRole.UserRole)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)          # enabled, not selectable
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        else:
            item.setData(False, HEADER_ROLE)
            item.setData(style, Qt.ItemDataRole.UserRole)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        model.appendRow(item)
    combo.setModel(model)
    combo.setItemDelegate(DitherStyleDelegate(combo))


class DitherStyleDelegate(QStyledItemDelegate):
    """Renders category header rows in bold and without the selection highlight."""

    def paint(self, painter, option, index):
        if index.data(HEADER_ROLE):
            option.font.setBold(True)
            option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return size
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_delegates.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/delegates.py tests/test_delegates.py
git commit -m "feat(ui): grouped dither combo delegate + populate"
```

---

### Task 5.8: Viewport (CustomGraphicsView with zoom/pan/inertia)

**Files:** Create `ditherzam/ui/viewport.py`; Test `tests/test_viewport.py`
**Interfaces:** `CustomGraphicsView(bg_color, friction, enable_inertia, velocity_scale,
max_velocity)`; `set_pixmap(QPixmap)`, `fit_image_to_viewport()`,
`current_zoom_percent()`, `zoom_in/out/reset_zoom`; signals `image_dropped(str)`,
`zoom_changed(int)`. Uses the pure math from Task 5.4.

- [ ] **Step 1: Write failing test — `tests/test_viewport.py`**

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")


def _pixmap(w, h):
    from PySide6.QtGui import QPixmap
    from ditherzam.ui.convert import numpy_to_qimage
    arr = np.random.RandomState(0).randint(0, 256, (h, w, 3), np.uint8)
    return QPixmap.fromImage(numpy_to_qimage(arr))


def test_set_pixmap_and_zoom(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView
    v = CustomGraphicsView()
    v.resize(200, 200)
    v.set_pixmap(_pixmap(64, 64))
    before = v.transform().m11()
    v.zoom_in()
    assert v.transform().m11() > before          # zoomed in


def test_zoom_percent_signal(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView
    v = CustomGraphicsView()
    v.resize(200, 200)
    seen = []
    v.zoom_changed.connect(seen.append)
    v.set_pixmap(_pixmap(64, 64))
    assert seen and isinstance(seen[-1], int)


def test_zoom_hard_cap_no_crash(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView
    v = CustomGraphicsView()
    v.resize(200, 200)
    v.set_pixmap(_pixmap(64, 64))
    for _ in range(200):                          # spam past the 100x cap
        v.zoom_in()
    assert v.transform().m11() <= 100.0
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_viewport.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.viewport'`

- [ ] **Step 3: Implement `ditherzam/ui/viewport.py`**

```python
from __future__ import annotations

import time

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .viewport_math import clamp_velocity, inertia_step, next_zoom, zoom_percent


class CustomGraphicsView(QGraphicsView):
    """Image viewport: Shift+wheel zoom, drag-pan with fling inertia, drag-drop import."""

    image_dropped = Signal(str)
    zoom_changed = Signal(int)

    def __init__(self, bg_color="#1f1f1f", friction=0.95, enable_inertia=True,
                 velocity_scale=0.5, max_velocity=2000.0, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self.setBackgroundBrush(QColor(bg_color))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAcceptDrops(True)

        self._friction = friction
        self._enable_inertia = enable_inertia
        self._velocity_scale = velocity_scale
        self._max_velocity = max_velocity
        self._velocity = QPointF(0.0, 0.0)
        self._last_pos: QPointF | None = None
        self._last_time: float | None = None
        self._panning = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_inertia_tick)

    # ---- image / zoom -------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pix_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pix_item.boundingRect())
        self.fit_image_to_viewport()

    def fit_image_to_viewport(self) -> None:
        if not self._pix_item.pixmap().isNull():
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.current_zoom_percent())

    def current_zoom_percent(self) -> int:
        return zoom_percent(self.transform().m11())

    def is_image_zoomed(self) -> bool:
        rect = self._pix_item.boundingRect()
        vp = self.viewport().rect()
        return (rect.width() * self.transform().m11() > vp.width()
                or rect.height() * self.transform().m22() > vp.height())

    def _apply_zoom(self, direction: int) -> None:
        if next_zoom(self.transform().m11(), direction) is None:
            return
        factor = 1.2 if direction > 0 else 0.8
        self.scale(factor, factor)
        self.zoom_changed.emit(self.current_zoom_percent())

    def zoom_in(self) -> None:
        self._apply_zoom(1)

    def zoom_out(self) -> None:
        self._apply_zoom(-1)

    def reset_zoom(self) -> None:
        self.fit_image_to_viewport()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._apply_zoom(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
        else:
            super().wheelEvent(event)

    # ---- pan + inertia ------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_pos = event.position()
            self._last_time = time.monotonic()
            self._velocity = QPointF(0.0, 0.0)
            self._timer.stop()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_pos is not None:
            delta = event.position() - self._last_pos
            self._last_pos = event.position()
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - int(delta.x()))
            vbar.setValue(vbar.value() - int(delta.y()))
            now = time.monotonic()
            dt = max(now - (self._last_time or now), 1e-3)
            self._velocity = QPointF(delta.x() / dt, delta.y() / dt)
            self._last_time = now
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._enable_inertia:
                vx = clamp_velocity(self._velocity.x(), self._velocity_scale, self._max_velocity)
                vy = clamp_velocity(self._velocity.y(), self._velocity_scale, self._max_velocity)
                self._velocity = QPointF(vx, vy)
                if abs(vx) >= 10 or abs(vy) >= 10:
                    self._timer.start(16)
        super().mouseReleaseEvent(event)

    def _on_inertia_tick(self):
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        nx, vx = inertia_step(hbar.value(), self._velocity.x(), hbar.maximum(), self._friction)
        ny, vy = inertia_step(vbar.value(), self._velocity.y(), vbar.maximum(), self._friction)
        hbar.setValue(int(nx))
        vbar.setValue(int(ny))
        self._velocity = QPointF(vx, vy)
        if abs(vx) < 10 and abs(vy) < 10:
            self._timer.stop()

    # ---- drag-and-drop import ----------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.image_dropped.emit(path)
                break
        event.acceptProposedAction()
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_viewport.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/viewport.py tests/test_viewport.py
git commit -m "feat(ui): viewport zoom/pan/inertia + drag-drop"
```

---

### Task 5.9: Control panel

**Files:** Create `ditherzam/ui/controls.py`; Test `tests/test_controls.py`
**Interfaces:** `ControlPanel(QWidget)` — builds dither + adjustments + color + effects
sections; holds `self.state: dict`; emits `changed` on any edit; exposes
`set_style(name)`, `set_registry_categories(by_category)`. Widgets never do image math.

- [ ] **Step 1: Write failing test — `tests/test_controls.py`**

```python
import pytest

pytest.importorskip("PySide6")


def test_control_panel_state_defaults(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    st = p.state
    assert st["contrast"] == 50 and st["scale"] == 5 and st["style"] == "None"
    assert st["invert"] is False and st["preview_disabled"] is False
    assert st["saturation"] == 50 and st["params"] == {}


def test_slider_edit_updates_state_and_emits(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    seen = []
    p.changed.connect(lambda: seen.append(True))
    p.contrast_slider.setValue(70)
    assert p.state["contrast"] == 70
    assert seen                       # changed fired


def test_set_style_updates_state(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    p.set_registry_categories({"Default": ["None"], "Error Diffusion": ["Atkinson"]})
    p.set_style("Atkinson")
    assert p.state["style"] == "Atkinson"


def test_invert_and_preview_toggles(qapp_fixture):
    from ditherzam.ui.controls import ControlPanel
    p = ControlPanel()
    p.invert_toggle.setChecked(True)
    p.preview_toggle.setChecked(True)
    assert p.state["invert"] is True
    assert p.state["preview_disabled"] is True
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_controls.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.controls'`

- [ ] **Step 3: Implement `ditherzam/ui/controls.py`**

```python
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .delegates import populate_dither_combo
from .widgets import (
    InvisibleSpinBox,
    NoScrollComboBox,
    ResettableGlowSlider,
)

# Adjustment sliders: (state_key, label, spin_display_max). All range 0..100, default 50.
_ADJUSTMENTS = [
    ("contrast", "Contrast", 250),
    ("midtones", "Midtones", 10),
    ("highlights", "Highlights", 50),
    ("luminance_threshold", "Luminance Threshold", 100),
    ("blur", "Blur", 100),
]

_PALETTES = ["grayscale", "gameboy", "cga", "pico8", "sepia"]
_COLOR_MODES = ["off", "nearest", "ordered", "diffused"]
_EFFECTS = ["Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch", "Epsilon Glow"]


class ControlPanel(QWidget):
    """The right-hand control panel. Every edit mutates ``self.state`` then emits
    ``changed`` — the window turns that into a debounced re-render."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("control_panel")
        self.state: dict = {
            "contrast": 50, "midtones": 50, "highlights": 50,
            "luminance_threshold": 50, "blur": 50, "saturation": 50,
            "invert": False, "preview_disabled": False,
            "style": "None", "scale": 5, "params": {},
            "palette": "grayscale", "color_mode": "off", "effects": [],
        }
        self._sliders: dict[str, ResettableGlowSlider] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._build_dither_section(layout)
        self._build_adjustments_section(layout)
        self._build_color_section(layout)
        self._build_effects_section(layout)
        layout.addStretch(1)

    # ---- section builders ---------------------------------------------------
    def _build_dither_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Dither Controls"))

        self.preview_toggle = QCheckBox("Disable Preview")
        self.preview_toggle.toggled.connect(self._on_preview_toggle)
        layout.addWidget(self.preview_toggle)

        self.dither_combo = NoScrollComboBox()
        populate_dither_combo(self.dither_combo, {"Default": ["None"]})
        self.dither_combo.currentIndexChanged.connect(self._on_style_changed)
        layout.addWidget(_labeled("Style", self.dither_combo))

        self.scale_slider = ResettableGlowSlider(default=5, glow_color="#5e89ed")
        self.scale_slider.setRange(1, 20)
        self.scale_spin = InvisibleSpinBox(max_display=20)
        self.scale_spin.setValue(round(5 / 20 * 100))
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        layout.addWidget(_labeled("Scale", self.scale_slider, self.scale_spin))

    def _build_adjustments_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Adjustments"))

        self.invert_toggle = QCheckBox("Invert Output")
        self.invert_toggle.toggled.connect(self._on_invert_toggle)
        layout.addWidget(self.invert_toggle)

        for key, label, disp_max in _ADJUSTMENTS:
            slider = ResettableGlowSlider(default=50, glow_color="#5e89ed")
            slider.setRange(0, 100)
            spin = InvisibleSpinBox(max_display=disp_max)
            spin.setValue(50)
            slider.valueChanged.connect(self._make_slider_handler(key))
            self._sliders[key] = slider
            layout.addWidget(_labeled(label, slider, spin))

        # expose the Contrast slider by name for the tests / window
        self.contrast_slider = self._sliders["contrast"]

    def _build_color_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Color"))

        self.palette_combo = NoScrollComboBox()
        self.palette_combo.addItems(_PALETTES)
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        layout.addWidget(_labeled("Palette", self.palette_combo))

        self.mode_combo = NoScrollComboBox()
        self.mode_combo.addItems(_COLOR_MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(_labeled("Mode", self.mode_combo))

        self.saturation_slider = ResettableGlowSlider(default=50, glow_color="#5e89ed")
        self.saturation_slider.setRange(0, 100)
        sat_spin = InvisibleSpinBox(max_display=100)
        sat_spin.setValue(50)
        self.saturation_slider.valueChanged.connect(self._make_slider_handler("saturation"))
        layout.addWidget(_labeled("Saturation", self.saturation_slider, sat_spin))

    def _build_effects_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(_header("Effects"))

        self.effects_list = QListWidget()
        layout.addWidget(self.effects_list)

        row = QHBoxLayout()
        self.effect_combo = NoScrollComboBox()
        self.effect_combo.addItems(_EFFECTS)
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        add_btn.clicked.connect(self._on_add_effect)
        remove_btn.clicked.connect(self._on_remove_effect)
        row.addWidget(self.effect_combo)
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        container = QWidget()
        container.setLayout(row)
        layout.addWidget(container)

    # ---- public API ---------------------------------------------------------
    def set_registry_categories(self, by_category) -> None:
        current = self.state["style"]
        self.dither_combo.blockSignals(True)
        populate_dither_combo(self.dither_combo, by_category)
        self.dither_combo.blockSignals(False)
        self.set_style(current)

    def set_style(self, name: str) -> None:
        model = self.dither_combo.model()
        for row in range(self.dither_combo.count()):
            idx = model.index(row, 0)
            if idx.data(Qt.ItemDataRole.UserRole) == name:
                self.dither_combo.setCurrentIndex(row)
                break
        self.state["style"] = name
        self.changed.emit()

    # ---- signal handlers ----------------------------------------------------
    def _on_style_changed(self, _index: int) -> None:
        data = self.dither_combo.currentData(Qt.ItemDataRole.UserRole)
        if data is not None:
            self.state["style"] = data
            self.changed.emit()

    def _on_scale_changed(self, value: int) -> None:
        self.state["scale"] = int(value)
        self.changed.emit()

    def _on_preview_toggle(self, checked: bool) -> None:
        self.state["preview_disabled"] = bool(checked)
        self.changed.emit()

    def _on_invert_toggle(self, checked: bool) -> None:
        self.state["invert"] = bool(checked)
        self.changed.emit()

    def _on_palette_changed(self, text: str) -> None:
        self.state["palette"] = text
        self.changed.emit()

    def _on_mode_changed(self, text: str) -> None:
        self.state["color_mode"] = text
        self.changed.emit()

    def _make_slider_handler(self, key: str):
        def handler(value: int) -> None:
            self.state[key] = int(value)
            self.changed.emit()
        return handler

    def _on_add_effect(self) -> None:
        name = self.effect_combo.currentText()
        self.effects_list.addItem(name)
        self.state["effects"] = self._effects_from_list()
        self.changed.emit()

    def _on_remove_effect(self) -> None:
        row = self.effects_list.currentRow()
        if row >= 0:
            self.effects_list.takeItem(row)
            self.state["effects"] = self._effects_from_list()
            self.changed.emit()

    def _effects_from_list(self) -> list[str]:
        return [self.effects_list.item(i).text() for i in range(self.effects_list.count())]


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
    return lbl


def _labeled(text: str, widget: QWidget, spin: QWidget | None = None) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel(text))
    row.addWidget(widget, 1)
    if spin is not None:
        row.addWidget(spin)
    return container
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_controls.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/controls.py tests/test_controls.py
git commit -m "feat(ui): control panel (dither/adjustments/color/effects)"
```

---

### Task 5.10: Main window, debounced render worker, app entry (offscreen smoke test)

**Files:** Create `ditherzam/ui/main_window.py`, `ditherzam/app.py`;
Test `tests/test_app_smoke.py`
**Interfaces:**
```python
class ImageEditor(QMainWindow):
    def load_array(self, gray_f32) -> None
    def set_style(self, name: str) -> None
    def render_now(self) -> QImage        # synchronous, for tests
    def schedule_render(self) -> None     # debounced async path
def main() -> int
```
- Consumes: `RenderPipeline`, `RenderSettings` (Phase 4), the shared dither `registry`
  (Phase 1/2), `settings_from_controls`, `numpy_to_qimage`, `CustomGraphicsView`,
  `ControlPanel`, `get_hotkeys`, `find_themes` / `load_theme`.

- [ ] **Step 1: Write failing test — `tests/test_app_smoke.py`** (offscreen; the env var
  is already set in `conftest.py` before any QApplication)

```python
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"


def test_window_constructs_and_renders(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    w = ImageEditor()
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))
    w.load_array(ramp)
    w.set_style("Floyd-Steinberg")
    img = w.render_now()
    assert img.width() == 64 and img.height() == 64


def test_none_style_renders_grayscale(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    w = ImageEditor()
    w.load_array(np.full((32, 48), 128.0, np.float32))
    w.set_style("None")
    img = w.render_now()
    assert img.width() == 48 and img.height() == 32


def test_main_entry_is_callable():
    from ditherzam import app
    assert callable(app.main)


def test_debounced_render_produces_image(qapp_fixture):
    from PySide6.QtCore import QCoreApplication
    from ditherzam.ui.main_window import ImageEditor
    w = ImageEditor()
    w.load_array(np.full((16, 16), 200.0, np.float32))
    w.set_style("Atkinson")
    w.schedule_render()
    # let the debounce timer fire and the worker finish
    for _ in range(50):
        QCoreApplication.processEvents()
    assert w.last_qimage is not None
    assert w.last_qimage.width() == 16
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_app_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.main_window'`

- [ ] **Step 3: Implement `ditherzam/ui/main_window.py`**

```python
from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QSplitter,
    QWidget,
)

from ditherzam.dithering import registry as _dither_registry
from ditherzam.render import RenderPipeline

from .controls import ControlPanel
from .convert import numpy_to_qimage
from .hotkeys import get_hotkeys
from .settings_map import settings_from_controls
from .viewport import CustomGraphicsView


class _RenderSignals(QObject):
    finished = Signal(QImage)


class _RenderWorker(QRunnable):
    """Runs one render off the GUI thread and emits the finished QImage."""

    def __init__(self, pipeline: RenderPipeline, base_gray: np.ndarray, settings):
        super().__init__()
        self._pipeline = pipeline
        self._base_gray = base_gray
        self._settings = settings
        self.signals = _RenderSignals()

    def run(self) -> None:
        rgb = self._pipeline.render(self._base_gray, self._settings)
        self.signals.finished.emit(numpy_to_qimage(rgb))


class ImageEditor(QMainWindow):
    def __init__(self, registry=None, color_engine=None, effect_stack=None,
                 debounce_ms: int = 20, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ditherzam")
        self._registry = registry or _dither_registry
        self.pipeline = RenderPipeline(self._registry, color_engine, effect_stack)
        self._base_gray: np.ndarray | None = None
        self.last_qimage: QImage | None = None
        self._pool = QThreadPool.globalInstance()
        self._debounce_ms = debounce_ms

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)

        self.viewport = CustomGraphicsView()
        self.panel = ControlPanel()
        self.panel.set_registry_categories(self._registry.by_category())
        self.panel.changed.connect(self.schedule_render)
        self.viewport.image_dropped.connect(self._on_image_dropped)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self.viewport)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        # single-child layout for the central widget
        from PySide6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_render)

        self._install_shortcuts()

    # ---- public API ---------------------------------------------------------
    def load_array(self, gray_f32) -> None:
        self._base_gray = np.asarray(gray_f32, dtype=np.float32)

    def set_style(self, name: str) -> None:
        self.panel.set_style(name)

    def render_now(self) -> QImage:
        """Synchronous render (used by tests and the initial paint)."""
        if self._base_gray is None:
            raise RuntimeError("No image loaded")
        settings = settings_from_controls(self.panel.state)
        rgb = self.pipeline.render(self._base_gray, settings)
        qimg = numpy_to_qimage(rgb)
        self.last_qimage = qimg
        self.viewport.set_pixmap(QPixmap.fromImage(qimg))
        return qimg

    def schedule_render(self) -> None:
        self._debounce.start(self._debounce_ms)

    # ---- internals ----------------------------------------------------------
    def _do_render(self) -> None:
        if self._base_gray is None:
            return
        settings = settings_from_controls(self.panel.state)
        worker = _RenderWorker(self.pipeline, self._base_gray, settings)
        worker.signals.finished.connect(self._on_rendered)
        self._pool.start(worker)

    def _on_rendered(self, qimg: QImage) -> None:
        self.last_qimage = qimg
        self.viewport.set_pixmap(QPixmap.fromImage(qimg))

    def _on_image_dropped(self, path: str) -> None:
        from PIL import Image

        from ditherzam.imaging import to_gray_f32
        try:
            self.load_array(to_gray_f32(Image.open(path)))
        except Exception:
            return
        self.render_now()

    def _install_shortcuts(self) -> None:
        hk = get_hotkeys(sys.platform)
        self._actions: dict[str, QAction] = {}
        bindings = {
            "zoom_in": self.viewport.zoom_in,
            "zoom_out": self.viewport.zoom_out,
            "zoom_reset": self.viewport.reset_zoom,
        }
        for action_name, slot in bindings.items():
            act = QAction(self)
            act.setShortcut(QKeySequence(hk[action_name]))
            act.triggered.connect(slot)
            self.addAction(act)
            self._actions[action_name] = act
```

- [ ] **Step 4: Implement `ditherzam/app.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.ui.theme import find_themes, load_theme

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("ditherzam")

    themes_root = Path(__file__).resolve().parent.parent / "themes"
    if "default" in find_themes(themes_root):
        app.setStyleSheet(load_theme(themes_root, "default").stylesheet)

    window = ImageEditor()
    window.resize(1100, 720)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run — expect PASS**

Run: `pytest tests/test_app_smoke.py -v`
Expected: `4 passed`

- [ ] **Step 6: Run the whole UI suite together**

Run: `pytest tests/test_convert.py tests/test_theme.py tests/test_hotkeys.py tests/test_viewport_math.py tests/test_settings_map.py tests/test_widgets.py tests/test_delegates.py tests/test_viewport.py tests/test_controls.py tests/test_app_smoke.py -v`
Expected: all green (no failures, no errors).

- [ ] **Step 7: Commit**

```bash
git add ditherzam/ui/main_window.py ditherzam/app.py tests/test_app_smoke.py
git commit -m "feat(ui): main window, debounced render worker, app entry"
```

---

## Subsystem Definition of Done (checklist)

- [ ] `pytest tests/test_convert.py tests/test_theme.py tests/test_hotkeys.py
  tests/test_viewport_math.py tests/test_settings_map.py tests/test_widgets.py
  tests/test_delegates.py tests/test_viewport.py tests/test_controls.py
  tests/test_app_smoke.py -v` is fully green (offscreen).
- [ ] `python -m ditherzam` / `ditherzam` launches a window (manual check on a real
  display): image loads, dither combo is grouped, sliders drive a live debounced
  preview, Shift+wheel zoom + drag fling work, default QSS applied.
- [ ] Pure helpers (`convert`, `theme`, `hotkeys`, `viewport_math`, `settings_map`)
  are unit-tested **without a QApplication** (convert only needs `QImage`).
- [ ] `themes/default/theme.yaml` contains the **complete** default QSS from the spec
  appendix (every rule, none abbreviated) plus `labels:` and `glow_color: "#5e89ed"`.
- [ ] Smoke test constructs `ImageEditor` offscreen and both `render_now()` (sync) and
  `schedule_render()` (debounced worker) produce a correctly-sized `QImage`.
- [ ] Qt import audit passes (see Self-Review).

## Self-Review

**Spec coverage (each UI spec area → task):**

| Spec area | Task |
|---|---|
| numpy ↔ QImage conversion | 5.1 |
| Theming system, `find_themes`/`load_theme`, default QSS + glow `#5e89ed` + labels | 5.2 |
| `get_hotkeys(platform)` Windows/macOS table (import/zoom/export/theme/restart/help) | 5.3 |
| Zoom (×1.2 / ×0.8, cap 100.0, floor 0.01, `int(m11*100)`) + inertia (dt 0.016, friction 0.95, velocity_scale 0.5, max_velocity 2000, stop <10) | 5.4 (math) + 5.8 (view) |
| Settings mapping to `RenderSettings`; grouped combo row model | 5.5 |
| Custom widgets: `GlowSlider`, `ResettableGlowSlider` (double-click reset), `InvisibleSpinBox` (friendly display), `ClickableLabel`, `NoScrollComboBox` (swallow wheel) | 5.6 |
| Grouped dither combo w/ non-selectable category headers (`DitherStyleDelegate`) | 5.7 |
| Viewport `CustomGraphicsView`: Shift+wheel zoom, drag-pan, fling inertia, drag-drop import, fit-to-viewport, zoom% | 5.8 |
| Dither/adjustments/color/effects control sections; each edit → debounced update | 5.9 |
| `ImageEditor(QMainWindow)`, debounced `QRunnable` render worker, `app.py` (`QApplication`, style "Fusion", app id `ditherzam`, load config/theme, show) | 5.10 |

**Placeholder scan:** every code step above is complete real code — no `TODO`,
no `pass`-stub, no "implement X here". `ui/__init__.py` is intentionally empty
(so pure submodules import without pulling in Qt widget modules).

**Type / contract consistency:**
- `settings_from_controls` returns `RenderSettings` with the exact field names and
  spec defaults (`contrast/midtones/highlights/blur/luminance_threshold=50`,
  `invert=False`, `saturation=50`, `style="None"`, `scale=5`,
  `preview_disabled=False`, `params={}`) — matches the FROZEN CONTRACT.
- `ImageEditor` builds `RenderPipeline(registry, color_engine=None, effect_stack=None)`
  and calls `pipeline.render(base_gray_f32, settings)` — matches the FROZEN CONTRACT
  (`render(self, base_gray_f32, settings, temporal_field=None) -> uint8 HxWx3`);
  `numpy_to_qimage` consumes that `uint8 HxWx3`.
- `next_zoom` / `inertia_step` signatures match the plan interfaces exactly.

**Qt-isolation audit (run before final commit):**

Run: `grep -rl "import PySide6\|from PySide6" ditherzam | sort`
Expected: only paths under `ditherzam/ui/` plus `ditherzam/app.py` (and, from later
phases, `ditherzam/video/workers.py`). `ditherzam/ui/viewport_math.py`,
`settings_map.py`, `theme.py`, `hotkeys.py` must **not** appear — they are pure.
`ditherzam/ui/convert.py` appears (needs `QImage`) but requires no `QApplication`.

**Clean-room note:** all QSS, labels, and strings are transcribed from the described
default-theme values (colors, radii, sizes) — no Dither Boy source, product strings,
URLs, or licensing/telemetry code is present.
