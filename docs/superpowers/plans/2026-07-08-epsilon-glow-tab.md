# Epsilon Glow Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `epsilon_glow` from a 2-param placeholder to a full-fidelity, threshold-driven bloom, and give it its own fully-customizable **Glow** tab in the UI.

**Architecture:** A Qt-free core function in `effects/post.py` (weighted-luminance soft-knee mask → emissive source → anisotropic multi-scale blur → epsilon hot core → additive combine). A pure, Qt-free param-mapping module bridges slider ints ↔ float params. A new `GlowPanel` Qt widget hosts eight sliders + enable + reset inside a `QTabWidget` added to the right pane. `main_window` appends the glow to the existing `EffectStack` when enabled and round-trips its params through presets.

**Tech Stack:** Python 3.12, numpy, Pillow (no scipy), PySide6 (UI only).

## Global Constraints

- **Clean-room.** Our own algorithm, reverse-engineered from public tutorial behaviour. No Dither Boy / Studio AAA code/strings/binaries. "Epsilon Glow" already exists in-tree; reuse it, add no new branding.
- **Qt-free core.** `ditherzam/effects/**` must NOT import PySide6. UI→param mapping is a pure function with its own unit test.
- **No new deps.** Available: numpy, numba, Pillow, PyYAML, platformdirs, PySide6. Anisotropy via non-uniform resize→isotropic-blur→resize.
- **`RenderPipeline` stage order and `RenderSettings` fields are FROZEN.** Glow is a post-effect in the existing `EffectStack`; do not touch `render.py`.
- **Python 3.12; TDD per task** (red → green → refactor, commit each green). Suite green under `NUMBA_DISABLE_JIT=1`. Runner: `./.venv/Scripts/python.exe -m pytest`.

---

## File Structure

```
ditherzam/effects/post.py         # MODIFY: full epsilon_glow(...) replaces placeholder   (Task 1)
ditherzam/effects/glow_params.py  # CREATE: pure forward/inverse slider<->param mapping    (Task 2)
ditherzam/ui/glow_panel.py        # CREATE: GlowPanel widget (8 sliders + enable + reset)   (Task 3)
ditherzam/ui/main_window.py       # MODIFY: QTabWidget, defaults, stack wiring, preset apply (Tasks 4,5)
ditherzam/ui/controls.py          # MODIFY: drop "Epsilon Glow" from name-only _EFFECTS      (Task 4)
tests/test_effects_post.py        # MODIFY: replace old epsilon tests with full-model tests  (Task 1)
tests/test_render_cache.py        # MODIFY: update epsilon_glow call to new signature         (Task 1)
tests/test_glow_params.py         # CREATE                                                   (Task 2)
tests/test_glow_panel.py          # CREATE                                                   (Task 3)
tests/test_glow_tab.py            # CREATE                                                   (Task 4)
tests/test_glow_preset_wiring.py  # CREATE                                                   (Task 5)
```

---

### Task 1: Full-fidelity `epsilon_glow` core

**Files:**
- Modify: `ditherzam/effects/post.py` (replace `epsilon_glow` at lines 44-49; `EFFECTS` dict unchanged)
- Modify: `tests/test_effects_post.py` (line 31 call; replace tests at lines 89-99)
- Modify: `tests/test_render_cache.py:30` (signature)

**Interfaces:**
- Produces: `epsilon_glow(rgb_u8, threshold=64.0, smoothing=32.0, radius=8.0, intensity=1.0, epsilon=0.4, falloff=0.5, distance_scale=1.0, aspect=1.0) -> np.ndarray` (uint8 RGB, shape preserved). Master gate: `intensity <= 0` ⇒ returns input unchanged.
- Helpers (module-private): `_smoothstep(edge0, edge1, x)`, `_aniso_blur(src_f, sigma, aspect)`.

- [ ] **Step 1: Write the failing tests** — replace the two old epsilon tests (lines 89-99) and fix the call at line 31 of `tests/test_effects_post.py`.

Change line 31 from `(epsilon_glow, {"radius": 3, "strength": 0.5}),` to:
```python
        (epsilon_glow, {"threshold": 32, "radius": 3, "intensity": 1.0}),
```

Replace `test_epsilon_glow_brightens_uniform_field` and `test_epsilon_glow_clips_to_255` (lines 89-99) with:
```python
def test_epsilon_glow_intensity_zero_is_identity():
    x = rand_img()
    np.testing.assert_array_equal(epsilon_glow(x, intensity=0.0), x)


def test_epsilon_glow_shape_dtype_all_params():
    x = rand_img()
    out = epsilon_glow(x, threshold=40, smoothing=20, radius=5, intensity=1.0,
                       epsilon=0.5, falloff=0.4, distance_scale=1.2, aspect=1.5)
    assert out.shape == x.shape and out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_epsilon_glow_higher_threshold_fewer_glow_pixels():
    # vertical luminance gradient 0..255 across the width
    grad = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    x = np.stack([grad, grad, grad], axis=-1)
    changed_low = int(np.count_nonzero(np.any(
        epsilon_glow(x, threshold=32, radius=4, intensity=1.0) != x, axis=-1)))
    changed_high = int(np.count_nonzero(np.any(
        epsilon_glow(x, threshold=200, radius=4, intensity=1.0) != x, axis=-1)))
    assert changed_high < changed_low          # higher threshold => fewer glowing pixels


def test_epsilon_glow_smoothing_zero_hard_cut_no_error():
    x = rand_img()
    out = epsilon_glow(x, threshold=100, smoothing=0, radius=3, intensity=1.0)
    assert out.dtype == np.uint8 and np.isfinite(out.astype(np.float64)).all()


def test_epsilon_glow_epsilon_raises_core_brightness():
    # one bright pixel over a dark field; brighter epsilon => brighter center
    x = np.zeros((32, 32, 3), np.uint8)
    x[16, 16] = 255
    dim = epsilon_glow(x, threshold=50, radius=6, intensity=1.0, epsilon=0.0)
    hot = epsilon_glow(x, threshold=50, radius=6, intensity=1.0, epsilon=1.0)
    assert int(hot[16, 16].sum()) >= int(dim[16, 16].sum())


def test_epsilon_glow_grayscale_input_ok():
    x = gray_img(180)
    out = epsilon_glow(x, threshold=50, radius=3, intensity=0.8)
    assert out.shape == x.shape and out.dtype == np.uint8


def test_chromatic_before_glow_differs_from_after():
    # emergent stack-order interaction: CA then Glow != Glow then CA
    x = rand_img()
    ca_then_glow = epsilon_glow(chromatic_aberration(x, shift=2),
                                threshold=40, radius=3, intensity=1.0)
    glow_then_ca = chromatic_aberration(
        epsilon_glow(x, threshold=40, radius=3, intensity=1.0), shift=2)
    assert not np.array_equal(ca_then_glow, glow_then_ca)
```

Also update `tests/test_render_cache.py:30` from
`s.add("Epsilon Glow", radius=3.0, strength=0.4)` to:
```python
    s.add("Epsilon Glow", threshold=40.0, radius=3.0, intensity=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_effects_post.py -q`
Expected: FAIL — old signature `strength` gone / new tests error (`TypeError` on unexpected kwargs).

- [ ] **Step 3: Implement the core** — replace `epsilon_glow` (post.py lines 44-49) with:

```python
def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """0..1 soft ramp between edge0 and edge1; hard step when edge1 <= edge0."""
    if edge1 <= edge0:
        return (x >= edge0).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _aniso_blur(src_f: np.ndarray, sigma: float, aspect: float) -> np.ndarray:
    """Gaussian blur wider along x by `aspect`, via resize->isotropic blur->resize.
    aspect > 1 stretches the glow horizontally; aspect == 1 is isotropic."""
    h, w = src_f.shape[:2]
    ax = max(float(aspect), 1e-3)
    new_w = max(1, int(round(w / ax)))
    img = Image.fromarray(np.clip(src_f, 0, 255).astype(np.uint8))
    if new_w != w:
        img = img.resize((new_w, h), Image.BILINEAR)
    img = img.filter(ImageFilter.GaussianBlur(float(max(sigma, 0.0))))
    if new_w != w:
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, np.float32)


def epsilon_glow(rgb_u8: np.ndarray, threshold: float = 64.0, smoothing: float = 32.0,
                 radius: float = 8.0, intensity: float = 1.0, epsilon: float = 0.4,
                 falloff: float = 0.5, distance_scale: float = 1.0,
                 aspect: float = 1.0) -> np.ndarray:
    """Threshold-driven bloom tuned for dithered art.

    Weighted luminance selects bright pixels through a soft knee; the emissive
    source is spread by a multi-scale anisotropic blur; `epsilon` adds a tight,
    hot core. `intensity` is the master gate (0 => identity).
    """
    if intensity <= 0:
        return rgb_u8
    base = rgb_u8.astype(np.float32)
    lum = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
    mask = _smoothstep(float(threshold), float(threshold) + float(smoothing), lum)
    src = base * mask[..., None]

    r = max(float(radius) * float(distance_scale), 0.0)
    scales = (0.5, 1.0, 2.0)
    base_w = np.array([1.0, 0.6, 0.35], np.float32)
    f = float(np.clip(falloff, 0.0, 1.0))
    bias = np.array([1.0 + f, 1.0, 1.0 - 0.5 * f], np.float32)
    weights = base_w * bias
    weights /= weights.sum()
    glow = np.zeros_like(base)
    for s, wgt in zip(scales, weights):
        glow += _aniso_blur(src, r * s + 1e-3, aspect) * float(wgt)

    k = 1.0 + float(np.clip(epsilon, 0.0, 1.0)) * 8.0
    core = base * (mask[..., None] ** k)
    core_glow = _aniso_blur(core, max(r * 0.25, 1e-3), aspect) * float(np.clip(epsilon, 0.0, 1.0))

    out = base + (glow + core_glow) * float(intensity)
    return np.clip(out, 0, 255).astype(np.uint8)
```

Update the `EFFECTS` dict entry — it already reads `"Epsilon Glow": epsilon_glow`, leave unchanged. `Image, ImageFilter` are already imported at the top of `post.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_effects_post.py tests/test_render_cache.py -q`
Expected: PASS (all effects tests + render-cache).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/effects/post.py tests/test_effects_post.py tests/test_render_cache.py
git commit -m "feat(effects): full-fidelity epsilon_glow (threshold/epsilon/falloff/aspect)"
```

---

### Task 2: Pure param-mapping module

**Files:**
- Create: `ditherzam/effects/glow_params.py`
- Test: `tests/test_glow_params.py`

**Interfaces:**
- Produces:
  - `GLOW_DEFAULTS: dict[str, int]` — default slider state (keys below).
  - `glow_params_from_state(state: dict) -> dict` — slider ints → `epsilon_glow` float kwargs (`threshold, smoothing, radius, intensity, epsilon, falloff, distance_scale, aspect`).
  - `glow_state_from_params(params: dict) -> dict` — inverse; returns the eight `glow_*` int keys plus `"glow_enabled": True`.
- State keys: `glow_enabled` (bool), `glow_threshold, glow_smoothing, glow_radius, glow_intensity, glow_epsilon, glow_falloff, glow_distance, glow_aspect` (ints).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_glow_params.py
import math
from ditherzam.effects.glow_params import (
    GLOW_DEFAULTS, glow_params_from_state, glow_state_from_params,
)


def test_defaults_map_to_neutral_params():
    p = glow_params_from_state(GLOW_DEFAULTS)
    assert p["threshold"] == 64.0
    assert p["intensity"] == 1.0            # glow_intensity 50 -> 50/50
    assert p["distance_scale"] == 1.0       # glow_distance 50 -> 50/50
    assert abs(p["aspect"] - 1.0) < 1e-6    # glow_aspect 50 -> 2**0
    assert p["epsilon"] == 0.4 and p["falloff"] == 0.5


def test_forward_scales_each_slider():
    state = dict(GLOW_DEFAULTS, glow_intensity=100, glow_epsilon=100,
                 glow_aspect=100, glow_radius=200)
    p = glow_params_from_state(state)
    assert p["intensity"] == 2.0
    assert p["epsilon"] == 1.0
    assert abs(p["aspect"] - 4.0) < 1e-6
    assert p["radius"] == 200.0


def test_roundtrip_state_params_state():
    state = dict(GLOW_DEFAULTS, glow_threshold=120, glow_radius=40,
                 glow_intensity=70, glow_aspect=75)
    back = glow_state_from_params(glow_params_from_state(state))
    for k in ("glow_threshold", "glow_radius", "glow_intensity", "glow_aspect"):
        assert abs(back[k] - state[k]) <= 1
    assert back["glow_enabled"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_params.py -q`
Expected: FAIL — `ModuleNotFoundError: ditherzam.effects.glow_params`.

- [ ] **Step 3: Implement**

```python
# ditherzam/effects/glow_params.py
"""Pure (Qt-free) mapping between Glow-tab slider ints and epsilon_glow params."""
from __future__ import annotations

import math

GLOW_DEFAULTS: dict = {
    "glow_enabled": False,
    "glow_threshold": 64,     # 0..255  -> threshold
    "glow_smoothing": 32,     # 0..128  -> smoothing
    "glow_radius": 8,         # 0..200  -> radius
    "glow_intensity": 50,     # 0..100  -> intensity = v/50   (0..2)
    "glow_epsilon": 40,       # 0..100  -> epsilon   = v/100  (0..1)
    "glow_falloff": 50,       # 0..100  -> falloff   = v/100  (0..1)
    "glow_distance": 50,      # 1..100  -> distance_scale = v/50 (0.02..2)
    "glow_aspect": 50,        # 0..100  -> aspect = 2**((v-50)/25) (0.25..4)
}


def glow_params_from_state(state: dict) -> dict:
    g = lambda k: state.get(k, GLOW_DEFAULTS[k])
    return {
        "threshold": float(g("glow_threshold")),
        "smoothing": float(g("glow_smoothing")),
        "radius": float(g("glow_radius")),
        "intensity": g("glow_intensity") / 50.0,
        "epsilon": g("glow_epsilon") / 100.0,
        "falloff": g("glow_falloff") / 100.0,
        "distance_scale": g("glow_distance") / 50.0,
        "aspect": 2.0 ** ((g("glow_aspect") - 50) / 25.0),
    }


def glow_state_from_params(params: dict) -> dict:
    return {
        "glow_enabled": True,
        "glow_threshold": int(round(params["threshold"])),
        "glow_smoothing": int(round(params["smoothing"])),
        "glow_radius": int(round(params["radius"])),
        "glow_intensity": int(round(params["intensity"] * 50.0)),
        "glow_epsilon": int(round(params["epsilon"] * 100.0)),
        "glow_falloff": int(round(params["falloff"] * 100.0)),
        "glow_distance": int(round(params["distance_scale"] * 50.0)),
        "glow_aspect": int(round(50 + 25.0 * math.log2(max(params["aspect"], 1e-6)))),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_params.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/effects/glow_params.py tests/test_glow_params.py
git commit -m "feat(effects): pure slider<->param mapping for the Glow tab"
```

---

### Task 3: `GlowPanel` widget

**Files:**
- Create: `ditherzam/ui/glow_panel.py`
- Test: `tests/test_glow_panel.py`

**Interfaces:**
- Consumes: `ditherzam.effects.glow_params.GLOW_DEFAULTS`; `ditherzam.ui.widgets.ResettableGlowSlider, InvisibleSpinBox`.
- Produces: `GlowPanel(QWidget)` with `.state` (a copy of `GLOW_DEFAULTS`), `changed = Signal()`, `.enable_toggle` (QCheckBox), `._sliders: dict[str, ResettableGlowSlider]`, and `.reset_all()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_glow_panel.py
import pytest

pytest.importorskip("PySide6")


def test_glow_panel_has_eight_sliders_and_toggle(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    p = GlowPanel()
    keys = ("glow_threshold", "glow_smoothing", "glow_radius", "glow_intensity",
            "glow_epsilon", "glow_falloff", "glow_distance", "glow_aspect")
    for k in keys:
        assert k in p._sliders
    assert p.state["glow_enabled"] is False


def test_glow_slider_edit_updates_state_and_emits(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    p = GlowPanel()
    seen = []
    p.changed.connect(lambda: seen.append(True))
    p._sliders["glow_threshold"].setValue(120)
    assert p.state["glow_threshold"] == 120
    assert seen


def test_glow_enable_toggle_updates_state(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    p = GlowPanel()
    p.enable_toggle.setChecked(True)
    assert p.state["glow_enabled"] is True


def test_glow_reset_all_restores_defaults(qapp_fixture):
    from ditherzam.ui.glow_panel import GlowPanel
    from ditherzam.effects.glow_params import GLOW_DEFAULTS
    p = GlowPanel()
    p._sliders["glow_radius"].setValue(200)
    p.reset_all()
    assert p.state["glow_radius"] == GLOW_DEFAULTS["glow_radius"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_panel.py -q`
Expected: FAIL — `ModuleNotFoundError: ditherzam.ui.glow_panel`.

- [ ] **Step 3: Implement**

```python
# ditherzam/ui/glow_panel.py
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..effects.glow_params import GLOW_DEFAULTS
from .widgets import InvisibleSpinBox, ResettableGlowSlider

# (state_key, label, min, max, spin_display_max)
_GLOW_SLIDERS = [
    ("glow_threshold", "Threshold", 0, 255, 255),
    ("glow_smoothing", "Smoothing", 0, 128, 128),
    ("glow_radius", "Radius", 0, 200, 200),
    ("glow_intensity", "Intensity", 0, 100, 100),
    ("glow_epsilon", "Epsilon", 0, 100, 100),
    ("glow_falloff", "Falloff", 0, 100, 100),
    ("glow_distance", "Distance Scale", 1, 100, 100),
    ("glow_aspect", "Aspect", 0, 100, 100),
]


class GlowPanel(QWidget):
    """Dedicated, fully-customizable Epsilon Glow controls (its own tab)."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glow_panel")
        self.state: dict = dict(GLOW_DEFAULTS)
        self._sliders: dict[str, ResettableGlowSlider] = {}
        self._spins: dict[str, InvisibleSpinBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Epsilon Glow")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        self.enable_toggle = QCheckBox("Enable Glow")
        self.enable_toggle.toggled.connect(self._on_enable)
        layout.addWidget(self.enable_toggle)

        for key, label, lo, hi, disp in _GLOW_SLIDERS:
            default = int(GLOW_DEFAULTS[key])
            slider = ResettableGlowSlider(default=default, glow_color="#5e89ed")
            slider.setRange(lo, hi)
            slider.setValue(default)
            spin = InvisibleSpinBox(max_display=disp)
            spin.setValue(int(round(default / max(hi, 1) * 100)))
            slider.valueChanged.connect(self._make_handler(key))
            slider.valueChanged.connect(
                lambda v, s=spin, h=hi: s.setValue(int(round(v / max(h, 1) * 100))))
            self._sliders[key] = slider
            self._spins[key] = spin
            layout.addWidget(_row(label, slider, spin))

        self.reset_btn = QPushButton("Reset all")
        self.reset_btn.clicked.connect(self.reset_all)
        layout.addWidget(self.reset_btn)
        layout.addStretch(1)

    def _make_handler(self, key: str):
        def handler(value: int) -> None:
            self.state[key] = int(value)
            self.changed.emit()
        return handler

    def _on_enable(self, checked: bool) -> None:
        self.state["glow_enabled"] = bool(checked)
        self.changed.emit()

    def reset_all(self) -> None:
        for key, slider in self._sliders.items():
            slider.setValue(int(GLOW_DEFAULTS[key]))
        self.enable_toggle.setChecked(bool(GLOW_DEFAULTS["glow_enabled"]))


def _row(text: str, widget: QWidget, spin: QWidget) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(QLabel(text))
    row.addWidget(widget, 1)
    row.addWidget(spin)
    return container
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/glow_panel.py tests/test_glow_panel.py
git commit -m "feat(ui): GlowPanel widget — enable + 8 glow sliders + reset"
```

---

### Task 4: Tab integration + drop Glow from name-only effects

**Files:**
- Modify: `ditherzam/ui/main_window.py` (right-pane layout at lines ~106-120; `_EFFECT_DEFAULTS` at lines 36-42)
- Modify: `ditherzam/ui/controls.py:37` (`_EFFECTS` list)
- Test: `tests/test_glow_tab.py`

**Interfaces:**
- Consumes: `ditherzam.ui.glow_panel.GlowPanel`.
- Produces: `ImageEditor.glow_panel` (a `GlowPanel`); `ImageEditor.tabs` (a `QTabWidget` with tabs "Editor" and "Glow"); `glow_panel.changed` connected to `schedule_render`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_glow_tab.py
import pytest

pytest.importorskip("PySide6")


def test_editor_has_glow_tab(qapp_fixture):
    from PySide6.QtWidgets import QTabWidget
    from ditherzam.ui.main_window import ImageEditor
    from ditherzam.ui.glow_panel import GlowPanel
    ed = ImageEditor()
    assert isinstance(ed.tabs, QTabWidget)
    titles = [ed.tabs.tabText(i) for i in range(ed.tabs.count())]
    assert "Editor" in titles and "Glow" in titles
    assert isinstance(ed.glow_panel, GlowPanel)


def test_epsilon_glow_removed_from_name_only_effects(qapp_fixture):
    from ditherzam.ui.controls import _EFFECTS
    assert "Epsilon Glow" not in _EFFECTS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_tab.py -q`
Expected: FAIL — `ImageEditor` has no attribute `tabs` / `_EFFECTS` still contains "Epsilon Glow".

- [ ] **Step 3a: controls.py** — change line 37 from
`_EFFECTS = ["Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch", "Epsilon Glow"]` to:
```python
_EFFECTS = ["Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch"]
```

- [ ] **Step 3b: main_window.py** — replace the QScrollArea/splitter block (lines 114-120) with a `QTabWidget` that holds the Editor scroll area and the Glow panel scroll area:

```python
        from PySide6.QtWidgets import QTabWidget
        from .glow_panel import GlowPanel

        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setWidget(self.panel)

        self.glow_panel = GlowPanel()
        self.glow_panel.changed.connect(self.schedule_render)
        glow_scroll = QScrollArea()
        glow_scroll.setWidgetResizable(True)
        glow_scroll.setWidget(self.glow_panel)

        self.tabs = QTabWidget()
        self.tabs.addTab(editor_scroll, "Editor")
        self.tabs.addTab(glow_scroll, "Glow")

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self.viewport)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
```

Leave the `_EFFECT_DEFAULTS["Epsilon Glow"]` entry in place but update its value so any preset-loaded glow still renders with the new signature — change line 41 to:
```python
    "Epsilon Glow": {"threshold": 64.0, "smoothing": 32.0, "radius": 8.0,
                     "intensity": 1.0, "epsilon": 0.4, "falloff": 0.5,
                     "distance_scale": 1.0, "aspect": 1.0},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_tab.py tests/test_controls.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/main_window.py ditherzam/ui/controls.py tests/test_glow_tab.py
git commit -m "feat(ui): Editor/Glow QTabWidget; Glow owned by its tab, not the effects list"
```

---

### Task 5: Effect-stack wiring + preset round-trip

**Files:**
- Modify: `ditherzam/ui/main_window.py` (`_current_effect_stack` lines 260-268; `_apply_preset` lines 291-311)
- Test: `tests/test_glow_preset_wiring.py`

**Interfaces:**
- Consumes: `ditherzam.effects.glow_params.glow_params_from_state, glow_state_from_params`; `ImageEditor.glow_panel`.
- Produces: when `glow_panel.state["glow_enabled"]` is True, `_current_effect_stack()` includes `("Epsilon Glow", <live params>)`; `_apply_preset` routes a loaded "Epsilon Glow" effect's params into `glow_panel` (enable + sliders) instead of the name-only list.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_glow_preset_wiring.py
import pytest

pytest.importorskip("PySide6")


def test_enabled_glow_appears_in_effect_stack(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    ed = ImageEditor()
    ed.glow_panel.enable_toggle.setChecked(True)
    ed.glow_panel._sliders["glow_threshold"].setValue(120)
    stack = ed._current_effect_stack()
    names = [n for n, _ in stack.items]
    assert "Epsilon Glow" in names
    params = dict(stack.items)["Epsilon Glow"]
    assert params["threshold"] == 120.0


def test_disabled_glow_absent_from_stack(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    ed = ImageEditor()
    assert ed.glow_panel.state["glow_enabled"] is False
    stack = ed._current_effect_stack()
    names = [] if stack is None else [n for n, _ in stack.items]
    assert "Epsilon Glow" not in names


def test_apply_preset_routes_glow_into_tab(qapp_fixture):
    from ditherzam.ui.main_window import ImageEditor
    ed = ImageEditor()
    settings = ed._collect_settings()
    effects = [("Epsilon Glow", {"threshold": 100.0, "smoothing": 20.0,
                                  "radius": 30.0, "intensity": 1.4, "epsilon": 0.6,
                                  "falloff": 0.3, "distance_scale": 1.0, "aspect": 1.0})]
    ed._apply_preset(settings, None, effects)
    assert ed.glow_panel.state["glow_enabled"] is True
    assert ed.glow_panel.state["glow_threshold"] == 100
    # not added to the name-only effects list
    assert "Epsilon Glow" not in ed.panel.state.get("effects", [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_preset_wiring.py -q`
Expected: FAIL — glow not added to stack / not routed into tab.

- [ ] **Step 3a: `_current_effect_stack`** — replace lines 260-268 with:

```python
    def _current_effect_stack(self):
        from ..effects.stack import EffectStack
        from ..effects.glow_params import glow_params_from_state
        names = self.panel.state.get("effects", []) or []
        glow_on = bool(self.glow_panel.state.get("glow_enabled"))
        if not names and not glow_on:
            return None
        stack = EffectStack()
        for name in names:
            stack.add(name, **_EFFECT_DEFAULTS.get(name, {}))
        if glow_on:
            stack.add("Epsilon Glow", **glow_params_from_state(self.glow_panel.state))
        return stack
```

- [ ] **Step 3b: `_apply_preset`** — in the effects loop (lines 303-306), split "Epsilon Glow" out of the name-only list and route it into the glow tab. Replace those four lines with:

```python
        from ..effects.glow_params import glow_state_from_params, GLOW_DEFAULTS
        panel.effects_list.clear()
        glow_state = None
        non_glow = []
        for name, params in effects:
            if name == "Epsilon Glow":
                glow_state = glow_state_from_params(params)
            else:
                panel.effects_list.addItem(name)
                non_glow.append(name)
        panel.state["effects"] = non_glow
        # push glow params into the Glow tab (enable + sliders), or disable if absent
        gp = self.glow_panel
        gp.enable_toggle.setChecked(bool(glow_state))
        if glow_state:
            for key, slider in gp._sliders.items():
                slider.setValue(int(glow_state.get(key, GLOW_DEFAULTS[key])))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_glow_preset_wiring.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (full suite green, JIT-off).

```bash
git add ditherzam/ui/main_window.py tests/test_glow_preset_wiring.py
git commit -m "feat(ui): wire Glow tab into effect stack + preset round-trip"
```

---

## Self-Review notes

- **Spec coverage:** core algorithm (Task 1), pure Qt-free mapping (Task 2), GlowPanel + 8 sliders + enable + reset (Task 3), QTabWidget Editor/Glow + single-source-of-truth removal from `_EFFECTS` (Task 4), effect-stack wiring + preset round-trip + chromatic interaction test (Task 1 step) — all covered.
- **Signature break handled:** old `strength` tests replaced (Task 1) and `test_render_cache.py:30` updated; `_EFFECT_DEFAULTS` updated (Task 4) so any lingering preset with the name still renders.
- **Type consistency:** state keys (`glow_*`) identical across Tasks 2/3/5; `glow_params_from_state` / `glow_state_from_params` names consistent; `epsilon_glow` kwargs identical in post.py, tests, `_EFFECT_DEFAULTS`, and mapping output.
