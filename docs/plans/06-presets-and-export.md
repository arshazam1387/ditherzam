# Phase 6 — Presets & Export — Implementation Plan

> REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Read
> [`00-ROADMAP.md`](00-ROADMAP.md); complete Phases 1–5.

**Goal:** Save/load/import/export YAML presets (adjustments + dither + color +
effects), export the rendered image to PNG/JPG and to optimized SVG, and batch a
folder of same-sized images.

**Architecture:** `presets.py` (de)serializes a `RenderSettings` (+ palette +
effect stack) to YAML with range-clamping on load. `export/raster.py` and
`export/vector.py` are pure functions over `uint8` arrays. Batch is a headless
loop reused by the UI.

**Tech Stack:** PyYAML · Pillow · NumPy · pytest.

## Global Constraints
See roadmap. Presets live in `user_data_dir/PRESETS/*.yaml`. Export functions are
pure and Qt-free.

---

## File structure (this phase)

- Create `ditherzam/presets.py`
- Create `ditherzam/export/__init__.py`, `raster.py`, `vector.py`
- Create `ditherzam/batch.py`
- Tests: `tests/test_presets.py`, `tests/test_export_raster.py`, `tests/test_export_vector.py`, `tests/test_batch.py`

---

### Task 1: Preset (de)serialization with clamping

**Files:** Create `ditherzam/presets.py`; Test `tests/test_presets.py`
**Interfaces:**
```python
def settings_to_preset(settings, palette=None, effect_stack=None) -> dict
def preset_to_settings(preset: dict) -> tuple[RenderSettings, Palette|None, list]
class PresetManager:
    def __init__(self, presets_dir): ...
    def save(self, name, preset: dict) -> Path
    def load(self, name) -> dict
    def list(self) -> list[str]
    def import_file(self, src) -> str        # returns imported name; raises on invalid
```

- [ ] **Step 1: Failing test**

```python
from ditherzam.render import RenderSettings
from ditherzam.presets import settings_to_preset, preset_to_settings, PresetManager

def test_roundtrip_clamps_out_of_range():
    s = RenderSettings(contrast=70, style="Atkinson", scale=3)
    d = settings_to_preset(s)
    d["adjustments"]["contrast"] = 9999          # out of range
    s2, pal, fx = preset_to_settings(d)
    assert 0 <= s2.contrast <= 100
    assert s2.style == "Atkinson" and s2.scale == 3

def test_preset_manager_save_list_load(tmp_path):
    m = PresetManager(tmp_path)
    m.save("mine", {"adjustments": {}, "dither": {"style": "None"}})
    assert "mine" in m.list()
    assert m.load("mine")["dither"]["style"] == "None"

def test_import_invalid_raises(tmp_path):
    import pytest
    bad = tmp_path/"bad.yaml"; bad.write_text("just a string")
    with pytest.raises(ValueError):
        PresetManager(tmp_path).import_file(bad)
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement** — YAML schema (spec §10.1 extended with `color` + `effects`):

```yaml
adjustments: {contrast, midtones, highlights, luminance_threshold, blur, saturation, invert}
dither: {style, scale, preview_disabled, params: {...}}
color: {mode, palette: {name, colors: [[r,g,b],...]}}   # optional
effects: [{name, params: {...}}, ...]                    # optional
```
Clamp each adjustment to 0..100, scale to 1..20 on load; validate top-level is a mapping.
- [ ] **Step 4: Pass. Step 5: Commit** `feat(presets): YAML presets with clamping + PresetManager`

---

### Task 2: Raster export (PNG/JPG)

**Files:** Create `ditherzam/export/raster.py`; Test `tests/test_export_raster.py`
**Interfaces:** `save_raster(rgb_u8, path)` (format by extension).

- [ ] **Step 1: Failing test** — save `(8,8,3)` array to `tmp/x.png` and `x.jpg`;
  reopen with PIL; PNG round-trips exactly, JPG matches shape.
- [ ] **Step 2–5:** implement (`Image.fromarray(rgb_u8).save(path)`), pass, commit
  `feat(export): PNG/JPG raster export`.

---

### Task 3: Vector export (optimized SVG)

**Files:** Create `ditherzam/export/vector.py`; Test `tests/test_export_vector.py`
**Interfaces:** `raster_to_svg(gray_u8, threshold: int, invert: bool) -> str`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from ditherzam.export.vector import raster_to_svg

def test_svg_header_and_size():
    a = np.array([[0,255],[0,255]], np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.startswith("<svg") and 'width="2"' in svg and svg.rstrip().endswith("</svg>")

def test_vertical_run_merged_into_one_rect():
    a = np.zeros((4,1), np.uint8)             # one filled column, 4 tall
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 2            # background + one merged run
```

- [ ] **Step 2–3:** implement the run-merge algorithm (spec §11.3): background rect +
  for each column, merge consecutive filled pixels (`< threshold`) into a single
  `<rect x y width=1 height=runlen>`; colors flip with `invert`.
- [ ] **Step 4: pass. Step 5: commit** `feat(export): optimized SVG vector export`

---

### Task 4: Batch processing (headless)

**Files:** Create `ditherzam/batch.py`; Test `tests/test_batch.py`
**Interfaces:**
```python
def batch_process(folder, out_folder, settings, pipeline, ref_size) -> tuple[int,int]
# returns (processed, skipped); only images whose size == ref_size are processed
```

- [ ] **Step 1: Failing test** — create 3 PNGs (2 matching ref size, 1 different) in a
  tmp folder; assert `(2, 1)` and that 2 output files exist.
- [ ] **Step 2–5:** implement (glob `*.png/.jpg/.jpeg/.webp`, load gray, `pipeline.render`,
  save into `out_folder`), pass, commit `feat(batch): folder batch processing`.

---

### Task 5: Wire presets/export into the UI

**Files:** Modify `ditherzam/ui/main_window.py`; (smoke) `tests/test_app_smoke.py`
- [ ] Add menu actions Save/Load/Import/Export Preset, Export PNG/JPG, Export SVG,
  Batch → Select Folder, calling the headless functions above. Smoke-test that the
  actions exist. Commit `feat(ui): preset + export menu wiring`.

---

## Phase 6 Self-Review
- [ ] Presets round-trip and clamp; invalid import raises.
- [ ] PNG round-trips exactly; SVG merges vertical runs and has correct header.
- [ ] Batch returns correct (processed, skipped) and writes outputs.
- [ ] Export functions Qt-free; UI only wires them.
