# Phase 6 — Presets & Export — Completion Task List

Save/load/import/export YAML presets (with range-clamping on load), export the
rendered image to PNG/JPG and to optimized **run-merged** SVG, batch a folder of
same-sized images, and wire all of it into the Qt UI (UI layer only).

## Prereqs

- **Phases 1–5 green** (`NUMBA_DISABLE_JIT=1 pytest` clean). This phase consumes:
  - `ditherzam.render.RenderSettings` + `RenderPipeline` (Phase 4).
  - `ditherzam.color.palette.Palette` (Phase 3).
  - `ditherzam.effects.stack.EffectStack` (Phase 4).
  - `ditherzam.imaging.to_gray_f32` (Phase 1).
  - `ditherzam.dithering.registry` shared registry instance (Phase 1/2).
- **Python 3.12** on PATH (Numba/PySide6 target). If `python --version` is not
  `3.12.x`, use the project venv at
  `…/scratchpad/dbwork/py312/python.exe` and prefix commands with it, e.g.
  `NUMBA_DISABLE_JIT=1 …/py312/python.exe -m pytest …`.
- Tests run with `NUMBA_DISABLE_JIT=1`. Shell shown is **Git Bash**.

### FROZEN CONTRACTS honored verbatim by this phase

```python
# ditherzam/render.py
@dataclass
class RenderSettings:
    contrast=50; midtones=50; highlights=50; blur=50; luminance_threshold=50
    invert=False; saturation=50; style="None"; scale=5; preview_disabled=False; params={}
class RenderPipeline:
    def __init__(self, registry, color_engine=None, effect_stack=None)
    def render(self, base_gray_f32, settings, temporal_field=None) -> np.ndarray  # uint8 HxWx3

# ditherzam/color/palette.py
class Palette: name: str; colors: np.ndarray  # float32[K,3]

# ditherzam/effects/stack.py
class EffectStack:
    items: list[tuple[str, dict]]
    def add(self, name, **params); def move(self, i, j); def remove(self, i)
```

**Slider ranges used for clamping** (from `RenderSettings` defaults / spec §7.1 §10.2):
`contrast, midtones, highlights, luminance_threshold, blur, saturation` → **0..100**;
`scale` → **1..20**; `invert, preview_disabled` → bool; `style` → str; `params` → passthrough dict.

**Files created this phase**
- `ditherzam/presets.py`
- `ditherzam/export/__init__.py`, `ditherzam/export/raster.py`, `ditherzam/export/vector.py`
- `ditherzam/batch.py`
- `ditherzam/ui/export_actions.py`  *(Qt — ui layer only)*
- Tests: `tests/test_presets.py`, `tests/test_export_raster.py`,
  `tests/test_export_vector.py`, `tests/test_batch.py`, `tests/test_ui_export_actions.py`

---

### Task 6.0: Sanity — prerequisite subsystems import

**Files:** Test `tests/test_presets.py` (created here, extended by 6.1/6.2)
**Interfaces:** Consumes Phase 1–5 modules; Produces nothing.

- [ ] **Step 1: Write failing test** — `tests/test_presets.py`

```python
def test_prereqs_import():
    from ditherzam.render import RenderSettings, RenderPipeline
    from ditherzam.color.palette import Palette
    from ditherzam.effects.stack import EffectStack
    s = RenderSettings()
    assert s.contrast == 50 and s.scale == 5 and s.style == "None"
    assert s.saturation == 50 and s.invert is False
    assert isinstance(s.params, dict)
```

- [ ] **Step 2: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_presets.py::test_prereqs_import -v`
  → expect **PASS** if Phases 1–5 are green. If it **FAILs** with `ImportError`,
  stop and finish the missing prerequisite phase before continuing.
- [ ] **Step 3: Implement** — none (verification only).
- [ ] **Step 4: Run** → expect `1 passed`.
- [ ] **Step 5: Commit**
  `git add tests/test_presets.py && git commit -m "test(presets): prerequisite import sanity check"`

---

### Task 6.1: Preset (de)serialization with clamping

**Files:** Create `ditherzam/presets.py`; extend Test `tests/test_presets.py`
**Interfaces:**
```python
# Produces (Qt-free, pure):
def settings_to_preset(settings, palette=None, effect_stack=None, color_mode="off") -> dict
def preset_to_settings(preset: dict) -> tuple[RenderSettings, Palette | None, list[tuple[str, dict]]]
# Consumes: RenderSettings, Palette, EffectStack.items
```
Preset YAML schema (spec §10.1 extended with `color` + `effects`):
```yaml
adjustments: {contrast, midtones, highlights, luminance_threshold, blur, saturation, invert}
dither:      {style, scale, preview_disabled, params: {...}}
color:       {mode, palette: {name, colors: [[r,g,b], ...]}}   # optional
effects:     [{name, params: {...}}, ...]                      # optional
```

- [ ] **Step 1: Write failing test** — append to `tests/test_presets.py`

```python
import numpy as np
import pytest
from ditherzam.render import RenderSettings
from ditherzam.presets import settings_to_preset, preset_to_settings


def test_roundtrip_preserves_values():
    s = RenderSettings(contrast=70, midtones=30, highlights=60,
                       luminance_threshold=40, blur=10, saturation=80,
                       invert=True, style="Atkinson", scale=3,
                       preview_disabled=True, params={"dither_parameter": 2})
    d = settings_to_preset(s)
    s2, pal, fx = preset_to_settings(d)
    assert s2.contrast == 70 and s2.midtones == 30 and s2.highlights == 60
    assert s2.luminance_threshold == 40 and s2.blur == 10 and s2.saturation == 80
    assert s2.invert is True and s2.style == "Atkinson" and s2.scale == 3
    assert s2.preview_disabled is True
    assert s2.params == {"dither_parameter": 2}
    assert pal is None and fx == []


def test_roundtrip_clamps_out_of_range():
    s = RenderSettings(contrast=70, style="Atkinson", scale=3)
    d = settings_to_preset(s)
    d["adjustments"]["contrast"] = 9999          # out of range high
    d["adjustments"]["blur"] = -50               # out of range low
    d["dither"]["scale"] = 0                      # below 1
    s2, pal, fx = preset_to_settings(d)
    assert 0 <= s2.contrast <= 100 and s2.contrast == 100
    assert 0 <= s2.blur <= 100 and s2.blur == 0
    assert 1 <= s2.scale <= 20 and s2.scale == 1
    assert s2.style == "Atkinson"


def test_missing_sections_fall_back_to_defaults():
    s2, pal, fx = preset_to_settings({"dither": {"style": "Floyd-Steinberg"}})
    d = RenderSettings()
    assert s2.contrast == d.contrast and s2.saturation == d.saturation
    assert s2.style == "Floyd-Steinberg" and s2.scale == d.scale


def test_non_mapping_preset_raises():
    with pytest.raises(ValueError):
        preset_to_settings("just a string")


def test_palette_and_effects_roundtrip():
    from ditherzam.color.palette import Palette
    from ditherzam.effects.stack import EffectStack
    pal = Palette(name="mini", colors=np.array([[0, 0, 0], [255, 255, 255]], np.float32))
    stack = EffectStack()
    stack.add("Blur", radius=2)
    stack.add("Sharpen", amount=1)
    d = settings_to_preset(RenderSettings(), palette=pal, effect_stack=stack, color_mode="nearest")
    assert d["color"]["mode"] == "nearest"
    assert d["color"]["palette"]["name"] == "mini"
    assert d["color"]["palette"]["colors"] == [[0, 0, 0], [255, 255, 255]]
    assert d["effects"] == [{"name": "Blur", "params": {"radius": 2}},
                            {"name": "Sharpen", "params": {"amount": 1}}]
    s2, pal2, fx = preset_to_settings(d)
    assert pal2 is not None and pal2.name == "mini"
    assert pal2.colors.shape == (2, 3) and pal2.colors.dtype == np.float32
    assert fx == [("Blur", {"radius": 2}), ("Sharpen", {"amount": 1})]
```

- [ ] **Step 2: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_presets.py -v`
  → expect **FAIL** (`ModuleNotFoundError: No module named 'ditherzam.presets'`).

- [ ] **Step 3: Implement** — `ditherzam/presets.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .render import RenderSettings
from .color.palette import Palette

# Allowed ranges used to clamp presets on load (spec §10.2).
_ADJ_RANGE: dict[str, tuple[int, int]] = {
    "contrast": (0, 100),
    "midtones": (0, 100),
    "highlights": (0, 100),
    "luminance_threshold": (0, 100),
    "blur": (0, 100),
    "saturation": (0, 100),
}
_SCALE_RANGE: tuple[int, int] = (1, 20)


def _clamp_int(value, lo: int, hi: int) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        v = lo
    return max(lo, min(hi, v))


def settings_to_preset(settings: RenderSettings, palette: Palette | None = None,
                       effect_stack=None, color_mode: str = "off") -> dict:
    """Serialize a RenderSettings (+ optional palette/effect stack) to a preset dict."""
    preset: dict = {
        "adjustments": {
            "contrast": int(settings.contrast),
            "midtones": int(settings.midtones),
            "highlights": int(settings.highlights),
            "luminance_threshold": int(settings.luminance_threshold),
            "blur": int(settings.blur),
            "saturation": int(settings.saturation),
            "invert": bool(settings.invert),
        },
        "dither": {
            "style": str(settings.style),
            "scale": int(settings.scale),
            "preview_disabled": bool(settings.preview_disabled),
            "params": dict(settings.params),
        },
    }
    if palette is not None:
        preset["color"] = {
            "mode": str(color_mode),
            "palette": {
                "name": str(palette.name),
                "colors": np.asarray(palette.colors, dtype=np.float32)
                            .round().astype(int).reshape(-1, 3).tolist(),
            },
        }
    if effect_stack is not None:
        preset["effects"] = [
            {"name": str(name), "params": dict(params)}
            for name, params in effect_stack.items
        ]
    return preset


def preset_to_settings(preset: dict) -> tuple[RenderSettings, Palette | None, list[tuple[str, dict]]]:
    """Deserialize a preset dict into RenderSettings, clamping every value to range."""
    if not isinstance(preset, dict):
        raise ValueError("Not a valid preset file.")
    adj = preset.get("adjustments") or {}
    dit = preset.get("dither") or {}
    if not isinstance(adj, dict) or not isinstance(dit, dict):
        raise ValueError("Not a valid preset file.")

    defaults = RenderSettings()

    def adj_val(key: str) -> int:
        lo, hi = _ADJ_RANGE[key]
        return _clamp_int(adj.get(key, getattr(defaults, key)), lo, hi)

    settings = RenderSettings(
        contrast=adj_val("contrast"),
        midtones=adj_val("midtones"),
        highlights=adj_val("highlights"),
        luminance_threshold=adj_val("luminance_threshold"),
        blur=adj_val("blur"),
        saturation=adj_val("saturation"),
        invert=bool(adj.get("invert", defaults.invert)),
        style=str(dit.get("style", defaults.style)),
        scale=_clamp_int(dit.get("scale", defaults.scale), *_SCALE_RANGE),
        preview_disabled=bool(dit.get("preview_disabled", defaults.preview_disabled)),
        params=dict(dit.get("params", {}) or {}),
    )

    palette: Palette | None = None
    color = preset.get("color")
    if isinstance(color, dict) and isinstance(color.get("palette"), dict):
        pdata = color["palette"]
        colors = np.asarray(pdata.get("colors", []), dtype=np.float32)
        if colors.size:
            colors = colors.reshape(-1, 3)
        else:
            colors = colors.reshape(0, 3)
        palette = Palette(name=str(pdata.get("name", "preset")), colors=colors)

    effects: list[tuple[str, dict]] = []
    for item in preset.get("effects", []) or []:
        if isinstance(item, dict) and "name" in item:
            effects.append((str(item["name"]), dict(item.get("params", {}) or {})))

    return settings, palette, effects
```

- [ ] **Step 4: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_presets.py -v`
  → expect **PASS** (`6 passed` including the 6.0 sanity test).
- [ ] **Step 5: Commit**
  `git add ditherzam/presets.py tests/test_presets.py && git commit -m "feat(presets): settings<->preset (de)serialization with range clamping"`

---

### Task 6.2: PresetManager (save / list / load / delete / import)

**Files:** Modify `ditherzam/presets.py`; extend Test `tests/test_presets.py`
**Interfaces:**
```python
class PresetManager:
    def __init__(self, presets_dir): ...
    def save(self, name, preset: dict) -> Path
    def load(self, name) -> dict
    def list(self) -> list[str]
    def delete(self, name) -> bool
    def import_file(self, src) -> str        # returns imported name; raises ValueError on invalid
```

- [ ] **Step 1: Write failing test** — append to `tests/test_presets.py`

```python
from ditherzam.presets import PresetManager


def test_preset_manager_save_list_load(tmp_path):
    m = PresetManager(tmp_path)
    p = m.save("mine", {"adjustments": {}, "dither": {"style": "None"}})
    assert p.exists() and p.suffix == ".yaml"
    assert "mine" in m.list()
    assert m.load("mine")["dither"]["style"] == "None"


def test_preset_manager_list_is_sorted(tmp_path):
    m = PresetManager(tmp_path)
    m.save("zeta", {"dither": {}})
    m.save("alpha", {"dither": {}})
    assert m.list() == ["alpha", "zeta"]


def test_preset_manager_delete(tmp_path):
    m = PresetManager(tmp_path)
    m.save("temp", {"dither": {}})
    assert m.delete("temp") is True
    assert "temp" not in m.list()
    assert m.delete("temp") is False          # already gone


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PresetManager(tmp_path).load("nope")


def test_import_valid_yaml_returns_name(tmp_path):
    src = tmp_path / "cool.yaml"
    src.write_text("dither:\n  style: Atkinson\n", encoding="utf-8")
    m = PresetManager(tmp_path / "store")
    name = m.import_file(src)
    assert name == "cool"
    assert m.load("cool")["dither"]["style"] == "Atkinson"


def test_import_invalid_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("just a string", encoding="utf-8")
    with pytest.raises(ValueError):
        PresetManager(tmp_path / "store2").import_file(bad)


def test_import_wrong_extension_raises(tmp_path):
    bad = tmp_path / "notpreset.txt"
    bad.write_text("dither: {}", encoding="utf-8")
    with pytest.raises(ValueError):
        PresetManager(tmp_path / "store3").import_file(bad)
```

- [ ] **Step 2: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_presets.py -k preset_manager or import or load_missing -v`
  → expect **FAIL** (`ImportError: cannot import name 'PresetManager'`).

- [ ] **Step 3: Implement** — append to `ditherzam/presets.py`

```python
class PresetManager:
    """Filesystem-backed store of preset YAML files (spec §10)."""

    def __init__(self, presets_dir) -> None:
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.presets_dir / f"{name}.yaml"

    def save(self, name: str, preset: dict) -> Path:
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(preset, f, sort_keys=False, allow_unicode=True)
        return path

    def load(self, name: str) -> dict:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Preset not found: {name}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Not a valid preset file.")
        return data

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.presets_dir.glob("*.yaml"))

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if path.is_file():
            path.unlink()
            return True
        return False

    def import_file(self, src) -> str:
        src = Path(src)
        if src.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Not a valid preset file.")
        try:
            data = yaml.safe_load(src.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ValueError("Not a valid preset file.") from e
        if not isinstance(data, dict):
            raise ValueError("Not a valid preset file.")
        name = src.stem
        self.save(name, data)
        return name
```

- [ ] **Step 4: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_presets.py -v`
  → expect **PASS** (`13 passed`).
- [ ] **Step 5: Commit**
  `git add ditherzam/presets.py tests/test_presets.py && git commit -m "feat(presets): PresetManager save/list/load/delete/import"`

---

### Task 6.3: Raster export (PNG / JPG)

**Files:** Create `ditherzam/export/__init__.py`, `ditherzam/export/raster.py`; Test `tests/test_export_raster.py`
**Interfaces:**
```python
# Produces (Qt-free, pure):
def save_raster(rgb_u8: np.ndarray, path) -> Path   # format chosen by file extension
```

- [ ] **Step 1: Write failing test** — `tests/test_export_raster.py`

```python
import numpy as np
from PIL import Image
from ditherzam.export.raster import save_raster


def test_png_round_trips_exactly(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = save_raster(a, tmp_path / "x.png")
    assert out.exists()
    back = np.array(Image.open(out).convert("RGB"))
    assert back.shape == (8, 8, 3)
    np.testing.assert_array_equal(back, a)          # PNG is lossless


def test_jpg_matches_shape(tmp_path):
    a = np.full((8, 8, 3), 120, dtype=np.uint8)
    out = save_raster(a, tmp_path / "x.jpg")
    assert out.exists()
    back = np.array(Image.open(out).convert("RGB"))
    assert back.shape == (8, 8, 3)                   # JPEG is lossy, shape stable


def test_jpeg_extension_also_works(tmp_path):
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    out = save_raster(a, tmp_path / "y.jpeg")
    assert out.exists() and out.suffix == ".jpeg"


def test_accepts_non_uint8_input(tmp_path):
    a = np.full((4, 4, 3), 200.0, dtype=np.float32)
    out = save_raster(a, tmp_path / "z.png")
    back = np.array(Image.open(out).convert("RGB"))
    assert back.dtype == np.uint8 and int(back[0, 0, 0]) == 200
```

- [ ] **Step 2: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_export_raster.py -v`
  → expect **FAIL** (`ModuleNotFoundError: No module named 'ditherzam.export'`).

- [ ] **Step 3: Implement**

Create `ditherzam/export/__init__.py`:

```python
"""Image/vector export functions (pure, Qt-free)."""
```

Create `ditherzam/export/raster.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_raster(rgb_u8: np.ndarray, path) -> Path:
    """Save an HxWx3 (or HxW grayscale) uint8 array as PNG/JPG by file extension."""
    path = Path(path)
    arr = np.clip(np.asarray(rgb_u8), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        img.convert("RGB").save(path, "JPEG", quality=95)
    else:
        img.save(path)
    return path
```

- [ ] **Step 4: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_export_raster.py -v`
  → expect **PASS** (`4 passed`).
- [ ] **Step 5: Commit**
  `git add ditherzam/export/__init__.py ditherzam/export/raster.py tests/test_export_raster.py && git commit -m "feat(export): PNG/JPG raster export"`

---

### Task 6.4: Vector export — optimized run-merged SVG

**Files:** Create `ditherzam/export/vector.py`; Test `tests/test_export_vector.py`
**Interfaces:**
```python
# Produces (Qt-free, pure):
def raster_to_svg(gray_u8: np.ndarray, threshold: int, invert: bool = False) -> str
```
Algorithm (spec §11.3): pixels **< threshold** are "filled". Background rect first;
then for each **column**, consecutive filled pixels are merged into a single
`<rect x y width="1" height="runlen">`. `invert=False` → white bg / black fills;
`invert=True` → black bg / white fills.

- [ ] **Step 1: Write failing test** — `tests/test_export_vector.py`

```python
import numpy as np
import pytest
from ditherzam.export.vector import raster_to_svg


def test_svg_header_and_size():
    a = np.array([[0, 255], [0, 255]], np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.startswith("<svg")
    assert 'width="2"' in svg and 'height="2"' in svg
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert svg.rstrip().endswith("</svg>")


def test_background_colors_and_invert():
    a = np.array([[0, 255]], np.uint8)
    normal = raster_to_svg(a, threshold=128, invert=False)
    assert 'fill="#ffffff"' in normal            # white background
    assert 'fill="#000000"' in normal            # black filled rect
    inverted = raster_to_svg(a, threshold=128, invert=True)
    assert '<rect width="100%" height="100%" fill="#000000"/>' in inverted
    assert 'fill="#ffffff"' in inverted          # white filled rect


def test_vertical_run_merged_into_one_rect():
    a = np.zeros((4, 1), np.uint8)               # one filled column, 4 tall
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 2               # background + one merged run
    assert 'height="4"' in svg                    # the whole column is one rect


def test_gap_splits_column_into_two_runs():
    # column pattern filled/empty/filled/filled -> two separate runs, not merged across the gap
    a = np.array([[0], [255], [0], [0]], np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 3               # background + 2 runs
    assert 'y="0" width="1" height="1"' in svg   # first run: single pixel at top
    assert 'y="2" width="1" height="2"' in svg   # second run: 2 pixels merged


def test_runs_not_merged_across_columns():
    # two adjacent full columns must stay as two rects (horizontal merge is NOT done)
    a = np.zeros((3, 2), np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 3               # background + one rect per column
    assert 'x="0" y="0" width="1" height="3"' in svg
    assert 'x="1" y="0" width="1" height="3"' in svg


def test_empty_image_has_only_background():
    a = np.full((3, 3), 255, np.uint8)           # nothing below threshold
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 1               # background only


def test_threshold_boundary_is_exclusive():
    # value == threshold is NOT filled (strictly < threshold)
    a = np.array([[128]], np.uint8)
    svg = raster_to_svg(a, threshold=128, invert=False)
    assert svg.count("<rect") == 1               # background only


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        raster_to_svg(np.zeros((2, 2, 3), np.uint8), threshold=128)
```

- [ ] **Step 2: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_export_vector.py -v`
  → expect **FAIL** (`ModuleNotFoundError: No module named 'ditherzam.export.vector'`).

- [ ] **Step 3: Implement** — `ditherzam/export/vector.py`

```python
from __future__ import annotations

import numpy as np

_WHITE = "#ffffff"
_BLACK = "#000000"


def raster_to_svg(gray_u8: np.ndarray, threshold: int, invert: bool = False) -> str:
    """Convert a 2-D grayscale array to an optimized SVG.

    Pixels strictly below ``threshold`` are "filled". Vertical runs of adjacent
    filled pixels in the same column are merged into a single <rect> to keep the
    element count low (spec §11.3). No external dependencies.
    """
    arr = np.asarray(gray_u8)
    if arr.ndim != 2:
        raise ValueError("raster_to_svg expects a 2-D grayscale array")
    h, w = arr.shape

    bg = _BLACK if invert else _WHITE
    fg = _WHITE if invert else _BLACK
    filled = arr < threshold

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" version="1.1">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
    ]
    for x in range(w):
        col = filled[:, x]
        y = 0
        while y < h:
            if col[y]:
                start = y
                while y < h and col[y]:
                    y += 1
                run = y - start
                parts.append(
                    f'<rect x="{x}" y="{start}" width="1" height="{run}" fill="{fg}"/>'
                )
            else:
                y += 1
    parts.append("</svg>")
    return "\n".join(parts)
```

- [ ] **Step 4: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_export_vector.py -v`
  → expect **PASS** (`8 passed`).
- [ ] **Step 5: Commit**
  `git add ditherzam/export/vector.py tests/test_export_vector.py && git commit -m "feat(export): optimized run-merged SVG vector export"`

---

### Task 6.5: Batch processing (headless)

**Files:** Create `ditherzam/batch.py`; Test `tests/test_batch.py`
**Interfaces:**
```python
# Produces (Qt-free):
def batch_process(folder, out_folder, settings, pipeline, ref_size) -> tuple[int, int]
# returns (processed, skipped). Only images whose (w,h) == ref_size are processed;
# each output is saved as PNG in out_folder under the source stem.
```

- [ ] **Step 1: Write failing test** — `tests/test_batch.py`

```python
import numpy as np
from PIL import Image
from ditherzam.batch import batch_process
from ditherzam.render import RenderSettings, RenderPipeline
from ditherzam.dithering import registry


def _write_png(path, size_wh, value=127):
    w, h = size_wh
    Image.fromarray(np.full((h, w, 3), value, np.uint8)).save(path)


def test_batch_processes_matching_and_skips_others(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _write_png(src / "a.png", (8, 8))
    _write_png(src / "b.png", (8, 8))
    _write_png(src / "c.png", (16, 16))          # different size -> skipped
    (src / "notes.txt").write_text("ignore me")  # non-image -> ignored

    out = tmp_path / "out"
    pipeline = RenderPipeline(registry)
    settings = RenderSettings(style="None")

    processed, skipped = batch_process(src, out, settings, pipeline, (8, 8))
    assert (processed, skipped) == (2, 1)
    assert (out / "a.png").exists() and (out / "b.png").exists()
    assert not (out / "c.png").exists()


def test_batch_creates_output_folder(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _write_png(src / "only.png", (8, 8))
    out = tmp_path / "made" / "here"
    pipeline = RenderPipeline(registry)
    processed, skipped = batch_process(src, out, RenderSettings(style="None"),
                                       pipeline, (8, 8))
    assert out.is_dir() and processed == 1 and skipped == 0
    result = np.array(Image.open(out / "only.png").convert("RGB"))
    assert result.shape == (8, 8, 3)
```

- [ ] **Step 2: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_batch.py -v`
  → expect **FAIL** (`ModuleNotFoundError: No module named 'ditherzam.batch'`).

- [ ] **Step 3: Implement** — `ditherzam/batch.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import to_gray_f32
from .export.raster import save_raster

_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def batch_process(folder, out_folder, settings, pipeline, ref_size) -> tuple[int, int]:
    """Render every image in ``folder`` whose (w,h) equals ``ref_size``.

    Returns ``(processed, skipped)``. Outputs are written as PNG into
    ``out_folder`` using each source file's stem. Non-image files are ignored;
    images with a mismatched size are counted as skipped.
    """
    folder = Path(folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for src in sorted(folder.iterdir()):
        if not src.is_file() or src.suffix.lower() not in _EXTS:
            continue
        with Image.open(src) as im:
            if im.size != tuple(ref_size):
                skipped += 1
                continue
            gray = to_gray_f32(im)
        result = pipeline.render(gray, settings)
        save_raster(np.asarray(result, dtype=np.uint8), out_folder / f"{src.stem}.png")
        processed += 1
    return processed, skipped
```

- [ ] **Step 4: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_batch.py -v`
  → expect **PASS** (`2 passed`).
- [ ] **Step 5: Commit**
  `git add ditherzam/batch.py tests/test_batch.py && git commit -m "feat(batch): folder batch processing with size gating"`

---

### Task 6.6: Wire presets/export into the UI (Qt — ui layer only)

**Files:** Create `ditherzam/ui/export_actions.py`; Modify `ditherzam/ui/main_window.py`;
Test `tests/test_ui_export_actions.py`
**Interfaces:**
```python
# ditherzam/ui/export_actions.py  (the ONLY PySide6 import in this phase)
MENU_SPEC: list[tuple[str, str, str | None]]     # (key, label, shortcut)
def create_export_menu(menubar, handlers: dict[str, Callable]) -> tuple[QMenu, dict[str, QAction]]
```
The pure preset/export/batch functions from Tasks 6.1–6.5 do all the work; this
task only builds the menu and connects each `QAction.triggered` to a handler
callable. Actions covered (keys): `save_preset`, `load_preset`, `import_preset`,
`export_preset`, `export_png`, `export_jpg`, `export_svg`, `batch_folder`.

- [ ] **Step 1: Write failing test** — `tests/test_ui_export_actions.py`

```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMainWindow
from ditherzam.ui.export_actions import create_export_menu, MENU_SPEC

_app = QApplication.instance() or QApplication([])


def test_menu_spec_has_all_actions():
    keys = [k for k, _label, _sc in MENU_SPEC]
    for expected in ("save_preset", "load_preset", "import_preset", "export_preset",
                     "export_png", "export_jpg", "export_svg", "batch_folder"):
        assert expected in keys


def test_create_menu_builds_actions_with_labels():
    win = QMainWindow()
    calls = []
    handlers = {k: (lambda key=k: calls.append(key)) for k, _l, _s in MENU_SPEC}
    menu, actions = create_export_menu(win.menuBar(), handlers)
    assert menu.title() == "&Export"
    assert set(actions.keys()) == {k for k, _l, _s in MENU_SPEC}
    for key, label, _sc in MENU_SPEC:
        assert actions[key].text() == label


def test_triggering_action_calls_handler():
    win = QMainWindow()
    calls = []
    handlers = {k: (lambda key=k: calls.append(key)) for k, _l, _s in MENU_SPEC}
    _menu, actions = create_export_menu(win.menuBar(), handlers)
    actions["export_svg"].trigger()
    actions["save_preset"].trigger()
    assert calls == ["export_svg", "save_preset"]


def test_missing_handler_action_is_disabled():
    win = QMainWindow()
    _menu, actions = create_export_menu(win.menuBar(), {"export_png": lambda: None})
    assert actions["export_png"].isEnabled() is True
    assert actions["export_svg"].isEnabled() is False
```

- [ ] **Step 2: Run** — `NUMBA_DISABLE_JIT=1 pytest tests/test_ui_export_actions.py -v`
  → expect **FAIL** (`ModuleNotFoundError: No module named 'ditherzam.ui.export_actions'`).

- [ ] **Step 3: Implement**

Create `ditherzam/ui/export_actions.py`:

```python
"""Preset/export menu builder. This is the only PySide6 import in Phase 6.

The pure functions in ``ditherzam.presets``, ``ditherzam.export`` and
``ditherzam.batch`` do all the real work; this module only wires QActions.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QAction, QKeySequence

# (key, menu label, keyboard shortcut or None)
MENU_SPEC: list[tuple[str, str, str | None]] = [
    ("save_preset",   "Save Preset...",   None),
    ("load_preset",   "Load Preset...",   None),
    ("import_preset", "Import Preset(s)...", None),
    ("export_preset", "Export Preset...", None),
    ("export_png",    "Export PNG...",    "Ctrl+Shift+S"),
    ("export_jpg",    "Export JPG...",    None),
    ("export_svg",    "Export as Vector (SVG)...", None),
    ("batch_folder",  "Batch — Select Folder...", None),
]


def create_export_menu(menubar, handlers: dict[str, Callable]):
    """Add an "&Export" menu to ``menubar`` and connect each action to a handler.

    Actions without a handler are created disabled so the menu still lists them.
    Returns ``(menu, {key: QAction})``.
    """
    menu = menubar.addMenu("&Export")
    actions: dict[str, QAction] = {}
    for key, label, shortcut in MENU_SPEC:
        action = QAction(label, menu)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        handler = handlers.get(key)
        if handler is None:
            action.setEnabled(False)
        else:
            action.triggered.connect(lambda _checked=False, h=handler: h())
        menu.addAction(action)
        actions[key] = action
    return menu, actions
```

Modify `ditherzam/ui/main_window.py` — add the import near the other UI imports:

```python
from .export_actions import create_export_menu
from ..presets import PresetManager, settings_to_preset, preset_to_settings
from ..export.raster import save_raster
from ..export.vector import raster_to_svg
from ..batch import batch_process
```

Then, inside `MainWindow.__init__` (after the menu bar and pipeline exist), add:

```python
        # --- Presets & export wiring (Phase 6) ---
        from platformdirs import user_data_dir
        from pathlib import Path
        self._preset_manager = PresetManager(Path(user_data_dir("ditherzam")) / "PRESETS")
        self._export_menu, self._export_actions = create_export_menu(
            self.menuBar(),
            {
                "save_preset":   self._on_save_preset,
                "load_preset":   self._on_load_preset,
                "import_preset": self._on_import_preset,
                "export_preset": self._on_export_preset,
                "export_png":    lambda: self._on_export_raster("PNG Files (*.png)", ".png"),
                "export_jpg":    lambda: self._on_export_raster("JPEG Files (*.jpg)", ".jpg"),
                "export_svg":    self._on_export_svg,
                "batch_folder":  self._on_batch_folder,
            },
        )
```

And add these handler methods to `MainWindow` (they rely on the Phase-5 hooks
`self._collect_settings()`, `self._current_palette()`, `self._current_effect_stack()`,
`self._color_mode()`, `self._rendered_rgb()`, `self._base_gray()`, `self._pipeline`,
`self._reference_size()`, and `self._apply_preset(settings, palette, effects)`):

```python
    def _on_save_preset(self):
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")
        if not ok or not name:
            return
        preset = settings_to_preset(
            self._collect_settings(), self._current_palette(),
            self._current_effect_stack(), self._color_mode(),
        )
        self._preset_manager.save(name, preset)
        QMessageBox.information(self, "Presets", f"Preset '{name}' saved successfully!")

    def _on_load_preset(self):
        from PySide6.QtWidgets import QInputDialog
        names = self._preset_manager.list()
        if not names:
            return
        name, ok = QInputDialog.getItem(self, "Load Preset", "Preset:", names, 0, False)
        if not ok:
            return
        settings, palette, effects = preset_to_settings(self._preset_manager.load(name))
        self._apply_preset(settings, palette, effects)

    def _on_import_preset(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(self, "Import Preset(s)", "",
                                              "Preset Files (*.yaml *.yml)")
        if not path:
            return
        try:
            name = self._preset_manager.import_file(path)
        except ValueError:
            QMessageBox.warning(self, "Presets", "Not a valid preset file.")
            return
        QMessageBox.information(self, "Presets", f"Imported preset '{name}'.")

    def _on_export_preset(self):
        from PySide6.QtWidgets import QFileDialog
        import yaml
        path, _ = QFileDialog.getSaveFileName(self, "Export Preset", "",
                                              "Preset Files (*.yaml)")
        if not path:
            return
        preset = settings_to_preset(
            self._collect_settings(), self._current_palette(),
            self._current_effect_stack(), self._color_mode(),
        )
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(preset, f, sort_keys=False, allow_unicode=True)

    def _on_export_raster(self, file_filter, ext):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", file_filter)
        if not path:
            return
        save_raster(self._rendered_rgb(), path)

    def _on_export_svg(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        cont = QMessageBox.warning(
            self, "Export as Vector",
            "WARNING: vector export is experimental. For best results use a larger "
            "scale and fewer fine details. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if cont != QMessageBox.Yes:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Vector", "",
                                              "SVG Files (*.svg)")
        if not path:
            return
        settings = self._collect_settings()
        gray = self._base_gray()
        threshold = int(settings.luminance_threshold / 100.0 * 255.0)
        import numpy as np
        svg = raster_to_svg(np.asarray(gray).astype("uint8"), threshold,
                            bool(settings.invert))
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)

    def _on_batch_folder(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        from pathlib import Path
        out = Path(folder) / "batch_processed"
        processed, skipped = batch_process(
            folder, out, self._collect_settings(),
            self._pipeline, self._reference_size(),
        )
        QMessageBox.information(
            self, "Batch",
            f"Processed {processed} images. Skipped {skipped} images.",
        )
```

- [ ] **Step 4: Run** — `NUMBA_DISABLE_JIT=1 QT_QPA_PLATFORM=offscreen pytest tests/test_ui_export_actions.py -v`
  → expect **PASS** (`4 passed`).
- [ ] **Step 5: Commit**
  `git add ditherzam/ui/export_actions.py ditherzam/ui/main_window.py tests/test_ui_export_actions.py && git commit -m "feat(ui): preset + export menu wiring"`

---

## Phase 6 — Full-suite gate

- [ ] Run `NUMBA_DISABLE_JIT=1 pytest tests/test_presets.py tests/test_export_raster.py tests/test_export_vector.py tests/test_batch.py tests/test_ui_export_actions.py -v`
  → expect **all pass** (`13 + 4 + 8 + 2 + 4 = 31 passed`).
- [ ] Run the whole suite `NUMBA_DISABLE_JIT=1 pytest -q` → expect green (no regressions).

---

## Subsystem Definition of Done  (checklist)

- [ ] `settings_to_preset` / `preset_to_settings` round-trip all `RenderSettings`
      fields; out-of-range values are **clamped** (`contrast/…/saturation` → 0..100,
      `scale` → 1..20); non-mapping presets raise `ValueError`.
- [ ] Palette and effect stack round-trip through the optional `color` / `effects`
      sections.
- [ ] `PresetManager` saves/lists (sorted)/loads/deletes; `import_file` returns the
      name for valid `*.yaml` and raises `ValueError` ("Not a valid preset file.")
      for a non-mapping body or wrong extension; `load` of a missing preset raises
      `FileNotFoundError`.
- [ ] `save_raster` writes PNG (lossless round-trip) and JPG/JPEG (shape stable),
      accepting non-uint8 input.
- [ ] `raster_to_svg` emits a correct header (`width`/`height`/`xmlns`), a background
      rect, and **vertical-run-merged** fills; gaps split runs; columns are not
      merged horizontally; `threshold` is exclusive (`< threshold`); `invert` swaps
      bg/fg; non-2-D input raises. **SVG run-merging is directly tested** (see
      `test_vertical_run_merged_into_one_rect`, `test_gap_splits_column_into_two_runs`,
      `test_runs_not_merged_across_columns`).
- [ ] `batch_process` returns `(processed, skipped)`, processes only size-matching
      images, writes PNG outputs, creates the output folder, and ignores non-images.
- [ ] UI menu builder is the only PySide6 import; actions exist, carry the right
      labels/shortcuts, disable when unhandled, and trigger their handlers; verified
      offscreen.
- [ ] **Core stays Qt-free:** `presets.py`, `export/*.py`, `batch.py` import no
      PySide6 (only `ditherzam/ui/export_actions.py` + `main_window.py` do).

---

## Self-Review

**Spec-coverage matrix (spec section → task):**

| Spec § | Requirement | Task |
|---|---|---|
| §10 / §10.1 | Preset YAML schema (adjustments/dither + color/effects) | 6.1 |
| §10.2 | Clamp every value to slider range on apply | 6.1 |
| §10 | `PresetManager` get_all / save / import + "Not a valid preset file." | 6.2 |
| §10.3 | Preset menu (Save/Load/Import/Export) UI | 6.6 |
| §11.2 / §17.7 | PNG (+ JPG) raster export | 6.3 |
| §11.3 | `raster_to_svg` optimized **vertical run-merge**, bg/fg, invert, header | 6.4 |
| §11.7 | Batch: process size-matching images, `batch_processed` folder, report | 6.5, 6.6 |
| §17.7 | Export formats PNG / JPG / SVG | 6.3, 6.4 |

**Placeholder scan:** no `TODO` / `pass`-body / "implement here" — every code step
is complete and runnable.

**Type consistency vs FROZEN CONTRACTS:** `RenderSettings` field names/defaults used
verbatim; `Palette(name, colors: float32[K,3])` reconstructed with `float32`;
`EffectStack.items` = `list[tuple[str, dict]]` matches serialize/deserialize;
`RenderPipeline.render(base_gray_f32, settings)` used positionally in batch.

**Clean-room:** no Dither Boy / Studio AAA code, strings, URLs, or binaries; SVG,
PNG/JPG and preset logic are original and dependency-light (NumPy/Pillow/PyYAML).

**Out-of-scope (documented deferrals):** Export-to-Photoshop (§11.4), chroma-drop
transparent PNG (§11.5), copy-to-clipboard (§11.6), and the "large vector file" size
warning (§11.3) are UI conveniences not required by this subsystem's DoD; add later
if desired.
