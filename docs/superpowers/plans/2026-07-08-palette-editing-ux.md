# Palette Editing UX (Sub-project B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make palettes editable and user-owned in the app — a user-owned palette store, in-app swatch add/remove/recolor, per-swatch lock + shuffle, and "From Image" generation.

**Architecture:** A new Qt-free `PaletteStore` provides per-palette user YAML files that shadow read-only builtins (the copy-on-edit fork mechanism). A new `SwatchStrip` Qt widget edits an in-memory working `Palette` and emits it on every change. `ControlPanel` owns the store + working palette and drives live re-render; `MainWindow` consumes the working palette for rendering and retains the source RGB image for From-Image extraction. No dithering/ramp math changes; the frozen render path is untouched.

**Tech Stack:** Python 3.12, NumPy, PySide6 (UI only), PyYAML, Numba (unaffected), pytest.

## Global Constraints

- **Clean-room** — no Dither Boy / Studio AAA code, strings, or binaries. Public algorithm techniques only.
- **Qt-free core** — `ditherzam/color/**` and `ditherzam/dithering/**` must not import PySide6. Only `ditherzam/ui/`, `ditherzam/app.py`, `ditherzam/video/workers.py` may import Qt.
- **Frozen render path** — do not change `RenderPipeline.render()` stage order or `RenderSettings` fields. Palette editing touches palette *provisioning*, not the pipeline.
- **Python 3.12; TDD** — red → green → refactor, commit after each green.
- **Test runner:** `./.venv/Scripts/python.exe -m pytest`. `conftest.py` sets `NUMBA_DISABLE_JIT=1` and `QT_QPA_PLATFORM=offscreen` by default. Qt tests take the `qapp_fixture` fixture.
- **Regression gate:** the full suite must stay green (currently **511** passing) in **both** JIT-on (`NUMBA_DISABLE_JIT=0`) and JIT-off modes. Run both only at the final task; per-task runs use the default (JIT-off) mode.
- **Palettes are RGB `float32[K,3]` in 0..255**; builtins live at `ditherzam/color/builtin/*.yaml`.

## File Structure

- **Create** `ditherzam/color/palette_store.py` — `PaletteStore`: user-owned palette repository over a config dir; builtin↔user shadowing; save/delete/reset. Qt-free.
- **Modify** `ditherzam/color/palette.py` — add pure `generate_palette(rgb_u8, unit, value, name)` dispatcher over the existing `extract_palette` / `source_palette`.
- **Create** `ditherzam/ui/palette_editor.py` — `SwatchStrip(QWidget)`: edits an in-memory working `Palette`, emits `edited(Palette)`.
- **Modify** `ditherzam/ui/controls.py` — own a `PaletteStore` + working palette; populate the palette combo from the store; embed `SwatchStrip`; add Save/Reset/Shuffle/From-Image controls; add `palette_autosave` and `extract_unit` state keys; drop the hardcoded `_PALETTES`.
- **Modify** `ditherzam/ui/main_window.py` — retain the source RGB image; `_current_palette()` returns the panel's working palette; handle From-Image requests; restore the working palette on preset load.
- **Create** tests: `tests/test_palette_store.py`, `tests/test_palette_editor.py`, `tests/test_controls_palette.py`, `tests/test_main_window_palette.py`; **extend** `tests/test_palette.py`.

---

## Task 1: PaletteStore (Qt-free user palette repository)

**Files:**
- Create: `ditherzam/color/palette_store.py`
- Test: `tests/test_palette_store.py`

**Interfaces:**
- Consumes: `ditherzam.color.palette.Palette`, `builtin_palettes()` (existing).
- Produces:
  - `class PaletteStore`
    - `__init__(self, user_dir: Path | None = None)` — `user_dir` defaults to the platform config dir (`%APPDATA%/ditherzam/palettes` on Windows, else `~/.config/ditherzam/palettes`). Directory is **not** created until `save`.
    - `list(self) -> list[str]` — sorted union of builtin names and user names.
    - `get(self, name: str) -> Palette` — user copy if a user YAML exists, else the builtin; always returns a fresh copy (copied `colors` array). Raises `KeyError` if neither exists.
    - `is_user(self, name: str) -> bool` — a user YAML exists for `name`.
    - `is_builtin(self, name: str) -> bool` — a builtin exists for `name`.
    - `save(self, palette: Palette) -> None` — writes `user_dir/<palette.name>.yaml` (creates dir lazily).
    - `delete(self, name: str) -> None` — removes the user YAML if present (no error if absent).
    - `reset_to_builtin(self, name: str) -> Palette` — deletes the user YAML and returns a copy of the builtin; raises `KeyError` if there is no builtin of that name.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_palette_store.py`:

```python
import numpy as np
import pytest

from ditherzam.color.palette import Palette
from ditherzam.color.palette_store import PaletteStore


def _store(tmp_path):
    return PaletteStore(user_dir=tmp_path / "palettes")


def test_list_includes_builtins(tmp_path):
    s = _store(tmp_path)
    names = s.list()
    for b in ("grayscale", "gameboy", "cga", "pico8", "sepia"):
        assert b in names


def test_get_builtin_returns_copy(tmp_path):
    s = _store(tmp_path)
    a = s.get("gameboy")
    b = s.get("gameboy")
    assert a.colors is not b.colors           # independent arrays
    a.colors[0, 0] = 123.0
    assert s.get("gameboy").colors[0, 0] != 123.0   # source not mutated


def test_get_unknown_raises(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(KeyError):
        s.get("no-such-palette")


def test_save_then_get_roundtrip(tmp_path):
    s = _store(tmp_path)
    p = Palette.from_list("mine", [[1, 2, 3], [4, 5, 6]])
    s.save(p)
    assert s.is_user("mine")
    got = s.get("mine")
    np.testing.assert_array_equal(got.colors, p.colors)
    assert "mine" in s.list()


def test_user_file_shadows_builtin(tmp_path):
    s = _store(tmp_path)
    fork = Palette.from_list("gameboy", [[0, 0, 0], [255, 255, 255]])
    s.save(fork)
    assert s.is_user("gameboy") and s.is_builtin("gameboy")
    np.testing.assert_array_equal(s.get("gameboy").colors, fork.colors)
    assert s.list().count("gameboy") == 1     # de-duplicated


def test_delete_reveals_builtin_again(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("gameboy", [[0, 0, 0], [255, 255, 255]]))
    s.delete("gameboy")
    assert not s.is_user("gameboy")
    assert s.get("gameboy").colors.shape[0] > 2   # original builtin restored


def test_reset_to_builtin_returns_builtin_and_drops_fork(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("gameboy", [[0, 0, 0], [255, 255, 255]]))
    restored = s.reset_to_builtin("gameboy")
    assert not s.is_user("gameboy")
    assert restored.colors.shape[0] > 2


def test_reset_to_builtin_without_builtin_raises(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("customonly", [[1, 1, 1]]))
    with pytest.raises(KeyError):
        s.reset_to_builtin("customonly")


def test_user_dir_not_created_until_save(tmp_path):
    d = tmp_path / "palettes"
    s = PaletteStore(user_dir=d)
    s.list()                     # read-only ops must not create the dir
    assert not d.exists()
    s.save(Palette.from_list("x", [[0, 0, 0]]))
    assert d.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.color.palette_store'`.

- [ ] **Step 3: Write the implementation**

Create `ditherzam/color/palette_store.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

from .palette import Palette, builtin_palettes


def _default_user_dir() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / "ditherzam" / "palettes"
    return Path.home() / ".config" / "ditherzam" / "palettes"


class PaletteStore:
    """User-owned palette repository. User YAML files shadow read-only builtins."""

    def __init__(self, user_dir: Path | None = None) -> None:
        self.user_dir = Path(user_dir) if user_dir is not None else _default_user_dir()

    # -- discovery ------------------------------------------------------------
    def _user_path(self, name: str) -> Path:
        return self.user_dir / f"{name}.yaml"

    def _user_names(self) -> list[str]:
        if not self.user_dir.is_dir():
            return []
        return [p.stem for p in self.user_dir.glob("*.yaml")]

    def is_user(self, name: str) -> bool:
        return self._user_path(name).is_file()

    def is_builtin(self, name: str) -> bool:
        return name in builtin_palettes()

    def list(self) -> list[str]:
        return sorted(set(builtin_palettes().keys()) | set(self._user_names()))

    # -- access ---------------------------------------------------------------
    def get(self, name: str) -> Palette:
        if self.is_user(name):
            p = Palette.load(self._user_path(name))
        else:
            builtins = builtin_palettes()
            if name not in builtins:
                raise KeyError(name)
            p = builtins[name]
        return Palette(name=p.name, colors=p.colors.copy())

    # -- mutation -------------------------------------------------------------
    def save(self, palette: Palette) -> None:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        palette.to_yaml(self._user_path(palette.name))

    def delete(self, name: str) -> None:
        path = self._user_path(name)
        if path.is_file():
            path.unlink()

    def reset_to_builtin(self, name: str) -> Palette:
        if name not in builtin_palettes():
            raise KeyError(name)
        self.delete(name)
        return self.get(name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_store.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/palette_store.py tests/test_palette_store.py
git commit -m "feat(color): PaletteStore user-owned palette repository with builtin shadowing"
```

---

## Task 2: `generate_palette` dispatcher (From-Image extraction, pure)

**Files:**
- Modify: `ditherzam/color/palette.py` (append a function)
- Test: `tests/test_palette.py` (append tests)

**Interfaces:**
- Consumes: existing `extract_palette(rgb_u8, k)` and `source_palette(rgb_u8, completeness)`.
- Produces:
  - `generate_palette(rgb_u8: np.ndarray, unit: str, value: int, name: str = "from image") -> Palette`
    - `unit == "k"`: exactly `value` colors via `extract_palette(rgb_u8, k=value)` (value clamped to `>= 1`).
    - `unit == "pct"`: `source_palette(rgb_u8, completeness=value / 100.0)`.
    - any other `unit`: raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_palette.py`:

```python
def test_generate_palette_k_gives_exact_count():
    from ditherzam.color.palette import generate_palette
    img = np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    p = generate_palette(img, "k", 8)
    assert p.colors.shape == (8, 3)
    assert p.name == "from image"


def test_generate_palette_pct_maps_to_source_palette():
    from ditherzam.color.palette import generate_palette, source_palette
    img = np.random.default_rng(1).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    got = generate_palette(img, "pct", 50)
    expect = source_palette(img, completeness=0.5)
    assert got.colors.shape == expect.colors.shape


def test_generate_palette_bad_unit_raises():
    from ditherzam.color.palette import generate_palette
    img = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        generate_palette(img, "nonsense", 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette.py -k generate_palette -q`
Expected: FAIL — `ImportError: cannot import name 'generate_palette'`.

- [ ] **Step 3: Write the implementation**

Append to `ditherzam/color/palette.py`:

```python
def generate_palette(rgb_u8: np.ndarray, unit: str, value: int,
                     name: str = "from image") -> "Palette":
    """Extract a palette from an image. ``unit`` is 'k' (exact colors) or 'pct'."""
    if unit == "k":
        return extract_palette(rgb_u8, k=max(1, int(value)), name=name)
    if unit == "pct":
        return source_palette(rgb_u8, completeness=float(value) / 100.0, name=name)
    raise ValueError(f"unknown unit: {unit!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette.py -k generate_palette -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/palette.py tests/test_palette.py
git commit -m "feat(color): generate_palette dispatcher for From-Image extraction"
```

---

## Task 3: SwatchStrip widget (edit a working palette)

**Files:**
- Create: `ditherzam/ui/palette_editor.py`
- Test: `tests/test_palette_editor.py`

**Interfaces:**
- Consumes: `ditherzam.color.palette.Palette`; PySide6.
- Produces:
  - `class SwatchStrip(QWidget)` with signal `edited = Signal(object)` (emits the working `Palette` after every mutation).
    - `set_palette(self, palette: Palette) -> None` — adopts a **copy** as the working palette; rebuilds cells; resets locks; does **not** emit `edited`.
    - `palette(self) -> Palette` — the current working palette.
    - `set_swatch_color(self, i: int, rgb) -> None` — set row `i` to `rgb` (len-3, 0..255); emits `edited`. (The click handler calls this after `QColorDialog`; tests call it directly.)
    - `add_swatch(self) -> None` — append a copy of the last color; emits `edited`.
    - `remove_swatch(self, i: int) -> None` — remove row `i` (no-op if it would drop below 1 swatch); reindexes locks; emits `edited`.
    - `toggle_lock(self, i: int) -> None` — flip lock on row `i`.
    - `locked(self) -> set[int]` — currently locked indices.
    - `shuffle(self, rng=None) -> None` — `Palette.shuffle(self.locked(), rng or np.random.default_rng())`; adopts the result; emits `edited`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_palette_editor.py`:

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")

from ditherzam.color.palette import Palette


def _strip():
    from ditherzam.ui.palette_editor import SwatchStrip
    s = SwatchStrip()
    s.set_palette(Palette.from_list("t", [[0, 0, 0], [128, 128, 128], [255, 255, 255]]))
    return s


def test_set_palette_adopts_copy(qapp_fixture):
    p = Palette.from_list("t", [[10, 20, 30], [40, 50, 60]])
    from ditherzam.ui.palette_editor import SwatchStrip
    s = SwatchStrip()
    s.set_palette(p)
    p.colors[0, 0] = 200
    assert s.palette().colors[0, 0] == 10          # independent from the source


def test_recolor_updates_and_emits(qapp_fixture):
    s = _strip()
    seen = []
    s.edited.connect(lambda pal: seen.append(pal))
    s.set_swatch_color(1, (10, 20, 30))
    np.testing.assert_array_equal(s.palette().colors[1], [10, 20, 30])
    assert len(seen) == 1


def test_add_swatch_appends(qapp_fixture):
    s = _strip()
    s.add_swatch()
    assert s.palette().colors.shape[0] == 4


def test_remove_swatch_respects_minimum(qapp_fixture):
    s = _strip()
    s.remove_swatch(0)
    s.remove_swatch(0)
    assert s.palette().colors.shape[0] == 1
    s.remove_swatch(0)                              # would drop to 0 -> no-op
    assert s.palette().colors.shape[0] == 1


def test_locked_indices_survive_shuffle(qapp_fixture):
    s = _strip()
    s.toggle_lock(1)
    assert s.locked() == {1}
    before = s.palette().colors[1].copy()
    s.shuffle(rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s.palette().colors[1], before)


def test_remove_reindexes_locks(qapp_fixture):
    s = _strip()
    s.toggle_lock(2)
    s.remove_swatch(0)                              # index 2 -> now index 1
    assert s.locked() == {1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_editor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ditherzam.ui.palette_editor'`.

- [ ] **Step 3: Write the implementation**

Create `ditherzam/ui/palette_editor.py`:

```python
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from ..color.palette import Palette

_MIN_SWATCHES = 1


class SwatchStrip(QWidget):
    """Editable row of palette swatches over an in-memory working Palette."""

    edited = Signal(object)   # emits the working Palette

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette = Palette.from_list("empty", [[0, 0, 0]])
        self._locked: set[int] = set()
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(2)
        self._buttons: list[QPushButton] = []
        self._rebuild()

    # -- public API -----------------------------------------------------------
    def set_palette(self, palette: Palette) -> None:
        self._palette = Palette(name=palette.name, colors=palette.colors.copy())
        self._locked = set()
        self._rebuild()

    def palette(self) -> Palette:
        return self._palette

    def locked(self) -> set[int]:
        return set(self._locked)

    def set_swatch_color(self, i: int, rgb) -> None:
        self._palette.colors[i] = np.asarray(rgb, dtype=np.float32)
        self._rebuild()
        self.edited.emit(self._palette)

    def add_swatch(self) -> None:
        last = self._palette.colors[-1:].copy()
        self._palette = Palette(
            name=self._palette.name,
            colors=np.vstack([self._palette.colors, last]).astype(np.float32),
        )
        self._rebuild()
        self.edited.emit(self._palette)

    def remove_swatch(self, i: int) -> None:
        if self._palette.colors.shape[0] <= _MIN_SWATCHES:
            return
        self._palette = Palette(
            name=self._palette.name,
            colors=np.delete(self._palette.colors, i, axis=0).astype(np.float32),
        )
        self._locked = {j - 1 if j > i else j for j in self._locked if j != i}
        self._rebuild()
        self.edited.emit(self._palette)

    def toggle_lock(self, i: int) -> None:
        if i in self._locked:
            self._locked.discard(i)
        else:
            self._locked.add(i)
        self._rebuild()

    def shuffle(self, rng=None) -> None:
        rng = rng if rng is not None else np.random.default_rng()
        self._palette = self._palette.shuffle(self._locked, rng)
        self._rebuild()
        self.edited.emit(self._palette)

    # -- rendering ------------------------------------------------------------
    def _rebuild(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._buttons = []
        for i in range(self._palette.colors.shape[0]):
            r, g, b = (int(round(c)) for c in self._palette.colors[i])
            btn = QPushButton()
            btn.setFixedSize(22, 22)
            border = "2px solid #f0d000" if i in self._locked else "1px solid #333"
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: {border};")
            btn.clicked.connect(lambda _=False, idx=i: self._pick_color(idx))
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda _pos, idx=i: self.remove_swatch(idx))
            self._row.addWidget(btn)
            self._buttons.append(btn)
        add = QPushButton("+")
        add.setFixedSize(22, 22)
        add.clicked.connect(lambda _=False: self.add_swatch())
        self._row.addWidget(add)

    def _pick_color(self, i: int) -> None:
        c = self._palette.colors[i]
        initial = QColor(int(c[0]), int(c[1]), int(c[2]))
        chosen = QColorDialog.getColor(initial, self, "Pick swatch color")
        if chosen.isValid():
            self.set_swatch_color(i, (chosen.red(), chosen.green(), chosen.blue()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_editor.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/palette_editor.py tests/test_palette_editor.py
git commit -m "feat(ui): SwatchStrip editable palette widget (recolor/add/remove/lock/shuffle)"
```

---

## Task 4: Wire the store + SwatchStrip into ControlPanel

**Files:**
- Modify: `ditherzam/ui/controls.py`
- Test: `tests/test_controls_palette.py`

**Interfaces:**
- Consumes: `PaletteStore` (Task 1), `SwatchStrip` (Task 3).
- Produces (new on `ControlPanel`):
  - `__init__(self, parent=None, store: PaletteStore | None = None)` — `store` defaults to a fresh `PaletteStore()`.
  - `self.store: PaletteStore`, `self.working_palette: Palette`, `self.swatch_strip: SwatchStrip`.
  - `self.palette_combo` populated from `store.list()`.
  - New `state` keys: `"palette_autosave": False`, `"extract_unit": "k"`.
  - `set_working_palette(self, palette: Palette) -> None` — adopt `palette` as working, push to the strip, emit `changed` (used by MainWindow after From-Image / preset load).
  - Signal `from_image_requested = Signal()` — emitted by the "From Image" button.
  - `self.extract_slider` — a `ResettableGlowSlider` for the From-Image count.

**Notes for the implementer:**
- Remove the module-level `_PALETTES` list entirely.
- The working palette lives on the panel object, **not** in the plain `state` dict (it is a `Palette`, not a scalar). The scalar `state["palette"]` still holds the selected name for settings/preset naming.
- Repopulating the combo after Save must **block signals** so it doesn't fire `_on_palette_changed` and clobber the working copy.
- `palette_autosave` / `extract_unit` are UI-session prefs; they are NOT added to `RenderSettings` (frozen). They only live in `state`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_controls_palette.py`:

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")

from ditherzam.color.palette import Palette
from ditherzam.color.palette_store import PaletteStore


def _panel(tmp_path):
    from ditherzam.ui.controls import ControlPanel
    return ControlPanel(store=PaletteStore(user_dir=tmp_path / "pal"))


def test_combo_populated_from_store(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    items = [panel.palette_combo.itemText(i) for i in range(panel.palette_combo.count())]
    assert "gameboy" in items and "pico8" in items


def test_new_state_defaults(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    assert panel.state["palette_autosave"] is False
    assert panel.state["extract_unit"] == "k"


def test_selecting_palette_sets_working_copy(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    assert panel.working_palette.name == "gameboy"
    assert panel.swatch_strip.palette().name == "gameboy"


def test_swatch_edit_updates_working_and_emits_changed(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.swatch_strip.set_swatch_color(0, (7, 8, 9))
    np.testing.assert_array_equal(panel.working_palette.colors[0], [7, 8, 9])
    assert seen                                  # changed fired


def test_save_palette_persists_and_refreshes_combo(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    panel.swatch_strip.set_swatch_color(0, (1, 2, 3))
    panel._on_save_palette()
    assert panel.store.is_user("gameboy")
    # combo still holds a single "gameboy" entry, still selected
    items = [panel.palette_combo.itemText(i) for i in range(panel.palette_combo.count())]
    assert items.count("gameboy") == 1
    assert panel.palette_combo.currentText() == "gameboy"


def test_autosave_writes_on_edit_when_enabled(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    panel.state["palette_autosave"] = True
    panel.swatch_strip.set_swatch_color(0, (1, 2, 3))
    assert panel.store.is_user("gameboy")


def test_reset_to_builtin_drops_fork(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_combo.setCurrentText("gameboy")
    panel.swatch_strip.set_swatch_color(0, (1, 2, 3))
    panel._on_save_palette()
    panel._on_reset_palette()
    assert not panel.store.is_user("gameboy")


def test_set_working_palette_pushes_to_strip(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    seen = []
    panel.changed.connect(lambda: seen.append(1))
    panel.set_working_palette(Palette.from_list("from image", [[9, 9, 9], [1, 1, 1]]))
    assert panel.working_palette.name == "from image"
    assert panel.swatch_strip.palette().colors.shape[0] == 2
    assert seen
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_controls_palette.py -q`
Expected: FAIL — `ControlPanel()` has no `store`/`working_palette`/`swatch_strip`/`_on_save_palette`.

- [ ] **Step 3: Edit `controls.py`**

3a. Update imports at the top of the file — add:

```python
from ..color.palette_store import PaletteStore
from .palette_editor import SwatchStrip
```

3b. Delete the `_PALETTES = [...]` line (line 33).

3c. Change the class signature and `__init__` head. Replace:

```python
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("control_panel")
```

with:

```python
    changed = Signal()
    from_image_requested = Signal()

    def __init__(self, parent=None, store: PaletteStore | None = None):
        super().__init__(parent)
        self.setObjectName("control_panel")
        self.store = store if store is not None else PaletteStore()
```

3d. Add the two new keys to the `self.state` dict literal (inside the existing dict):

```python
            "depth": 2, "color_mapping": "match",
            "palette_autosave": False, "extract_unit": "k",
```

3e. After `self._spins` / before the layout build (still in `__init__`), add:

```python
        self.working_palette = self.store.get(self.state["palette"])
```

Place this line immediately before `layout = QVBoxLayout(self)`.

3f. Replace the whole `_build_color_section` method body's palette-combo block. Replace:

```python
        self.palette_combo = NoScrollComboBox()
        self.palette_combo.addItems(_PALETTES)
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        layout.addWidget(_labeled("Palette", self.palette_combo))
```

with:

```python
        self.palette_combo = NoScrollComboBox()
        self.palette_combo.addItems(self.store.list())
        self.palette_combo.setCurrentText(self.state["palette"])
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        layout.addWidget(_labeled("Palette", self.palette_combo))

        self.swatch_strip = SwatchStrip()
        self.swatch_strip.set_palette(self.working_palette)
        self.swatch_strip.edited.connect(self._on_palette_edited)
        layout.addWidget(self.swatch_strip)

        pal_btns = QHBoxLayout()
        self.shuffle_btn = QPushButton("Shuffle")
        self.save_palette_btn = QPushButton("Save palette")
        self.reset_palette_btn = QPushButton("Reset to builtin")
        self.from_image_btn = QPushButton("From Image")
        self.shuffle_btn.clicked.connect(lambda: self.swatch_strip.shuffle())
        self.save_palette_btn.clicked.connect(self._on_save_palette)
        self.reset_palette_btn.clicked.connect(self._on_reset_palette)
        self.from_image_btn.clicked.connect(lambda: self.from_image_requested.emit())
        for b in (self.shuffle_btn, self.save_palette_btn,
                  self.reset_palette_btn, self.from_image_btn):
            pal_btns.addWidget(b)
        pal_container = QWidget()
        pal_container.setLayout(pal_btns)
        layout.addWidget(pal_container)

        self.extract_slider = ResettableGlowSlider(default=8, glow_color="#5e89ed")
        self.extract_slider.setRange(2, 64)
        layout.addWidget(_labeled("From-Image Colors", self.extract_slider))

        self._update_reset_enabled()
```

3g. Replace the existing `_on_palette_changed` handler:

```python
    def _on_palette_changed(self, text: str) -> None:
        self.state["palette"] = text
        self.changed.emit()
```

with:

```python
    def _on_palette_changed(self, text: str) -> None:
        self.state["palette"] = text
        self.working_palette = self.store.get(text)
        self.swatch_strip.set_palette(self.working_palette)
        self._update_reset_enabled()
        self.changed.emit()

    def _on_palette_edited(self, palette) -> None:
        self.working_palette = palette
        if self.state.get("palette_autosave"):
            self.store.save(palette)
            self._update_reset_enabled()
        self.changed.emit()

    def set_working_palette(self, palette) -> None:
        self.working_palette = palette
        self.state["palette"] = palette.name
        self.swatch_strip.set_palette(palette)
        self.changed.emit()

    def _on_save_palette(self) -> None:
        self.store.save(self.working_palette)
        self._refresh_palette_combo(self.working_palette.name)
        self._update_reset_enabled()

    def _on_reset_palette(self) -> None:
        name = self.working_palette.name
        if self.store.is_user(name) and self.store.is_builtin(name):
            self.working_palette = self.store.reset_to_builtin(name)
            self.swatch_strip.set_palette(self.working_palette)
        self._refresh_palette_combo(name if self.store.is_builtin(name) else None)
        self._update_reset_enabled()
        self.changed.emit()

    def _refresh_palette_combo(self, select: str | None) -> None:
        self.palette_combo.blockSignals(True)
        self.palette_combo.clear()
        self.palette_combo.addItems(self.store.list())
        if select is not None:
            self.palette_combo.setCurrentText(select)
        self.palette_combo.blockSignals(False)

    def _update_reset_enabled(self) -> None:
        name = self.working_palette.name
        self.reset_palette_btn.setEnabled(
            self.store.is_user(name) and self.store.is_builtin(name))
```

3h. Add the needed Qt imports. The `from PySide6.QtWidgets import (...)` block already imports `QPushButton`, `QHBoxLayout`, `QWidget` — no change needed. `ResettableGlowSlider` is already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_controls_palette.py tests/test_controls.py tests/test_controls_ramp.py -q`
Expected: PASS (new file green; existing controls tests still green).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/controls.py tests/test_controls_palette.py
git commit -m "feat(ui): wire PaletteStore + SwatchStrip into ControlPanel"
```

---

## Task 5: MainWindow wiring — working palette, source RGB, From-Image, preset restore

**Files:**
- Modify: `ditherzam/ui/main_window.py`
- Test: `tests/test_main_window_palette.py`

**Interfaces:**
- Consumes: `ControlPanel.working_palette`, `ControlPanel.set_working_palette`, `ControlPanel.from_image_requested`, `ControlPanel.state["extract_unit"]`, `ControlPanel.extract_slider`; `generate_palette` (Task 2).
- Produces (behavioral):
  - `MainWindow._base_rgb: np.ndarray | None` — source RGB retained on image load.
  - `load_array(self, gray_f32, rgb_u8=None)` — optional second arg stores `_base_rgb`.
  - `_current_palette()` returns `self.panel.working_palette` when color mode != "off", else `None`.
  - From-Image request → `generate_palette(self._base_rgb, unit, value)` → `panel.set_working_palette(...)`.
  - Preset load restores the working palette from the preset's `Palette`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main_window_palette.py`:

```python
import numpy as np
import pytest

pytest.importorskip("PySide6")


def _win(qapp_fixture):
    from ditherzam.ui.main_window import MainWindow
    return MainWindow()


def test_current_palette_is_working_palette(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel.palette_combo.setCurrentText("gameboy")
    win.panel.swatch_strip.set_swatch_color(0, (5, 6, 7))
    pal = win._current_palette()
    np.testing.assert_array_equal(pal.colors[0], [5, 6, 7])


def test_current_palette_none_when_off(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "off"
    assert win._current_palette() is None


def test_load_array_retains_rgb(qapp_fixture):
    win = _win(qapp_fixture)
    rgb = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    gray = rgb.mean(axis=2).astype(np.float32)
    win.load_array(gray, rgb)
    assert win._base_rgb is not None
    assert win._base_rgb.shape == (8, 8, 3)


def test_from_image_request_sets_working_palette(qapp_fixture):
    win = _win(qapp_fixture)
    rgb = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    win.load_array(rgb.mean(axis=2).astype(np.float32), rgb)
    win.panel.state["extract_unit"] = "k"
    win.panel.extract_slider.setValue(6)
    win._on_from_image_requested()
    assert win.panel.working_palette.colors.shape == (6, 3)
    assert win.panel.working_palette.name == "from image"


def test_from_image_no_image_is_noop(qapp_fixture):
    win = _win(qapp_fixture)
    win._base_rgb = None
    win._on_from_image_requested()      # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_main_window_palette.py -q`
Expected: FAIL — no `_base_rgb`, `_current_palette` re-looks-up by name, no `_on_from_image_requested`.

- [ ] **Step 3: Edit `main_window.py`**

3a. In `__init__`, next to `self._base_gray: np.ndarray | None = None` (line ~80), add:

```python
        self._base_rgb: np.ndarray | None = None
```

3b. Connect the From-Image signal. Find where the panel is created and its signals are connected in `__init__` (near where `self.panel = ControlPanel(...)` and `self.panel.changed.connect(...)` appear) and add:

```python
        self.panel.from_image_requested.connect(self._on_from_image_requested)
```

3c. Replace `_current_palette`:

```python
    def _current_palette(self):
        if self._color_mode() == "off":
            return None
        from ..color.palette import builtin_palettes
        return builtin_palettes().get(self.panel.state.get("palette"))
```

with:

```python
    def _current_palette(self):
        if self._color_mode() == "off":
            return None
        return self.panel.working_palette
```

3d. Replace `load_array`:

```python
    def load_array(self, gray_f32) -> None:
        self._base_gray = np.asarray(gray_f32, dtype=np.float32)
        self.pipeline.clear_cache()  # drop the previous image's cached intermediates
```

with:

```python
    def load_array(self, gray_f32, rgb_u8=None) -> None:
        self._base_gray = np.asarray(gray_f32, dtype=np.float32)
        self._base_rgb = None if rgb_u8 is None else np.asarray(rgb_u8, dtype=np.uint8)
        self.pipeline.clear_cache()  # drop the previous image's cached intermediates
```

3e. In `_on_image_dropped`, pass the RGB through. Replace:

```python
        try:
            self.load_array(to_gray_f32(Image.open(path)))
        except Exception:
            return
```

with:

```python
        try:
            img = Image.open(path)
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            self.load_array(to_gray_f32(img), rgb)
        except Exception:
            return
```

3f. Add the From-Image handler. Place it next to `_current_palette` (in the pure-accessor / handler region):

```python
    def _on_from_image_requested(self) -> None:
        if self._base_rgb is None:
            return
        from ..color.palette import generate_palette
        unit = str(self.panel.state.get("extract_unit", "k"))
        value = int(self.panel.extract_slider.value())
        palette = generate_palette(self._base_rgb, unit, value)
        self.panel.set_working_palette(palette)
```

3g. Restore the working palette on preset load. In `_apply_preset`, replace:

```python
        if palette is not None:
            panel.state["palette"] = palette.name
```

with:

```python
        if palette is not None:
            panel.set_working_palette(palette)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_main_window_palette.py tests/test_main_window_render_wiring.py tests/test_presets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/main_window.py tests/test_main_window_palette.py
git commit -m "feat(ui): MainWindow uses working palette + From-Image extraction + preset restore"
```

---

## Task 6: Full-suite regression gate (both JIT modes)

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite JIT-off (default)**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all green, count ≥ 511 + the new tests.

- [ ] **Step 2: Run the full suite JIT-on**

Run (Git Bash): `NUMBA_DISABLE_JIT=0 ./.venv/Scripts/python.exe -m pytest -q`
Expected: same pass count, all green.

- [ ] **Step 3: Headless smoke — palette store round-trip through a MainWindow**

Confirm no console errors editing/saving/resetting a palette and running a render. (The subagent-driven review handles this; if running inline, drive a `MainWindow`, select a palette, recolor a swatch, save, reset, and render once.)

- [ ] **Step 4: Commit (if any smoke fixups were needed)**

```bash
git add -A
git commit -m "test: full-suite regression gate green in both JIT modes for palette editing"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** store boundary → Task 1; builtin shadowing / copy-on-edit / reset → Tasks 1+4; swatch add/remove/recolor/lock/shuffle → Task 3; From-Image (k + %) → Tasks 2+5; explicit Save + autosave setting → Task 4; extract-unit setting → Task 4 (`state["extract_unit"]`) + Task 5 (consumed); combo from store + name-resolution seam → Tasks 4+5; preset round-trip → Task 5 (already serialized by `presets.py`). All spec sections mapped.
- **Deferred (C), intentionally absent:** drag-reorder, categories/picker, hover-scroll preview, import/share.
- **Type consistency:** `PaletteStore` method names identical across Tasks 1/4/5; `SwatchStrip.set_palette/set_swatch_color/shuffle/edited` identical across Tasks 3/4; `generate_palette(rgb_u8, unit, value, name)` identical across Tasks 2/5; `set_working_palette`/`from_image_requested`/`extract_slider`/`working_palette` consistent between Tasks 4 and 5.
- **UI-pref placement:** `palette_autosave` / `extract_unit` kept in `panel.state`, never added to `RenderSettings` — frozen render path preserved.
- **Qt boundary:** new core file `palette_store.py` and `generate_palette` import no Qt; only `palette_editor.py`, `controls.py`, `main_window.py` touch PySide6.
