# Palette Library (Sub-project C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the flat palette list into an organized library — categories, a category-organized tree picker with per-palette swatch previews, hover/scroll preview on the current image, and drag-reorder swatches.

**Architecture:** A `category` field is added to the Qt-free `Palette`/store. A new `PalettePicker(QTreeWidget)` replaces the flat combo in `controls.py`. Hover/scroll emits a transient `preview` up to `main_window`, which renders a preview palette through `_current_palette()` without mutating the working palette. `SwatchStrip` gains `move_swatch`.

**Tech Stack:** Python 3.12, PySide6, NumPy, PyYAML, pytest.

## Global Constraints

- **Clean-room** — our own code; Dither Boy is behaviour inspiration only. No Studio/Dither-Boy code, strings, or binaries.
- **Qt-free core** — `ditherzam/color/**` (incl. `palette.py`, `palette_store.py`) must not import PySide6. Only `ui/`, `app.py`, `video/workers.py` import Qt.
- **Frozen render path** — do NOT change `RenderPipeline.render()` stage order or `RenderSettings` color fields. UI-session prefs live in `panel.state` only, never in `RenderSettings`.
- **Python 3.12; TDD per task** (red → green → refactor, commit per green).
- **Test runner:** `./.venv/Scripts/python.exe -m pytest`. Tests run `NUMBA_DISABLE_JIT=1` (set in `tests/conftest.py`); Qt tests `QT_QPA_PLATFORM=offscreen`; session fixture `qapp_fixture`.
- **Green means JIT-off.** Full suite is currently **545** JIT-off. 7 kernel JIT-**on** failures are pre-existing (`special.py`) and NOT this work's concern.
- Number-display widgets need an explicit `valueChanged.connect` (memory 016). C adds no value sliders, only checkboxes/combos.

---

### Task 1: `Palette.category` field (Qt-free core)

**Files:**
- Modify: `ditherzam/color/palette.py`
- Test: `tests/test_palette.py`

**Interfaces:**
- Produces:
  - `Palette` dataclass gains `category: str = ""` (field order: `name`, `colors`, `category`).
  - `Palette.from_list(cls, name, rgb_list, category: str = "") -> Palette`
  - `to_yaml` writes a `category:` key **only when `self.category` is non-empty**.
  - `load` reads `category` (default `""`).
  - `shuffle` preserves `category`.
  - `extract_palette(rgb_u8, k=16, name="source", category="user")`, `source_palette(..., category="user")`, `generate_palette(..., category="user")` — results carry `category="user"`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_palette.py`:

```python
def test_category_defaults_empty():
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3]])
    assert p.category == ""


def test_category_yaml_roundtrip(tmp_path):
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3]], category="retro")
    dest = tmp_path / "x.yaml"
    p.to_yaml(dest)
    assert "category: retro" in dest.read_text(encoding="utf-8")
    assert Palette.load(dest).category == "retro"


def test_empty_category_not_written(tmp_path):
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3]])
    dest = tmp_path / "x.yaml"
    p.to_yaml(dest)
    assert "category" not in dest.read_text(encoding="utf-8")


def test_load_legacy_yaml_has_empty_category(tmp_path):
    from ditherzam.color.palette import Palette
    dest = tmp_path / "legacy.yaml"
    dest.write_text("name: legacy\ncolors:\n  - [1, 2, 3]\n", encoding="utf-8")
    assert Palette.load(dest).category == ""


def test_shuffle_preserves_category():
    import numpy as np
    from ditherzam.color.palette import Palette
    p = Palette.from_list("x", [[1, 2, 3], [4, 5, 6]], category="retro")
    out = p.shuffle(locked={0}, rng=np.random.default_rng(0))
    assert out.category == "retro"


def test_extract_palette_is_user_category():
    import numpy as np
    from ditherzam.color.palette import extract_palette
    rgb = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert extract_palette(rgb, k=4).category == "user"
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette.py -k "category" -v`
Expected: FAIL (`TypeError` for unexpected `category` kwarg / `AttributeError`).

- [ ] **Step 3: Implement** in `ditherzam/color/palette.py`:

Dataclass + `from_list`:
```python
@dataclass
class Palette:
    """An RGB palette: ``colors`` is float32[K, 3] in the 0..255 range."""

    name: str
    colors: np.ndarray
    category: str = ""

    @classmethod
    def from_list(cls, name: str, rgb_list, category: str = "") -> "Palette":
        arr = np.asarray(rgb_list, dtype=np.float32).reshape(-1, 3)
        return cls(name=name, colors=arr, category=category)
```

`to_yaml` (write category only when non-empty):
```python
    def to_yaml(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name}
        if self.category:
            data["category"] = self.category
        data["colors"] = [[int(round(c)) for c in row] for row in self.colors.tolist()]
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
```

`load`:
```python
    @classmethod
    def load(cls, path) -> "Palette":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Palette file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = data.get("name", path.stem)
        return cls.from_list(name, data["colors"], category=data.get("category", ""))
```

`shuffle` — change the returned constructor to `Palette(name=self.name, colors=new, category=self.category)`.

`extract_palette` — add `category: str = "user"` param and build `Palette(name=name, colors=colors, category=category)`.
`source_palette` — add `category: str = "user"`, pass through to `extract_palette`.
`generate_palette` — add `category: str = "user"`, pass through on both branches.

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/palette.py tests/test_palette.py
git commit -m "feat(color): add Palette.category field + user category on extraction"
```

---

### Task 2: Builtin categories + `PaletteStore.list_by_category`

**Files:**
- Modify: `ditherzam/color/builtin/gameboy.yaml`, `pico8.yaml`, `cga.yaml`, `grayscale.yaml`, `sepia.yaml`
- Modify: `ditherzam/color/palette_store.py`
- Test: `tests/test_palette_store.py`

**Interfaces:**
- Consumes: `Palette.category` (Task 1), `builtin_palettes()`.
- Produces:
  - `PaletteStore.get(name)` now carries `category` on the returned copy.
  - `PaletteStore.list_by_category() -> dict[str, list[str]]` — category → sorted names; empty category bucketed under `"uncategorized"`; categories ordered alphabetically with `"uncategorized"` pinned last; user file's category wins for a shadowed name.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_palette_store.py`:

```python
def test_get_carries_category(tmp_path):
    s = _store(tmp_path)
    assert s.get("gameboy").category == "retro"


def test_list_by_category_groups_builtins(tmp_path):
    s = _store(tmp_path)
    cats = s.list_by_category()
    assert set(cats["retro"]) >= {"gameboy", "pico8", "cga"}
    assert set(cats["mono"]) >= {"grayscale", "sepia"}


def test_list_by_category_uncategorized_last(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("loner", [[1, 1, 1]]))   # empty category
    cats = s.list_by_category()
    assert "loner" in cats["uncategorized"]
    assert list(cats.keys())[-1] == "uncategorized"


def test_user_category_wins_for_shadowed_name(tmp_path):
    s = _store(tmp_path)
    s.save(Palette.from_list("gameboy", [[0, 0, 0]], category="favourites"))
    cats = s.list_by_category()
    assert "gameboy" in cats["favourites"]
    assert "gameboy" not in cats.get("retro", [])
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_store.py -k "category" -v`
Expected: FAIL (`AttributeError: list_by_category` / category is `""`).

- [ ] **Step 3a: Add `category:` to each builtin YAML.** Insert the line directly under `name:`.

`gameboy.yaml`, `pico8.yaml`, `cga.yaml` → `category: retro`
`grayscale.yaml`, `sepia.yaml` → `category: mono`

Example (`gameboy.yaml`):
```yaml
name: gameboy
category: retro
colors:
  - [15, 56, 15]
  - [48, 98, 48]
  - [139, 172, 15]
  - [155, 188, 15]
```

- [ ] **Step 3b: Update `palette_store.py`.**

`get()` — carry category:
```python
        return Palette(name=p.name, colors=p.colors.copy(), category=p.category)
```

Add `list_by_category`:
```python
    def list_by_category(self) -> dict[str, list[str]]:
        cats: dict[str, list[str]] = {}
        for name in self.list():
            key = self.get(name).category or "uncategorized"
            cats.setdefault(key, []).append(name)
        ordered = sorted(k for k in cats if k != "uncategorized")
        if "uncategorized" in cats:
            ordered.append("uncategorized")
        return {k: sorted(cats[k]) for k in ordered}
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_store.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/color/palette_store.py ditherzam/color/builtin/*.yaml tests/test_palette_store.py
git commit -m "feat(color): seed builtin categories + PaletteStore.list_by_category"
```

---

### Task 3: `PalettePicker` tree widget

**Files:**
- Create: `ditherzam/ui/palette_picker.py`
- Test: `tests/test_palette_picker.py`

**Interfaces:**
- Consumes: `PaletteStore.list_by_category()`, `PaletteStore.get(name)` (Task 2).
- Produces `PalettePicker(QTreeWidget)`:
  - Signals: `selected(str)` (commit), `preview(object)` (a `Palette`, or `None` to revert).
  - `populate(store)` — rebuild grouped tree, signals blocked.
  - `select(name)` — highlight without emitting `selected`/`preview`.
  - `set_preview_enabled(bool)`, `set_wheel_cycle(bool)`.
  - Palette rows store the name in `Qt.ItemDataRole.UserRole`; category headers store `None`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_palette_picker.py`:

```python
import pytest

pytest.importorskip("PySide6")

from ditherzam.color.palette_store import PaletteStore


def _picker(tmp_path):
    from ditherzam.ui.palette_picker import PalettePicker
    p = PalettePicker()
    p.populate(PaletteStore(user_dir=tmp_path / "pal"))
    return p


def _palette_items(picker):
    from PySide6.QtCore import Qt
    out = []
    for i in range(picker.topLevelItemCount()):
        top = picker.topLevelItem(i)
        for j in range(top.childCount()):
            out.append(top.child(j))
    return out


def test_headers_are_categories(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    headers = [picker.topLevelItem(i).text(0) for i in range(picker.topLevelItemCount())]
    assert "retro" in headers and "mono" in headers


def test_palette_rows_present(qapp_fixture, tmp_path):
    from PySide6.QtCore import Qt
    picker = _picker(tmp_path)
    names = [it.data(0, Qt.ItemDataRole.UserRole) for it in _palette_items(picker)]
    assert "gameboy" in names and "grayscale" in names


def test_click_palette_emits_selected(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.selected.connect(seen.append)
    item = next(it for it in _palette_items(picker)
                if it.data(0, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ItemDataRole.UserRole) == "gameboy")
    picker._on_item_clicked(item, 0)
    assert seen == ["gameboy"]


def test_header_click_does_not_emit_selected(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.selected.connect(seen.append)
    picker._on_item_clicked(picker.topLevelItem(0), 0)
    assert seen == []


def test_hover_emits_preview_when_enabled(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.preview.connect(seen.append)
    item = next(it for it in _palette_items(picker)
                if it.data(0, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ItemDataRole.UserRole) == "gameboy")
    picker._on_item_entered(item, 0)
    assert seen and seen[-1] is not None and seen[-1].name == "gameboy"


def test_hover_suppressed_when_disabled(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    picker.set_preview_enabled(False)
    seen = []
    picker.preview.connect(seen.append)
    item = _palette_items(picker)[0]
    picker._on_item_entered(item, 0)
    assert seen == []


def test_select_does_not_emit(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen_sel, seen_prev = [], []
    picker.selected.connect(seen_sel.append)
    picker.preview.connect(seen_prev.append)
    picker.select("gameboy")
    assert seen_sel == [] and seen_prev == []


def test_wheel_cycle_previews_only_when_on(qapp_fixture, tmp_path):
    picker = _picker(tmp_path)
    seen = []
    picker.preview.connect(seen.append)
    picker.set_wheel_cycle(True)
    picker.select("gameboy")
    picker._cycle(1)                     # step to next palette
    assert seen and seen[-1] is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_picker.py -v`
Expected: FAIL (`ModuleNotFoundError: ditherzam.ui.palette_picker`).

- [ ] **Step 3: Implement** — create `ditherzam/ui/palette_picker.py`:

```python
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

_NAME_ROLE = Qt.ItemDataRole.UserRole


def _swatch_icon(colors: np.ndarray, w: int = 64, h: int = 16) -> QIcon:
    pix = QPixmap(w, h)
    pix.fill(QColor(0, 0, 0, 0))
    n = max(1, int(colors.shape[0]))
    painter = QPainter(pix)
    cell = w / n
    for i in range(n):
        r, g, b = (int(round(c)) for c in colors[i])
        painter.fillRect(int(i * cell), 0, int(cell) + 1, h, QColor(r, g, b))
    painter.end()
    return QIcon(pix)


class PalettePicker(QTreeWidget):
    """Category-grouped palette tree. Hover/scroll preview, click to commit."""

    selected = Signal(str)      # committed palette name
    preview = Signal(object)    # a Palette to preview, or None to revert

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setMouseTracking(True)
        self.setIconSize(QSize(64, 16))
        self._preview_enabled = True
        self._wheel_cycle = False
        self._store = None
        self.itemClicked.connect(self._on_item_clicked)
        self.itemActivated.connect(self._on_item_clicked)   # Enter / dbl-click
        self.currentItemChanged.connect(self._on_current_changed)
        self.itemEntered.connect(self._on_item_entered)

    # -- config ---------------------------------------------------------------
    def set_preview_enabled(self, on: bool) -> None:
        self._preview_enabled = bool(on)

    def set_wheel_cycle(self, on: bool) -> None:
        self._wheel_cycle = bool(on)

    # -- population -----------------------------------------------------------
    def populate(self, store) -> None:
        self._store = store
        self.blockSignals(True)
        self.clear()
        for category, names in store.list_by_category().items():
            header = QTreeWidgetItem([category])
            header.setFlags(Qt.ItemFlag.ItemIsEnabled)     # not selectable
            header.setData(0, _NAME_ROLE, None)
            self.addTopLevelItem(header)
            for name in names:
                child = QTreeWidgetItem([name])
                child.setData(0, _NAME_ROLE, name)
                child.setIcon(0, _swatch_icon(store.get(name).colors))
                header.addChild(child)
            header.setExpanded(True)
        self.blockSignals(False)

    def select(self, name: str) -> None:
        item = self._find_item(name)
        if item is not None:
            self.blockSignals(True)
            self.setCurrentItem(item)
            self.blockSignals(False)

    # -- helpers --------------------------------------------------------------
    def _palette_items(self) -> list[QTreeWidgetItem]:
        out = []
        for i in range(self.topLevelItemCount()):
            top = self.topLevelItem(i)
            for j in range(top.childCount()):
                out.append(top.child(j))
        return out

    def _find_item(self, name: str):
        for it in self._palette_items():
            if it.data(0, _NAME_ROLE) == name:
                return it
        return None

    @staticmethod
    def _name_of(item):
        return None if item is None else item.data(0, _NAME_ROLE)

    def _emit_preview(self, name) -> None:
        if not self._preview_enabled or self._store is None:
            return
        self.preview.emit(self._store.get(name) if name is not None else None)

    # -- signal handlers ------------------------------------------------------
    def _on_item_clicked(self, item, _col) -> None:
        name = self._name_of(item)
        if name is not None:
            self.selected.emit(name)

    def _on_current_changed(self, current, _prev) -> None:
        self._emit_preview(self._name_of(current))

    def _on_item_entered(self, item, _col) -> None:
        self._emit_preview(self._name_of(item))

    # -- events ---------------------------------------------------------------
    def leaveEvent(self, event):
        if self._preview_enabled:
            self.preview.emit(None)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if not self._wheel_cycle:
            super().wheelEvent(event)
            return
        self._cycle(-1 if event.angleDelta().y() > 0 else 1)
        event.accept()

    def _cycle(self, step: int) -> None:
        items = self._palette_items()
        if not items:
            return
        cur = self.currentItem()
        idx = items.index(cur) if cur in items else -1
        idx = max(0, min(len(items) - 1, idx + step))
        self.setCurrentItem(items[idx])   # fires _on_current_changed -> preview
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_picker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/palette_picker.py tests/test_palette_picker.py
git commit -m "feat(ui): PalettePicker category tree with hover/wheel preview"
```

---

### Task 4: Rewire `controls.py` — picker replaces combo, category combo, new settings

**Files:**
- Modify: `ditherzam/ui/controls.py`
- Test (rewrite combo refs): `tests/test_controls_palette.py`
- Test (new): add cases to `tests/test_controls_palette.py`

**Interfaces:**
- Consumes: `PalettePicker` (Task 3), `PaletteStore.list_by_category` (Task 2).
- Produces on `ControlPanel`:
  - Attribute `palette_picker: PalettePicker` (replaces `palette_combo`).
  - Attribute `category_combo: NoScrollComboBox` (editable).
  - Attributes `palette_preview_toggle`, `wheel_cycle_toggle` (QCheckBox).
  - New signal `palette_preview = Signal(object)` re-emitting the picker's preview.
  - New `state` keys `palette_preview: bool = True`, `palette_wheel_cycle: bool = False`.
  - `set_working_palette` keeps its signature; internally uses the picker.

- [ ] **Step 1: Rewrite the combo-referencing tests + add new ones** in `tests/test_controls_palette.py`.

Replace the four combo tests with picker equivalents (delete `test_combo_populated_from_store`, `test_selecting_palette_sets_working_copy`, `test_save_palette_persists_and_refreshes_combo`, `test_set_working_palette_syncs_combo`; keep the rest). Add at top a helper and the new tests:

```python
def _picker_names(panel):
    from PySide6.QtCore import Qt
    out = []
    pk = panel.palette_picker
    for i in range(pk.topLevelItemCount()):
        top = pk.topLevelItem(i)
        for j in range(top.childCount()):
            out.append(top.child(j).data(0, Qt.ItemDataRole.UserRole))
    return out


def test_picker_populated_from_store(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    names = _picker_names(panel)
    assert "gameboy" in names and "pico8" in names


def test_selecting_palette_sets_working_copy(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel._on_palette_changed("gameboy")
    assert panel.working_palette.name == "gameboy"
    assert panel.swatch_strip.palette().name == "gameboy"


def test_save_palette_persists_and_refreshes_picker(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel._on_palette_changed("gameboy")
    panel.swatch_strip.set_swatch_color(0, (1, 2, 3))
    panel._on_save_palette()
    assert panel.store.is_user("gameboy")
    assert _picker_names(panel).count("gameboy") == 1


def test_set_working_palette_sets_name(qapp_fixture, tmp_path):
    from ditherzam.color.palette import Palette
    panel = _panel(tmp_path)
    panel.set_working_palette(Palette.from_list("from image", [[1, 1, 1], [2, 2, 2]]))
    assert panel.working_palette.name == "from image"


def test_preview_defaults(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    assert panel.state["palette_preview"] is True
    assert panel.state["palette_wheel_cycle"] is False


def test_preview_toggle_widget(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.palette_preview_toggle.setChecked(False)
    assert panel.state["palette_preview"] is False


def test_wheel_cycle_toggle_widget(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel.wheel_cycle_toggle.setChecked(True)
    assert panel.state["palette_wheel_cycle"] is True


def test_save_uses_category_combo(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    panel._on_palette_changed("gameboy")
    panel.category_combo.setCurrentText("favourites")
    panel._on_save_palette()
    assert panel.store.get("gameboy").category == "favourites"


def test_palette_preview_signal_reemitted(qapp_fixture, tmp_path):
    panel = _panel(tmp_path)
    seen = []
    panel.palette_preview.connect(seen.append)
    panel.palette_picker.preview.emit(None)
    assert seen == [None]
```

Also update `test_swatch_edit_updates_working_and_emits_changed`, `test_autosave_writes_on_edit_when_enabled`, `test_reset_to_builtin_drops_fork`, `test_set_working_palette_pushes_to_strip` to call `panel._on_palette_changed("gameboy")` instead of `panel.palette_combo.setCurrentText("gameboy")`.

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_controls_palette.py -v`
Expected: FAIL (`AttributeError: palette_picker` / `palette_preview` / `category_combo`).

- [ ] **Step 3: Implement `controls.py` changes.**

Imports — add:
```python
from .palette_picker import PalettePicker
```

`changed = Signal()` block — add:
```python
    changed = Signal()
    from_image_requested = Signal()
    palette_preview = Signal(object)
```

`state` dict — add keys:
```python
            "palette_autosave": False, "extract_unit": "k",
            "palette_preview": True, "palette_wheel_cycle": False,
```

In `_build_color_section`, replace the `palette_combo` block (the `NoScrollComboBox()` + `addItems(self.store.list())` + `_labeled("Palette", ...)`) with:
```python
        self.palette_picker = PalettePicker()
        self.palette_picker.populate(self.store)
        self.palette_picker.select(self.state["palette"])
        self.palette_picker.selected.connect(self._on_palette_changed)
        self.palette_picker.preview.connect(self.palette_preview.emit)
        layout.addWidget(_labeled("Palette", self.palette_picker))
```

After the palette button row (before/after the extract widgets), add the category combo:
```python
        self.category_combo = NoScrollComboBox()
        self.category_combo.setEditable(True)
        self.category_combo.addItems(sorted(self.store.list_by_category().keys()))
        layout.addWidget(_labeled("Category", self.category_combo))
```

After the `autosave_toggle` block, add the two preview toggles:
```python
        self.palette_preview_toggle = QCheckBox("Preview on hover")
        self.palette_preview_toggle.setChecked(True)
        self.palette_preview_toggle.toggled.connect(self._on_palette_preview_toggled)
        layout.addWidget(self.palette_preview_toggle)

        self.wheel_cycle_toggle = QCheckBox("Wheel cycles palettes")
        self.wheel_cycle_toggle.toggled.connect(self._on_wheel_cycle_toggled)
        layout.addWidget(self.wheel_cycle_toggle)
```

Replace `_on_palette_changed` body's combo assumptions (it already takes `text`):
```python
    def _on_palette_changed(self, text: str) -> None:
        self.state["palette"] = text
        self.working_palette = self.store.get(text)
        self.swatch_strip.set_palette(self.working_palette)
        self.palette_preview.emit(None)                # clear any hover preview
        self._sync_category_combo(self.working_palette.category)
        self._update_reset_enabled()
        self.changed.emit()
```

Replace `set_working_palette`:
```python
    def set_working_palette(self, palette) -> None:
        self.working_palette = palette
        self.state["palette"] = palette.name
        self.swatch_strip.set_palette(palette)
        self.palette_picker.select(palette.name)
        self._sync_category_combo(getattr(palette, "category", ""))
        self.changed.emit()
```

Delete `_sync_palette_combo`. Replace `_refresh_palette_combo` with a picker refresh:
```python
    def _refresh_palette_picker(self, select: str | None) -> None:
        self.palette_picker.populate(self.store)
        if select is not None:
            self.palette_picker.select(select)
```

Add category-combo sync helper:
```python
    def _sync_category_combo(self, category: str) -> None:
        self.category_combo.blockSignals(True)
        self.category_combo.setCurrentText(category or "")
        self.category_combo.blockSignals(False)
```

`_on_save_palette` — set category from the combo before saving:
```python
    def _on_save_palette(self) -> None:
        self.working_palette.category = self.category_combo.currentText().strip()
        self.store.save(self.working_palette)
        self._refresh_palette_picker(self.working_palette.name)
        self._update_reset_enabled()
```

`_on_reset_palette` — swap combo refresh for picker:
```python
    def _on_reset_palette(self) -> None:
        name = self.working_palette.name
        if self.store.is_user(name) and self.store.is_builtin(name):
            self.working_palette = self.store.reset_to_builtin(name)
            self.swatch_strip.set_palette(self.working_palette)
        self._refresh_palette_picker(name if self.store.is_builtin(name) else None)
        self._update_reset_enabled()
        self.changed.emit()
```

Add the two toggle handlers (near `_on_autosave_toggled`):
```python
    def _on_palette_preview_toggled(self, checked: bool) -> None:
        self.state["palette_preview"] = bool(checked)
        self.palette_picker.set_preview_enabled(bool(checked))

    def _on_wheel_cycle_toggled(self, checked: bool) -> None:
        self.state["palette_wheel_cycle"] = bool(checked)
        self.palette_picker.set_wheel_cycle(bool(checked))
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_controls_palette.py tests/test_controls.py tests/test_controls_ramp.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/controls.py tests/test_controls_palette.py
git commit -m "feat(ui): ControlPanel uses PalettePicker + category combo + preview toggles"
```

---

### Task 5: Preview seam in `main_window.py`

**Files:**
- Modify: `ditherzam/ui/main_window.py`
- Test: `tests/test_main_window_palette.py`

**Interfaces:**
- Consumes: `ControlPanel.palette_preview(object)` signal (Task 4).
- Produces on `ImageEditor`:
  - `self._preview_palette` attribute (default `None`).
  - `_current_palette()` returns `_preview_palette` when set, else `panel.working_palette` (still `None` when color mode off).
  - `_on_palette_preview(palette)` slot honouring the `palette_preview` setting.

- [ ] **Step 1: Rewrite the combo-referencing test + add preview tests** in `tests/test_main_window_palette.py`.

Replace `win.panel.palette_combo.setCurrentText("gameboy")` in `test_current_palette_is_working_palette` with `win.panel._on_palette_changed("gameboy")`. Add:

```python
def test_preview_palette_overrides_current(qapp_fixture):
    from ditherzam.color.palette import Palette
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    win._on_palette_preview(Palette.from_list("temp", [[9, 9, 9]]))
    assert win._current_palette().name == "temp"
    assert win.panel.working_palette.name == "gameboy"   # unchanged


def test_preview_none_reverts(qapp_fixture):
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    win._on_palette_preview(None)
    assert win._current_palette().name == "gameboy"


def test_preview_ignored_when_disabled(qapp_fixture):
    from ditherzam.color.palette import Palette
    win = _win(qapp_fixture)
    win.panel.state["color_mode"] = "nearest"
    win.panel._on_palette_changed("gameboy")
    win.panel.state["palette_preview"] = False
    win._on_palette_preview(Palette.from_list("temp", [[9, 9, 9]]))
    assert win._current_palette().name == "gameboy"
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_main_window_palette.py -v`
Expected: FAIL (`AttributeError: _on_palette_preview`).

- [ ] **Step 3: Implement.**

In `ImageEditor.__init__`, after `self._base_rgb = None`:
```python
        self._preview_palette = None
```

Connect the signal near the other `self.panel.*.connect(...)` lines:
```python
        self.panel.palette_preview.connect(self._on_palette_preview)
```

Update `_current_palette`:
```python
    def _current_palette(self):
        if self._color_mode() == "off":
            return None
        if self._preview_palette is not None:
            return self._preview_palette
        return self.panel.working_palette
```

Add the slot (near `_current_palette`):
```python
    def _on_palette_preview(self, palette) -> None:
        if not self.panel.state.get("palette_preview", True):
            return
        self._preview_palette = palette          # a Palette, or None to revert
        if self._base_gray is not None:
            self.schedule_render()
```

Note: `_on_palette_changed` already emits `palette_preview(None)` (Task 4), which clears `_preview_palette` on commit.

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_main_window_palette.py tests/test_main_window_render_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/ui/main_window.py tests/test_main_window_palette.py
git commit -m "feat(ui): hover-preview seam via _preview_palette + _current_palette"
```

---

### Task 6: Drag-reorder swatches (`SwatchStrip.move_swatch`)

**Files:**
- Modify: `ditherzam/ui/palette_editor.py`
- Test: `tests/test_palette_editor.py`

**Interfaces:**
- Produces `SwatchStrip.move_swatch(src_i: int, dst_i: int)` — reorders `colors`, remaps `_locked`, `_rebuild()`, emits `edited(palette)`. The drop gesture calls it.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_palette_editor.py`:

```python
def test_move_swatch_reorders_colors(qapp_fixture):
    s = _strip()   # colors [[0,0,0],[128,128,128],[255,255,255]]
    s.move_swatch(0, 2)
    np.testing.assert_array_equal(s.palette().colors[2], [0, 0, 0])
    np.testing.assert_array_equal(s.palette().colors[0], [128, 128, 128])


def test_move_swatch_emits_edited(qapp_fixture):
    s = _strip()
    seen = []
    s.edited.connect(lambda pal: seen.append(pal))
    s.move_swatch(2, 0)
    assert len(seen) == 1


def test_move_swatch_remaps_locks(qapp_fixture):
    s = _strip()
    s.toggle_lock(0)              # lock the first swatch
    s.move_swatch(0, 2)          # it moves to index 2
    assert s.locked() == {2}


def test_move_swatch_noop_same_index(qapp_fixture):
    s = _strip()
    before = s.palette().colors.copy()
    s.move_swatch(1, 1)
    np.testing.assert_array_equal(s.palette().colors, before)
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_editor.py -k move_swatch -v`
Expected: FAIL (`AttributeError: move_swatch`).

- [ ] **Step 3: Implement `move_swatch`** in `SwatchStrip` (after `remove_swatch`):

```python
    def move_swatch(self, src_i: int, dst_i: int) -> None:
        n = self._palette.colors.shape[0]
        if not (0 <= src_i < n and 0 <= dst_i < n) or src_i == dst_i:
            return
        order = list(range(n))
        order.insert(dst_i, order.pop(src_i))
        new_colors = self._palette.colors[order].astype(np.float32)
        remap = {old: new for new, old in enumerate(order)}
        self._locked = {remap[i] for i in self._locked}
        self._palette = Palette(
            name=self._palette.name, colors=new_colors,
            category=self._palette.category,
        )
        self._rebuild()
        self.edited.emit(self._palette)
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_editor.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the drag gesture (no test — thin Qt glue).**

In `_rebuild`, after creating each swatch `btn`, enable dragging and make the strip accept drops. Minimal implementation using Qt's drag-drop with the source index in the mime text. Add at the end of `__init__`: `self.setAcceptDrops(True)`. On each `btn`, install press-to-drag via a small helper; on the strip, `dropEvent` reads the source index and computes the target from `event.position()` against button geometries, then calls `move_swatch`. Keep the gesture code contained; `move_swatch` is the tested contract.

```python
    # in __init__, last line:
    self.setAcceptDrops(True)

    # new methods on SwatchStrip:
    def _target_index(self, x: float) -> int:
        for i, b in enumerate(self._buttons):
            if x < b.x() + b.width() / 2:
                return i
        return len(self._buttons) - 1

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        try:
            src = int(event.mimeData().text())
        except (TypeError, ValueError):
            return
        dst = self._target_index(event.position().x())
        self.move_swatch(src, dst)
        event.acceptProposedAction()
```

And start a drag from a swatch on left-press-drag. In `_rebuild`, replace the plain `btn.clicked.connect(...)` swatch with a subclass press that begins a `QDrag` carrying `str(i)` when the mouse moves with the button held; simplest is to set `btn.setProperty("swatch_index", i)` and start the drag from the strip's `mousePressEvent`/`mouseMoveEvent`. Implementation detail is left to the engineer; the only hard requirement is that a completed drop calls `move_swatch(src, dst)`. Import `QDrag`, `QMimeData` from `PySide6.QtGui`/`PySide6.QtCore` as needed.

- [ ] **Step 6: Run the full editor test file + commit**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_palette_editor.py -v`
Expected: PASS.

```bash
git add ditherzam/ui/palette_editor.py tests/test_palette_editor.py
git commit -m "feat(ui): drag-reorder swatches via move_swatch + drop gesture"
```

---

### Task 7: Preset round-trip of `category` + final regression

**Files:**
- Modify: `ditherzam/presets.py`
- Test: `tests/test_presets.py` (or the existing preset test file)

**Interfaces:**
- Consumes: `Palette.category` (Task 1).
- Produces: `settings_to_preset` writes `palette.category`; `preset_to_settings` reads it back (default `""`).

- [ ] **Step 1: Locate the preset test file and write the failing test.**

Run: `./.venv/Scripts/python.exe -m pytest --collect-only -q | grep -i preset` to find the file (likely `tests/test_presets.py`). Append:

```python
def test_preset_roundtrips_palette_category():
    from ditherzam.presets import settings_to_preset, preset_to_settings
    from ditherzam.render import RenderSettings
    from ditherzam.color.palette import Palette
    pal = Palette.from_list("mine", [[1, 2, 3], [4, 5, 6]], category="retro")
    preset = settings_to_preset(RenderSettings(), pal, None, "ramp")
    _settings, out_pal, _effects = preset_to_settings(preset)
    assert out_pal.category == "retro"
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_presets.py -k category -v`
Expected: FAIL (`category == ""`).

- [ ] **Step 3: Implement.**

In `settings_to_preset`, the `preset["color"]["palette"]` dict — add category:
```python
            "palette": {
                "name": str(palette.name),
                "category": str(getattr(palette, "category", "") or ""),
                "colors": np.asarray(palette.colors, dtype=np.float32)
                            .round().astype(int).reshape(-1, 3).tolist(),
            },
```

In `preset_to_settings`, where `palette` is reconstructed:
```python
        palette = Palette(name=str(pdata.get("name", "preset")), colors=colors,
                          category=str(pdata.get("category", "")))
```

- [ ] **Step 4: Run to verify pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_presets.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite regression gate**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, count ≥ 545 + the new tests (JIT-off). If a legacy test still references `palette_combo`, update it to the picker equivalent.

- [ ] **Step 6: Commit**

```bash
git add ditherzam/presets.py tests/test_presets.py
git commit -m "feat(io): round-trip Palette.category through presets"
```

---

## Self-Review

**Spec coverage:**
- Categories field + YAML → Task 1. Builtins seeded + `list_by_category` → Task 2. ✅
- Tree picker with swatch previews, hover/wheel preview, select-without-emit → Task 3. ✅
- controls.py rewire, category combo, `palette_preview`/`palette_wheel_cycle` settings + toggles → Task 4. ✅
- Preview seam (`_preview_palette`, `_current_palette`, master-toggle honoured) → Task 5. ✅
- Drag-reorder (`move_swatch` + gesture) → Task 6. ✅
- Preset round-trip of category + regression gate → Task 7. ✅
- Session prefs stay in `panel.state` (never `RenderSettings`); frozen render path untouched — held across Tasks 4/5/7. ✅
- Import/share explicitly out of scope — no task. ✅

**Placeholder scan:** Task 6 Step 5 leaves the exact drag *gesture* wiring to the engineer, but pins the tested contract (`move_swatch`) and gives concrete `dragEnterEvent`/`dropEvent`/`_target_index` code — acceptable, as the gesture is untestable headless glue over a fully-specified, tested method.

**Type consistency:** `_on_palette_changed(text: str)` is fed by `palette_picker.selected(str)`; `palette_preview(object)` carries a `Palette`-or-`None`; `move_swatch(int, int)`; `list_by_category() -> dict[str, list[str]]` consumed by `populate`. Names consistent across tasks.
