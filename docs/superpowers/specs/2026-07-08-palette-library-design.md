# Palette Library (Sub-project C) — Design

**Date:** 2026-07-08
**Status:** Approved (brainstorming) → ready for implementation plan
**Sub-project:** C of 3 (A = depth-ramp engine [shipped/merged], B = palette editing UX [shipped/merged], C = palette library)

## Goal

Turn the flat palette list into an organized, browsable library. Today the UI
picks a palette from a flat `NoScrollComboBox` populated by `PaletteStore.list()`
([`controls.py`](../../../ditherzam/ui/controls.py)), with no grouping and no way
to compare palettes against the current image before committing. C adds palette
**categories**, a **category-organized picker** with per-palette swatch previews,
**hover/scroll preview** of a palette applied to the loaded image (without
committing it), and **drag-to-reorder swatches** (deferred from B; consumed by the
`glitch`/`banded` stored-order ramp mappings).

C builds directly on B's `PaletteStore` and `SwatchStrip`. It adds no dithering or
ramp math.

### In scope (C)
- **Categories:** an optional `category` field on `Palette` (in YAML), builtins
  seeded, free-form user-assignable on save. `PaletteStore.list_by_category()`.
- **Picker:** an always-visible `PalettePicker(QTreeWidget)` grouping palettes by
  category, each row showing a mini swatch preview. Replaces the flat combo.
- **Hover/scroll preview:** hovering a palette row previews it on the current image
  via a transient render that never mutates the working palette; a master on/off
  setting (default on) and a separate wheel-cycle setting (default off).
- **Drag-reorder swatches:** reorder `SwatchStrip` swatches, reindexing locks.

### Out of scope (deferred, YAGNI here)
- **Import / share** of palette files (and therefore the deferred filename
  sanitization — category strings are not filenames, so no new sanitization risk
  is introduced by C).
- CMYK halftone; animating ramp `phase`.

## Invariants (must hold)

- **Clean-room** — Dither Boy is behaviour inspiration only; no code, strings, or
  binaries.
- **Qt-free core** — `color/**` (incl. `palette.py`, `palette_store.py`) stays
  Qt-free; only `ui/`, `app.py`, `video/workers.py` import PySide6.
- **Frozen `RenderPipeline.render()` stage order** unchanged; no change to
  `RenderSettings` color fields. Preview lives entirely in the window layer via
  `_current_palette()`; UI-session prefs stay in `panel.state`, never in
  `RenderSettings`.
- Python 3.12; TDD per task (red → green → refactor, commit per green); tests run
  `NUMBA_DISABLE_JIT=1`, Qt tests `QT_QPA_PLATFORM=offscreen`, runner
  `./.venv/Scripts/python.exe -m pytest`. Full suite must stay green (currently
  **545** in the default JIT-off mode).

## Known pre-existing breakage (not C's)

Under `NUMBA_DISABLE_JIT=0` (JIT-on) 7 kernel tests fail (`special.py` float-index).
Pre-existing on `main`; the default JIT-off mode is green. "Keep the suite green"
means **JIT-off**. Do not attribute those failures to C.

## Architecture

### Component 1 — `ditherzam/color/palette.py` + `palette_store.py` (Qt-free core)

**`Palette` dataclass gains `category: str = ""`.**
- `from_list(name, rgb_list, category="")` — optional arg, default empty.
- `to_yaml` writes `category` **only when non-empty** (keeps existing empty-category
  YAML byte-stable / uncluttered); `load` reads `category`, default `""`.
- `shuffle` preserves `category` on the returned copy.
- `extract_palette` / `source_palette` / `generate_palette` set `category="user"`
  on their results so from-image palettes group sensibly.

**`PaletteStore` gains categorization:**
```
def list_by_category(self) -> dict[str, list[str]]
    # category -> sorted palette names.
    # A palette's category is read from its Palette.category.
    # Empty category is bucketed under "uncategorized".
    # Categories ordered alphabetically, with "uncategorized" pinned last.
    # User files still shadow builtins by name; the shadowing user file's
    # category wins for that name.
```
- Existing flat `list()` is retained (back-compat; other callers/tests use it).
- `get()` already returns a copy; it now also carries `category` through.
- Builtin YAMLs (`ditherzam/color/builtin/*.yaml`) get a `category:` line:
  `gameboy`, `pico8`, `cga` → `retro`; `grayscale`, `sepia` → `mono`.

`palette.py`'s ramp/extraction math is otherwise unchanged.

### Component 2 — `ditherzam/ui/palette_picker.py` (new Qt widget) + `controls.py`

**`PalettePicker(QTreeWidget)`** replaces the flat `palette_combo`.
- Top-level items are **category headers** (expandable, non-selectable, no palette
  payload). Child items are palettes, each with a **mini swatch-strip pixmap** icon
  (drawn from the palette's colors) plus the palette name; the palette name is
  stored in the item's `UserRole`.
- Signals:
  - `selected(str)` — user committed a palette (click / Enter on a palette row).
  - `preview(object)` — emit a `Palette` to preview, or `None` to revert. Only
    emitted when the master preview setting is on.
- `populate(store)` rebuilds the tree from `store.list_by_category()`, blocking
  signals during the rebuild so it never fires a spurious `selected` that would
  reset the working copy.
- `select(name)` programmatically highlights a palette row without emitting
  `selected` (used by `set_working_palette` / preset restore).
- Preview triggers:
  - **Hover** (`itemEntered`, mouse tracking on): emit `preview(store.get(name))`
    for the hovered palette row; on leave / hovering a header, emit `preview(None)`.
  - **Keyboard** selection change (arrow keys) also emits `preview(...)`.
  - **Wheel** (`wheelEvent`): if the wheel-cycle setting is on, step the highlighted
    palette row ±1 and emit `preview(...)` (commit on click/Enter); otherwise fall
    through to default list scrolling.

**`controls.py` rewire:**
- Remove `palette_combo`; add `self.palette_picker = PalettePicker()` and call
  `palette_picker.populate(self.store)`.
- `palette_picker.selected` → `_on_palette_changed(name)` (unchanged body: set
  `working = store.get(name)`, update swatch strip, reset-enabled, `changed`).
- `palette_picker.preview` → re-emit on a new `ControlPanel.palette_preview(object)`
  signal that `main_window` connects to (keeps `controls.py` free of render logic).
- `set_working_palette(palette)` keeps its public signature; internally calls
  `palette_picker.select(palette.name)` instead of `_sync_palette_combo`. The
  combo-era helpers (`_sync_palette_combo`, `_refresh_palette_combo`) become picker
  repopulate/select calls; `_on_save_palette` / `_on_reset_palette` repopulate then
  select.
- **Category assignment UI:** an editable `NoScrollComboBox` ("Category") next to
  *Save palette*, its items seeded from the store's existing categories. On save,
  the working palette's `category` is set from this combo's text before
  `store.save(working)`. Selecting a palette syncs the combo to that palette's
  category.
- New `state` keys: `palette_preview: bool` (default **True**),
  `palette_wheel_cycle: bool` (default **False**). Two checkboxes wire them and pass
  the current values down to the picker.

### Component 3 — Hover/scroll preview seam (`main_window.py`)

- `ImageEditor.__init__` adds `self._preview_palette = None`.
- `_current_palette()` returns `self._preview_palette` when it is not `None`, else
  `self.panel.working_palette` (still returns `None` when color mode is `off`).
- Connect `panel.palette_preview(object)` →
  ```
  def _on_palette_preview(self, palette):
      if not self.panel.state.get("palette_preview", True):
          return
      self._preview_palette = palette          # Palette or None
      self.render_now()                          # existing debounced, proxied path
  ```
- `panel.changed` already fires on commit (`selected`); the commit handler clears
  `_preview_palette` (set to `None`) so the committed working palette renders.
- No synchronous per-pixel rendering: hover reuses the existing render debounce and
  preview proxy from the optimization pass.

### Component 4 — Drag-reorder swatches (`ditherzam/ui/palette_editor.py`)

- Add `move_swatch(src_i: int, dst_i: int)` to `SwatchStrip`:
  - Reorder the `colors` array by moving row `src_i` to position `dst_i`.
  - Remap the `_locked` index set to follow the move (same discipline as
    `remove_swatch`'s reindex).
  - `_rebuild()` then `edited.emit(self._palette)`.
- The drag gesture: swatch buttons become drag sources (mouse-press starts a
  `QDrag`) and drop targets; a drop computes the target index and calls
  `move_swatch`. Headless tests exercise `move_swatch` directly (the gesture is thin
  glue over it).
- This reorder is what `glitch` / `banded` ramp mappings consume (they use stored
  palette order).

### Component 5 — Round-trip (`presets.py`, `settings_map.py`)

- `settings_to_preset`: the serialized palette blob gains `category`
  (`palette.category`).
- `preset_to_settings`: reads `category` back into the reconstructed `Palette`
  (default `""`), so an edited/from-image palette's category round-trips.
- `palette_preview` and `palette_wheel_cycle` are session prefs carried by
  `settings_map` / preset-apply where the other session prefs already flow; they are
  **never** added to `RenderSettings`.

## Testing (TDD)

Every task writes tests first. All pass JIT-off; Qt tests offscreen.

- **Core (no Qt):**
  - `Palette.category` YAML round-trip, including the empty default (no `category:`
    key written when empty; `load` of an old category-less YAML yields `""`).
  - `list_by_category` groups by category in the defined order, buckets empty under
    `uncategorized`, and a user file shadows the builtin of the same name within its
    category.
  - `get()` carries `category`; `extract_palette`/`generate_palette` produce
    `category == "user"`.
- **Picker (offscreen):**
  - Tree has one header per category (defined order) and correct palette children.
  - Clicking a palette row emits `selected(name)`; header rows are non-selectable.
  - Hover emits `preview(Palette)` only when `palette_preview` is on; leaving emits
    `preview(None)`.
  - Wheel steps selection + previews only when `palette_wheel_cycle` is on; else the
    event is not consumed as a cycle.
  - `populate`/`select` do not emit `selected`.
- **Preview seam (offscreen):** setting `_preview_palette` makes `_current_palette`
  return it without changing `panel.working_palette`; `preview(None)` reverts;
  preview is ignored when `palette_preview` is off.
- **Drag-reorder:** `move_swatch(src, dst)` reorders colors and remaps locks
  correctly (incl. moving a locked swatch and moving across a locked one); emits
  `edited`.
- **Round-trip:** `category` and both toggles survive preset/settings round-trips.
- **Regression:** full suite stays green (≥545, JIT-off).

## Risks / gotchas

- **Combo → tree churn** — several `controls.py` methods and their tests reference
  `palette_combo`. Keeping `set_working_palette`'s public signature contains the
  blast radius (`main_window` preset restore is unchanged); picker-specific tests
  are rewritten.
- **Signal loops on repopulation** — `populate`/`select` must block signals so a
  rebuild never fires a spurious `selected` that resets the working copy (same
  discipline the old combo used with `blockSignals`).
- **Preview render cost** — mitigated by reusing the existing debounce + preview
  proxy; nothing renders synchronously on each hover pixel. Preview is skippable
  entirely via the master setting.
- **Preview vs color-mode-off** — when color mode is `off`, `_current_palette`
  returns `None`, so hovering shows no change. Intended: preview is for comparing
  palettes while color is active.
- **Number-display widgets** — any new slider's number display needs an explicit
  `valueChanged.connect` (memory 016); C adds no new value sliders, only checkboxes
  and combos, so this is a watch-item, not a task.

## Build order note

C depends on A and B (both shipped/merged). C is the last color-system sub-project;
import/share remains the only major deferred color item after C.
