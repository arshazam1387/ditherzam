---
type: progress
phase: 8
status: done
date: 2026-07-08
---

**Sub-project B (palette editing UX) shipped & MERGED to main** (merge commit e406a7d;
6 commits a43f9c8..0782ef2). Built via subagent-driven-development (fresh implementer +
reviewer per task, opus final whole-branch review). **545 green** in the default JIT-off
mode. Spec `docs/superpowers/specs/2026-07-08-palette-editing-ux-design.md`, plan
`docs/superpowers/plans/2026-07-08-palette-editing-ux.md`.

**What landed:**
- `color/palette_store.py` — new Qt-free `PaletteStore(user_dir=None)`: per-palette user
  YAML in a config dir (`%APPDATA%/ditherzam/palettes`, POSIX fallback). User files
  **shadow** builtins = the copy-on-edit fork mechanism. `list/get/is_user/is_builtin/
  save/delete/reset_to_builtin`. `get()` always returns an independent copy (no builtin
  aliasing). Dir created lazily on first `save`.
- `color/palette.py` — added pure `generate_palette(rgb_u8, unit, value, name="from image")`
  dispatcher: unit `"k"`→`extract_palette(k)`, `"pct"`→`source_palette(completeness)`.
- `ui/palette_editor.py` — new `SwatchStrip(QWidget)`, signal `edited(Palette)`. Edits an
  in-memory working Palette: recolor (`set_swatch_color`, separate from the `QColorDialog`
  so it's testable), add/remove (1-swatch min, lock reindex), per-swatch lock, `shuffle`.
  `set_palette` adopts a copy and does NOT emit (prevents signal recursion).
- `ui/controls.py` — `ControlPanel(store=None)` owns the store + `working_palette`; palette
  combo populated from `store.list()` (dropped hardcoded `_PALETTES`); embeds SwatchStrip;
  Shuffle/Save/Reset/From-Image buttons; `extract_slider` + **k/% unit combo** (adapts
  range 2-64 vs 0-100) + **Autosave checkbox**. New state keys `palette_autosave`,
  `extract_unit` live in `panel.state` ONLY (never in RenderSettings — frozen path).
  `set_working_palette` syncs the combo label (`_sync_palette_combo`, blocked signals).
- `ui/main_window.py` (class `ImageEditor`) — retains source RGB (`_base_rgb`) via
  `load_array(gray, rgb_u8=None)` + `_on_image_dropped`; `_current_palette()` returns
  `panel.working_palette`; `_on_from_image_requested` runs `generate_palette` on the loaded
  RGB; preset load restores via `set_working_palette`. Presets already serialize palette
  colors → edited/from-image palettes round-trip for free.

**Deferred to Sub-project C (next):** drag-reorder swatches, palette categories/picker,
hover-scroll preview on current image, import/share. Also deferred minors: cache
`builtin_palettes()` (re-globs each call); sanitize palette name→filename (no rename path
until C). See [[017-color-depth-ramp-shipped]]; next = C.
