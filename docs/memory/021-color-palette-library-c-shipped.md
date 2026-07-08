---
type: progress
phase: 8
status: done
date: 2026-07-08
---

Sub-project C (palette library) shipped + MERGED to main (merge commit `007f84d`),
7 TDD tasks + a final-review fix; 578 green JIT-off. Builds on [[019-color-palette-editing-b-shipped]].

What C added:
- `Palette.category` field (Qt-free; YAML writes it only when non-empty; legacy files
  load as ""); `extract_palette`/`source_palette`/`generate_palette` tag results `"user"`.
- Builtin category seeds (gameboy/pico8/cga→`retro`, grayscale/sepia→`mono`) +
  `PaletteStore.list_by_category()` (alpha order, `uncategorized` pinned last, user shadow wins).
- `ditherzam/ui/palette_picker.py` — `PalettePicker(QTreeWidget)` category-grouped picker
  with swatch icons; signals `selected(str)` + `preview(object)`; hover/keyboard/wheel-cycle
  preview. REPLACES the old flat `palette_combo` in controls.py.
- controls.py: editable Category combo (set on Save); session prefs `palette_preview`
  (default True) + `palette_wheel_cycle` (default False) as checkboxes — in `panel.state` ONLY.
- main_window preview seam: `_preview_palette` + `_on_palette_preview`; `_current_palette()`
  returns the preview palette when set without mutating `working_palette`.
- `SwatchStrip.move_swatch()` drag-reorder (+ `_SwatchButton` drag gesture, `dragMoveEvent`).
- Preset round-trip of `Palette.category`.

Deferred (cosmetic/non-blocking polish, from the opus whole-branch review): from-image
palette not-in-tree leaves a stale picker highlight; itemClicked+itemActivated can double-emit
`selected` (idempotent); category_combo suggestions not refreshed after save; picker test uses
private handlers (wheelEvent/leaveEvent uncovered); double YAML read per populate.

**Import/share is now the only major deferred color item** (was C's 4th candidate, cut from scope).
