---
type: progress
phase: 6
status: done
date: 2026-07-05
---

Phase 6 presets & export done. Core (Qt-free): `ditherzam/presets.py`
(`settings_to_preset`/`preset_to_settings` with range clamping, `PresetManager`
save/list/load/delete/import), `ditherzam/export/raster.py` (`save_raster` PNG/JPG),
`ditherzam/export/vector.py` (`raster_to_svg`, vertical run-merged, threshold
exclusive, invert), `ditherzam/batch.py` (`batch_process` size-gated → PNG).
UI: `ditherzam/ui/export_actions.py` (`create_export_menu`/`MENU_SPEC`, only PySide6
import this phase) + handlers wired into existing `ImageEditor` via `_wire_export()`
(reuses `settings_from_controls`, `self.pipeline`; adds `_apply_preset`, palette from
`builtin_palettes()`). 258 tests green headless + offscreen. Next: Phase 7.
