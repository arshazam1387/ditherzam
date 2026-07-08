# Handoff — Color System Sub-project C (Palette Library)

Paste this into a fresh Claude Code session in `C:\Users\arsha\Desktop\custom dither`
to build the last part of the color system.

---

We're extending the ditherzam color system. **Sub-project A (depth-ramp engine) and
B (palette editing UX) are both built and merged to `main`.** Now build **C (palette
library)** as its own brainstorm → spec → plan → build cycle.

**Start by invoking the `zam-memory` skill (recall)** to load project state, then read:
- the A spec `docs/superpowers/specs/2026-07-07-color-system-core-design.md` (its
  out-of-scope list seeds C), and
- the B spec `docs/superpowers/specs/2026-07-08-palette-editing-ux-design.md` +
  plan `docs/superpowers/plans/2026-07-08-palette-editing-ux.md` (C builds directly on
  B's `PaletteStore` and `SwatchStrip`).

## What already exists (build on it, don't rebuild)

**Model / core (Qt-free):**
- `ditherzam/color/palette.py` — `Palette` dataclass (float32[K,3]; `from_list`, YAML
  `load`/`to_yaml`, `shuffle(locked, rng)`), `extract_palette` (median-cut),
  `source_palette(completeness)`, `builtin_palettes()`, and `generate_palette(rgb_u8,
  unit, value, name)` (From-Image dispatcher: `"k"`/`"pct"`).
- `ditherzam/color/palette_store.py` — **`PaletteStore(user_dir=None)`** (B): per-palette
  user YAML in a config dir (`%APPDATA%/ditherzam/palettes`, POSIX fallback). User files
  **shadow** builtins (copy-on-edit fork). API: `list()`, `get(name)` (independent copy),
  `is_user`, `is_builtin`, `save(palette)`, `delete(name)`, `reset_to_builtin(name)`.
  **This is C's foundation** — categories/import/share extend this store.
- `ditherzam/color/ramp.py`, `engine.py` — ramp engine (A); untouched by C.

**UI (Qt):**
- `ditherzam/ui/palette_editor.py` — **`SwatchStrip(QWidget)`** (B), signal `edited(Palette)`;
  recolor/add/remove/per-swatch-lock/shuffle over an in-memory working `Palette`.
- `ditherzam/ui/controls.py` — `ControlPanel(store=None)` owns a `PaletteStore` +
  `working_palette`; palette combo from `store.list()`; Shuffle/Save/Reset/From-Image
  buttons; k/% unit combo + Autosave checkbox; `set_working_palette(palette)` (syncs the
  combo via `_sync_palette_combo`). State keys `palette_autosave`, `extract_unit` live in
  `panel.state` ONLY (never in RenderSettings). `NoScrollComboBox`, `ResettableGlowSlider`,
  `InvisibleSpinBox` (number display needs an explicit `valueChanged.connect` — memory 016).
- `ditherzam/ui/main_window.py` — class **`ImageEditor`** (NOT `MainWindow`). Retains source
  RGB (`_base_rgb`) via `load_array(gray, rgb_u8=None)` + `_on_image_dropped`;
  `_current_palette()` returns `panel.working_palette`; `_on_from_image_requested` runs
  `generate_palette` on `_base_rgb`; preset load restores via `set_working_palette`.
- `settings_map.py` + `presets.py` — round-trip settings incl. `depth`/`color_mapping`;
  presets serialize palette **colors** (so edited/from-image palettes round-trip already).

## Sub-project C — Palette library (scope to brainstorm)

From the A/B out-of-scope lists, C covers:
- **Categories** — group palettes (e.g. retro, source, user) and a picker organized by
  category. Decide where category metadata lives: a field in each palette YAML, or a
  directory-per-category layout under the store. This is the main design decision — extend
  `PaletteStore` (its `list()` currently returns a flat sorted name list).
- **Hover / scroll preview** — preview a palette applied to the **current image** on hover
  or scroll, without committing it (transient render). Note `_current_palette()` /
  `working_palette` is the live seam; preview must not clobber the working palette.
- **Import / share** — import a palette file into the store and export one out. YAML
  `load`/`to_yaml` + `PaletteStore.save`/`get` exist; add the UI + a file dialog. Fold in
  the deferred **filename sanitization** (B note: `save()` writes `<palette.name>.yaml`
  verbatim — sanitize names with `/`, `:`, etc. once import/rename can introduce them).
- **Drag-reorder swatches** — deferred from B; only affects `glitch`/`banded` mappings
  (which use STORED palette order). Wire drag-drop in `SwatchStrip`, reindexing the locked
  set the same way `remove_swatch` does.

## Invariants (must hold — verify against the tree, don't trust from memory)

- **Clean-room**: our own implementation; no Studio/Dither-Boy code, strings, or binaries.
  Dither Boy is inspiration for *behavior* only.
- **Qt-free core**: `color/**` (incl. `palette_store.py`) and `dithering/**` must not import
  PySide6; only `ui/`, `app.py`, `video/workers.py` import Qt.
- **Python 3.12; TDD per task.** Tests run `NUMBA_DISABLE_JIT=1`; Qt tests
  `QT_QPA_PLATFORM=offscreen`; `qapp_fixture` session fixture. Runner:
  `./.venv/Scripts/python.exe -m pytest`. Keep the full suite green (currently **545** in
  the default JIT-off mode).
- Don't touch the frozen `RenderPipeline.render()` stage order or `RenderSettings` color
  fields. UI-session prefs stay in `panel.state`, never in `RenderSettings`.

## ⚠️ Known pre-existing breakage (memory 020 — NOT from color work)

Under `NUMBA_DISABLE_JIT=0` (JIT-on) **7 kernel tests fail** — `special.py` `_topography`/
`_wireframe_alt`/`_diagonal` index arrays with a float64 subscript that a stricter Numba
now rejects. Reproduces on `main` before B. The default test mode (JIT-off) is fully green.
So "keep the suite green" currently means **JIT-off**. If you need both modes green, fix
`special.py` (cast indices to `int`) as its own task first — don't attribute those failures
to your C work.

## Workflow that worked for A and B (reuse it)

1. `superpowers:brainstorming` → design doc in `docs/superpowers/specs/`, approval, commit.
   Ask clarifying questions ONE at a time before proposing a design.
2. `superpowers:writing-plans` → bite-sized TDD plan in `docs/superpowers/plans/` with full
   code per step. Verify class/name references against the tree (B's plan mislabeled
   `ImageEditor` as `MainWindow` — cost a mid-task escalation).
3. `superpowers:subagent-driven-development` → **branch off `main` in-place** (NOT a
   worktree — the repo-root `.venv` won't exist in one). Fresh implementer subagent per task
   (haiku for pure transcription tasks, sonnet for integration), a task reviewer subagent
   after each, fix loop, ledger at `.superpowers/sdd/progress.md`, **opus** final
   whole-branch review, then `finishing-a-development-branch`.
4. Record results via `zam-memory` and update `docs/memory/INDEX.md`.

First step: invoke `zam-memory` (recall), then `superpowers:brainstorming` for **Sub-project
C**. Ask me clarifying questions one at a time before proposing a design.
