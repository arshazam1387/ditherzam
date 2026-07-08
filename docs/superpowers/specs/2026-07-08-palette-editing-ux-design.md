# Palette Editing UX (Sub-project B) — Design

**Date:** 2026-07-08
**Status:** Approved (brainstorming) → ready for implementation plan
**Sub-project:** B of 3 (A = depth-ramp engine [shipped/merged], B = palette editing UX, C = palette library)

## Goal

Make palettes editable and user-owned inside the app. Today the color system can
*use* palettes (Sub-project A's ramp engine), but the UI only picks one of five
hardcoded builtin names ([`controls.py`](../../../ditherzam/ui/controls.py) `_PALETTES`) and
[`main_window._current_palette`](../../../ditherzam/ui/main_window.py) resolves the name
straight out of `builtin_palettes()`. There is no way to recolor a swatch, add/remove
colors, lock+shuffle in the UI, generate a palette from the current image, or keep any of
it. B adds an editable, user-owned palette store plus the swatch-editing UI on top of it.

The model layer already supports the operations — `Palette.shuffle(locked, rng)`,
`extract_palette(rgb_u8, k)`, `source_palette(rgb_u8, completeness)`,
`Palette.from_list/to_yaml/load`. B is mostly the **store boundary** and the **Qt UI** that
drives those existing primitives. No new dithering or ramp math.

### In scope (B)
- New Qt-free `color/palette_store.py`: user-owned palette repository (per-palette YAML in a
  user config dir), with builtin↔user shadowing and fork/reset semantics.
- Swatch-editing UI: add / remove / recolor swatches, per-swatch lock, Shuffle button.
- "From Image" palette generation (median-cut extraction on the loaded image), with a count
  control whose framing (`k` exact colors vs `%` completeness) is a user setting.
- Copy-on-edit: editing a builtin lazily forks it into an in-memory working palette; the
  builtin is never mutated and is recoverable.
- Persistence: explicit "Save palette" action by default, plus an autosave setting.
- Wire the palette combo + name-resolution seam through the store; preset round-trip keeps
  working (presets already serialize the `Palette` object, not just its name).

### Out of scope (deferred to C, YAGNI here)
- Drag-to-reorder swatches (only affects `glitch`/`banded` stored-order mappings).
- Palette library: categories, palette picker, hover-scroll preview on the current image.
- Import / share of palette files.
- CMYK halftone; animating ramp `phase` (separate deferred items).

## Invariants (must hold)

- **Clean-room** — Dither Boy is behaviour inspiration only; no code, strings, or binaries.
- **Qt-free core** — `color/palette_store.py` and everything under `color/**` and
  `dithering/**` stay Qt-free; only `ui/`, `app.py`, `video/workers.py` import PySide6.
- **Frozen `RenderPipeline.render()` stage order** unchanged; no change to `RenderSettings`
  color fields or the render path. B touches palette *provisioning*, not the pipeline.
- Python 3.12; TDD per task (red → green → refactor, commit per green); tests run
  `NUMBA_DISABLE_JIT=1`, Qt tests `QT_QPA_PLATFORM=offscreen`, runner
  `./.venv/Scripts/python.exe -m pytest`. Full suite must stay green (currently **511**) in
  **both** JIT-on and JIT-off modes.

## Architecture

### Component 1 — `ditherzam/color/palette_store.py` (new, Qt-free)

A dumb, testable file repository over the user config dir. Composes the existing
`palette.py` primitives; adds no palette math.

```
class PaletteStore:
    def __init__(self, user_dir: Path | None = None)   # default: user config dir
    def list(self) -> list[str]                 # builtin names ∪ user names (user shadows)
    def get(self, name: str) -> Palette         # user copy if present, else builtin; a COPY
    def is_user(self, name: str) -> bool
    def is_builtin(self, name: str) -> bool
    def save(self, palette: Palette) -> None    # write user_dir/<name>.yaml (Palette.to_yaml)
    def delete(self, name: str) -> None         # remove user yaml; builtin re-emerges
    def reset_to_builtin(self, name: str) -> Palette   # delete fork, return builtin copy
```

- **User dir:** platform config path; on Windows `%APPDATA%/ditherzam/palettes/`. Resolve the
  path directly from `os`/`Path` (e.g. `%APPDATA%`, fallback `~/.config/ditherzam/palettes`)
  to avoid adding a `platformdirs` dependency unless one already exists — settle in planning.
  Directory is created lazily on first `save`.
- **Schema:** one YAML per palette, identical to builtins, so `Palette.load`/`to_yaml` are
  reused unchanged.
- **Shadowing (the fork mechanism):** `list()` returns builtin names plus user names,
  de-duplicated; a user file with a builtin's name shadows it. `get()` prefers the user copy.
- **`get()` always returns a copy** (`Palette` with a copied `colors` array) so callers can
  mutate a working palette without touching the source (builtin arrays especially).
- Copy-on-edit itself lives in the UI layer; the store only sees a palette on explicit
  `save()`. This keeps the store decoupled from the render loop and trivially unit-testable
  with a `tmp_path` `user_dir`.

`palette.py` is unchanged.

### Component 2 — `ditherzam/ui/palette_editor.py` (new Qt widget)

Keeps `controls.py` from bloating. Exposes a `SwatchStrip(QWidget)` and the From-Image group.

- **SwatchStrip:** renders the working palette as one clickable cell per color, each with a
  lock toggle. Holds a reference to the working `Palette`. Emits `edited(Palette)` on any
  mutation.
  - click swatch → `QColorDialog` → recolor that row
  - `+` cell → append a swatch (copy of the last color)
  - ✕ / right-click on a swatch → remove it (minimum 1 swatch enforced)
  - per-cell lock toggle; the locked index set drives Shuffle
- **Buttons:** `Shuffle` (→ `Palette.shuffle(locked, rng)`), `Save palette` (→ `store.save`),
  `Reset to builtin` (enabled only when the current name shadows a builtin), `From Image`.
- **From Image group:** button + one count slider. The slider's mode is read from the
  `extract_unit` setting:
  - `k`: slider 2–64, exact colors → `extract_palette(rgb_u8, k=value)`.
  - `pct`: slider 0–100 completeness → `source_palette(rgb_u8, completeness=value/100)`
    (internally maps to k, clamped to a sane maximum).
  Both paths produce a new working `Palette`; result becomes an editable working copy with
  fork/collision handling identical to builtin edits.

### Component 3 — `controls.py` + `main_window.py` wiring

- **`controls.py`:**
  - Delete the hardcoded `_PALETTES` list. Populate `palette_combo` from `store.list()`.
  - Add the `SwatchStrip` + From-Image group under the Palette dropdown.
  - Hold the working `Palette` object on the panel (not in the plain-value `state` dict).
  - New `state` keys: `palette_autosave: bool` (default `False`), `extract_unit: str`
    (`"k"` | `"pct"`, default `"k"`).
- **Data flow (copy-on-edit):**
  1. Selecting a palette name → panel sets `working = store.get(name)` (a fresh copy).
  2. Any swatch edit / shuffle / From-Image mutates `working` in place → `changed` emitted →
     existing debounced re-render (live). If `palette_autosave`, the same debounce also calls
     `store.save(working)`.
  3. `Save palette` → `store.save(working)`; combo refreshes so the fork appears/shadows.
  4. `Reset to builtin` → `working = store.reset_to_builtin(name)`.
- **`main_window.py`:** owns one `PaletteStore`. `_current_palette()` returns the panel's
  working `Palette` (already resolved) instead of re-looking-up the name via
  `builtin_palettes()`. `_build_color_engine` / `_current_color_engine` downstream unchanged.
- **Settings/preset round-trip:** presets already serialize the actual `Palette` (colors), so
  a preset saved with a user/edited palette restores correctly. The two new setting toggles
  persist through `settings_map`. No change to the frozen render path or `RenderSettings`
  color fields.

## Testing (TDD)

Every task writes tests first. All pass in both JIT modes; Qt tests offscreen.

- **`palette_store.py` (no Qt):** `save`→`list`→`get` round-trip; a user file shadows the
  builtin of the same name in both `list` and `get`; `delete` makes the builtin re-emerge;
  `reset_to_builtin` returns the builtin and drops the fork; `is_user`/`is_builtin`; each test
  uses an isolated `tmp_path` `user_dir`.
- **Copy-on-edit semantics:** mutating a working copy from `get()` never changes the builtin;
  two `get()` calls on the same name return independent `colors` arrays.
- **UI (offscreen):** recolor updates the working palette and emits `changed`; add/remove
  respects the 1-swatch minimum; the locked index set is passed to `shuffle`; From-Image in
  both `k` and `pct` modes yields the expected color count; combo repopulates after `save`.
- **Settings:** `palette_autosave` and `extract_unit` survive a settings round-trip.
- **Regression:** full suite stays green (≥511) in both JIT modes.

## Risks / gotchas

- **Working-copy aliasing** — the panel must hold a *copy*, not the store's/builtin's array,
  or edits corrupt the builtin. Enforced by `get()` copying + a copy-on-edit test.
- **Combo repopulation on save** — after `save`/`delete` the combo must refresh without firing
  a spurious palette-change that resets the working copy; block signals during repopulation.
- **Autosave disk churn** — gated behind the setting and share the render debounce; off by
  default.
- **Config-dir portability** — resolve `%APPDATA%` with a POSIX fallback; create lazily so
  a read-only environment (tests) never writes unless `save` is called.

## Build order note

B depends on A (shipped). C (palette library: categories/picker, hover-scroll preview,
import/share, drag-reorder) builds on B's `PaletteStore` and swatch UI.
