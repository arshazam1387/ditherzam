---
type: progress
phase: 8
status: done
date: 2026-07-08
---

Built-in palette library expanded + palette picker enlarged (commit `bac7b7c`, on local main).

- 6 new builtin YAMLs in `ditherzam/color/builtin/`: **c64**, **zxspectrum** (category `retro`);
  **nord**, **solarized** (NEW category `cool`); **greencrt**, **ambercrt** (`mono` CRT ramps).
  Categories come straight from each YAML's `category:` field (PaletteStore.list_by_category
  reads it; no hardcoded seed map). Library is now cool(2) / mono(4: ambercrt, grayscale,
  greencrt, sepia) / retro(5: c64, cga, gameboy, pico8, zxspectrum).
- `ui/palette_picker.py`: `setMinimumHeight(260)` + `setIconSize(112,22)` and `_swatch_icon`
  defaults 112x22 — taller list, bigger swatches ("make the palette section bigger").
- Tests: added to `tests/test_palette.py` (new builtins valid+categorised) and
  `tests/test_palette_picker.py` (picker enlarged, `cool` header + new names visible). Existing
  builtin tests use `in`/per-palette asserts (no exact count), so adding palettes is safe.
  606 green JIT-off.

Chosen for good luminance spread so they read well through the tone-based colour engine
([[023-binary-kernels-ignore-depth-fix]]). Still open if wanted: raise default Depth so palettes
show colour out-of-the-box (currently depth defaults to 2 → 2 tones until raised). Builds on
[[021-color-palette-library-c-shipped]].
