# Handoff — Color System Sub-projects B & C

Paste this into a fresh Claude Code session in `C:\Users\arsha\Desktop\custom dither`
to build the next parts of the color system.

---

We're extending the ditherzam color system. **Sub-project A (the depth-ramp engine)
is already built and merged to `main`.** Now build **B (palette editing UX)** and
then **C (palette library)** — each as its own brainstorm → spec → plan → build cycle.

**Start by invoking the `zam-memory` skill (recall)** to load project state, then read
the A spec at `docs/superpowers/specs/2026-07-07-color-system-core-design.md` (its
"Build order note" and out-of-scope list define B and C).

## What already exists (build on it, don't rebuild)

- `ditherzam/color/palette.py` — `Palette` dataclass (float32[K,3], `from_list`,
  YAML `load`/`to_yaml`, `shuffle(locked, rng)` already supports per-swatch locks at the
  MODEL layer), plus `extract_palette` (median-cut), `source_palette(completeness)`,
  `builtin_palettes()` (loads `color/builtin/*.yaml`).
- `ditherzam/color/ramp.py` — `build_ramp(palette, depth, mapping, phase)` +
  `RAMP_MODES`. `phase` (0..1) rotates the ramp; currently unused, reserved for animation.
- `ditherzam/color/engine.py` — `ColorEngine(palette, mode, depth, mapping, phase)`,
  mode `"ramp"`.
- UI: `ditherzam/ui/controls.py` `ControlPanel` — plain `self.state` dict; `_COLOR_MODES`
  includes `"ramp"`; has a Palette combo (`_PALETTES` hardcoded list), Mode combo, Depth
  slider, Mapping combo. `ResettableGlowSlider` + `InvisibleSpinBox` (number display needs
  an explicit `valueChanged.connect` — see memory 016). `NoScrollComboBox` for dropdowns.
- `ditherzam/ui/main_window.py` — `_build_color_engine` builds `ColorEngine(palette, mode)`
  from panel state; `_current_palette` reads `builtin_palettes().get(state["palette"])`.
- `ditherzam/ui/settings_map.py` + `ditherzam/presets.py` — round-trip settings incl.
  `depth`/`color_mapping`.

## Sub-project B — Palette editing UX (do this first)

Goal: edit palettes in-app. Scope to brainstorm:
- In-app **swatch editing** (add/remove/recolor individual colors; color picker).
- Surface **per-swatch lock + shuffle** in the UI (model `Palette.shuffle(locked, rng)`
  exists — wire lock toggles + a Shuffle button; Dither Boy locks a swatch then randomizes
  the rest). Consider the same lock+randomize idea for adjustment sliders.
- **Source/auto palette generation** UI (wire `source_palette`/`extract_palette` from the
  loaded image; a "completeness"/K control).
- The main cost per the A spec is the **UI state refactor** from the single hardcoded
  `_PALETTES` list + `state["palette"]` string to an editable, user-owned palette object
  (likely a per-session palette store on disk under a user palettes dir). Design that
  boundary carefully in brainstorming.

## Sub-project C — Palette library (after B)

Goal: browse/manage many palettes. Scope to brainstorm:
- Palette **categories** (e.g. retro), a picker organized by category.
- **Hover/scroll preview** of a palette applied to the current image.
- **Import / share** (load a palette file, export one) — YAML `load`/`to_yaml` exist.

## Invariants (must hold — verify against the tree, don't trust from memory)

- **Clean-room**: our own implementation; no Studio/Dither-Boy code, strings, or binaries.
  Dither Boy is inspiration for *behavior* only.
- **Qt-free core**: `color/**` and `dithering/**` must not import PySide6; only `ui/`,
  `app.py`, `video/workers.py` import Qt.
- **Python 3.12; TDD per task.** Tests run `NUMBA_DISABLE_JIT=1`; Qt tests
  `QT_QPA_PLATFORM=offscreen`. Runner: `./.venv/Scripts/python.exe -m pytest`. Keep the
  full suite green (currently **511**) in both JIT modes.
- Don't touch the frozen `RenderPipeline.render()` stage order.

## Workflow that worked for A (reuse it)

1. `superpowers:brainstorming` → design doc in `docs/superpowers/specs/`, get approval,
   commit.
2. `superpowers:writing-plans` → bite-sized TDD plan in `docs/superpowers/plans/`.
3. `superpowers:subagent-driven-development` → branch off `main` first (do NOT build on
   main; the repo-root `.venv` means use a **branch in-place**, not a worktree — a worktree
   won't have the venv). Fresh implementer subagent per task (sonnet; plan has full code),
   then a task reviewer subagent, fix loop, ledger at `.superpowers/sdd/progress.md`,
   final whole-branch review on opus, then `finishing-a-development-branch`.
4. Record results via `zam-memory` and update `docs/memory/INDEX.md`.

## Known deferred items to fold in if relevant

- On-engine ramp cache is bypassed in the live path (`main_window` rebuilds the engine per
  render) — negligible, but if B/C change engine construction, consider fixing.
- `settings_from_controls` still defaults `blur` to 50 when the key is absent (latent; panel
  provides 0). Out of scope unless you touch it.

First step: invoke `zam-memory` (recall), then `superpowers:brainstorming` for **Sub-project
B**. Ask me clarifying questions one at a time before proposing a design.
