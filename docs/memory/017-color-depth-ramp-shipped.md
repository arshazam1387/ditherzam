---
type: progress
phase: 8
status: done
date: 2026-07-08
---

**Depth-ramp color system (Sub-project A) shipped on branch `feat/color-depth-ramp`**
(9 commits d481b10..304e869, NOT yet merged to main). Reverse-engineered from
Dither Boy 6.0: dither luminance into N tonal levels (1–64) and map a palette
across them as a tone **ramp**. Built via subagent-driven-development (implementer
+ reviewer per task); **511 green** both JIT modes; headless smoke passed (depth
sweep, all 6 mappings, match≠glitch, render==render_cached).

**What landed:**
- `color/ramp.py` — `build_ramp(palette, depth, mapping, phase=0.0) -> f32[depth,3]`
  + `RAMP_MODES = (match, interpolated, glitch, reverse, hue_cycle, banded)`
  (single source of truth for UI combo, preset validation, dispatch). Rec.601
  luminance sort (stable). `match`=nearest-swatch (hard palette colors, bounded by
  K), `interpolated`=linear blend, `glitch`=raw `palette[i%K]`, `reverse`=match
  flipped, `hue_cycle`=palette-independent HSV sweep, `banded`=lum-sorted repeat
  every K. `phase` wired but unused (reserved for animation — a later cycle).
- `color/engine.py` — new `ColorEngine` mode `"ramp"`: bins the dithered grayscale
  `level=clip(round(gray/255*(depth-1)),0,depth-1)` → `ramp[level]`; ramp cached on
  engine keyed by (palette bytes, depth, mapping, phase). Other modes untouched.
- N-level dithering: `dithering/nlevels.py` njit `quantize_to_levels`; generalized
  the shared cores `_floyd_steinberg/_atkinson/_diffuse` (error_diffusion.py) and
  both `_ordered` (ordered.py + error_diffusion.py). **`levels<=2` calls the exact
  original body → golden fixtures byte-identical.** 19 kernels marked
  `supports_levels=True` (14 error-diffusion incl bayer_4, 5 ordered).
- Plumbing: `DitherEntry.supports_levels`; `apply_dither(..., levels=2)`;
  `RenderSettings.depth`/`.color_mapping`; render + render_cached forward levels to
  both dither calls and sync ramp params; cache sigs include depth+mapping.
- UI: added `"ramp"` to `_COLOR_MODES` (makes it reachable), Depth slider 1–64
  (number-display sync wired per [[016-slider-number-display-fix]]), Mapping combo
  from RAMP_MODES. settings_map + presets round-trip depth+color_mapping.

**End-to-end wiring:** `main_window._build_color_engine` builds
`ColorEngine(palette, mode)`; when mode=="ramp" the pipeline syncs depth/mapping
from settings each render — so no main_window change was needed beyond the combo.

**Next:** merge to main; then Sub-project B (palette editing UX) and C (palette
library) — see [[005-project-state]] and the spec
`docs/superpowers/specs/2026-07-07-color-system-core-design.md`. Deferred Minor: on-engine
ramp cache is bypassed in the live path (main_window rebuilds the engine per
render) — negligible now, revisit if a phase/animation cycle needs it.
