# HANDOFF PROMPT — Produce heavily detailed completion task lists for ditherzam

> Paste everything below the line into a fresh Claude Code session (or a subagent
> prompt) that has the `ditherzam` repo as its working directory. It carries all
> context needed to produce the deliverable without re-deriving anything.

---

## Your role & deliverable

You are picking up **ditherzam**, an open-source PySide6 pixel-dither studio. The
architecture, spec, and phase plans already exist. **Your job is NOT to write app
code yet.** Produce a **heavily detailed, execution-ready TASK LIST for the
completion of each of the 8 subsystems** — one checklist document per subsystem,
at the same granularity as the existing `docs/plans/01-foundation-and-dither-core.md`
(bite-sized TDD steps with *complete, real code in every code step* — no
placeholders, no "TODO", no "implement X here").

Save them to `docs/tasklists/01..08-<subsystem>-tasks.md`. Each item is a
checkbox (`- [ ]`). Each subsystem's list must be independently executable by a
developer who has zero prior context, using only that document + the shared
contracts below.

**Definition of "detailed enough":** every kernel/function/class gets (1) its own
failing test with concrete assertions, (2) the exact command to run it and the
expected fail message, (3) the full implementation code, (4) the pass command +
expected result, (5) a `git commit` with a conventional message. For repetitive
sets (e.g. the 63 dither kernels, the built-in palettes), enumerate EVERY item by
name with its exact parameters/coefficients — do not collapse into "repeat for the
rest."

## What already exists (read these first, do not duplicate)

- `DITHER_BOY_FULL_SPEC.md` — the full requirements spec. §1–§16 = verified exact
  behavior (adjustment formulas, slider ranges, pipeline, all algorithm names +
  categories + slider configs, zoom/inertia, video ffmpeg commands, export, themes).
  §17 = the newer 6.0-era additions (color engine, CMYK halftone, stackable
  effects, temporal animation) from public sources.
- `docs/plans/00-ROADMAP.md` — architecture, repo file map, global constraints,
  frozen interface contracts, phase order.
- `docs/plans/01-foundation-and-dither-core.md` — **fully step-expanded** — use it
  as the exact template for depth and formatting.
- `docs/plans/02..08-*.md` — the other subsystems at task/interface level. Your job
  is to EXPAND these to Phase-1 depth (every step, every line of code, every test).

## Hard rules (never violate)

- **Clean-room / legal:** never copy, embed, or ship any Dither Boy / Studio AAA
  code, verbatim strings, URLs, licensing endpoints, or binaries. Public-domain
  algorithm *techniques* (Floyd–Steinberg, Atkinson, Bayer, median-cut, etc.) are
  fine. Do NOT reintroduce any licensing/telemetry/network code.
- **No ffmpeg binaries committed** — resolve them at runtime (`assets/ffmpeg/` then
  `shutil.which`).
- **Core stays Qt-free.** Only `ditherzam/ui/`, `ditherzam/app.py`, and
  `ditherzam/video/workers.py` may import PySide6. Anything testable headlessly must
  be a pure function tested without Qt.
- **TDD always** (red → green → refactor), **commit after every green**, DRY, YAGNI.
- **No placeholders in any code step.** If a step changes code, show the whole code.

## Environment (this machine)

- Repo (working dir): `C:\Users\arsha\Desktop\custom dither` — git repo, remote
  `origin` = `https://github.com/arshazam1387/ditherzam` (private). `gh` is
  authenticated as `arshazam1387` (at `C:\Program Files\GitHub CLI\gh.exe`).
- Python **3.12** is required. On this box a clean 3.12 lives at
  `C:\Users\arsha\AppData\Local\Temp\claude\...\scratchpad\dbwork\py312\python.exe`;
  system Pythons are 3.11 and 3.13 (NOT usable for the Numba/PySide6 target). If no
  3.12 is on PATH, note it in the task list's Task 0 (env bootstrap).
- Tests run with `NUMBA_DISABLE_JIT=1` for speed/coverage. Shell is Git Bash + PowerShell.

## FROZEN CONTRACTS (every task list must honor these names/types verbatim)

```python
# ditherzam/dithering/registry.py
@dataclass(frozen=True)
class DitherEntry:
    name: str; category: str; dims: int
    param_sliders: tuple[str, ...]; func: Callable; param_func: Callable | None = None
class DitherRegistry:
    def register(self, name, category, dims=2, param_sliders=(), param_func=None): ...  # decorator
    def get_entry(self, name) -> DitherEntry | None: ...
    def list_dithers(self) -> list[str]: ...
    def by_category(self) -> dict[str, list[str]]: ...

# kernels: @njit(cache=True, parallel=True)
def kernel(image_array: 'float32[H,W]', parameter, luminance_threshold_value: float) -> 'float32[H,W] 0..255'

# ditherzam/dithering/pipeline.py
def apply_dither(gray_f32, *, style, scale, luminance_threshold, params, registry,
                 preview_disabled=False, threshold_field=None) -> np.ndarray

# ditherzam/adjustments.py  (float32, 0..255)
apply_contrast(img, value)      # img * (value/50)
apply_midtones(img, value)      # gamma=max(1+(value-50)/200,0.1); 255*(img/255)**(1/gamma)
apply_highlights(img, value)    # img * (1+(value-50)/100)
apply_blur(img, value)          # PIL GaussianBlur radius=(value/10)**2 ; 0 -> identity
apply_invert(img, enabled)      # 255-img
apply_saturation(rgb, value)    # lum + (rgb-lum)*(value/50)

# ditherzam/color/engine.py
class ColorEngine:
    palette: Palette; mode: str   # "off"|"nearest"|"ordered"|"diffused"
    def map(self, gray_or_rgb_f32) -> np.ndarray  # uint8 HxWx3
# ditherzam/color/palette.py
class Palette: name: str; colors: np.ndarray  # float32[K,3]
def builtin_palettes() -> dict[str, Palette]
def extract_palette(rgb_u8, k=16, name="source") -> Palette

# ditherzam/effects/stack.py
class EffectStack:
    items: list[tuple[str, dict]]
    def add(self, name, **params); def move(self, i, j); def remove(self, i)
    def apply(self, rgb_u8) -> np.ndarray
# ditherzam/effects/post.py  EFFECTS = {"Blur","Sharpen","Chromatic Aberration","JPEG Glitch","Epsilon Glow"}

# ditherzam/render.py
@dataclass
class RenderSettings:
    contrast=50; midtones=50; highlights=50; blur=50; luminance_threshold=50
    invert=False; saturation=50; style="None"; scale=5; preview_disabled=False; params={}
class RenderPipeline:
    def __init__(self, registry, color_engine=None, effect_stack=None)
    def render(self, base_gray_f32, settings, temporal_field=None) -> np.ndarray  # uint8 HxWx3

# ditherzam/animation/temporal.py
PATTERNS  # exactly 9
def temporal_noise(frame, shape, pattern, amplitude, seed=0) -> 'float32 HxW ~[-amp,amp]'
# ditherzam/animation/timeline.py
def ease(t, kind); class Keyframe(frame,field,value); class Timeline(length, add, value_at, settings_at)
```

**Render order (spec §8.1 + color/effects insert):**
contrast → midtones → highlights → blur → **dither(downscale→kernel→upscale)** →
color(map to palette) → saturation → effects stack → invert(last).

## The 8 subsystems to produce completion task lists for

For each, expand its plan to full Phase-1 depth. Minimum coverage per subsystem:

1. **Foundation & dither core** — already fully expanded in `01-*.md`; produce a
   *completion checklist* that verifies each of its 7 tasks is done + adds any gaps
   (e.g., `NUMBA_DISABLE_JIT` fallback test, `param_func` path test).
2. **Dither kernel library (~63)** — one task block PER kernel, grouped by the five
   modules. Enumerate all names from `DITHER_BOY_FULL_SPEC.md` §9.2 and §6.2 with
   their category, dims, `param_sliders`, and (label,min,max,default) where relevant.
   Include the canonical diffusion matrices (JJN/Stucki/Burkes/Sierra*/Stevenson-Arce)
   as literal offset+weight arrays. Add the golden-array harness + `>=63` count test.
3. **Color engine** — saturation, Palette (model/IO/built-ins: grayscale, gameboy,
   cga, pico8, sepia — give exact RGB lists), median-cut extraction, source/complete,
   ColorEngine nearest/ordered/diffused, lock+shuffle. One test per behavior.
4. **Effects stack & render pipeline** — the 5 effect functions (full code),
   EffectStack add/move/remove/apply, RenderPipeline compose + order test.
5. **UI shell** — pure helpers first (convert, theme, hotkeys, viewport math,
   settings map) each TDD-tested; then widgets, viewport, controls, main window,
   debounced worker, app entry (smoke test offscreen). Paste the full default QSS
   from spec Appendix A into `themes/default/theme.yaml`.
6. **Presets & export** — preset (de)serialize+clamp, PresetManager, PNG/JPG,
   optimized run-merged SVG, batch, UI wiring.
7. **Video** — ffmpeg command builders (tested without running ffmpeg), import
   limits (60fps/60s + expert bypass), per-frame dither with cancel/progress,
   assemble+audio mux (integration, skip if ffmpeg missing), Qt workers + UI.
8. **Animation & temporal** — 9 deterministic noise patterns, threshold-field hook
   (backward-compatible when None), timeline+easing, `render_animation` generator,
   UI timeline/playback/MP4 export.

## Output format for each task list document

```
# Phase N — <Subsystem> — Completion Task List
<one-line goal>
## Prereqs: <phases that must be green first>
### Task N.M: <name>
**Files:** Create/Modify/Test: <exact paths>
**Interfaces:** Consumes / Produces (exact signatures)
- [ ] Step 1: Write failing test  <full test code>
- [ ] Step 2: Run  `NUMBA_DISABLE_JIT=1 pytest <path>::<name> -v`  → expect FAIL "<msg>"
- [ ] Step 3: Implement  <full code>
- [ ] Step 4: Run  → expect PASS (N passed)
- [ ] Step 5: Commit  `git commit -m "feat(<sub>): ..."`
...
## Subsystem Definition of Done  (checklist)
## Self-Review  (spec-coverage, placeholder scan, type-consistency)
```

## After generating all 8 documents

1. Run the self-review from the writing-plans skill against `DITHER_BOY_FULL_SPEC.md`:
   for each spec section, point to the task that implements it; list gaps and fix inline.
2. Verify type/name consistency against the FROZEN CONTRACTS above.
3. `git add docs/tasklists && git commit -m "docs(tasklists): heavily detailed per-subsystem completion checklists" && git push`.
4. Report a coverage matrix: spec section → task list task(s).

Begin with subsystem 2 (the kernel library) since it is the largest and most
mechanical, then 3→4→5→6→7→8, and finish subsystem 1's completion checklist.
