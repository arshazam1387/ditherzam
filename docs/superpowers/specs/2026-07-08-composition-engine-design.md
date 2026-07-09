# Spec — Composition Engine (Sub-project 1 of the Custom Video Filter system)

**Date:** 2026-07-08
**Status:** design approved, pending written-spec review
**Feature epic:** custom, timeline-driven dither videos where whole *looks* switch and
*morph* into one another over time.

---

## Context

Today the whole app renders through **one** configuration: a single dither `style`,
one `ColorEngine`, one `EffectStack`, one `RenderSettings`. The animation `Timeline`
can only keyframe *scalar* fields (contrast, blur, scale, depth…). It cannot switch
the dither algorithm, swap palettes/effects, or blend one look into another. The video
path (`video/frames.py::dither_frames`) renders every frame through one fixed
`settings`.

This epic adds a layer that lets a user place different **looks** on a timeline and
have them **transition/morph** between each other. It is split into three sub-projects,
each with its own spec → plan → build cycle (per `docs/superpowers/WORKFLOW-new-feature.md`):

1. **Composition engine (this spec)** — headless, Qt-free, TDD. The novel core.
2. **Video/still I/O + measured optimization** — decode→compose→encode, still mode,
   wire the Qt controller/workers, and profile-driven perf (cross-frame caching, frame
   parallelism, optional GPU). Optimization is a co-equal goal here, *not* a prior pass.
3. **Timeline editor UI** — visual clip-track editor (drag looks, insert/resize
   transitions, scrub preview, save/load).

This is **additive**. The existing "video with its own ffmpeg filters" path is
untouched; this is a new custom-look pipeline alongside it.

---

## Goal

A headless, fully unit-testable `ditherzam/composition/` package that, given an ordered
track of look-clips with transitions between them and a source of grayscale frames,
produces the final RGB frame for any timeline index — and can render/export a whole
composition to a video file. It **composes on top of** the existing `RenderPipeline`;
it never modifies it.

Success criteria:
- A single-clip composition renders **byte-identical** to `RenderPipeline.render()`.
- Each of the four transition types blends look A → look B correctly across `t ∈ [0,1]`.
- A whole composition renders to a sequence of frames and exports to MP4 via the
  existing `ffmpeg.assemble_video`.
- Zero changes to `render.py`, kernels, color, effects, or presets.

---

## In scope

- **`Look`** — a named, full-pipeline config built from a preset dict (reuses the preset
  serialization format and `preset_to_settings`). Builds its own `RenderSettings`,
  `ColorEngine` (or None), and `EffectStack`.
- **`LookClip`** — a `Look` occupying a half-open frame range `[start, end)`, with an
  optional scalar-automation `Timeline` (reuses the existing `animation.Timeline`).
- **`Composition`** — an ordered, contiguous track of `LookClip`s covering `[0, length)`,
  plus transitions attached at clip boundaries. Resolves "what is active at frame N".
- **`Transition`** — a pluggable strategy in a name→class registry (mirrors the kernel
  registry). Four built-ins: `crossfade`, `dither-dissolve`, `spatial-wipe`,
  `param-morph`.
- **`Compositor`** — `render_frame(idx, base_gray, …)`; owns a per-`Look` `RenderPipeline`
  cache. Plus `render_composition(...)` (generator) and `export_composition(...)`
  (reuses `ffmpeg.assemble_video`, injectable runner).
- **Frame source abstraction** — `render_composition` takes a `frame_source(idx) ->
  gray_f32` callable, so a held still (constant) and a decoded video (per-idx frame) use
  the identical code path.
- **Composition (de)serialization** — `composition_to_dict` / `composition_from_dict`
  (YAML-friendly; looks embed their preset dicts) for round-trip. Enables SP3 save/load
  and is pure/testable now.

## Out of scope (YAGNI — deferred to SP2/SP3)

- Any Qt/UI (timeline widget, drag-drop, scrubbing). → SP3.
- Real video decode/encode wiring and the Qt worker/controller integration. → SP2.
- Performance optimization: cross-frame cache reuse, frame parallelism, GPU kernels.
  The engine here is **correctness-first**; overlaps legitimately render the pipeline
  twice per frame. → SP2 (measured).
- New transition types beyond the four (the registry makes them drop-in later).
- Audio, easing curves beyond what `animation.ease` already provides.

---

## Invariants (mandatory — carried verbatim from WORKFLOW-new-feature.md)

- **Clean-room.** Our own code; Dither Boy is behaviour inspiration only. No Studio AAA
  code/strings/binaries.
- **Qt-free core.** `ditherzam/composition/**` must never import PySide6 (like
  `color/**`, `dithering/**`). Only `ui/`, `app.py`, `video/workers.py` import Qt.
- **Frozen `RenderPipeline.render()` stage order and `RenderSettings` fields are
  untouched.** The composition layer composes *on top of* the pipeline, never inside it.
  No new fields on `RenderSettings`.
- **UI-session prefs live in `panel.state` only, never `RenderSettings`** (relevant to
  SP3; noted here so the whole epic inherits it).
- **Python 3.12; TDD per task.** Suite green in **JIT-off** mode
  (`NUMBA_DISABLE_JIT=1`, set by `tests/conftest.py`). The 7 kernel tests that fail
  JIT-**on** (`special.py` float-index) are pre-existing and out of scope.

---

## Architecture

New package `ditherzam/composition/`:

```
composition/
  __init__.py      # public exports
  look.py          # Look
  clip.py          # LookClip, Composition, resolve()
  transitions.py   # Transition base + registry + 4 strategies + mask helpers
  compositor.py    # Compositor, render_composition, export_composition
  serialize.py     # composition_to_dict / composition_from_dict
```

### `look.py` — `Look`

Reuses the preset format so a look IS a preset (and any saved preset becomes a look).

```python
class Look:
    def __init__(self, name: str, preset: dict) -> None: ...
    name: str
    preset: dict
    @property
    def settings(self) -> RenderSettings: ...          # via preset_to_settings (memoized)
    def build_color_engine(self) -> ColorEngine | None:
        # palette from preset_to_settings; mode from preset["color"]["mode"]
        # None when mode == "off" or no palette
    def build_effect_stack(self) -> EffectStack:       # from the effects list
    @classmethod
    def from_preset(cls, name, manager: PresetManager) -> "Look": ...
```

Depends on: `presets.preset_to_settings`, `color.engine.ColorEngine`,
`effects.stack.EffectStack`, `color.palette.Palette`. Qt-free.

### `clip.py` — `LookClip`, `Composition`

```python
@dataclass
class LookClip:
    look: Look
    start: int            # inclusive
    end: int              # exclusive; end > start
    automation: Timeline | None = None   # optional scalar keyframes over this clip

@dataclass
class TransitionSpec:
    kind: str             # registry key
    duration: int         # frames; occupies the first `duration` frames of the later clip
    params: dict          # per-transition params (e.g. wipe direction, dissolve mask)

class Composition:
    length: int
    clips: list[LookClip]                 # sorted, contiguous, cover [0, length)
    transitions: dict[int, TransitionSpec]  # key = index of the *later* clip in `clips`

    def add_clip(self, clip: LookClip) -> None: ...
    def set_transition(self, later_clip_index: int, spec: TransitionSpec) -> None: ...
    def validate(self) -> None:            # contiguity, ordering, duration ≤ clip length
    def resolve(self, idx: int) -> Resolution: ...
```

`resolve(idx)` returns a small tagged result:
- `("hold", clip)` — idx is inside a clip and outside any incoming transition window.
- `("transition", clipA, clipB, t, spec)` — idx is in clipB's first `duration` frames and
  clipB has an incoming transition; `t = (idx - clipB.start) / duration` in `[0,1)`.

`clipB`'s automation (if any) is applied to its look's base settings via
`Timeline.settings_at`; likewise clipA in the transition case. Automation is resolved to
concrete `RenderSettings` before any transition math.

### `transitions.py` — `Transition` + registry

Contract (approved): a transition combines two look contexts at parameter `t`.

```python
class _LookContext:
    """A look ready to render at one frame index. Memoizes its own render."""
    settings: RenderSettings
    def frame(self) -> np.ndarray:            # render(settings) via the look's pipeline, memoized
    def render(self, settings) -> np.ndarray: # render arbitrary settings (for param-morph)

class Transition:
    def render_frame(self, t: float, a: _LookContext, b: _LookContext,
                     base_gray: np.ndarray, params: dict) -> np.ndarray:  # uint8 HxWx3
        raise NotImplementedError

TRANSITIONS: dict[str, Transition] = {}   # registry; register("crossfade")(CrossfadeTransition)
```

Strategies:
- **`crossfade`** — `lerp(a.frame(), b.frame(), t)` (rounded to uint8). Uniform alpha.
- **`dither-dissolve`** — signature move. A stable ordered mask `M ∈ [0,1)` (tiled Bayer
  by default; `params["mask"]="noise"` uses `animation.temporal_noise` normalized).
  Per pixel: take `b` where `M < t`, else `a`. Grain-by-grain flip. Deterministic.
- **`spatial-wipe`** — mask from a spatial coordinate: `params["mode"]` ∈
  `linear`(x/W)/`radial`(normalized distance from centre)/`luma`(base_gray normalized).
  Per pixel: take `b` where `coord ≤ t`, else `a`. Optional soft edge width.
- **`param-morph`** — interpolate the *numeric* `RenderSettings` fields of a and b
  (contrast, midtones, highlights, blur, luminance_threshold, saturation, scale, depth)
  and render **once**. Discrete fields (style, color_mapping) and palette:
  - same `style` → render once with interpolated numeric settings (uses A's discrete
    fields, or B's — they match). Palette colours interpolate only when both palettes
    have equal length; otherwise no palette interpolation.
  - different `style` (or non-interpolable palette) → **fallback**: render both looks
    with interpolated numeric settings and `crossfade` them by `t`. Correctness over
    cleverness; quality is preserved (never worse than crossfade).

All strategies return uint8; `t=0` yields exactly `a.frame()`-equivalent and `t=1`
yields `b.frame()`-equivalent (tested). Mask helpers (`_bayer_mask`, `_wipe_mask`) are
module-private and deterministic.

### `compositor.py` — `Compositor`, render/export

```python
class Compositor:
    def __init__(self, composition: Composition, registry=dither_registry, seed: int = 0): ...
    def render_frame(self, idx, base_gray_f32,
                     temporal_pattern=None, temporal_amplitude=0.0) -> np.ndarray:
        # resolve(idx) -> hold: pipeline.render(base_gray, settings)  [byte-identical path]
        #             -> transition: TRANSITIONS[spec.kind].render_frame(t, ctxA, ctxB, ...)
    def _pipeline_for(self, look: Look) -> RenderPipeline:   # cached per look id

def render_composition(compositor, frame_source, length,
                       temporal_pattern=None, temporal_amplitude=0.0, seed=0):
    # frame_source(idx) -> gray_f32 (still = constant fn, video = per-idx loader)
    # yields uint8 HxWx3 for idx in range(length)

def export_composition(compositor, frame_source, length, out_path, fps=24,
                       temporal_pattern=None, temporal_amplitude=0.0, seed=0,
                       progress=None, runner=run_command) -> str:
    # write each frame to PNG, then ffmpeg.assemble_video(frames_dir, fps, None, out_path, runner)
```

Temporal noise field: computed per `_LookContext` at that look's `scale` (looks in a
transition may differ), matching the existing `render_animation` behaviour.

The **hold path calls `pipeline.render()` directly with no blend**, guaranteeing the
byte-identical parity criterion.

### `serialize.py`

`composition_to_dict(comp) -> dict` / `composition_from_dict(d, manager=None) -> Composition`.
Each clip stores its look's preset dict inline (plus name), its `start`/`end`, optional
automation keyframes, and the boundary transitions. YAML-round-trippable; no Qt.

---

## Data flow

```
frame_source(idx) ─► base_gray_f32
                         │
        Composition.resolve(idx)
        ┌────────────┴─────────────┐
     hold                       transition
        │                           │
 pipeline.render(              TRANSITIONS[kind].render_frame(
   base_gray, settings)          t, ctxA, ctxB, base_gray, params)
        │                           │  (ctx.frame() -> pipeline.render each look)
        └──────────► uint8 HxWx3 ◄──┘
                         │
        render_composition ─► frames ─► export_composition ─► ffmpeg.assemble_video
```

---

## Testing (TDD per task; JIT-off green)

- **Look:** preset round-trip → correct `RenderSettings`/palette/effects; `mode="off"`
  or no palette → `build_color_engine()` is None; effects list → EffectStack contents.
- **Clip/Composition:** `validate()` rejects gaps/overlaps/oversized transitions;
  `resolve()` returns `hold` vs `transition` with correct `t` at boundaries
  (`t=0` at clipB.start, approaches 1 at `start+duration-1`); automation applied.
- **Transitions:** for each — `t=0` ≡ A, `t=1` ≡ B; dissolve/wipe masks monotonic in
  the B-fraction as `t↑`; dissolve deterministic under fixed seed; param-morph same-style
  renders once (spy on `_LookContext.render`), diff-style falls back to crossfade.
- **Compositor parity (critical):** single-clip composition renders **byte-identical** to
  `RenderPipeline.render()` (`assert_array_equal`). Transition-window frame differs from
  both pure looks.
- **render_composition:** yields exactly `length` frames; still vs video frame_source
  both drive it.
- **export_composition:** injected fake runner (like existing `test_ffmpeg_integration`)
  asserts `assemble_video` invoked with the frames dir + fps; no real ffmpeg spawn.

## Risks & mitigations

- **2× render cost in overlaps.** Accepted for SP1; explicitly SP2's optimization target.
  Kept honest by not caching across frames here (correctness-first).
- **param-morph discrete-field fallback.** Most complex piece; the fallback to crossfade
  guarantees it never looks worse than a plain crossfade. Fully unit-tested both paths.
- **uint8 rounding parity.** Hold path must call `pipeline.render` verbatim (no lerp) so
  the byte-identical test holds; blend/lerp only ever runs inside transitions.
- **Temporal field shape across differing scales.** Each `_LookContext` builds its own
  field at its look's scale — no shared-shape assumption.
- **Registry import order.** Transitions self-register on import of `transitions.py`;
  `__init__.py` imports it for side effects (mirrors the kernel registry pattern).
```
