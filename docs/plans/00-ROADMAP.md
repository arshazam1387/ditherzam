# ditherzam — Implementation Roadmap

> **For agentic workers:** Each subsystem below has its own plan file in this
> folder. Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to implement task-by-task. Steps use checkbox
> (`- [ ]`) syntax. Build the phases **in order** — later phases consume the
> interfaces produced by earlier ones.

**Goal:** A clean-room, open-source PySide6 pixel-dither studio with feature
parity to the current commercial landscape (~63 dither algorithms, a full color
engine, a stackable effects pipeline, tonal adjustments, presets, image/vector
export, video, and temporal animation). No third-party application code or
binaries are used; the source-of-truth requirements live in
[`../../DITHER_BOY_FULL_SPEC.md`](../../DITHER_BOY_FULL_SPEC.md).

**Architecture:** A pure-NumPy/Numba **core** (config, dither registry + kernels,
tonal adjustments, color engine, effects stack, render pipeline) that is 100%
headless and unit-testable, wrapped by a thin **PySide6 UI** layer. Everything
downstream of "load image → grayscale/float32 array" is deterministic and tested
against reference arrays; Qt only wires widgets to core functions. Workers run
off-thread via `QThreadPool`/`QRunnable`.

**Tech Stack:** Python 3.12 · PySide6 · NumPy · Numba (+llvmlite) · Pillow ·
PyYAML · platformdirs · bundled ffmpeg/ffprobe · pytest.

---

## Global Constraints (apply to EVERY task in EVERY phase)

- **Python 3.12** exactly (matches Numba/PySide6 wheels). Type hints everywhere.
- **Clean-room:** never import, copy, embed, or ship any Dither Boy / Studio AAA
  code, strings, URLs, licensing endpoints, or binaries. Algorithm *techniques*
  (Floyd–Steinberg, Atkinson, Bayer, median-cut, …) are public-domain and fine.
- **No licensing / telemetry / network calls** of any kind.
- **Dither kernels** are `@njit(cache=True, parallel=True)` functions taking
  `(image_array: float32[H,W], parameter, luminance_threshold_value: float)` and
  returning `float32[H,W]` in range `0..255`. `prange` on the outer loop.
- **Core is Qt-free.** Nothing under `ditherzam/` except `ditherzam/ui/`,
  `ditherzam/app.py`, and `ditherzam/video/workers.py` may import PySide6.
- **QApplication style = "Fusion"**; app id `ditherzam`.
- **TDD:** every task is red → green → refactor. **Commit after every green.**
- **Data dirs** via `platformdirs.user_data_dir("ditherzam")`:
  `PRESETS/`, `palettes/`, `logs/`, video temp under `tempfile.gettempdir()/ditherzam`.
- **DRY / YAGNI.** No placeholders in code — every function fully implemented.
- Commit message convention: `feat(<subsystem>): …` / `test(...)` / `refactor(...)`.

---

## Repository file map (created across all phases)

```
ditherzam/
├── pyproject.toml                      # phase 1
├── config/config.yaml                  # phase 1
├── themes/default/theme.yaml           # phase 5
├── assets/{icon.png,idle.gif,...}      # phase 5 (placeholder art ok)
├── ditherzam/
│   ├── __init__.py                     # phase 1
│   ├── app.py                          # phase 5
│   ├── config.py                       # phase 1  → ConfigLoader, AppConfig
│   ├── imaging.py                      # phase 1  → to_gray_f32, clamp, resize helpers
│   ├── adjustments.py                  # phase 1  → contrast/midtones/highlights/blur/invert/saturation
│   ├── dithering/
│   │   ├── __init__.py                 # phase 1
│   │   ├── registry.py                 # phase 1  → DitherRegistry, DitherEntry
│   │   ├── pipeline.py                 # phase 1  → apply_dither (downscale→kernel→upscale)
│   │   └── kernels/
│   │       ├── error_diffusion.py      # phase 1 (first 3) + phase 2 (rest)
│   │       ├── ordered.py              # phase 2
│   │       ├── pattern.py              # phase 2
│   │       ├── glitch.py               # phase 2
│   │       └── special.py              # phase 2
│   ├── color/
│   │   ├── __init__.py                 # phase 3
│   │   ├── palette.py                  # phase 3  → Palette, built-ins, extraction
│   │   └── engine.py                   # phase 3  → ColorEngine.map()
│   ├── effects/
│   │   ├── __init__.py                 # phase 4
│   │   ├── stack.py                    # phase 4  → EffectStack (add/reorder)
│   │   └── post.py                     # phase 4  → glow/chromatic/jpeg/blur/sharpen
│   ├── render.py                       # phase 4  → RenderPipeline (orchestrates all)
│   ├── presets.py                      # phase 6  → PresetManager, schema
│   ├── export/
│   │   ├── __init__.py                 # phase 6
│   │   ├── raster.py                   # phase 6  → PNG/JPG
│   │   └── vector.py                   # phase 6  → raster_to_svg
│   ├── video/
│   │   ├── __init__.py                 # phase 7
│   │   ├── ffmpeg.py                   # phase 7  → probe/extract/assemble
│   │   └── workers.py                  # phase 7  → QRunnable workers (Qt)
│   ├── animation/
│   │   ├── __init__.py                 # phase 8
│   │   ├── temporal.py                 # phase 8  → 9 noise patterns
│   │   └── timeline.py                 # phase 8  → keyframes + easing
│   └── ui/                             # phase 5
│       ├── main_window.py
│       ├── viewport.py
│       ├── widgets.py
│       ├── controls.py
│       ├── delegates.py
│       └── theme.py
└── tests/                              # every phase adds here
```

---

## Phase order & plan files

| # | Subsystem | Plan file | Depends on | Deliverable |
|---|---|---|---|---|
| 1 | Foundation & dither core | `01-foundation-and-dither-core.md` | — | `pip install -e .`; config loads; registry works; tone adjustments + 3 error-diffusion dithers pass reference tests |
| 2 | Full dither kernel library | `02-dither-kernel-library.md` | 1 | All ~63 kernels registered, each with a golden-array test |
| 3 | Color engine | `03-color-engine.md` | 1 | Palettes (built-in/extract/source), `ColorEngine.map()` with nearest + ordered color dither |
| 4 | Effects stack & render pipeline | `04-effects-stack.md` | 1,3 | `EffectStack` add/reorder; glow/chromatic/jpeg/blur/sharpen; `RenderPipeline.render()` end-to-end |
| 5 | UI shell | `05-ui-shell.md` | 1–4 | Runnable PySide6 app: load image, pick dither, adjust, live preview, zoom/pan, theme |
| 6 | Presets & export | `06-presets-and-export.md` | 1–5 | Save/load/import/export YAML presets; PNG/JPG/SVG export; batch |
| 7 | Video | `07-video.md` | 1–5 | Import video → per-frame dither → reassemble MP4 (audio preserved); live playback |
| 8 | Animation & temporal | `08-animation-temporal.md` | 1–5 | 9 temporal noise patterns; keyframe timeline w/ easing; animated preview & MP4 |

**Definition of done (whole project):** phases 1–8 green, `pytest` clean, the app
launches, loads an image, applies a color dither with stacked effects, and exports
PNG/SVG/MP4.

---

## Cross-cutting interfaces (the contracts every phase honors)

```python
# ditherzam/dithering/registry.py
@dataclass(frozen=True)
class DitherEntry:
    name: str
    category: str
    dims: int                     # 1 | 2 | 3  (all shipped kernels use 2)
    param_sliders: tuple[str, ...]
    func: Callable                # njit kernel
    param_func: Callable | None = None

class DitherRegistry:
    def register(self, name, category, dims=2, param_sliders=(), param_func=None): ...  # decorator
    def get_entry(self, name: str) -> DitherEntry | None: ...
    def list_dithers(self) -> list[str]: ...
    def by_category(self) -> dict[str, list[str]]: ...

# ditherzam/dithering/pipeline.py
def apply_dither(gray_f32, *, style, scale, luminance_threshold,
                 params, registry, preview_disabled=False) -> np.ndarray: ...  # float32 HxW

# ditherzam/adjustments.py   (all take & return float32 HxW or HxWx3, 0..255)
def apply_contrast(img, value): ...       # value 0..100, factor value/50
def apply_midtones(img, value): ...       # gamma = max(1+(value-50)/200, 0.1); 255*(img/255)**(1/gamma)
def apply_highlights(img, value): ...     # img * (1+(value-50)/100)
def apply_blur(img, value): ...           # PIL GaussianBlur radius=(value/10)**2
def apply_invert(img, enabled): ...       # 255-img
def apply_saturation(rgb, value): ...     # NEW color control (phase 3+)

# ditherzam/color/engine.py
class ColorEngine:
    palette: Palette
    mode: str    # "off" | "nearest" | "ordered" | "diffused"
    def map(self, gray_or_rgb_f32) -> np.ndarray: ...  # -> uint8 HxWx3

# ditherzam/effects/stack.py
class EffectStack:
    def add(self, name, **params): ...
    def move(self, index, new_index): ...
    def remove(self, index): ...
    def apply(self, img_u8_rgb) -> np.ndarray: ...

# ditherzam/render.py
class RenderPipeline:
    def render(self, base_gray_f32) -> np.ndarray: ...  # adjustments→dither→color→effects→invert
```

These signatures are **frozen contracts** — later phases rely on the exact names,
parameters, and return dtypes above.
