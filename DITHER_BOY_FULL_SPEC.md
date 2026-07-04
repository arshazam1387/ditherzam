# Dither Boy 3.0.2 — Complete Reverse-Engineered Specification

> Goal of this document: capture **every** feature, control, formula, constant,
> string, file, and pipeline of *Dither Boy 3.0.2 "Studio AAA"* in enough detail
> to build a **1:1 open-source clone** with no loss of functionality.
>
> Source of truth: the shipped PyInstaller build
> (`Dither Boy.exe`, Python 3.12, PySide6) plus `config/config.yaml` and the
> `themes/` folder. The application's own code lives in three modules:
> `main.py` (the GUI + all logic), `dither_registry.py` (the 53 dither
> algorithms), and `db_tools.py` (a one-function debug helper).
> All numbers, labels, formulas, and command lines below were extracted directly
> from the compiled bytecode and bundled data.

---

## 0. Executive summary

Dither Boy is a **desktop image & video "1-bit / pixel dither" studio**. You
load an image (or a short video), it is converted to grayscale, you tune tonal
adjustments and pick one of 53 dithering algorithms with live preview, then
export to PNG / SVG (vector) / Photoshop / clipboard / transparent "chroma drop"
PNG, or process a whole folder (batch) or a video frame-by-frame.

Key architectural facts to replicate:

- **GUI toolkit:** Qt via **PySide6**, `QApplication` style = **"Fusion"**.
- **Image math:** **NumPy** + **Pillow (PIL)**; dithers are **Numba `@njit(parallel=True)`** kernels (`prange`).
- **Video:** bundled **ffmpeg/ffprobe** (in `assets/ffmpeg/`).
- **Everything is grayscale mode `'L'` internally**; color is only re-expanded for 3-D dithers.
- **Threading:** `QThreadPool` + `QRunnable` workers; GUI updates via `Signal`.
- **Config-driven:** an external `config.yaml` and per-folder YAML themes; missing config is a fatal `ConfigurationError`.
- **Licensing:** an online license check gate (see §14) — **omit/replace this entirely in an OSS build.**

---

## 1. Application identity & packaging

From `config/config.yaml → app`:

| Field | Value |
|---|---|
| name | `Dither Boy` |
| author | `Studio AAA` |
| version | `3.0.2` |
| apple_name | `DitherBoy` |
| apple_bundle_identifier | `com.studioaaa.ditherboy` |

- Windows install dir (original): `C:\Program Files (x86)\Studio AAA\Dither Boy`
- User data dir: `%LOCALAPPDATA%\Studio AAA\Dither Boy` (via `appdirs.user_data_dir`)
  - `logs/` — failure logs `failure_log_YYYYMMDD_HHMMSS_fff.log`
  - `persistent_key.key` — Fernet key for the license cache
  - `PRESETS/` — user preset `.yaml` files
- Temp dir for video work: `%TEMP%\DitherBoy` and `ditherboy_video_*` temp folders.

### Dependencies (Python)
`PySide6`, `numpy`, `numba` (+ `llvmlite`), `Pillow`, `PyYAML`, `appdirs`,
`cryptography` (Fernet), `PyJWT` (`jwt`, RS256), `requests` (license HTTP),
`pywin32` (Windows Photoshop COM / `CREATE_NO_WINDOW`). Bundled binaries:
`assets/ffmpeg/ffmpeg.exe`, `assets/ffmpeg/ffprobe.exe`.

### Operating-system detection (`get_operating_system`)
Returns `"Windows"`, `"macOS"` (from `platform.system()=="Darwin"`), `"Linux"`, or `"Unknown"`.
On Windows all subprocesses use `CREATE_NO_WINDOW` and a hidden `startupinfo` so no console flashes.

---

## 2. Configuration system (`config/config.yaml`)

Loaded by `ConfigLoader(config_path, required_keys)`. The file **must exist and
be non-empty**, otherwise a `ConfigurationError` is raised and the app aborts
with a message box `Configuration Error: …`. `validate_config` checks every
required key per section. Full shipped file, verbatim:

```yaml
app:
  name: "Dither Boy"
  author: "Studio AAA"
  version: "3.0.2"
  apple_name: "DitherBoy"
  apple_bundle_identifier: "com.studioaaa.ditherboy"

rendering:
  force_software_rendering: false

dither_settings:
  default_dither_style: "None"
  default_dither_scale: 5

style_settings:
  image_editor_init: ""
  zoom_button_style: "font-size: 10px;"
  loading_label_style: "background-color: rgba(0, 0, 0, 150); color: white;"
  viewport_bg_color: "#1f1f1f"
  app_style: "Fusion"

downscale_settings:
  apply_downscale: false
  width: 1280
  height: 720

labels:
  dither_controls: ""
  disable_dither_preview: "Disable Preview"
  dither_style: "Style"
  dither_scale: "Scale"
  adjustments: ""
  invert_output: "Invert"
  reset_adjustments: "Reset All"
  restart_application: "Restart"

inertia_settings:
  enable_inertia: true
  friction: 0.95
  velocity_scale: 0.5
  max_velocity: 2000.0

other_settings:
  schedule_debounce_time: 20
  loading_animation_delay: 400
  use_viewport_proxy: false
```

**Required-keys validation** covers: `app`(name,author,version), `rendering`
(force_software_rendering), `dither_settings`(default_dither_style,
default_dither_scale), `style_settings`(image_editor_init, zoom_button_style,
loading_label_style, app_style, viewport_bg_color), `labels`(dither_controls,
adjustments, disable_dither_preview, invert_output, dither_style, dither_scale,
reset_adjustments), `other_settings`(schedule_debounce_time,
loading_animation_delay, use_viewport_proxy), `downscale_settings`(apply_downscale,
width, height), `inertia_settings`(enable_inertia, friction, velocity_scale,
max_velocity).

`force_software_rendering: true` sets the Qt software-OpenGL attribute before the
`QApplication` is created (for GPUs that mis-render).

---

## 3. Theming system (`themes/`)

`find_themes(THEMES_ROOT)` scans every subfolder of `themes/`. A **valid theme
folder must contain `theme.yaml`**. Invalid folders are reported but skipped; if
none are valid the app raises.

Each `theme.yaml` has:
- `app_stylesheet:` — a full Qt stylesheet (QSS) string applied app-wide.
- `labels:` — a map of control-label → bool (which labels are shown for the theme).
- `glow_color:` — hex color used for slider "glow" drop-shadow and the sub-page fill.
- optional `idle_gif:` — theme-specific idle animation (else global `assets/idle.gif`).

Shipped themes: `default` (dark, bg `#1f1f1f`, glow `#5e89ed`),
`theme_one` (lime `#b4ff00` with `idle.gif`), `theme_two` (with `idle.gif`).
The **full `default/theme.yaml` QSS is reproduced in Appendix A** — scrollbars,
sliders (radial-gradient handle), push-buttons (rounded 10px, `#292929`),
spinboxes (arrows hidden, 35px wide), combo boxes (`#292929`, 125px), labels
(bold 11px). Reuse it as-is.

- **`Ctrl+T`** cycles to the next theme (`cycle_theme`); the change-theme hotkey
  (`Ctrl+Shift+T` / `⌘+Shift+T`) also cycles.
- `load_theme(theme_name)` loads QSS + idle gif + glow color; on any failure it
  falls back to the `default` theme, and if that also fails shows a fatal error.

---

## 4. Main window & layout

Class `ImageEditor(QMainWindow)`. Construction order (`__init__` →
`_setup_main_layout`):

1. `_setup_image_viewport` — the left **image viewport** (`CustomGraphicsView` over a `QGraphicsScene`/`QGraphicsPixmapItem`); background = `viewport_bg_color` `#1f1f1f`.
2. `_setup_control_panel` — the right **control panel** inside a thin-scrollbar `QScrollArea` (`objectName="control_panel"`). Central widget `objectName="central_widget"`.
3. `_setup_import_export_zoom_section` — **Import** / **Export** buttons + zoom buttons.
4. `_setup_help_label` — "`Ctrl+H` for help" hint label.
5. `_setup_logo` — the animated idle GIF logo.
6. `_setup_dither_section` — dither controls (see §6).
7. `_setup_adjustments_section` — tonal adjustments (see §7).
8. `_setup_reset_section` — "Reset All" button.
9. `_setup_overlay` — the license overlay (see §14) & loading overlay.
10. `create_menus`, `_setup_shortcuts`.

Window flags: maximize-button hint set; minimum size enforced; `center_window()`
centers on screen. Window title prefix = `"Dither Boy"`; the title shows
`Dither Boy - <image name> (WxH)` and, while zoomed, `Dither Boy - [NN%]`.

`resizeEvent` re-fits the image; `set_controls_enabled(bool)` greys out all
controls until an image is loaded. Disabled sections get a `QGraphicsOpacityEffect`
(dither group, adjustments group, presets) so they look dimmed.

---

## 5. Image viewport — zoom, pan, inertia (`CustomGraphicsView`)

- **Drag-and-drop:** `dragEnterEvent`/`dragMoveEvent`/`dropEvent` accept image
  files dropped onto the viewport and import them.
- **Zoom:** `Shift + mouse wheel`. Wheel up → `zoom_in`, wheel down → `zoom_out`.
  - `zoom_in`: multiply view scale by **1.2** (both axes). Hard cap **max zoom = 100.0** (i.e. 10000%). Updates title to ` - [NN%]`.
  - `zoom_out`: multiply by **0.8**. Hard floor **min zoom = 0.01** (1%).
  - `reset_zoom` (`Ctrl+0`): `fit_image_to_viewport()` (fit whole image) and reset title.
  - Zoom % = `int(view.transform().m11() * 100)`.
- **Pan:** left-mouse drag scrolls the scrollbars; velocity is tracked in a `deque` for inertia.
- **Inertia (fling):** on mouse release, if `inertia_settings.enable_inertia`,
  a `QTimer` runs `apply_inertia`:
  - `delta_time = 0.016` (≈60 fps)
  - `new_scroll = current - velocity * delta_time`, clamped to `[0, scrollbar.maximum()]`
  - `velocity *= friction` (config `friction = 0.95`) each tick
  - stop when `abs(velocity_x) < 10 and abs(velocity_y) < 10`
  - `velocity_scale = 0.5`, `max_velocity = 2000.0` bound the initial fling.
- `is_image_zoomed()` → true when the pixmap exceeds the viewport.
- Plain wheel (no Shift) on spinboxes/combos is swallowed (`event.accept()`), so
  scrolling never accidentally changes a value (`NoScrollComboBox`, `InvisibleSpinBox`, `GlowSlider.wheelEvent`).

---

## 6. Dither controls section (`_setup_dither_section`)

Widgets, top to bottom:

- **Group header:** "Dither Controls".
- **"Disable Preview"** checkbox (`dither_preview_toggle`). When checked, the
  dither step is skipped in the live preview (faster scrubbing) but still applied on export.
- **"Dither Style"** combo (`dither_combo`) — a **`NoScrollComboBox`** populated
  from the registry, **grouped by category** with a custom `DitherStyleDelegate`
  (category headers are non-selectable; items store the style name in `userData`).
  Default selection from `dither_settings.default_dither_style` (`"None"`).
- **"Presets"** combo (`preset_selector`) with a leading `" None"` entry (see §10).
- A **"↔" shuffle icon** (`ClickableLabel`, font-size 17) → `shuffle_dither_settings` (randomizes style + params).
- **"Dither Scale"** slider — `ResettableGlowSlider`, **range 1–20**, default from
  `default_dither_scale` (**5**); paired **`InvisibleSpinBox`**. Double-click resets to default.
- **Dynamic parameter sliders** — created once, shown/hidden per selected style by
  `update_dither_scale_controls` (see §6.2). Each slider has a label + `InvisibleSpinBox`.

### 6.1 The full dither category list (combo contents)

Order and grouping exactly as registered (53 styles across 6 categories):

| Category | Styles |
|---|---|
| **Default** | None |
| **Error Diffusion** | Floyd-Steinberg, Atkinson, Jarvis-Judice-Ninke, Stucki, Burkes, Sierra, Sierra-Lite, Two-Row-Sierra, Stevenson-Arce, Ostromukhov, Gaussian |
| **Ordered Dither** | Bayer-Ordered, Bayer-Void, Random Ordered, Bit Tone, Mosaic, Bayer-Matrix 2x2, Bayer-Matrix 4x4, Bayer-Matrix 8x8, Bayer-Matrix 16x16 |
| **Glitch Effects** | Artifact Modulation, Atkinson-VHS, Glitch, Modulated Diffuse Y, Modulated Diffuse X, Uniform Modulation Y, Uniform Modulation X, Waveform, Waveform Alt, Ordered Modulation, Smooth Diffuse, Stucki Diffusion Lines, Atkinson Line Modulation, Contrast Aware Y, Contrast Aware X |
| **Patterned** | Checkers - Small, Checkers - Medium, Checkers - Large, Diamond, Gridlock/Traffic, Print Pattern, Block Tone, Stippling, Crosshatch |
| **Special Effects** | Radial Burst, Wave, Noise, Topography, Thresholder, Diagonal, Displace Contour, Sine Wave Modulation |

### 6.2 Per-style parameter slider (`STYLE_TILE_CONFIG`)

Styles that expose a **primary parameter slider** (`dither_parameter_slider`),
with `(label, min, max, default)`:

| Style | Label | min | max | default |
|---|---|---|---|---|
| Atkinson-VHS | Line Count | 1 | 20 | 1 |
| Bayer-Void | Warp Intensity | 1 | 50 | 10 |
| Bit Tone | Dot Size | 1 | 20 | 1 |
| Block Tone | Dot Size | 4 | 30 | 4 |
| Contrast Aware X | Line Scale | 1 | 20 | 1 |
| Contrast Aware Y | Line Scale | 1 | 20 | 1 |
| Crosshatch | Line Spacing | 1 | 20 | 1 |
| Diagonal | Edge Sensitivity | 1 | 20 | 1 |
| Gaussian | Distribution Spread | 1 | 20 | 1 |
| Glitch | Glitch Intensity | 1 | 20 | 1 |
| Modulated Diffuse X | Line Scale | 1 | 20 | 1 |
| Modulated Diffuse Y | Line Scale | 1 | 20 | 1 |
| Mosaic | Block Size | 1 | 50 | 10 |
| Stippling | Dot Density | 1 | 20 | 1 |
| Thresholder | Modulation Frequency | 1 | 20 | 1 |
| Topography | Warp Intensity | 1 | 20 | 1 |
| Uniform Modulation X | Line Scale | 1 | 20 | 1 |
| Uniform Modulation Y | Line Scale | 1 | 20 | 1 |
| Waveform | Wave Density | 1 | 20 | 1 |
| Waveform Alt | Modulation Blend | 1 | 20 | 1 |

The **secondary/extra sliders** (created in `_setup_dither_section`, shown per
style) — `(label, min, max, default)`:

| Slider (attr) | Label | min | max | default | Used by |
|---|---|---|---|---|---|
| dither_parameter_slider | *(per table above)* | — | — | — | many |
| contour_thresh_slider | Contour Threshold | 0 | 100 | 50 | Displace Contour |
| line_mode_slider | Line Mode | 1 | 3 | 1 | Displace Contour |
| smoothing_slider | Smoothing | 0 | 5 | 0 | Displace Contour |
| line_space_slider | Line Spacing | 1 | 5 | 1 | Displace Contour |
| smoothness_slider | Smoothness | 1 | 10 | 5 | Smooth Diffuse |
| matrix_size_slider | Matrix Size | 2 | 3 | 2 | Modulated Bayer Dither |
| wave_frequency_slider | Wave Frequency | 1 | 20 | 5 | Sine Wave Modulation |
| wave_threshold_slider | Wave Threshold | 1 | 30 | 10 | Sine Wave Modulation |
| line_emphasis_slider | Line Emphasis | 1 | 10 | 5 | Stucki Diffusion Lines |
| modulation_strength_slider | Modulation Strength | 1 | 10 | 5 | Atkinson Line Modulation |
| horizontal_bias_slider | Horizontal Bias | 1 | 10 | 5 | Atkinson Line Modulation |
| smoothing_factor_slider | Smoothing Factor | 0 | 1 | 0 | Uniform Modulation X/Y |
| bleed_fraction_slider | Bleed Fraction | 0 | 100 | 0 | Uniform Modulation X/Y |
| *(Tile Size)* | Tile Size | 1 | 50 | 1 | generic tile |

Also: the **Dither Scale slider is enabled only** when a real style is selected
(`style not in {None,"None"}`) **and** preview is not disabled. The
**Luminance Threshold** slider is disabled/irrelevant for
`{None, "None", "Displace Contour", "Atkinson Line Modulation", "Sine Wave Modulation", "Ordered Modulation"}`.

---

## 7. Adjustments section (`_setup_adjustments_section`)

Header "Adjustments" + a "↔" shuffle icon → `shuffle_adjustments`.
An **"Invert Output"** checkbox (`invert_output_toggle`, default off) sits above
the sliders. Five `ResettableGlowSlider`s, each with an `InvisibleSpinBox` and a
double-click-to-reset `ResettableLabel`:

| Slider | Range | Default | Spinbox display max |
|---|---|---|---|
| **Contrast** | 0–100 | 50 | 250 |
| **Midtones** | 0–100 | 50 | 10 |
| **Highlights** | 0–100 | 50 | 50 |
| **Luminance Threshold** | 0–100 | 50 | 100 |
| **Blur** | 0–100 | 50 | 100 |

(The spinbox shows a "friendly" number via `InvisibleSpinBox.textFromValue`
mapping the 0–100 slider onto `max_display_value`.) Labels `Denoise` and
`Sharpen` exist in the code as reserved/planned adjustments but are **not** wired
to live sliders in 3.0.2 — a clone may add them or ignore them.

Changing any slider calls `schedule_image_update` (debounced) and toggling
Invert also calls `reset_dither_cache`.

### 7.1 Exact adjustment formulas (verbatim from bytecode)

All operate on a float32 grayscale array `img` (0–255):

- **Contrast** (`apply_contrast`):
  ```
  contrast = sliders["Contrast"].value() / 50.0     # slider 0..100 → factor 0.0..2.0 (50→1.0)
  return img * contrast
  ```
- **Midtones / gamma** (`apply_midtones`):
  ```
  midtones = sliders["Midtones"].value() - 50       # -50..+50
  gamma    = 1.0 + midtones / 200.0                 # 0.75..1.25
  gamma    = max(gamma, 0.1)
  return 255 * (img / 255.0) ** (1.0 / gamma)
  ```
- **Highlights / brightness** (`apply_highlights`):
  ```
  highlights        = sliders["Highlights"].value() - 50   # -50..+50
  brightness_factor = 1 + highlights / 100.0               # 0.5..1.5
  return img * brightness_factor
  ```
- **Blur** (`apply_blur`) — Pillow Gaussian:
  ```
  blur_value  = sliders["Blur"].value()             # 0..100
  blur_factor = (blur_value / 10.0) ** 2            # 0 .. 100 (radius)
  if blur_factor > 0:
      pil = Image.fromarray(np.clip(img, 0, 255).astype(uint8))
      pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_factor))
      return np.array(pil).astype(float32)
  else:
      return img                                    # "No Blur Applied."
  ```
- **Invert** (`apply_invert`): if `invert_output_toggle.isChecked()` → `255 - img`, else unchanged.
- **Luminance Threshold**: not a standalone step — it is passed **into the dither
  kernel** as `tval` (see §8): `tval = (sliders["Luminance Threshold"].value() / 100.0) * 255.0`.

`get_current_adjustments()` returns the tuple `(Contrast, Midtones, Highlights, Blur)` slider values.

---

## 8. The rendering pipeline

### 8.1 Effect order (`EffectRegistry`)
In `ImageEditor.__init__`, effects are registered **in this exact order** and
`effect_registry.apply_all(img)` runs them sequentially:

```
1. apply_contrast
2. apply_midtones
3. apply_highlights
4. apply_blur
5. apply_dither
6. apply_invert
```

- **Preview:** `apply_all_effects()` = `preview_image_array.copy()` → `effect_registry.apply_all`.
- **Full-size export:** `apply_effects_to_full_image()` runs the same chain on the full-resolution grayscale array.
- Processing runs off the GUI thread (`ImageProcessor(QRunnable)` in the `QThreadPool`), emitting the finished `QImage` via signals. A debounce (`schedule_debounce_time = 20 ms` + `QTimer`) coalesces rapid slider changes; a loading GIF appears after `loading_animation_delay = 400 ms`.

### 8.2 Dither step (`apply_dither`) — the pixelation + dither core

```
style = dither_combo.itemData(currentIndex())
entry = dither_registry.get_entry(style)
if style == "None" or not entry or dither_preview_toggle.isChecked():
    return img                                   # no dithering

dims = entry["dims"]                             # 1, 2, or 3
tval = float(sliders["Luminance Threshold"].value() / 100.0 * 255.0)
factor = max(1, int(linear_scale(dither_scale_slider.value())))   # block size ≥ 1

# 1) DOWNSCALE (pixelate) with NEAREST by `factor`
orig_pil = Image.fromarray(np.clip(img, 0, 255).astype(uint8))
small = orig_pil.resize((max(1, w // factor), max(1, h // factor)), Image.NEAREST)
arr = np.array(small, dtype=np.float32)

# 2) DIMENSION FIX
if dims == 3 and arr.ndim == 2:   arr = np.stack([arr]*3, axis=-1)   # gray→RGB
elif dims in (1, 2) and arr.ndim == 3: arr = arr[..., 0]             # RGB→gray

# 3) PARAM EXTRACTION
#    if entry has param_func: param = param_func(self)
#    else: param = value(s) pulled from entry["param_sliders"] (single value, or tuple)

# 4) CALL the kernel (dispatch depends on dims and special-cased styles):
#    dims == 2/3, generic:            out = func(arr, param, tval)
#    dims == 1 (per-scanline):        out = np.stack([func(row, param, tval) for row in arr], axis=0)
#      · "Sine Wave Modulation":      func(row, param[0], param[1])   # (freq, threshold), no tval
#    dims == 2 special multi-arg:
#      · "Sine Wave Modulation":      func(arr, param[0], param[1])
#      · "Smooth Diffuse":            func(arr, param[0], tval, param[1])
#      · "Modulated Bayer Dither":    func(arr, param[0], tval, param[1])
#      · "Atkinson Line Modulation":  func(arr, param[0], param[1])
#      · "Uniform Modulation X/Y":    func(arr, param[0], tval, param[1], param[2])   # (line_scale, thr, smoothing_factor, bleed_fraction)
#      · "Displace Contour":          func(arr, param, tval)          # param is 4-tuple (contour_thresh,line_mode,smoothing,line_space)
#      · default:                     func(arr, param, tval)

# 5) UPSCALE BACK to original size with NEAREST (so blocks stay crisp)
mode = "RGB" if dims == 3 else "L"
result = Image.fromarray(np.clip(out, 0, 255).astype(uint8), mode=mode)
result = result.resize(orig_pil.size, Image.NEAREST)
return np.array(result, dtype=np.float32)
```

**Key insight for a clone:** "Dither Scale" is a **pixel-block size**. The image
is nearest-neighbor **downscaled by `factor`**, dithered at that lower
resolution, then nearest-neighbor **upscaled back**, producing chunky
pixel-dither blocks. `linear_scale(scale, min_block, max_block)` simply clamps:
`max(min_block, min(scale, max_block))`.

---

## 9. The 53 dither algorithms (`dither_registry.py`)

### 9.1 Registry mechanics

```python
dither_registry = DitherRegistry()

@dither_registry.register(name, category, dims, param_sliders=[], param_func=None)
def some_dither(image_array, parameter, luminance_threshold_value): ...
```

- **`dims`**: `1` = per-scanline (1-D), `2` = full grayscale (2-D), `3` = full color (3-D). *All 53 shipped dithers use `dims = 2`* (the 1-D/3-D paths exist for extensibility; the per-scanline & multi-arg dispatch in §8.2 is driven by style name, not dims, in 3.0.2).
- **`param_sliders`**: list of slider-attribute names whose `.value()` is pulled to build `parameter`.
- **`param_func`**: optional `f(caller) → parameter` (none used in 3.0.2).
- `DISABLED_STYLES` / `unregister(name)` can permanently remove a dither.
- Helper functions: `clip(v, lo, hi)`, `_make_1d_kernel(sigma)`, `gaussian_blur(image, sigma)` (separable Gaussian used by several effects).
- All kernels are `@njit(parallel=True)` with `prange` outer loops and return a float32 array (0–255).

### 9.2 Full catalog (name · category · dims · param sliders · behavior)

Every registered style, with its docstring/behavior. Standard error-diffusion
kernels are given explicitly (verified against the embedded constants).

**Error Diffusion**

- **None** — Default — no-op passthrough (`no_dither`).
- **Floyd-Steinberg** — 2-D — classic. For each pixel: `new = 0 or 255` by threshold; error distributed **7/16 →(x+1), 3/16 ↙, 5/16 ↓, 1/16 ↘** (divisor 16). *(confirmed consts 7,16,3,5.)*
- **Atkinson** — 2-D — error split into **1/8** to 6 neighbors: (x+1,y),(x+2,y),(x-1,y+1),(x,y+1),(x+1,y+1),(x,y+2); only 6/8 of error propagates (divisor 8). *(confirmed 8.0.)*
- **Jarvis-Judice-Ninke** — 2-D — 12-neighbor diffusion, divisor **48**: row0 `[_,_,X,7,5]`, row1 `[3,5,7,5,3]`, row2 `[1,3,5,3,1]`.
- **Stucki** — 2-D — divisor **42**: row0 `[_,_,X,8,4]`, row1 `[2,4,8,4,2]`, row2 `[1,2,4,2,1]`.
- **Burkes** — 2-D — divisor **32**: row0 `[_,_,X,8,4]`, row1 `[2,4,8,4,2]`.
- **Sierra** (Sierra-3) — 2-D — divisor **32**: row0 `[_,_,X,5,3]`, row1 `[2,4,5,4,2]`, row2 `[_,2,3,2,_]`.
- **Sierra-Lite** — 2-D — divisor **4**: `X 2 / 1 1`.
- **Two-Row-Sierra** — 2-D — divisor **16**: row0 `[_,_,X,4,3]`, row1 `[1,2,3,2,1]`.
- **Stevenson-Arce** — 2-D — wide hex-lattice diffusion, divisor **200**, weights include 32,12,26,30,16,5 (verified consts 32,200,12,26,30,16,5).
- **Ostromukhov** — 2-D — variable-coefficient error diffusion; coefficients depend on the current pixel value (per Ostromukhov's tables).
- **Gaussian** — 2-D — `param_sliders=[dither_parameter_slider]` (Distribution Spread 1–20) — Gaussian-distributed probabilistic dithering for a smoother/natural look.

**Ordered Dither**

- **Bayer-Ordered** — 2-D — 4×4 Bayer matrix, Numba-parallelized (`custom_ordered_dither`; consts 0.5, 16.0, 255.0).
- **Bayer-Void** — 2-D — `param_sliders=[dither_parameter_slider]` (Warp Intensity 1–50, def 10). **CRT-bend + ordered grid**: each row shifted right by `∝ (y/height)²` (CRT warp), then `final_threshold = luminance_threshold + (grid_value - 128) * grid_strength` from a normalized 4×4 Bayer matrix.
- **Random Ordered** — 2-D — per-pixel random threshold matrix.
- **Bit Tone** — 2-D — Dot Size 1–20 — threshold-matrix "bit tone".
- **Mosaic** — 2-D — Block Size 1–50 (def 10) — breaks image into mosaic chunks.
- **Bayer-Matrix 2x2 / 4x4 / 8x8 / 16x16** — 2-D — recursive Bayer matrices built by a nested `build_bayer(n)`; ordered-dither at the chosen matrix order.

**Glitch Effects**

- **Artifact Modulation** — 2-D — Dither Param — horizontal Atkinson-VHS wavy variant (`horizontal_atkinson_vhs_wavy`).
- **Atkinson-VHS** — 2-D — Line Count 1–20 — stylized VHS/CRT: `line_count` horizontal tracking lines across the image (`vhs_linecount_dither`).
- **Glitch** — 2-D — Glitch Intensity 1–20 — random horizontal/vertical shifts (digital corruption) + binary dither.
- **Modulated Diffuse Y** — 2-D — Line Scale 1–20 — 1-D horizontal error diffusion → banded line patterns (density ∝ brightness).
- **Modulated Diffuse X** — 2-D — Line Scale 1–20 — column-wise (vertical) variant.
- **Uniform Modulation Y** — 2-D — sliders `[dither_parameter_slider, smoothing_factor_slider, bleed_fraction_slider]` — 1-D row diffusion; horizontal bands from raw error; smoothing+bleed govern vertical error carry.
- **Uniform Modulation X** — 2-D — same three sliders — column variant with EMA-smoothed bleed.
- **Waveform** — 2-D — Wave Density 1–20 — brightness → sine-wave frequency (dense in darks, sparse in lights); CRT/analog look.
- **Waveform Alt** — 2-D — Modulation Blend 1–20 — phase warped by brightness + local horizontal gradient; wider freq range; pure 0/255.
- **Ordered Modulation** — 2-D — Dither Param — `hybrid_wavy_ordered_diffusion`.
- **Smooth Diffuse** — 2-D — sliders `[dither_parameter_slider, smoothness_slider]` (Smoothness 1–10) — 1-D diffusion with smoothing for softer line transitions.
- **Stucki Diffusion Lines** — 2-D — Line Emphasis 1–10 (def 5) — Stucki modified to emphasize horizontal error → line patterns.
- **Atkinson Line Modulation** — 2-D — sliders `[modulation_strength_slider, horizontal_bias_slider]` (each 1–10) — Atkinson variant with horizontal emphasis + modulation.
- **Contrast Aware Y** — 2-D — Line Scale 1–20 — 1-D diffusion + local-contrast-aware warp (inline clipping for Numba).
- **Contrast Aware X** — 2-D — Line Scale 1–20 — X-axis variant.

**Patterned**

- **Checkers - Small / Medium / Large** — 2-D — checkerboard alternation at three board sizes.
- **Diamond** — 2-D — diamond-shaped pattern.
- **Gridlock/Traffic** — 2-D — structured geometric "traffic jam" appearance.
- **Print Pattern** — 2-D — CMYK halftone simulation with angled screens (`halftone_cmyk_simulation`, dot_scale 3–8 recommended).
- **Block Tone** — 2-D — Dot Size 4–30 (def 4) — classic round-dot halftone (`classic_halftone`); dot size varies with darkness.
- **Stippling** — 2-D — Dot Density 1–20 — dot density ∝ darkness.
- **Crosshatch** — 2-D — Line Spacing 1–20 — angled hatch lines by intensity.

**Special Effects**

- **Radial Burst** — 2-D — burst radiating from center.
- **Wave** — 2-D — pixel values modulated by a wave pattern.
- **Noise** — 2-D — random noise added per pixel (grainy).
- **Topography** — 2-D — Warp Intensity 1–20 — warping contour lines (analog/topographic look) (`analog_contour_dither`).
- **Thresholder** — 2-D — Modulation Frequency 1–20 — position/content-dependent varying thresholds.
- **Diagonal** — 2-D — Edge Sensitivity 1–20 — wireframe edge-detection + dither (`wireframe_dither`).
- **Displace Contour** — 2-D — sliders `[contour_thresh_slider, line_mode_slider, smoothing_slider, line_space_slider]` — vectorized contour-line dither: `parameters = (contour_threshold, line_thickness/line_mode, smoothing σ, line_spacing)`; binary 0/255 output (`contour_line_dither`).
- **Sine Wave Modulation** — 2-D — sliders `[wave_frequency_slider, wave_threshold_slider]` (Wave Frequency 1–20 def 5, Wave Threshold 1–30 def 10) — flowing sine-wave line density ∝ brightness.

> For an exact clone of every kernel body, the complete `@njit` disassembly of
> all 53 functions is available in the extracted `dither_registry.pyc`. The
> standard error-diffusion kernels above are canonical and match the embedded
> constants; the "custom/glitch" kernels follow the docstring semantics listed here.

---

## 10. Presets (`PresetManager` + menu)

Presets are YAML files in `%LOCALAPPDATA%\Studio AAA\Dither Boy\PRESETS\` (also
importable/exportable). `PresetManager(app_name, app_author, install_dir)` handles
`get_all_presets()`, `save_preset(dict, name)`, `import_preset(source_path)`
(validates `*.yaml`; "Not a valid preset file." on failure).

### 10.1 Preset dict schema (`export_current_preset_dict` → YAML)

```yaml
adjustments:
  Contrast: <int 0-100>
  Midtones: <int 0-100>
  Highlights: <int 0-100>
  Luminance Threshold: <int 0-100>
  Blur: <int 0-100>
  invert_output: <bool>
dither:
  style: <style name or "None">
  scale: <int 1-20>
  preview_disabled: <bool>
  params:                       # one entry per param_slider of the style
    <slider_name w/o "_slider">: <value>   # e.g. dither_parameter, contour_thresh, ...
```

`params` are built by iterating the style's `param_sliders`, stripping the
`_slider` suffix, and reading `getattr(self, slider_name).value()`.

### 10.2 Apply (`apply_preset`) — **clamps** every value to the current slider's
allowed range before setting it, so out-of-range presets are safely coerced.

### 10.3 Preset UI
- **Presets combo** in the dither section — selecting loads (`ui_load_preset`); index 0 = `" None"`.
- **"Save Preset"** button (`ui_save_preset_button`) → prompts for a name (`QInputDialog`, "Enter preset name:") → "Preset '…' saved successfully!".
- **Presets menu** (`create_presets_menu` / `populate_presets_submenu`): Load Preset, Save Preset, Bulk Import Presets, Import Preset(s), Export Preset, and a "Loaded Presets" submenu of saved presets.

---

## 11. Import / Export

### 11.1 Image import (`open_file_dialog` → `import_image` → `ImageLoader` worker)
- Dialog "Open Image", filter **`Images (*.jpeg *.jpg *.png *.webp)`**.
- Loaded off-thread (`ImageLoader(QRunnable)`); converted to grayscale `'L'`
  (logs "Converting image from `<mode>` to 'L' (grayscale).").
- **Auto-upscale for small images:** images **< 700×700** trigger a notice
  "Resolution Notice – Automatic Upscaling Applied" and the **preview** is upscaled
  to improve dither accuracy (export still uses the original full-size array).
- Optional **downscale** (`downscale_settings.apply_downscale`, default false) to
  `width×height` (1280×720) via `downscale_image(image, max_w, max_h, apply)` — keeps aspect ratio.
- Stores original dimensions; title becomes `Dither Boy - <name> (WxH)`.

### 11.2 Image export (`export_image` / `ImageExporter`)
- Dialog "Save Image", filter **`PNG Files (*.png);;All Files (*)`**.
- Runs `apply_effects_to_full_image()` and saves the full-resolution result as PNG.

### 11.3 Vector / SVG export (`export_vector` → `raster_to_svg`)
- Menu `Ctrl+Alt+P`? — actually **Export as Vector** (experimental). Shows a
  warning dialog first: *"WARNING: The vector export is highly experimental … try
  exporting with a larger scale and fewer fine details. Do you want to continue?"*
- Dialog filter **`SVG Files (*.svg);;All Files (*)`**.
- `raster_to_svg(image_array, threshold, invert)`:
  - Pixels **< threshold** are "filled". `invert=True` → background black, rects white; else background white, rects black.
  - **Optimization:** merges **vertical runs** of filled pixels into single `<rect>` elements to slash element count. No extra dependencies. Output:
    ```
    <svg xmlns="http://www.w3.org/2000/svg" width="W" height="H" version="1.1">
      <rect width="100%" height="100%" fill="BG"/>
      <rect x=".." y=".." width="1" height="runlen" fill="FG"/> ...
    </svg>
    ```
  - If the file is large it warns "Your export is a large vector file: N MB …".

### 11.4 Export to Photoshop (`export_image_to_ps`)
- Saves a temp PNG, then hands it to Photoshop. **Windows:** COM `Photoshop.Application`. **macOS:** `NSAppleScript` `tell application "Adobe Photoshop" … open theFile`. **Fails silently** if Photoshop isn't running.

### 11.5 Chroma-drop / transparent PNG (`export_chroma_drop_black` → `ChromaDropExporter`)
- Menu "Save as PNG", filter **`PNG Files (*.png);;All Files (*)`**.
- Produces an **RGBA** image where black becomes transparent (chroma-key drop of black), for overlay use. "Transparent image saved successfully."

### 11.6 Copy to clipboard (`copy_to_clipboard`, `Ctrl+Shift+C`)
- Converts the processed PIL image → PNG bytes → `QPixmap.loadFromData` → clipboard. Debug-logged if `CLIPBOARD_DEBUG` env is set; errors surfaced as "Couldn't copy image; please check app_debug.log".

### 11.7 Batch processing (`batch_process_images`, menu Batch → Select Folder)
- Requires an image already imported (as the **reference dimensions**).
- Confirmation: "This will process every image in the selected folder that shares
  the same dimensions as the input image…".
- Processes every `*.png/.jpg/.jpeg/.webp` whose dimensions match the reference,
  applying the **current** settings; writes into a `batch_processed` subfolder;
  skips mismatched sizes; final report "Processed N images. Skipped M images."

---

## 12. Video pipeline

Uses bundled `ffmpeg`/`ffprobe`. Signals classes: `VideoDitherSignals`,
`VideoAssembleSignals`, `WorkerSignals`. Workers: `VideoImportWorker`,
`VideoDitherWorker`, `VideoAssembleWorker` (all `QRunnable`). `video_mode`,
`video_framerate`, `video_input_file`, `video_temp_dir` track state.

### 12.1 Probe (`get_video_info`)
```
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 <file>   # → fps (parse "num/den")
ffprobe -v error -select_streams v:0 -show_entries format=duration     -of default=noprint_wrappers=1:nokey=1 <file>   # → duration seconds
```

### 12.2 Import (`import_video`)
- Dialog "Import Video", filter **`Video Files (*.mp4 *.avi *.mov *.mkv)`**, default dir = Movies.
- **Constraints (unless Expert Mode):** reject fps **> 60** ("Sorry, videos with a
  framerate above 60 fps aren't supported.") and duration **> 60 s** ("Sorry,
  videos longer than 60 seconds aren't supported."). Expert Mode prints
  "Expert mode active: skipping video importer limitations. 🤘" and bypasses both.
- Sets `video_framerate = fps`, enables the Export Video action, shows an
  indefinite "Extracting frames..." progress dialog (window-modal), runs
  `VideoImportWorker`.

### 12.3 Frame extraction (`VideoImportWorker.run`)
- Temp dir `ditherboy_video_*/original_frames/`.
- `ffmpeg -i <video> -qscale:v 2 <dir>/frame%06d.png` (`frame` + 6-digit index).
  On nonzero exit → "FFmpeg failed with exit code N".
- Preview: scan frames for the first with **average intensity > 5** (skip mostly-black
  frames) via `detect_preview_frame`; else "Could not detect a non-black frame for preview."

### 12.4 Dither each frame (`VideoDitherWorker.run`)
- Reads `original_frames/`, writes `dithered_frames/`.
- Per frame: `Image.open` → `np.array(float32)` → **`effect_registry.apply_all`**
  (same 6-step chain) → clip → uint8 → **resize back to `original_image_width×height` with NEAREST** → save. Emits progress per frame; supports **cancel** (`cancel()` sets `_is_canceled`), which cleans partial dithered frames and re-enables Export Video.

### 12.5 Reassemble (`VideoAssembleWorker.run`, `export_video`)
- Dialog "Export Video", filter **`MP4 Files (*.mp4)`**.
- Silent video encode:
  `ffmpeg -y -framerate <fps> -i dithered_frames/frame%06d.png -c:v libx264 -pix_fmt yuv420p temp_video.mp4`
- **Audio preservation:** probe original for an audio stream
  (`ffprobe … -select_streams a -show_entries stream=codec_type …`). If present:
  - Try stream copy: `ffmpeg -i <orig> -vn -acodec copy audio.m4a`; on failure re-encode `-vn -c:a aac -f adts`.
  - Merge: `ffmpeg -y -i temp_video.mp4 -i audio.m4a -c copy -shortest <out>`.
  - If no audio, just move `temp_video.mp4` → `<out>`.
- Progress dialogs: "Processing frames..." (definite bar) then "Reassembling video..." (indefinite). Final "Video export complete!".

### 12.6 Expert Mode (`toggle_expert_mode`, menu Extras → Expert Mode)
- Confirmation lists what it unlocks:
  *"Enabling Expert Mode will remove video importer limitations: • 60 fps videos
  longer than 60 seconds • 30 fps videos longer than 120 seconds • Only 60/30 fps
  videos are normally supported."*
- On: "Expert mode enabled: video importer limitations removed. 😉". Off:
  "Expert mode disabled. Back to safe defaults. 😌".

### 12.7 Temp cleanup (`purge_temp_folders`, `purge_temp_folders` menu)
- Deletes the `ditherboy_video_*` temp dirs; reports "Purged N temporary folder(s).".

---

## 13. Menus, shortcuts & help

### 13.1 Menu bar (`create_menus`)
- **&File:** &Open (`Ctrl+O`), &Save (`Ctrl+S`), Export to Photoshop (`Ctrl+Alt+P`), Export as Vector (Experimental), Save as PNG (chroma-drop), **Video** submenu (Import Video, Export Video), E&xit.
- **&Batch:** Select Folder.
- **&Adjustments:** Shuffle Adjustments, Shuffle Dither.
- **Presets:** (see §10.3).
- **&Extras:** Discord, Expert Mode ("Enable expert mode: remove video importer limits (fps/duration)"), Restart Application, Purge Temp Folders.
- **&Help:** Shortcuts (Show Keybinds), Check for Updates, Guide, Tutorials, &EULA, &About.

### 13.2 Hotkeys (`get_hotkeys(platform)`) — Windows / macOS

| Action | Windows | macOS |
|---|---|---|
| change_theme | `Ctrl+Shift+T` | `⌘+Shift+T` |
| export_image | `Ctrl+Shift+S` | `⌘+Shift+S` |
| copy_to_clipboard | `Ctrl+Shift+C` | `⌘+Shift+C` |
| import_image | `Ctrl+I` | `⌘+I` |
| restart_application | `Ctrl+Alt+R` | `⌘+Option+R` |
| zoom_in | `Ctrl+=` | `⌘+=` |
| zoom_out | `Ctrl+-` | `⌘+-` |
| zoom_reset | `Ctrl+0` | `⌘+0` |
| show_help | `Ctrl+Shift+/` | `⌘+Shift+/` |

Additional menu accelerators: Open `Ctrl+O`, Save `Ctrl+S`, Export-PS `Ctrl+Alt+P`,
Cycle theme `Ctrl+T`, Help `Ctrl+H`.

### 13.3 Help dialog (`HelpDialog`)
Shows "<platform> Keybinds" with `<h2>Keybinds</h2>`, a styled table of each
binding (keys rendered as pill buttons joined by "＋"; `⌘` on mac), a link
*"tutorial at studioaaa.com/ditherboyhelp"*, and a Close button.

### 13.4 About & external links
- **About** dialog: `Dither Boy v3.0.2\nBy Studio AAA`.
- External URLs (`QDesktopServices.openUrl`):
  - Discord: `https://studioaaa.com/wCw6FBDaTt`
  - Check for Updates: `https://studioaaa.com/product/dither-boy/#tab-wd_custom_tab`
  - EULA: `https://studioaaa.com/dither-boy-eula/`
  - Guide: `https://studioaaa.com/ditherboyhelp`
  - Tutorials (YouTube playlist): `https://youtube.com/playlist?list=PLvfny-DAmKHLwjpjyI0vuLGNyWZOvVY3J`

### 13.5 Shuffle / randomize
- `shuffle_adjustments`: randomizes the five adjustment sliders.
- `shuffle_dither_settings`: randomizes dither style + its parameter slider(s).

---

## 14. Licensing (OMIT / REPLACE for open source)

The shipped app is gated by an online license check. **An open-source clone
should delete this entire subsystem** (`LicensingManager`, `LicenseWidget`,
`LicenseValidationError`, the overlay, and the `manager` argument to
`ImageEditor`). Documented here only so behavior is understood and cleanly removed.

- `LicensingManager(app_name, app_author, verify_ssl)`; endpoint
  `https://licensing.studioaaa.com/api/v1/license/validate` (+ `/request-challenge`, `/verify`).
- Machine fingerprint → machine key; license cache `license_data.enc` encrypted
  with **Fernet** using `persistent_key.key`; server issues a **JWT (RS256)** token
  validated with a bundled public key hash (`268e2c39…fd9b9`) and `verify_exp/verify_iat`.
- Flow: `try_auto_validate()` (offline cache) → else `LicenseWidget` overlay asks
  for **Email + License Key**, a nonce challenge is requested, a signature computed
  and verified, and a token stored. On success `on_license_valid()` reveals the app;
  the whole UI is disabled behind the overlay until valid.
- `DB_DEV_MODE` env → dev bypass; failure logs written to `logs/failure_log_*.log`
  with `license_key` **censored** to `****CENSORED****`.
- The bundled build was distributed with a "DBPatcher.exe" that neutralizes this
  check — irrelevant to a clean OSS reimplementation (just don't build the gate).

---

## 15. Miscellaneous behaviors to preserve

- **Custom widgets:**
  - `GlowSlider` / `ResettableGlowSlider` — sliders with a colored `QGraphicsDropShadowEffect` "glow" (`glow_color`), double-click resets to default, wheel disabled.
  - `InvisibleSpinBox` — borderless spinbox with hidden arrows; shows a scaled "display" value; reveals on hover/focus.
  - `ClickableLabel` — label emitting `clicked` (used for "↔" shuffle icons), with hover color change and disabled-state color.
  - `NoScrollComboBox` — combo that ignores wheel scroll.
  - `ResettableLabel` / `ResettableSlider` — double-click resets.
- **Error handling:** a global `sys.excepthook` (`global_exception_hook`) logs to
  `crash_backtrace.log` / `app_debug.log` and shows an `ErrorDialog`
  (`assets/error_icon.png`). `CustomApplication.notify` wraps event dispatch to
  catch stray exceptions ("Exception in notify"). A humorous "You're working too fast!" guard exists.
- **Loading overlay:** a `QMovie` GIF (`assets/loading.gif`) shown after 400 ms with
  a semi-transparent label; idle logo uses `assets/idle.gif` (or theme's `idle.gif`).
- **`restart_application`** (`Ctrl+Alt+R`): closes and relaunches the process.
- **`use_viewport_proxy`** (default false): optional proxy widget for the viewport.
- **`pil_to_qimage` / `numpy_to_qimage`**: fast conversions without extra copies;
  unexpected modes are coerced to `'L'`.
- **Assets referenced:** `assets/icon.ico`, `assets/icon.png`, `assets/idle.gif`,
  `assets/loading.gif`, `assets/error_icon.png`, `assets/ffmpeg/{ffmpeg,ffprobe}.exe`.

---

## 16. Suggested open-source project layout

```
dither-boy-oss/
├── main.py                 # ImageEditor + all workers/widgets (from §4–§13, §15)
├── dither_registry.py      # DitherRegistry + 53 @njit kernels (from §9)
├── config/config.yaml      # §2 (ship defaults)
├── themes/
│   ├── default/theme.yaml  # Appendix A
│   ├── theme_one/{theme.yaml, idle.gif}
│   └── theme_two/{theme.yaml, idle.gif}
├── assets/
│   ├── icon.{ico,png}, idle.gif, loading.gif, error_icon.png
│   └── ffmpeg/{ffmpeg,ffprobe}[.exe]
└── requirements.txt        # PySide6, numpy, numba, llvmlite, Pillow, PyYAML, appdirs
                            #   (drop cryptography/PyJWT/requests — licensing removed)
```

**Build:** PyInstaller one-dir, entry `main.py`, `QApplication` style "Fusion",
Python 3.12. Bundle `assets/`, `themes/`, `config/`.

---

## Appendix A — `themes/default/theme.yaml` (verbatim QSS)

Reproduce exactly for a pixel-identical default look. `glow_color: "#5e89ed"`.
`labels:` for the default theme enables: Disable Preview, Dither Style, Dither
Scale, Tile Size, Invert Output, Contrast, Midtones, Highlights, Blur, Luminance
Threshold (all `true`).

```yaml
app_stylesheet: |
    QWidget#control_panel { background-color: #1f1f1f !important; border: none; }
    QWidget#central_widget { background-color: #1f1f1f !important; border: none; }

    QScrollBar:vertical, QScrollBar:horizontal { background: transparent; margin: 0px; }
    QScrollBar:vertical { width: 12px; }
    QScrollBar:horizontal { height: 12px; }
    QScrollBar::handle:vertical {
        background-color: #444444; border: 1px solid transparent;
        border-radius: 6px; min-height: 20px; margin: 2px;
    }
    QScrollBar::handle:horizontal {
        background-color: #444444; border: 1px solid transparent;
        border-radius: 6px; min-width: 20px; margin: 2px;
    }
    QScrollBar::handle:hover { background-color: #888888; }
    QScrollBar::handle:pressed { background-color: #555555; }
    QScrollBar::add-line, QScrollBar::sub-line {
        background: none; border: none; height: 0px; width: 0px; subcontrol-position: none;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

    QSlider { min-height: 18px; margin: 2px 0; }
    QSlider::groove:horizontal { background-color: #2e2e2e; height: 4px; border-radius: 2px; }
    QSlider::groove:horizontal:disabled { background-color: #222222; height: 4px; border-radius: 2px; }
    QSlider::sub-page:horizontal { background-color: #5e89ed; border-radius: 2px; }
    QSlider::sub-page:horizontal:disabled { background-color: #555555; border-radius: 2px; }
    QSlider::handle:horizontal {
        background: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5,
            stop:0 #a0a0a0, stop:0.2 #c0c0c0, stop:0.5 #f0f0f0, stop:1 #ffffff);
        border: 1px solid #555555; height: 14px; width: 14px; margin: -5px 0; border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5,
            stop:0 #b0b0b0, stop:0.2 #d0d0d0, stop:0.5 #f8f8f8, stop:1 #ffffff);
        border: 1px solid #777777; height: 14px; width: 14px; margin: -5px 0; border-radius: 7px;
    }
    QSlider::handle:horizontal:pressed {
        background: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.8, fx:0.5, fy:0.5,
            stop:0 #909090, stop:0.2 #b0b0b0, stop:0.5 #d0d0d0, stop:1 #e0e0e0);
        border: 1px solid #555555; height: 14px; width: 14px; margin: -5px 0; border-radius: 7px;
    }
    QSlider::handle:horizontal:disabled {
        background: #888888; border: 1px solid #777777; height: 14px; width: 14px;
        margin: -5px 0; border-radius: 7px;
    }

    QPushButton { background-color: #292929; border: none; padding: 5px 10px; border-radius: 10px; color: white; }
    QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); color: white; }
    QPushButton:pressed { background-color: #555555; }
    QPushButton:disabled { background-color: #292929; color: #888888; }

    QSpinBox {
        background-color: transparent; border: none; color: white; font-size: 12px;
        padding: 0px; margin: 0px; min-width: 35px; max-width: 35px;
    }
    QSpinBox:disabled { color: #888888; background-color: transparent; }
    QSpinBox::up-button, QSpinBox::down-button {
        width: 0px; height: 0px; border: none; background: transparent; subcontrol-origin: none;
    }
    QSpinBox::up-arrow, QSpinBox::down-arrow { width: 0px; height: 0px; background: transparent; }
    QHBoxLayout[objectName="slider_layout"] { spacing: 0; }

    QComboBox {
        background-color: #292929; border: none; padding: 5px; border-radius: 6px;
        color: white; width: 125px; font-size: 10.5px;
    }
    QComboBox:disabled { background-color: #444444; color: #888888; }
    QComboBox::drop-down { border: none; width: 20px; border-radius: 8px; }
    QComboBox QAbstractItemView {
        background-color: #292929; border: none; selection-background-color: #777777;
        selection-color: white; color: white; border-radius: 6px; padding: 5px;
    }
    QComboBox:on { border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; }
    QComboBox QAbstractItemView {
        border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;
        border-top-left-radius: 0px; border-top-right-radius: 0px;
    }

    QLabel { font-weight: bold; font-size: 11px; }

glow_color: "#5e89ed"
```

`theme_one` overrides panels to lime `#b4ff00` with dark labels `#292929`, thin
3px scrollbars, and ships an `idle.gif`; `theme_two` similar with its own `idle.gif`.

## Appendix B — Data/state directories

| Path | Contents |
|---|---|
| `<install>/config/config.yaml` | app config (§2) |
| `<install>/themes/*/theme.yaml` | themes (§3) |
| `<install>/assets/…` | icons, gifs, ffmpeg |
| `%LOCALAPPDATA%/Studio AAA/Dither Boy/PRESETS/*.yaml` | user presets (§10) |
| `%LOCALAPPDATA%/Studio AAA/Dither Boy/logs/failure_log_*.log` | error logs |
| `%LOCALAPPDATA%/Studio AAA/Dither Boy/persistent_key.key` | (licensing — drop) |
| `%TEMP%/DitherBoy`, `ditherboy_video_*` | video frame temp dirs (§12) |

---

*End of specification. Every UI control, slider range, formula, algorithm,
command line, file path, string, shortcut, and pipeline stage above was extracted
directly from Dither Boy 3.0.2's shipped bytecode and data files.*
