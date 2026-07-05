---
type: gotcha
phase: 5
status: done
date: 2026-07-05
---

Two live-app bugs found by running the real GUI (all tests were green because they
passed explicit values / used `NUMBA_DISABLE_JIT=1`; the defaults were never
exercised end-to-end). Both fixed with regression tests; suite now **333 passed**.

1. **Blur applied to every render by default.** `apply_blur(img, value)` uses
   `radius = (value/10)**2`, so the default Blur of **50 → a 25px Gaussian blur**.
   Blur runs BEFORE dither, so an untouched image previewed blurry, dithers/effects
   looked muddy, and exports saved the blurred result. **Why:** blur's identity is
   `0`, not `50` (tonal sliders contrast/midtones/highlights are identity at 50).
   **Fix:** default the Blur control (`controls.py` state + slider) and
   `RenderSettings.blur` to `0`. See [[009-phase4-effects-render-done]].

2. **Palette/Mode + Effects never reached the pipeline.** `ImageEditor` built
   `RenderPipeline(registry, None, None)` once and never refreshed
   `color_engine`/`effect_stack`, so the Color and Effects panels did nothing; post
   effects also had no default params (would crash on naive wiring). **Fix:**
   `_sync_pipeline()` rebuilds color_engine + effect_stack from `panel.state` before
   every render (`render_now`/`_do_render`/`_rendered_rgb`); added `_current_color_engine()`
   and per-effect `_EFFECT_DEFAULTS` in `main_window.py`.

**How to apply / verify:** run the actual app (`.venv/Scripts/pythonw.exe -m
ditherzam.app`) — tests alone missed these because they exercised neither the UI
defaults nor the live JIT/worker path. Drive with a SHARP source image to expose
blur; grab `viewport.grab()` offscreen to inspect the preview deterministically.
