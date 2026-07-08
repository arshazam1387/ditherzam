# Handoff — Diagnose & fix "app gets stuck the more you use it"

Paste this into a fresh Claude Code session in `C:\Users\arsha\Desktop\custom dither`.

---

We have a bug in ditherzam (PySide6 image-dither app). **This is a bug fix, not a
feature** — do NOT use the new-feature workflow; use **`superpowers:systematic-debugging`**.
Find the root cause before touching anything (Iron Law: no fixes without root-cause
investigation). **First invoke the `zam-memory` skill (recall)** to load project state.

## Symptom (from the user)

"The more you use the program, it starts to get stuck." User's guesses: a memory
leak, or "it's not removing the past filters after finding a new one" — some kind of
problem with the render engine. It degrades/freezes with continued use (changing
dither styles, stacking Effects, and hovering palettes — the palette hover-preview is
brand-new, just shipped in Sub-project C).

## Investigation already done (start here — don't re-derive)

**Ruled out:** the render cache is NOT a leak. `RenderPipeline._cache`
([ditherzam/render.py:67](../../ditherzam/render.py), `render_cached` at L117-222) is a
single dict with a fixed key set that overwrites in place — it does not grow per edit.

**Leading hypothesis — the render coalescer wedges permanently on a worker exception:**

- Render requests go through `RenderCoalescer`
  ([ditherzam/ui/render_scheduler.py](../../ditherzam/ui/render_scheduler.py)). It keeps
  at most one render in flight: `request()` returns `None` (coalesced, no worker) when
  `self._busy` is True. `_busy` is set True in `_begin()` and is **only ever reset to
  False in `on_finished()`**.
- `on_finished()` is called from `ImageEditor._on_rendered`
  ([ditherzam/ui/main_window.py:459](../../ditherzam/ui/main_window.py)), which is the
  slot connected to the worker's `finished` signal.
- The worker `_RenderWorker.run` ([main_window.py:63-69](../../ditherzam/ui/main_window.py))
  has **no try/except**:
  ```python
  def run(self) -> None:
      if self._mode == "proxy":
          rgb = render_preview(self._pipeline, self._base_gray, self._settings, self._proxy_max_side)
      else:
          rgb = self._pipeline.render_cached(self._base_gray, self._settings)
      self.signals.finished.emit(numpy_to_qimage(rgb), self._token)
  ```
  It runs on a `QThreadPool` thread. **If `render_preview` / `render_cached` /
  `numpy_to_qimage` raises, `finished` is never emitted → `on_finished()` never runs →
  `_busy` stays `True` forever → every subsequent `request()` returns `None` → no more
  workers launch → the preview stops updating = "stuck."** The exception dies silently
  on the pool thread, so there is no visible error.
- Aggravating: `RenderCoalescer.invalidate()` bumps `_gen` but does **not** reset
  `_busy`/`_pending`. And the new palette hover-preview
  (`ImageEditor._on_palette_preview` → `schedule_render()` on every hovered row) fires
  renders far more often than before and routes a *preview* palette through the color
  engine — increasing both the frequency of renders and the chance of hitting a
  throwing one. That likely explains why the freeze showed up / worsened after
  Sub-project C.

**Not yet confirmed (this is where to pick up):** *which* render input actually
throws. Was about to reproduce headlessly. Prime suspects, in order:
1. **Proxy path on a downscaled image.** `render_preview`
   ([ditherzam/ui/preview.py:36](../../ditherzam/ui/preview.py)) calls `pipeline.render()`
   (NOT `render_cached`) on a `nearest_downscale`d base with a reduced `scale`. An
   effect or kernel that assumes a minimum size could throw on a tiny proxy — matches
   "the more filters you add." Test each of the 5 Effects (Blur, Sharpen, Chromatic
   Aberration, JPEG Glitch, Epsilon Glow) and each dither style at small proxy sizes.
2. **Thread-safety.** `render()` (proxy path) does NOT take `_cache_lock` and reads/mutates
   `pipeline.color_engine` (the ramp `ColorEngine` mutates its own `depth`/`mapping` and
   an internal ramp cache). If a GUI-thread `render_now()` (which calls `invalidate()`
   but does not stop the running worker) overlaps an in-flight worker, two threads touch
   the engine/cache at once — possible corruption or throw.
3. `numpy_to_qimage` on a non-contiguous/unexpected array from upscale.

## How to get evidence (Phase 1)

Reproduce headlessly (no Qt window needed for the render paths):
- Build the pipeline like the app does (default dither registry + a `ColorEngine` +
  an `EffectStack`); see `ImageEditor._current_color_engine` / `_current_effect_stack`
  / `_sync_pipeline` in [main_window.py](../../ditherzam/ui/main_window.py) for exact
  construction, and `_EFFECT_DEFAULTS` (top of that file) for effect params.
- Drive `render_preview(pipeline, base_gray, settings, max_side)` and
  `pipeline.render_cached(...)` across a matrix: every color `mode`
  (off/nearest/ordered/diffused/ramp) × several `depth`/`color_mapping` × every dither
  `style` × each Effect stacked, at **small** base sizes (e.g. 12×12, 30×40) so proxy
  factor > 1 kicks in. Catch and print any exception + the settings that triggered it.
- Also write a focused test that proves the coalescer wedge directly: simulate a
  worker whose `run` raises, and assert the coalescer never recovers (`request()`
  keeps returning `None`). This is the real defect regardless of which input throws.

## The fix (Phase 4 — after root cause is confirmed, TDD)

Likely two parts, but confirm with evidence first:
1. **Make the worker exception-safe** so the coalescer can never wedge: wrap
   `_RenderWorker.run` in try/except and always emit `finished` (or add a `failed`
   signal) so `on_finished()` runs and `_busy` resets even on error. Consider having
   `RenderCoalescer.invalidate()` also reset `_busy`/`_pending` for robustness.
2. **Fix the actual throwing render** you found (e.g. guard the effect/kernel for tiny
   proxy sizes, or fix the thread-safety overlap). Fix at the source, not the symptom.

Write the failing test first (TDD), fix the root cause, verify. Keep the suite green
in **JIT-off** mode (`NUMBA_DISABLE_JIT=1`, set by `tests/conftest.py`; currently
**578** passing). Runner: `./.venv/Scripts/python.exe -m pytest`.
**Launch the app to reproduce visually:**
`./.venv/Scripts/python.exe -c "from ditherzam.app import main; main()"`.

## Invariants (verify against the tree, don't trust memory)

- Clean-room; Qt-free core (`ditherzam/color/**`, `dithering/**` never import PySide6).
- Don't change the frozen `RenderPipeline.render()` STAGE ORDER or `RenderSettings`
  fields. (You MAY add try/except in the UI worker and touch the coalescer — those are
  UI-layer, not the frozen render contract.) `render_cached` must stay byte-identical
  to `render` (there's a `test_render_cache` proving it).
- Python 3.12; TDD per fix.
- Pre-existing: 7 kernel tests fail JIT-**on** (`special.py` float-index) — not your bug.

## When done

Record via `zam-memory` (a `gotcha` entry for the coalescer-wedge + the throwing
input, and a `progress` entry), update `docs/memory/INDEX.md`, then
`superpowers:finishing-a-development-branch`.
