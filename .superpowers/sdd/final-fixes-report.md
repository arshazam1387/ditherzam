# Smart Mask final fixes report

Date: 2026-07-12

## Resolved final-review findings

- `ModelManifest` now validates approved model provenance, pinned upstream commit,
  lowercase SHA-256 values, non-negative byte count, fixed tensor contracts, safe
  relative paths, and finalized output-name structure during direct construction.
  YAML loading and synthetic adapter fixtures continue to use the same release
  invariants.
- Preset application restores both Qt signal-blocking states and
  `_applying_preset` in `finally`, including when a future control application
  raises. The successful path still performs one mask lifecycle and one render.
- `VideoController` retains its Import Video action and refreshes its enabled,
  tooltip, and status-tip state alongside Export Video. Export retains its
  independent imported-media prerequisite and both handlers retain guards.
- `boundary_f_score` rejects boolean tolerances rather than treating `True` as
  integer tolerance 1.

## Verification

JIT-off/offscreen focused command:

```text
QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 .venv/Scripts/python.exe -m pytest -q tests/test_mask_model_assets.py tests/test_mask_adapter_contract.py tests/test_mask_quality_metrics.py tests/test_glow_preset_wiring.py tests/test_mask_scope_gating.py tests/test_video_controller.py --basetemp=.pytest-final-fixes-2
```

Result: **107 passed in 14.64s**.

`git diff --check` passed. No thresholds, model assets, packaging scope, or
licensed-weight decisions changed.
