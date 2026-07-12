# SM-15 Implementation Report

## Outcome

- `save_raster` now accepts only non-empty HxW, HxWx3, or HxWx4 numeric
  uint8-like arrays and only PNG/JPEG extensions. Boolean and non-finite floating
  arrays fail explicitly before clipping/casting.
- PNG writes explicit lossless straight RGBA without modifying any channel.
- JPEG RGBA uses the shared byte-exact `flatten_rgba_white` implementation before
  deterministic Pillow quality-95 encoding; grayscale and RGB retain historical
  behavior.
- JPEG export is visibly labeled as flattening transparency onto white and emits
  one nonblocking status-bar notice per editor session when an RGBA result is saved.
  The notice follows the actual case-insensitive selected filename suffix rather
  than the originating menu action/filter.
- Tests cover exact four-channel PNG round trips, shared flatten math, transparent
  corners, thin soft edges, channel/rank/extension rejection, and historical
  grayscale/RGB paths.

## Verification

- JIT-off focused: `32 passed` for `tests/test_export_raster.py` and
  `tests/test_mask_render_integration.py`.
- Real-JIT focused: `49 passed` for `tests/test_export_raster.py` and
  `tests/test_mask_composite.py`.
- `git diff --check`: clean.
- A full-suite JIT-off run exceeded the initial 120-second execution bound before
  producing a summary. The orchestrator accepted the focused and real-JIT gates
  against the established 71-failure baseline and directed termination of the
  longer retry.

## Scope

No model assets, binaries, `.codex/`, image inputs, or temporary pytest directories
were staged.
