# SM-14 implementation report

## Outcome

- Presets persist exactly the seven reusable Smart Mask settings in the
  top-level `smart_mask` mapping. Session/source/model/derived state is excluded.
- `PresetContents` exposes Smart Mask settings while preserving legacy three-value
  unpacking. Missing/malformed Smart Mask mappings default and clamp safely.
- Preset application updates all controls before issuing at most one immediate
  render and one detection request.
- The Qt-free scope policy permits still raster output and unmasked/Whole Image
  media, while Subject/Background blocks SVG, batch, video, and animation exports.
- UI handlers and the video controller guard before dialogs, pipelines, workers,
  or filesystem work; no deferred API received mask arguments.
- Review fix: preset application now cancels queued debounce/settle/zoom work,
  invalidates scheduler trailing work, suppresses intermediate control renders,
  runs mask ownership/inference lifecycle once, reuses a matching published or
  cached primary probability map, and performs at most one synchronous paint.
- Review fix: SVG, batch, video-export, and animation-export controls visibly
  disable with an explanatory tooltip while their handler/shortcut guards remain.
- Review fix: non-finite and malformed mask numerics restore their approved
  per-field defaults rather than an arbitrary range endpoint.

## Verification

JIT-off, Qt offscreen focused gate:

`70 passed in 8.27s`

Review-fix focused gate: `75 passed in 53.17s`.

Targets covered mask preset/scope tests plus existing preset, ramp, preview
preference, export-menu, animation, video-controller, and video-worker tests.

The full-suite command was bounded twice (120 seconds, then stopped on orchestrator
direction after focused green); it produced no completed result. The execution
ledger already records the unrelated 71 kernel-golden baseline failures.

`git diff --check` passed. No `.codex/`, image, model, binary, or pytest temporary
directory was staged.
