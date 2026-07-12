# SM-12 implementation report

Implemented editor-owned Smart Mask inference coordination and source invalidation.

## Result

- Added an immutable optional `MaskContext` snapshot to `RenderRequest`; existing
  callers continue to default to `None`.
- Connected the Smart Mask panel to a dedicated latest-wins inference scheduler,
  resilient worker terminals, injected offline adapter/model identity, probability
  cache, and render scheduling.
- New sources invalidate scheduler publication and source-specific cached state
  before replacement fields become visible.
- Disabled and Whole Image states launch no inference. Setting edits reuse the
  last valid probability map; re-detect, cancellation, no-subject, model failure,
  and generic failure never discard a prior valid result.
- Stale terminals cannot publish. A valid active terminal promotes at most one
  complete newest trailing request.
- Instantiated actual editor caches from `editor_cache_allocation`: disabled is
  192 MiB render / 0 MiB mask; enabled is 128 MiB render / 64 MiB mask. Both
  actual budgets and retained-byte sums are tested against the 192 MiB ceiling.
- Render workers receive the frozen request context and do not read live mask UI.
  SM-13 owns mask derivation, compositing, and overlay.

## Verification

- JIT-off focused editor/lifecycle/request/worker/scheduler/thread-safety/
  resilience/cancellation/cache-budget: **53 passed**.
- Real-JIT required editor/thread-safety/cancellation targets: **17 passed**.
- `py_compile` and `git diff --check`: passed.
- The bounded full-suite gate was attempted twice (120 s and 300 s) and reached
  its time bounds without producing a summary under quiet capture. No focused or
  named real-JIT regression was observed; the ledger's 71 known golden failures
  remain the comparison baseline for the next wave gate.

No model weights, binaries, generated fixtures, `.codex/`, image, or temporary
pytest directories are included.
