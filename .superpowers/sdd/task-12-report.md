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
- Review fix: disabled source loads perform no source hashing or other mask work;
  source identity is derived lazily only when enabled detection is requested.
- Review fix: disabling clears probability authority, source identity, and all mask
  caches, returning to an actual 192/0 allocation. Re-enabling redetects, and every
  retained probability array is the same object charged by the bounded cache.
- Review fix: workers publish monotonic coarse safe-boundary progress (0/10/90/100),
  while the editor rejects stale source/model/generation progress and clears it on
  terminals. Inference uses an editor-owned single-thread pool so blocking inference
  cannot consume the render pool and trailing work remains serialized.

## Verification

- JIT-off focused editor/lifecycle/request/worker/scheduler/thread-safety/
  resilience/cancellation/cache-budget after review fixes: **56 passed**.
- Real-JIT required targets after review fixes: **19 passed**.
- `py_compile` and `git diff --check`: passed.
- The bounded full-suite gate was attempted twice (120 s and 300 s) and reached
  its time bounds without producing a summary under quiet capture. No focused or
  named real-JIT regression was observed; the ledger's 71 known golden failures
  remain the comparison baseline for the next wave gate.

No model weights, binaries, generated fixtures, `.codex/`, image, or temporary
pytest directories are included.
