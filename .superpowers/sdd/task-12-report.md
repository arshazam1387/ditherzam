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
- Final review fix: cache admission is now the sole probability publication
  boundary. Oversized/rejected maps never become direct editor authority or render
  context, while an already-accounted prior result remains valid on ordinary
  re-detect failures.
- Final review fix: editor close marks inference as nonpublishable, invalidates and
  clears scheduled work, and polls the owned pool asynchronously before a guarded
  final close. A controlled
  blocking inference test proves the global render pool remains available and the
  GUI heartbeat remains responsive through multiple ignored-close polls; after the
  blocker releases, the editor closes and the inference pool reaches zero active
  threads.

## Verification

- JIT-off focused editor/lifecycle/request/worker/scheduler/thread-safety/
  resilience/cancellation/cache-budget after final review fixes: **58 passed**.
- Real-JIT required targets after final review fixes: **21 passed**.
- Async-close focused lifecycle/worker/render gate: **41 passed JIT-off** and the
  named lifecycle/thread-safety/cancellation gate remains **21 passed real-JIT**.
- `py_compile` and `git diff --check`: passed.
- The bounded full-suite gate was attempted twice (120 s and 300 s) and reached
  its time bounds without producing a summary under quiet capture. No focused or
  named real-JIT regression was observed; the ledger's 71 known golden failures
  remain the comparison baseline for the next wave gate.

No model weights, binaries, generated fixtures, `.codex/`, image, or temporary
pytest directories are included.
