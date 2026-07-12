# SM-13 Implementation Report

## Outcome

Implemented immutable Smart Mask integration at the outer boundary of proxy,
Full, synchronous, and exact still renders. The core `RenderPipeline` and its
frozen stage order/cache keys remain unchanged. Display overlay is applied only
to a fresh post-composite result and is absent from exact export authority.

Background workers render through a request-local, zero-retention pipeline so
concurrent GUI reassignment cannot alter snapshotted source/color/effect/mask/
outside state and cannot increase the retained editor cache budget.

## Verification

- JIT-off focused mask/render/thread/zoom gate: 74 passed; one pre-existing
  SM-12 heartbeat timing test failed under the 120-second aggregate load
  (`len(heartbeats) == 1`, expected at least 2).
- JIT-off directly impacted gate after final functional changes: 31 passed.
- `git diff --check`: passed.
- Named real-JIT gate could not start because the existing `.venv` base Python
  3.12 interpreter was deleted from Claude's temporary scratch directory after
  the successful focused runs. This is an environment failure before pytest
  collection, not a test failure; dependencies and the virtualenv were not
  mutated to conceal it.

No model weights, binaries, `.codex/`, `purple harrow.png`, or pytest temporary
directories are included.
