# SM-09 implementation report

Implemented immutable inference request and terminal contracts plus a dedicated,
Qt-free latest-wins inference scheduler.

## Delivered

- `CancellationToken`: thread-safe, idempotent advisory cancellation.
- `InferenceRequest`: frozen request including source identity, exact model identity,
  preprocessing version, owned read-only RGBA snapshot, generation, and per-run token.
  The full content hash, dimensions, and alpha participation are re-derived from the
  owned snapshot and must exactly match the supplied identity.
- `InferenceTerminal` / `InferenceOutcome`: distinct success, no-subject, cancelled,
  and failed terminal values with fail-closed payload validation.
- `InferenceScheduler`: lock-protected one-active/one-newest-trailing scheduling,
  cooperative obsolescence, source invalidation, generation/source/model-hash
  publication matching, duplicate-terminal suppression, and terminal recovery.
- RenderScheduler was not modified or generalized.

## Verification

- Focused JIT-off gate: `19 passed` for `test_inference_request.py`,
  `test_inference_scheduler.py`, and `test_render_scheduler.py`.
- Full JIT-off suite was attempted twice but exceeded the bounded 120s/300s task
  windows without producing a result; no process was left running. The execution
  ledger already documents the unrelated 71 golden-kernel baseline failures.
- `git diff --check` clean.

Review fix: adversarial same-shape/different-content and alpha-mismatch identities
are rejected; terminal duplication is guarded by active request identity without an
unbounded generation-history set.

## Scope

Only SM-09 product files, tests, and this report are included. Existing `.codex/`,
pytest temporary directories, and `purple harrow.png` remain untracked and unstaged.
