# SM-10 implementation report

## Result

Implemented `InferenceSignals` and `InferenceWorker` as a Qt-only asynchronous
boundary over a frozen `InferenceRequest` and injected segmentation adapter.

- Exactly one mutually exclusive terminal signal is emitted for success,
  no-subject, cooperative cancellation, unavailable model assets, or unexpected
  failure.
- Every signal carries an immutable `InferenceOutcome`, so the coordinating slot
  can release the exact scheduler request on every path.
- Cancellation is checked before adapter entry and after runtime return; the
  adapter receives only the request token's safe-boundary callback. ONNX is never
  forcibly terminated.
- Expected local asset failures remain distinct from logged runtime failures.
- The worker reads no editor, widget, pipeline, or live settings state.
- Review fix binds every successful `ProbabilityMap` identity to the frozen
  request and moves result validation/outcome construction into the protected
  decision path. Invalid results fail once, while cancellation wins races with
  no-subject, unavailable-model, and unexpected-error outcomes.

## Verification

`QT_QPA_PLATFORM=offscreen NUMBA_DISABLE_JIT=1 .venv/Scripts/python.exe -m pytest -q tests/test_mask_worker.py tests/test_mask_session.py tests/test_render_resilience.py --basetemp=.pytest-tmp-sm10b`

Result: **14 passed** in 4.55s.

Review-fix final rerun of the same targets: **24 passed** in 4.36s, including malformed
adapter returns, every identity mismatch, and cancellation/terminal races.

The full JIT-off suite was started with an isolated basetemp and produced no
early output before the bounded orchestration timeout; it was stopped on parent
instruction. The execution ledger already establishes the unrelated 71-failure
golden baseline. No full-suite result is claimed here.

## Files

- `ditherzam/ui/mask_workers.py`
- `tests/test_mask_worker.py`
- `.superpowers/sdd/task-10-report.md`
