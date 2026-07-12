# SM-17 implementation report â€” skeleton only

Date: 2026-07-12

Commit scope: frozen Windows packaging/offline/E2E certification scaffolding.
This report makes no release-readiness, licensing, model-selection, quality, or
performance claim.

## Added

- Fail-closed frozen bundle verifier requiring approved status, pinned
  `onnxruntime==1.22.1`, content-addressed manifest/model, LICENSE, NOTICE,
  provenance, and at least one content-addressed ORT DLL.
- Pending release-lock example with no asset or binary claims.
- Socket-denied local verification test and missing/corrupt/path-escape gates.
- A collected 60-case real-model E2E/resilience matrix which skips only while no
  release lock exists and fails closed if a configured bundle is invalid.
- Full dated acceptance report template with every automated/manual gate pending.

## Verification

`NUMBA_DISABLE_JIT=1 pytest -q tests -k "mask or offline_security"`

Result: **302 passed, 61 skipped, 1303 deselected**. Sixty skips are the explicit
real-model E2E matrix; the other skip pre-existed. No selected asset, weights,
binaries, ORT installation, or network operation was introduced.

The known 71 unrelated kernel-golden failures were outside this focused run and
remain flagged in the execution ledger.

## Stop gate

Certification remains blocked until the user supplies/approves licensed weights,
redistribution evidence, selected bakeoff winner, exact seven output names, frozen
runtime inventory, and human inclusion/sign-off. At that point the example lock
must be replaced by `packaging/smart-mask-release.lock.json`, the real matrix
adapters completed, and every report row populated with dated evidence.
