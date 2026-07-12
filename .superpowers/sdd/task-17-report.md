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
- Minimal Windows/Python 3.12 PyInstaller spec, pre-build verifier, and offline
  frozen smoke launcher. The build entry point verifies an approved release lock
  before importing/invoking PyInstaller.
- Exact two-DLL ORT inventory, schema/content relationship, byte-count, unique-path,
  and fixed `onnxruntime/capi` gates, with successful and per-component-corruption
  synthetic bundle coverage.
- Offscreen editor missing-model smoke proving load/render stays unmasked and an
  explicit detection request presents Model unavailable without creating sockets.
- Concrete certification hooks and an expanded disabled/target/outside/surface/
  lifecycle/cache/performance E2E matrix; approved bundles call hooks rather than
  an unconditional placeholder failure.
- Certification evidence is now fail-closed: every approved case must carry an
  executed flag, exact case identity, output hashes/dimensions/formats, canonical
  evidence hash, and resilience/cache/performance fields where applicable.
- The bounded `--offline-smoke` path verifies the lock, decodes the locked fixture,
  runs a local adapter, produces masked PNG plus disabled JPEG, emits hashed JSON,
  and exits without entering the Qt event loop. Synthetic injection tests it.
- PyInstaller includes only the two exact locked DLL paths; a post-build verifier
  rejects missing, corrupt, or extra DLLs.

## Verification

`NUMBA_DISABLE_JIT=1 pytest -q tests -k "mask or offline_security"`

Result after review fixes: **310 passed, 309 skipped, 1303 deselected**. 308 skips
are the expanded explicit real-model E2E matrix; the other skip pre-existed. No selected asset, weights,
binaries, ORT installation, or network operation was introduced.

The known 71 unrelated kernel-golden failures were outside this focused run and
remain flagged in the execution ledger.

## Stop gate

Certification remains blocked until the user supplies/approves licensed weights,
redistribution evidence, selected bakeoff winner, exact seven output names, frozen
runtime inventory, and human inclusion/sign-off. At that point the example lock
must be replaced by `packaging/smart-mask-release.lock.json`, the real matrix
adapters completed, and every report row populated with dated evidence.
