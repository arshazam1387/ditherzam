# SM-16 implementation report — skeleton only

Status: complete within the binding asset gate; real release acceptance unavailable.

The commit records only the developer converter, benchmark/report harness, metadata
contract, and asset-gated tests. It does not claim a selected model.

## Delivered

- Local-only reproducible TorchScript-to-ONNX converter with hashes and graph-
  discovered exact seven output names/order.
- Complete quality/performance report schema and honest nonzero selection gate.
- Executable local-asset path validates fixture provenance/checksums and both finalized
  candidate manifests, injects adapters, measures per-fixture/category quality plus
  the full timing/memory/resilience schema, and invokes the frozen SM-02 policy.
- Release validation reconstructs candidate reports and recomputes thresholds,
  category floors, latency gates, identities, provenance, approval, and winner; it
  does not trust serialized `within_budgets` or `selection` fields.
- Heartbeat and cancellation remain explicit `None`/pending unless dedicated
  injected probes measure event-loop gaps and cancellation-to-terminal latency.
  Process RSS values are real deltas, and 50-cycle growth is populated only after
  at least 50 cycles. The release gate enforces the complete ten-category/manual
  matrix, exact local file hashes/sizes, and all frozen latency/memory/geometry gates.
  Cached geometry is separately capped at 50 ms (1080p) and 150 ms (4K), in
  addition to the 100/300 ms combined geometry-plus-composite limits. Windows peak
  delta uses `PeakWorkingSetSize`; POSIX retained/growth remain pending because
  `ru_maxrss` is only a lifetime peak.
- Exact ordered output-name enforcement in the live adapter; the schema remains
  explicitly unfinalized (`output_names: null`) until approved conversion.
- Dated pending acceptance template and opt-in real-asset test.

Boundary tolerance remains provisional at 2 px pending fixture review. No weights,
fixtures, binaries, downloads, license judgments, or threshold changes were made.

## Verification

Focused JIT-off benchmark, quality, manifest, and adapter tests are green with the
real-asset test skipped. `python -m benchmarks.smart_mask --require-selection`
returns exit 2 while licensed inputs and user sign-off are absent, as required.
