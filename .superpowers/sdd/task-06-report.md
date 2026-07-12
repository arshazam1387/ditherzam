# SM-06 implementation report

Implemented three explicit Smart Mask cache partitions in
`ditherzam/masking/cache.py` while leaving `RenderPipeline` and its staged
`RenderCache` mask-unaware.

- Inference entries key directly on complete `InferenceIdentity` values.
- Derived masks key on complete `MaskIdentity` values (which deliberately
  excludes `OutsideMode`).
- Outer composites use `CompositeIdentity`, covering rendered output identity,
  mask identity, outside mode, source identity, and alpha algorithm version.
- All partitions share one bounded LRU and unique NumPy backing-store accounting.
  The standalone/mask-disabled staged render-cache default remains 192 MiB.
  An explicit editor allocation contract yields 192/0 MiB when masking is
  disabled and 128/64 MiB when enabled, rejecting custom sums over 192 MiB.
- Oversized entries are not retained, eviction is entry-atomic, source-scoped
  clearing spans all partitions, and cached derived/composite arrays are owned
  immutable snapshots.

Verification (JIT disabled, foreground):

`pytest -q tests/test_mask_cache.py tests/test_render_cache.py tests/test_render_cache_budget.py --basetemp=.pytest-sm06`

Result: **20 passed** in 1.76s.

Reviewer follow-up fixed inference accounting by explicitly charging
`ProbabilityMap.values` (the probability value object itself is not a generic
container). Tests now cover oversized inference rejection, replacement/alias
accounting, and a 50-source probability-map soak. The editor allocation policy
is regression-tested at 192/0 MiB when disabled and 128/64 MiB when enabled.
SM-12 must instantiate the actual editor-owned cache instances from this
contract and test their summed instance budgets; SM-06 does not claim that
later integration has already occurred.

Follow-up result: **24 passed** in 3.13s using the same focused targets.

Final allocation-contract follow-up: **25 passed** in 1.82s. This restores the
standalone 192 MiB render default while retaining the explicit bounded editor
split for SM-12 integration.

The optional full suite was also started in the foreground with the documented
environment and a dedicated basetemp, but exceeded the 120-second command limit
before pytest emitted its summary. No full-suite result is claimed here; the
required focused regression gate is green.
