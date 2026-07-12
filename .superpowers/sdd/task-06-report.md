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
  The default mask sub-budget is 64 MiB and cannot exceed the 192 MiB editor
  retained-cache ceiling.
- Oversized entries are not retained, eviction is entry-atomic, source-scoped
  clearing spans all partitions, and cached derived/composite arrays are owned
  immutable snapshots.

Verification (JIT disabled, foreground):

`pytest -q tests/test_mask_cache.py tests/test_render_cache.py tests/test_render_cache_budget.py --basetemp=.pytest-sm06`

Result: **20 passed** in 1.76s.

The optional full suite was also started in the foreground with the documented
environment and a dedicated basetemp, but exceeded the 120-second command limit
before pytest emitted its summary. No full-suite result is claimed here; the
required focused regression gate is green.
