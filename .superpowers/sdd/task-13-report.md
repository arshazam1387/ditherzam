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

## Important-finding correction

The post-implementation review's four Important findings were corrected in a
follow-up commit. Request-local pipeline facades now freeze color/effect/source
context while sharing the editor's one locked, bounded staged-cache owner.
Derived masks and composites use SM-06's bounded caches and complete identities;
overlay is applied only after cache lookup. Exact and synchronous paths capture
all mutable inputs once. Cancellation gates surround every mask boundary and
cache publication is deferred until a complete result exists.

Additional integration coverage proves cache reuse, bounded shared ownership,
concurrent reassignment isolation, cancellation without partial publication,
and existing stage-order/cache-key invariance. Final gates:

- Focused JIT-off integration/render suite: 46 passed.
- Final cache/order correction subset: 18 passed.
- Named real-JIT mask/color/style suite: 16 passed.

The broken temporary Python base was safely replaced with a local uv-managed
Python 3.12.13 path in ignored `.venv/pyvenv.cfg`; no environment file is
committed.

## Cache-key and proxy-reuse correction

The final review findings removed all short-lived object IDs from masked render
identities. Signatures now use source identity, every render-setting value,
content-based color context, effect values, target geometry, mode, and an
algorithm version. Source Colors hashes its actual pixel content.

Masked previews now query the composite cache before rendering. Composite
misses reuse the complete proxy branch through the existing bounded staged
cache using an explicit stable cache key; sensitivity, feather, expansion,
invert, outside, and overlay edits therefore do not rerun creative stages.
Creative settings, source identity, and proxy geometry remain key partitions.

- Focused JIT-off cache/integration/order gate: 21 passed.
- Named real-JIT integration/color/style gate: 17 passed.

## Stable exact identity and cold disabled path

Exact masked export now uses only SourceIdentity, stable render-setting and
color/effect content signatures, geometry, and an algorithm version. Equal
independently reconstructed contexts hit the composite cache; creative changes
miss. Disabled proxy, Full, synchronous, and export paths branch before reading
`rendered_identity`, so they perform no mask signature or source-content hash.

- Focused JIT-off gate: 23 passed.
- Named real-JIT gate: 19 passed.
