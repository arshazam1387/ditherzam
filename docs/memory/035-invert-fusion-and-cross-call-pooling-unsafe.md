---
type: gotcha
phase: 8
status: done
date: 2026-07-10
---

Task 3.4 workspace/buffer reuse (`b2e1dc9`): the ONLY additional safe+measured exact
scratch win beyond 3.2's tonal `out=buf` was the **invert stage**. `apply_invert`
gained an `out=` param and `imaging.clamp_u8` gained `inplace=`; `render()` (L9) and
`render_cached()` (L6) now do `clamp_u8(apply_invert(fx, out=buf), inplace=True)`,
collapsing 5 full-RGB allocations (`.astype(f32)` + `255.0 - x` + redundant recast +
clamp copies) into one call-private scratch buffer + the final uint8 result.
Byte-identical (255-x on values 0..255 is exact in float32). Measured **−39% time /
−58% peak** at 4K in isolation, **−41%** in-pipeline. `buf = np.empty_like(...)` is
call-private and never reaches a cached slot; invert is recomputed every call (never
cached), so nothing across calls can alias it.

**Why the other candidates were REJECTED (durable invariant):** color map, dither,
and saturation outputs are all **stored in `render_cached()`'s LRU cache**
(`c["colored"]`, `c["d"]`, `c["satout"]`, also `c["g"]`, `c["fx"]`). A cross-call
pooled/leased buffer handed to two consecutive `render_cached()` calls would let call
2 overwrite call 1's still-cached intermediate → silent cache corruption. So **only
within-call, call-private scratch for stages whose output is NOT cached is safe to
reuse**; a module-global or cross-call buffer pool is forbidden here. `apply_blur`
stays PIL-backed (no `out=` hook, and its identity path returns its input unchanged).

**Fix/Apply:** future "reuse a buffer" ideas in render.py must prove the buffer never
backs a value stored in the render cache or returned to the caller. Related:
[[032-tonal-one-pass-changes-pixels]] (the first proven workspace win),
[[033-qimage-zero-copy-rejected]] (same aliasing-safety discipline). Tests:
`tests/test_render_scratch_reuse.py`.
