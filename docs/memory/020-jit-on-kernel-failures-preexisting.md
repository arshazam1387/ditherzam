---
type: gotcha
phase: 8
status: done
date: 2026-07-10
---

**RESOLVED (2026-07-10).** All 66 kernels now compile under JIT-ON; full dithering
surface green both JIT modes. Fix below.


**7 kernel tests fail under JIT-ON (`NUMBA_DISABLE_JIT=0`) — PRE-EXISTING, not from any
recent feature.** Surfaced during Sub-project B's regression gate; verified to reproduce
identically on `main` at cb72c32 (before B). The default test mode (JIT-off, set by
`tests/conftest.py`) is fully green (545). This contradicts [[017-color-depth-ramp-shipped]]
which claimed "511 green both JIT modes" — the machine's Numba appears to have gotten
stricter since A shipped.

**Why:** `dithering/kernels/special.py` `_topography` (and `_wireframe_alt`, `_diagonal`)
index arrays with a **float64** subscript, e.g. `img[yd, sx]` where `sx`/index is a float →
Numba `TypingError: Unsupported array index type float64`. Fine when JIT is disabled (plain
Python), rejected by nopython compilation.

**Affected:** test_kernels_all (Diagonal, Topography, Wireframe Alt),
test_kernels_special::test_special_single_binary_shape, test_pipeline_levels (Topography,
Diagonal, Wireframe Alt).

**Fix applied:** cast the derived ROW index to `int(...)` **at the getitem site**, e.g.
`img[int(yd), x]` — NOT `yd = int(y+1) ...` at assignment. Numba's parfor `native_parfor_lowering`
re-types a conditionally-derived loop-index variable (`yd = y+1 if ... else y`) to float64 when
it is the first array subscript; casting the *variable* doesn't stick, casting *at the index*
does. Only `_topography`/`_diagonal`/`_wireframe_alt` hit it (their ternary else-branch is the
raw parfor index). Values are byte-identical (int truncation deterministic; golden tests pass
under JIT-ON). 915 green JIT-off; full dithering surface green JIT-ON.
