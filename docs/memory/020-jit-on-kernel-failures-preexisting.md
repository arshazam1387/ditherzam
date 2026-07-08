---
type: gotcha
phase: 8
status: in-progress
date: 2026-07-08
---

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

**Fix (own task, separate from color work):** cast the offending indices to `int(...)` in
special.py so those kernels compile under njit; then re-verify BOTH JIT modes green. Until
then, "keep N green both JIT modes" can only be honored in JIT-off mode.
