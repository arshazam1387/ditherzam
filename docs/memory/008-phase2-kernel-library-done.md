---
type: progress
phase: 2
status: done
date: 2026-07-05
---

Phase 2 (full dither kernel library) done. Registry now holds **66** uniquely-named
kernels (contract `>=63` met). Category breakdown: Error Diffusion 16 · Ordered
Dither 12 · Patterned 11 · Glitch Effects 15 · Special Effects 12.

Modules under `ditherzam/dithering/kernels/`: `error_diffusion.py` (adds `_diffuse`/
`_diffuse_row` helpers + None/JJN/Stucki/Burkes/Sierra family/Stevenson-Arce/
Ostromukhov/Gaussian/Fan/Shiau-Fan/False-FS/Atkinson-Light; keeps P1's
Floyd-Steinberg/Atkinson/Bayer-4x4), `ordered.py`, `pattern.py`, `glitch.py`,
`special.py`. All five are wired in `ditherzam/dithering/__init__.py` in order
error_diffusion → ordered → pattern → glitch → special (glitch imports `_BAYER4`
from ordered, so ordered must precede it).

Golden fixtures live in `tests/golden/*.npy` (66 files) driven by
`tests/test_kernels_all.py` + `tests/golden_harness.py`. Full suite green:
`NUMBA_DISABLE_JIT=1 pytest -q` → 154 passed.

**Decision/deviation:** the tasklist deferred `__init__.py` wiring to Task 2.7, but
each task's "expect PASS" needs its module imported, so wiring was added
incrementally per task; 2.7's final import block already matched. No P1 kernel was
re-registered (registry dict would otherwise silently overwrite). Next: Phase 4.
