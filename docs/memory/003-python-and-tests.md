---
type: constraint
phase: "-"
status: n/a
date: 2026-07-04
---

Target runtime is Python **3.12** (matches Numba + PySide6 wheels). On this machine
system Pythons are 3.11 and 3.13 (NOT usable); a clean 3.12 is available in the
session scratchpad at `dbwork/py312/python.exe`. Dither kernels are
`@njit(cache=True, parallel=True)`, float32 I/O, range 0..255. Run tests with
`NUMBA_DISABLE_JIT=1 pytest` for speed and coverage. TDD every task: red → green →
refactor, commit after each green.
