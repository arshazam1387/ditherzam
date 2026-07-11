---
type: gotcha
phase: 8
status: done
date: 2026-07-09
---

Removing `numpy_to_qimage(...).copy()` is rejected for the current Qt path.

**Why:** direct borrowing fails source-mutation isolation. A safe private
NumPy-owned buffer passed GC and queued-signal lifetime tests but was slower at
4K (about 14.26 ms versus 11.35 ms for the existing copy path).

**Fix/Apply:** keep the existing owned `QImage.copy()` conversion. Capped preview
dimensions already remove the costly full-4K display allocation; do not revisit
zero-copy without new Qt ownership evidence and a measured win. See commit
`bd0cdc1`.
