---
type: decision
phase: 8
status: in-progress
date: 2026-07-09
---

The high-resolution performance program is approved: capped drag and settled
previews; persisted Auto/480/720/1080/1440/2160/Full outside presets; one-off Full
Quality Preview (`Ctrl+Enter`); optional persisted zoom refinement default Off;
asynchronous capped initial load; exact isolated exports; immutable prioritized
render requests; content-keyed color reuse; exact diffusion/allocation
optimizations; a 192 MiB retained-cache budget; and selective measured threading
without simultaneous interactive renders.

**Why:** measured exact 4K work takes roughly 0.86–1.08 s, Python RGB diffusion
takes 5.56 s at only 480p, current capped previews still allocate a full-source RGB
display result, and the 4K staged cache retains about 182 MiB.

**Apply:** execute the dependency-ordered TDD program in
`docs/superpowers/plans/2026-07-09-high-resolution-preview-performance.md` against
the approved design in
`docs/superpowers/specs/2026-07-09-high-resolution-preview-performance-design.md`.
Preserve exact pixels/exports, frozen stage order, terminal worker signals, and
one-read engine/effect snapshots. Related: [[025-high-res-optimization-handoff]],
[[022-render-worker-wedge-fix]], [[023-binary-kernels-ignore-depth-fix]].
