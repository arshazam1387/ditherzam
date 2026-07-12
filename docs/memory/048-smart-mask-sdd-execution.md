---
type: progress
phase: 8
status: in-progress
date: 2026-07-11
---

Smart Subject/Background Masking is under subagent-driven TDD execution on branch
`feat/smart-subject-masking` (off `4b99c4d`). Plan: 17 tasks SM-01..SM-17 in
`docs/superpowers/plans/2026-07-11-smart-subject-masking.md`. Live ledger (source of
truth, recovery map) at `.superpowers/sdd/progress.md`; per-task briefs at
`.superpowers/sdd/task-NN-brief.md`.

**User-approved execution scope (2026-07-11):** autonomously build SM-01..SM-15 fully
green; SM-16 (model bakeoff) and SM-17 (packaging/E2E) as CODE SKELETONS ONLY
(converter tool, benchmark harness, packaging config, tests red-pending-asset), then
STOP for the user to supply/approve licensed U2NET weights + license/redistribution
sign-off. Do NOT fetch weights, make licensing judgments, or commit model binaries
autonomously. onnxruntime not installed; unit tasks use fake/injected sessions.

**Done + reviewer-approved:** SM-01 offline provenance gate (`3fad7ed`), SM-02 quality
metrics + winner policy (`9eca9f6`), SM-03 immutable contracts + settings (`2a9b217`
+ fix `c9f8922`). HEAD `c9f8922`. Next task: SM-04 deterministic mask geometry.

**Gotcha to carry:** branch base (creative-dither commit `0335773`) has **71
pre-existing `test_kernels_all` golden-fixture failures** JIT-off, unrelated to masking
(golden fixtures need regeneration after the generative/creative-dither kernel changes).
Every masking task confirms it adds ZERO new failures beyond those 71. FLAG to user
before merging either branch to main.

**Orchestration handoff for continuation:**
`docs/superpowers/HANDOFF-smart-subject-masking-orchestration.md`.
Related: [[047-smart-mask-planning-handoff]].
