---
type: progress
phase: 8
status: in-progress
date: 2026-07-12
---

Smart Subject/Background Masking reached the approved pre-asset checkpoint on branch
`feat/smart-subject-masking` (off `4b99c4d`; implementation/review checkpoint
`ad1f418`, durable ledger commit/branch HEAD `a0a09da`). Plan: 17 tasks SM-01..SM-17 in
`docs/superpowers/plans/2026-07-11-smart-subject-masking.md`. Live ledger (source of
truth, recovery map) at `.superpowers/sdd/progress.md`; per-task briefs at
`.superpowers/sdd/task-NN-brief.md`.

**User-approved execution scope (2026-07-11):** autonomously build SM-01..SM-15 fully
green; SM-16 (model bakeoff) and SM-17 (packaging/E2E) as CODE SKELETONS ONLY
(converter tool, benchmark harness, packaging config, tests red-pending-asset), then
STOP for the user to supply/approve licensed U2NET weights + license/redistribution
sign-off. Do NOT fetch weights, make licensing judgments, or commit model binaries
autonomously. onnxruntime not installed; unit tasks use fake/injected sessions.

**Done + reviewer-approved:** SM-01..SM-15 production implementation, plus SM-16
licensed-model bakeoff and SM-17 packaging/E2E as fail-closed code skeletons. Final
whole-branch review found no remaining Critical/Important/Minor findings after
hardening commit `ad1f418`. Independent broad mask/offline verification: 320 passed,
310 expected asset-gated skips. Disposable pytest directories were removed; only
user-owned `.codex/` and `purple harrow.png` remain untracked.

**STOP / next task:** user must supply and approve licensed U2NETP/full-U2NET weights
and licensed fixtures, written redistribution/provenance evidence, then run SM-16's
reproducible conversions and real Windows bakeoff. Finalize exact seven ONNX outputs,
hashes and winner only after thresholds/manual QA pass; separately approve binary
inclusion. Then run SM-17 frozen Windows offline build/smoke and per-case evidence,
4K/RSS/cancellation/PNG/JPEG QA. Do not merge before those gates or before resolving/
explicitly waiving the 71 pre-existing creative-kernel golden failures.

**Gotcha to carry:** branch base (creative-dither commit `0335773`) has **71
pre-existing `test_kernels_all` golden-fixture failures** JIT-off, unrelated to masking
(golden fixtures need regeneration after the generative/creative-dither kernel changes).
Every masking task confirms it adds ZERO new failures beyond those 71. FLAG to user
before merging either branch to main.

**Orchestration history:**
`docs/superpowers/HANDOFF-smart-subject-masking-orchestration.md`; the live ledger is
the recovery authority for commits and task reviews.
Related: [[047-smart-mask-planning-handoff]].
