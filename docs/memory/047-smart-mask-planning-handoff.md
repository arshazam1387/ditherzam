---
type: progress
phase: 8
status: done
date: 2026-07-12
---

Smart Subject/Background Masking planning is complete: approved specification at
`docs/superpowers/specs/2026-07-11-smart-subject-masking-design.md`, approved
17-task TDD plan at `docs/superpowers/plans/2026-07-11-smart-subject-masking.md`,
and execution ledger at `docs/tasklists/09-smart-subject-masking-tasks.md` plus live
SDD recovery state in `.superpowers/sdd/progress.md`. Approved v1 scope is still-image preview plus
exact PNG/JPEG export, automatic primary foreground, no manual refinement, and no
masked SVG/batch/video/animation. PNG preserves RGBA; JPEG flattens to white.
Implementation must be lean, optimized, and quality-first. Model selection is a
licensed U2NETP-versus-full-U2NET measured bakeoff through offline ONNX Runtime;
full U2NET wins for at least 0.03 absolute IoU or boundary-F improvement when it
also meets latency/memory budgets. Pretrained-weight redistribution confirmation,
hash/provenance manifest, Python 3.12 Windows packaging, and no runtime network or
download are hard gates. Execution status and remaining external gates are tracked
in [[048-smart-mask-sdd-execution]].
