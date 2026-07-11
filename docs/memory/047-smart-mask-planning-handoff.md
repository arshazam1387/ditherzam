---
type: progress
phase: 8
status: in-progress
date: 2026-07-11
---

Smart Subject/Background Masking specification is approved at
`docs/superpowers/specs/2026-07-11-smart-subject-masking-design.md`; the detailed
17-task TDD plan at `docs/superpowers/plans/2026-07-11-smart-subject-masking.md`
awaits approval before deriving the execution ledger. Approved v1 scope is still-image preview plus
exact PNG/JPEG export, automatic primary foreground, no manual refinement, and no
masked SVG/batch/video/animation. PNG preserves RGBA; JPEG flattens to white.
Implementation must be lean, optimized, and quality-first. Model selection is a
licensed U2NETP-versus-full-U2NET measured bakeoff through offline ONNX Runtime;
full U2NET wins for at least 0.03 absolute IoU or boundary-F improvement when it
also meets latency/memory budgets. Pretrained-weight redistribution confirmation,
hash/provenance manifest, Python 3.12 Windows packaging, and no runtime network or
download are hard gates. Next task after plan approval: derive the one-to-one
SM-01…SM-17 execution ledger; do not implement product code yet.
