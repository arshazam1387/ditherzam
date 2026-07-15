# HANDOFF — Feedback Smear rebuild (exact-look redo of the smear effect)

**Date:** 2026-07-14 · **Branch:** `feat/echo-smear-style` (based on `feat/smart-subject-masking`; merges only after the mask branch)

## Why this handoff exists
The user's requirement is that the effect look **exactly** like the TouchDesigner reference video ("Will definitely try to modulate everything at least once.mp4"). The shipped **Echo Smear** style (8 sliders, fully green, all reviews clean) is a parametric line effect and was judged not graphically similar — root cause: the reference is a *feedback loop* (accumulated displaced history), not repeated edge copies. The user chose a full rebuild as a NEW style, **Feedback Smear**, keeping Echo Smear as-is.

## Authoritative documents
- Spec: `docs/superpowers/specs/2026-07-14-feedback-smear-design.md` (user decisions + model + sliders)
- Plan: `docs/superpowers/plans/2026-07-14-feedback-smear.md` (5 tasks, complete code, TDD)
- Ledger (live, update per task): `.superpowers/sdd/progress-echo-smear.md` — Echo Smear history + fix log; append Feedback Smear tasks there. `.superpowers/sdd/progress.md` belongs to the Smart Mask project — never touch it.
- Reference frames: session scratchpad `refvid/frame_*.png`; binarize like `echo_smear_tower_demo.py` before feeding kernels.

## User decisions already made (do not re-ask)
1. Keep both styles; rebuild is "Feedback Smear".
2. Still image + Time scrub slider (0–360, non-looping); animation via existing system.
3. Quality first: Trail Length up to K=64.
4. **Ship gate = user A/B** against real reference frames (plan Task 4). Constants are tunable there; every retune re-runs gates + re-bakes the golden.

## Standards learned the hard way on this branch (bake these in)
- The user rejects "shortcut" looks: no per-pixel hash dicing of continuous lines; motion sliders must produce *visible, nameable motion* (travel/sway), not statistical shuffles; straight helper lines must blend (taper/dissolve/sway) with the composition.
- Prove behavior with discriminator tests whose failure on the old code is analytically derived — several "obvious" probes on this branch were geometrically vacuous.
- Phase/Time sliders: verify they visibly change output at EVERY other slider's zero (the Wave Phase dead-slider bug).
- Golden invariants: any claim "default output unchanged" must hold without re-baking, else STOP.
- Visual proof to the user as GIF/strips rendered from real kernels, not descriptions.

## Execution
Subagent-driven (fresh implementer per task, task review after each, sonnet reviewers, cheap implementers when the plan text contains the full code). Tasks 1–3 are mechanical from the plan. Task 4 is controller+user iteration. Task 5 = final whole-branch review + zam-memory + finishing-a-development-branch.

Suite baseline before this rebuild: 108/108 (echo file + kernels_all) both JIT modes; full suite green with ~310 asset-gated mask skips and one known load-dependent flake (`test_mask_editor_lifecycle.py::test_blocking_inference_pool_...`, passes in isolation).
