# HANDOFF — Feedback Smear: make lines smear INTO the object

**Date:** 2026-07-14 · **Branch:** `feat/echo-smear-style` (based on `feat/smart-subject-masking`; merges only after the mask branch) · **HEAD at handoff:** `960992e`

## Where this stands (one paragraph)
Two styles now exist on this branch. **Echo Smear** (parametric line effect, 8 sliders, complete, all reviews clean) — user keeps it but it is NOT the target look. **Feedback Smear** (new, simulates a TouchDesigner feedback loop by per-pixel backward walk through a value-noise field) — user verdict after A/B iteration 1: *"this is better, the effect is good, BUT there are still no lines smearing into the object — no line going and turning into it."* That missing element is the whole remaining job.

## What "lines smearing into the object" means (the acceptance target)
**FIRST ACTION: study the reference video yourself.** It is at
`C:\Users\arsha\Downloads\Phone Link\Will definitely try to modulate everything at least once.mp4`
(9s, 720x1280, 30fps — a phone capture of a TouchDesigner session; crop the app chrome, top ~95px and bottom ~320px).
Extract and LOOK at (Read tool renders PNGs):
1. `ffmpeg -vf fps=1` over the whole clip — overall look per second;
2. a dense burst, e.g. `ffmpeg -ss 3 -t 2 -vf fps=10` — consecutive frames are the ONLY way to see how the lines travel; diff neighboring frames mentally: which lines moved where, how fast, what merges into the tower.

What you must reproduce: the tower has a handful of LONG, CONTINUOUS, individually-readable wavy vertical lines to its right that (a) hug the silhouette profile, (b) visibly TRAVEL toward and merge INTO the subject as time advances, and (c) coexist with the dense dissolving speckle Feedback Smear already produces. We have the speckle fan; we lack the distinct traveling lines. Judge every iteration by rendering a Time-sweep GIF and comparing it against the dense-burst frames — not against a single still.

## Technical leads for the next iteration (in priority order)
1. **Time must advance the feedback age, not just slide the field.** Currently `fs_time_slider` only shifts noise coordinates (`tshift`) — the trail pattern wobbles but nothing marches inward. Add fractional-iteration travel exactly like Echo Smear's Wave Phase fix (commit `fbeaaeb`, reviewed clean): continuous echo index `e = k - t_frac`, first step partial, so copies march INTO the subject and a new one fades in at the far end. That fix's derivation lives in `.superpowers/sdd/echo-fix-4-brief.md`.
2. **Line continuity.** The trailing-edge marks fragment because neighboring rows walk slightly different noise paths and the survival dither breaks lines. For the first few k (near, high-survival copies): suppress the dither (solid if `surv*dgain*1.5 >= 1`, already true near) AND reduce per-row path divergence (e.g. sample the walk's noise at a y quantized to ~4px, or lower `namount`'s y-component further) so each copy reads as ONE continuous bent line.
3. **Consider a hybrid**: the reference look = feedback speckle (have it) + a few long continuous contour lines (Echo Smear's strength). If 1+2 don't get there, blend Echo-Smear-style continuous trailing-edge lines whose lateral offset follows the SAME `_vnoise` field (so they bend organically, not sinusoidally) into `_feedback_smear`. All the pieces exist in `ditherzam/dithering/kernels/special.py`.

## Ground truth / procedures (do not re-derive)
- **Ledger (source of truth):** `.superpowers/sdd/progress-echo-smear.md` — every task/fix, commits, review verdicts, minor-findings roll-up for the final review. `.superpowers/sdd/progress.md` is the Smart Mask project's — never touch.
- **Spec/plan:** `docs/superpowers/specs/2026-07-14-feedback-smear-design.md`, `docs/superpowers/plans/2026-07-14-feedback-smear.md`. Plan Task 4 (A/B loop, controller-run, user gate) is where we are; Task 5 (final whole-branch review → zam-memory → finishing-a-development-branch) remains.
- **Test procedure per kernel change:** JIT-off `NUMBA_DISABLE_JIT=1 .venv/Scripts/python.exe -m pytest tests/test_feedback_smear.py -q`; golden re-bake = delete `tests/golden/Feedback Smear.npy` + run `tests/test_kernels_all.py -k Feedback` twice; JIT-on parity = same two files without the env var; commit each iteration.
- **Renders for the user:** JIT-ON (kernel too slow interpreted), red-on-black (`rgb[out==0.0]=(255,40,30)`), clean synthetic cat + binarized reference tower (`gray>40 → 20.0 else 230.0` after cropping the phone-video chrome), Time-sweep GIFs. The tower source contains the reference's own baked trails — say so when comparing density.
- **Suite baseline:** gates 10/10, parity 90/90, full suite 1687 passed/311 skips + known load flake (`test_mask_editor_lifecycle.py::test_blocking_inference_pool_...`, passes isolated) + one-off `test_no_model_weights_or_binaries_are_committed` order-artifact (passes isolated and with clean tree — re-check on next full run).

## User standards (violating these caused every rejection so far)
- "Looks exactly like it" = match the reference's generative process; parametric shortcuts get rejected.
- Motion sliders must produce visible, nameable motion (travel/march), not statistical shuffles; verify at every other slider's zero.
- No per-pixel hash dicing of things that should be continuous lines; helper elements must blend (taper/dissolve/sway).
- Prove behavior with discriminator tests whose old-code failure is analytically derived; prove looks with rendered GIFs/strips, not descriptions.
- Feedback Smear does NOT use the ML mask (threshold = subject gate); Smart Mask integration waits on licensed U2NET weights (see `docs/memory/048…`).

## Execution
Subagent-driven (superpowers:subagent-driven-development): briefs/reports under `.superpowers/sdd/fs-*`, sonnet reviewers, cheap implementers when the brief contains complete code; kernel-look iterations in Task 4 may be controller-inline (they were for iteration 1) but every iteration ends with tests + re-bake + parity + commit. Ship gate stays: the USER says when the lines look right.
