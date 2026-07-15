---
type: progress
phase: 8
status: in-progress
date: 2026-07-14
---

Feedback Smear rebuild started on `feat/echo-smear-style`. The shipped Echo Smear
style (8 sliders, green, reviewed) was judged NOT visually equivalent to the
TouchDesigner reference — root cause: reference is a feedback loop (accumulated
displaced history), Echo Smear draws parametric edge copies. User decisions
(2026-07-14): keep both styles; rebuild = new "Feedback Smear" style simulating
K feedback iterations per pixel (backward noise walk + decay + erosion); still
image + Time scrub slider; quality-first K≤64; ship gate = user A/B against real
reference frames.

Docs: spec `docs/superpowers/specs/2026-07-14-feedback-smear-design.md`, plan
`docs/superpowers/plans/2026-07-14-feedback-smear.md` (5 tasks, full code),
handoff `docs/superpowers/HANDOFF-feedback-smear-rebuild.md`. Live ledger
`.superpowers/sdd/progress-echo-smear.md`.

**Gotcha (user standard):** parametric approximations of organic effects get
rejected — "looks exactly like it" means matching the generative process, and
motion sliders must produce visible nameable motion at every other slider's
zero. See the handoff's "standards learned the hard way".
Related: [[048-smart-mask-sdd-execution]] (branch base), [[049-classic-dither-defaults-restored]].
