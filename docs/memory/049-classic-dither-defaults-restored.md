---
type: progress
phase: 8
status: done
date: 2026-07-12
---

Classic dither defaults restored (user decision 2026-07-12), resolving the 71
golden-fixture failures from [[048-smart-mask-sdd-execution]]. The 0335773 quality
audit had changed 20 styles' default output; user chose per style:

- **Classic restored (12):** Artifact Modulation, Atkinson-VHS, Bayer-Ordered,
  Bit Tone, Block Tone, Diagonal, Glitch, Noise, Reaction-Diffusion, Topography,
  Uniform Modulation X, Wireframe Alt — verified byte-identical to pre-audit
  kernels on gradient + random inputs. Most were parameter-default fixes; real
  code reverts: Block Tone round-dot, Diagonal/Wireframe flat edge thresholds
  (200/s, 160/s), Reaction-Diffusion x3 iterations + seed_cutoff 9.
- **Audit look kept (5):** Checkers x3, Uniform Modulation Y, Halftone-Ordered.
- **Topography split:** classic is "Topography"; audit look is new style
  "Topography Alt" (registered in special.py, param metadata in parameters.py).
- Sine Wave Modulation + Atkinson Line Modulation were UNCHANGED at intended
  defaults (their "old default" was a broken param=0 path); goldens re-baked.
  Displace Contour's true old default was blank white → restored old algorithm
  at intended defaults (50,1,0,1,0).

**Test contract changed:** `tests/golden_harness.default_param` now probes REAL
app defaults via `parameters.parameter_specs` (was: 4 into every slider). All
goldens re-baked to guard the chosen default looks; 78 golden tests green.
Audit-era tests rewritten where they contradicted the user's choice
(test_glitch_special_audit.py companion baselines, Block Tone endpoint/
progression gates, RD literal-iterations test).

**Accepted regressions (flag if user complains):** at default sliders,
Artifact Modulation ≡ Waveform Alt, Modulated Diffuse X ≡ Uniform Modulation X,
and Bayer-Ordered ≡ Bayer-Matrix 4x4 ≡ Bit Tone; Diagonal/Wireframe render
smooth ramps as blank white. All diverge once sliders move.

**Gotcha:** `test_mask_editor_lifecycle.py::test_blocking_inference_pool_...`
is flaky under full-suite CPU load only; passes in isolation. Full suite
2026-07-12: 1642 passed / 310 asset-gated skips / that 1 flake. Committed on
`feat/smart-subject-masking` as `ab31e76`.
