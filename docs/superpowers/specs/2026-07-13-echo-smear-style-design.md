# Echo Smear dither style — design

**Date:** 2026-07-13
**Status:** approved by user (conversation, 2026-07-13)
**Reference:** phone capture of a TouchDesigner 2025 feedback/glitch patch
("Will definitely try to modulate everything at least once.mp4") — a
transmission tower rendered red-on-black whose silhouette dissolves into
wavy onion-skin echo outlines, thin full-height streaks, and drifting
speckle dust, cycling between solid and dissolved.

## Goal

One new binary dither style, **Echo Smear**, that recreates the reference's
smear/echo mechanic on still images. All three visual ingredients (contour
echo trails, vertical pixel-stretch streaks, speckle dissolution) are
slider-controlled. A single **Breath** parameter sweeps the whole look
solid ↔ dissolved so animating one slider reproduces the reference's
breathing; a **Wave Phase** slider scrubs the wave position manually on
stills. The red-on-black color is out of scope — it is an ordinary
2-color palette applied downstream.

## Placement

- Kernel `echo_smear` in `ditherzam/dithering/kernels/special.py`,
  registered `("Echo Smear", "Special Effects", dims=2)` like Topography.
- Binary (0/255) output: the existing `_binary_to_levels` promoter provides
  depth-ramp/palette support; no `supports_levels` work.
- Universal creative controls apply unchanged; `creative_orientation`
  quarter-turns give echo direction (default drift: +x, rightward).
- No changes to any existing kernel, pipeline stage, or shared code other
  than adding an `_unpack7` helper beside `_unpack5`/`_unpack6`.

## Native sliders (8)

| slider key | Label | min | max | default | Effect |
|---|---|---|---|---|---|
| `echo_count_slider` | Echo Count | 0 | 16 | 6 | number of silhouette echo outlines |
| `echo_spacing_slider` | Echo Spacing | 2 | 40 | 10 | pixels between successive echoes |
| `echo_wave_amount_slider` | Wave Amount | 0 | 32 | 8 | vertical wiggle amplitude of each echo |
| `echo_wave_phase_slider` | Wave Phase | 0 | 360 | 0 | travels the echo lines along the smear axis (0→360 marches each line one spacing inward, seamless loop) and scrolls the wiggle |
| `echo_streak_slider` | Streak Amount | 0 | 100 | 20 | density of 1-px drip streaks falling from the subject's lowest edge |
| `echo_dissolve_slider` | Dissolve Amount | 0 | 100 | 30 | how much of the subject body erodes into speckle dust |
| `echo_breath_slider` | Breath | 0 | 100 | 50 | master cycle: 0 = fully solid subject, 100 = fully dissolved |
| `echo_wave_frequency_slider` | Wave Frequency | 1 | 100 | 10 | wave frequency of the echo wiggle (value/100 rad per px; 10 = classic 0.1) |

Rows go in `parameters.py` `parameter_specs` exactly like other Special
Effects styles so the golden harness probes real defaults.

## Kernel algorithm (approach A — single fused gather)

`@njit(cache=True, parallel=True)`, float32 grayscale in/out, 0..255.
`luminance_threshold_value` is the subject tone gate (darkness above it =
subject). Two passes inside one kernel call:

1. **Column pass** (parallel over columns): hash-selected columns record
   their lowest subject row in `drip_from[w]`, driving drip streaks.
2. **Pixel pass** (parallel over rows). Let `b = Breath/100`. Pixel is ink
   (0.0) if ANY of:
   - **Body** — pixel is subject AND survives dissolve: deterministic
     hash-noise test with gate `∝ Dissolve × b`, biased so erosion is
     densest near the silhouette edge (edge proximity via the
     neighbor-band test at small radius).
   - **Echo n = 1..Echo Count** — sample source at
     `sx = x − (n − phase/360)·spacing − sin(y·f + phase_rad + e·Δ)·wave`
     (`e = n − phase/360` is the continuous echo index/line identity,
     `f` = Wave Frequency/100 rad/px, default 0.1, `Δ` a fixed per-echo
     phase stagger ≈0.7 rad so echoes don't align); the loop runs one
     extra line (`n = 1..count+1`) so the cycle wraps seamlessly, and any
     line whose travel distance `d = e·spacing` drops below 1px is
     skipped as arrived at the subject; ink where the sample is the
     subject's trailing (right) edge (`img[y,sx] < thr` and
     `img[y,sx+1] >= thr`) — echoes are wavy vertical strokes hugging the
     silhouette, on the smear side only; Breath sets the visible
     echo reach (`visible = b * count`) with only the outermost echo
     fading in stochastically (echoes vanish at Breath 0).
   - **Streak** — selected columns (hash-gated, probability
     `streak/100 * 0.12`) drip straight down from the column's lowest
     subject pixel to the frame bottom; never above or through the
     subject; independent of Breath.
   - **Dust** — hash-noise speckle with probability decaying with distance
     past the subject edge, scaled by `Dissolve × b`.
   Else background (255.0).

Hash noise uses the same integer-hash recipe as pipeline jitter
(`x·374761393 + y·668265263 ⊕ shifts`) with no RNG state, so output is
fully deterministic per pixel and byte-identical under JIT on/off. (The
universal `creative_seed` still perturbs input via pipeline jitter as with
every style; the kernel itself takes no seed.)

Endpoint semantics:
- Breath 0 → pure thresholded body (no echoes, no dust; streaks still
  honor Streak Amount).
- Breath 100 → no solid body; only echoes, dust, streaks.
- Echo Count 0 / Wave Amount 0 / Streak 0 / Dissolve 0 each cleanly
  remove their ingredient.

## Animation

No new animation code. Breath and Wave Phase are ordinary kernel sliders;
the existing animation system (temporal patterns / parameter animation)
drives them like any other style parameter. Animating Breath 0→100→0
reproduces the reference's breathing; animating Wave Phase drifts the
wiggle.

## Testing (TDD per task, `NUMBA_DISABLE_JIT=1`)

- Golden fixture at real app defaults via the existing
  `golden_harness.default_param` probe.
- JIT-on/off byte parity.
- Determinism: identical input → identical bytes across calls.
- Endpoint gates: Breath 0 equals plain threshold body (modulo streaks at
  default), Breath 100 contains no solid body run wider than echo lines.
- Audit gates: no-collapse (output not constant on gradient/photo
  fixtures) and not byte-duplicate of Topography/Topography Alt/Displace
  Contour at defaults.
- Slider monotonicity smoke: raising Echo Count/Streak/Dissolve increases
  their ingredient's ink share on a fixed fixture.

## Risks / non-goals

- No feedback/temporal state (approach B rejected — gather reproduces the
  static look; nothing in scope needs cross-frame state).
- Smart-mask integration is out of scope; the tone threshold is the
  subject gate. Masking can composite later like any style.
- Regression surface: additive only (new kernel + `_unpack7` + param
  rows + tests).
