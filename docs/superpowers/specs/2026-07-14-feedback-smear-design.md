# Feedback Smear dither style — design

**Date:** 2026-07-14
**Status:** approved direction (user decisions 2026-07-14); constants subject to the A/B acceptance loop
**Reference:** TouchDesigner feedback-loop glitch ("Will definitely try to modulate everything at least once.mp4"); extracted frames in the session scratchpad `refvid/`.

## Why a rebuild

Echo Smear (shipped on this branch) draws parametric copies of the silhouette edge along a sine — clean, periodic, line-art. The reference is a **feedback loop**: each frame is the previous frame displaced by animated noise, decayed, and re-stamped with the subject. Its trails are accumulated history — irregularly spaced, merged and thick near the subject, dissolving into dithered speckle with distance, wobbled organically. The generative process differs, so no tuning of Echo Smear can converge on the reference. User verdict on Echo Smear vs reference: not graphically similar; requirement is "looks exactly like it."

## User decisions (2026-07-14)

1. **Keep both styles.** Echo Smear stays untouched; the rebuild ships as a NEW style, **"Feedback Smear"**, Special Effects category.
2. **Still + scrub.** The kernel renders one frozen feedback state; a **Time** slider scrubs the simulation. Animating Time through the existing animation system gives the evolving smear (scrub is non-looping — the noise field slides, like the reference).
3. **Quality first.** Trail Length (iterations K) up to 64; previews rely on the existing capped-preview system; slower 4K exports accepted.
4. **Acceptance = side-by-side.** Render the binarized pylon frames from the reference video through the style; iterate constants until the user cannot tell the vibe apart in A/B. The user's eye is the ship gate.

## Model — simulated feedback by per-pixel backward walk

For output pixel p, walk backwards through K iterations of the feedback displacement: each step subtracts a base drift along the smear axis plus a 2D value-noise wobble (noise keyed by iteration k and slid by Time). Track survival `decay^k` (dithered per pixel by hash). The pixel is trail-ink at step k when the walk **crosses into** the subject (`inside(pos_k) and not inside(pos_{k-1})`) and the survival dither passes — that crossing is exactly "the k-times-displaced subject copy peeks out from behind the fresher copies", which is what feedback shows. Near region: many k's overlap → dense/merged trails. Far region: low survival → broken speckle. Both match the reference's density falloff.

The subject body erodes with a noise-field-modulated dithered gate (organic patches, not uniform static), scaled by an Erode slider — the reference's leading-side dust and eaten-away areas.

Universal creative controls apply; `creative_orientation` sets smear direction. Binary output; red-on-black via ordinary palettes; `_binary_to_levels` provides depth support.

## Native sliders (8)

| key | Label | min | max | default | effect |
|---|---|---|---|---|---|
| `fs_length_slider` | Trail Length | 4 | 64 | 32 | feedback iterations K |
| `fs_drift_slider` | Drift | 1 | 8 | 2 | px per iteration along the smear axis |
| `fs_noise_amount_slider` | Noise Amount | 0 | 24 | 6 | wobble amplitude px per iteration |
| `fs_noise_scale_slider` | Noise Scale | 1 | 100 | 20 | noise feature size (higher = finer) |
| `fs_decay_slider` | Decay | 50 | 100 | 88 | survival % per iteration (trail persistence) |
| `fs_time_slider` | Time | 0 | 360 | 0 | scrubs the simulation (slides the noise field) |
| `fs_erode_slider` | Erode | 0 | 100 | 30 | subject erosion / dust amount |
| `fs_density_slider` | Density | 0 | 200 | 100 | trail ink gain (survival dither gain) |

## Kernel sketch (authoritative structure; constants are A/B-tunable)

- `_vnoise(x, y, k, salt)`: hash-lattice 2D value noise with smoothstep bilinear interpolation, iteration index folded into the lattice with a large stride; built on the existing `_hash01`.
- `_feedback_smear(img, thr, K, drift, namount, nscale, decay, time, erode, density)`: `@njit(cache=True, parallel=True)`, float32 in/out, 0/255 binary.
  - Body: subject pixel survives unless `hash01 < erode/100 * (0.35 + 0.65 * vnoise(...))`.
  - Walk: `px -= drift + (nx-0.5)*2*namount`, `py -= (ny-0.5)*2*namount*0.6` per k with per-k noise; `surv *= decay/100`; early-out when `surv*density/100 < 0.02` or off-frame; ink on inside-crossing when `hash01(x,y,202+k) < surv * density/100`.
  - Time slides noise sample coordinates; k also staggers them (`k*0.618` lattice offset) so every iteration wobbles differently.
- Determinism: pure hash noise, no RNG state; byte-identical JIT on/off.

## Acceptance loop (the ship gate)

Harness renders the binarized reference pylon frame through Feedback Smear at defaults and at a small parameter sweep; results are shown side-by-side with the actual reference frames. The user judges; constants (noise octaves/scale mapping, survival floor, erosion band, drift/wobble ratio) iterate until pass. Only after the user's yes: golden bake is final, branch closes.

## Non-goals

- No cross-frame state (still + scrub decision); no changes to Echo Smear; no UI work beyond registration + param rows; clean-room as always.
