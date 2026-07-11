---
type: progress
phase: 8
status: done
date: 2026-07-10
---

High-resolution performance **Wave 4 is COMPLETE** on `feat/high-res-perf`
(HEAD `b5649b6`), finishing the whole program (Waves 1–4). Tasks:
- **4.1** capped animation/video previews ([[036-capped-media-previews-shipped]],
  `f5406f1` + review-fix `95ae30d`) — subagent-implemented + task-reviewed.
- **4.3** thread-scaling benchmark (`614bbcb`): `benchmarks/thread_scaling.py`
  subprocess matrix over 1/2/4/8 threads + `THREAD_SCALING_2026-07-10.md`. Findings:
  ordered scales best (2.57×@8t), nearest/ramp/preview plateau ~1.4× by 2–4t,
  diffusion flat (sequential), effects REGRESS past 2t (GIL/PIL-bound), heartbeat
  ~22ms healthy.
- **4.5** JIT warmup (`3b1d9c7`): `warmup.py` now also warms ordered/ramp/diffused
  color kernels + saturation (were cold; style loop only hit nearest).
- **4.4** bounded threading policy (`da5a3b6`): Qt-free `ditherzam/threading_policy.py`
  — interactive budget `min(4,cpu)` snapped to {1,2,4,8} installed as process
  default at startup; async video export drops to `cpu−interactive` via
  best-effort `numba_threads()` ctx mgr. NOTE `set_num_threads` is PROCESS-GLOBAL
  (lowers any concurrent render, not a hard reservation). Diffusion never
  parallelized.
- **4.2** dedicated exact export contexts (`62ef5ca`): `ImageEditor._export_pipeline()`
  snapshots color+effects into a fresh pipeline; still/batch/video/animation route
  through it (controllers gained optional `export_pipeline_provider`, fallback=live).
  Fixes real TOCTOU: async video export shared the live pipeline whose
  color_engine/effect_stack are reassigned by `_sync_pipeline` on every edit. SVG
  unchanged.
- **4.6** acceptance results (`0bb9bed`): `benchmarks/HIGH_RES_ACCEPTANCE_2026-07-10.md`.

Final opus whole-branch review (b2e1dc9..0bb9bed): **✅ ready to merge, 0
Critical/Important**; 2 minor doc/CI polish applied in `b5649b6`, 2 minors left
documented (unreachable nested-budget leak; preview-only anim live-pipeline read).
Acceptance: diffusion ~302× (≥5×), retained cache 142.4 MiB (≤192), capped previews
source-independent, exactness 172 green under real JIT, **full suite 915/0 JIT-off**.
7 `special.py` JIT-on failures remain pre-existing ([[020-jit-on-kernel-failures-preexisting]]).
**Remaining before merge = interactive 4K GUI QA (human sign-off) + finish the
branch.** Related: [[031-high-res-wave2-done-wave3-in-progress]],
[[026-high-res-performance-program-approved]].
