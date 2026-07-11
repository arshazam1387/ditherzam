---
type: gotcha
phase: 8
status: done
date: 2026-07-09
---

A true one-pass Numba fusion of contrast → midtones → highlights is **not exact**:
at 4K it changed about 5.93 million float values (maximum finite difference
0.00012207), which can change downstream dither thresholds.

**Why:** combining stages changes float32 rounding/libm behavior even when the
formula is algebraically equivalent.

**Fix/Apply:** retain three stage passes. The measured exact candidate reuses one
private buffer across three passes, is ~1.73× faster, and cuts traced peak memory
about 67%; implement that candidate while preserving public stage observability
and `RenderPipeline.STAGE_ORDER`. Evidence/fixtures are in commit `a9a80c9`.
