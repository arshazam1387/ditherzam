---
type: progress
phase: 8
status: done
date: 2026-07-11
---

Creative dither customization now has two metadata-driven layers: every real
style gets functional Dither Mix, Quarter Turns, Pattern Offset X/Y, Threshold
Jitter, and deterministic Variation Seed controls, followed by any native kernel
controls already declared by that style (up to four). The universal transforms
run in `dithering/pipeline.py`; all-default values take an allocation-free path
and are byte-identical to historical output. Dynamic UI values persist across
style switches and serialize through the existing preset `params` mapping, so
old preset keys remain compatible. Tone Bias was explicitly rejected as a fake
universal customization because many kernels ignore luminance threshold. Focused
verification: 98 cross-style/UI tests, 60 integration tests, and 3 real-JIT
creative-path tests green.

Related: [[018-nlevel-threshold-is-tone-bias]], [[038-generative-dither-styles-shipped]].
