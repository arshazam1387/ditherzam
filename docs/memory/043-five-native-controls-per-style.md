---
type: progress
phase: 8
status: done
date: 2026-07-11
---

All 73 active dither styles now expose at least five genuinely native controls in
addition to the shared creative controls. Every native slider is consumed by its
kernel, semantic metadata is explicit (no generic fallback), and declared
defaults preserve historical pixels. The UI namespaces live parameter values by
style so legacy shared IDs such as `dither_parameter_slider` cannot leak between
unrelated algorithms; presets retain their flat current-style `params` format.
Parallel family verification plus integration: 189 native/UI tests and 119
preset/render/pipeline tests green JIT-off; family agents also ran real-JIT
coverage for Ordered/Patterned, Glitch/Special, Error Diffusion, Generative, and
Bayer 4x4. Related: [[042-creative-dither-controls]].

Wave/modulation styles may expose a sixth `Line Spacing` control. Wave, Sine Wave
Modulation, Waveform, Waveform Alt, Artifact Modulation, and Ordered Modulation
use a wavelength multiplier where 100 preserves historical output. Modulated
Diffuse X/Y and Contrast Aware X/Y use a separate geometric spacing control where
1 preserves output and N retains every Nth modulation row/column with real white
gaps; their original first parameter is correctly labeled `Error Divisor`, not
spacing. Partial preset params are positionally filled from style metadata so a
later spacing key cannot be misread as the first kernel argument.
