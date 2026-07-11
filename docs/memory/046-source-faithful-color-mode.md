---
type: progress
phase: 8
status: done
date: 2026-07-11
---

`Source Colors` mode fixes From Image palettes placing hues in the wrong regions.
The old modes color a grayscale dither by tone, so extracted reds/blues could be
reassigned wherever their luminance matched. Source mode instead quantizes each
original RGB pixel to its nearest extracted palette color at the same spatial
position. `From-Image Colors` remains the simplification control (low count =
more reduced; high count = more nuance), and pressing From Image automatically
selects Source Colors. The `Colored Dither` control colors the dither marks
themselves: the locally simplified source supplies each mark's hue/chroma and the
selected dither supplies its lightness. Value 0 is exact simplified source color;
100 (the default) is fully foreground-colored dither. This replaces the rejected
workaround that merely overlaid luminance texture on a finished color image. The
control is visible only in Source Colors mode and participates in cache signatures.
The retained source is
nearest-resized to capped preview
dimensions and exact-sized for export; source identity participates in render
cache signatures. Colored-dither verification: 165 integration tests plus 4
real-JIT source-mode tests green.
