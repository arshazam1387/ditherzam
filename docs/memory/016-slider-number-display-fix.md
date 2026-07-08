---
type: gotcha
phase: 5
status: done
date: 2026-07-07
---

**Slider number displays were never wired to their sliders** (bug present since
Phase 5; found by the user running the GUI after the optimization pass, same
lesson as [[014-live-app-bugs-fixed]]). Each adjustment/scale/saturation slider in
`ui/controls.py` has an `InvisibleSpinBox` beside it that shows
`round(value/100 * max_display)`, but nothing connected `slider.valueChanged` to
`spin.setValue`, so the number sat frozen at its initial value while the slider
moved.

**Why tests missed it:** no test asserted the *displayed text*; they only checked
`state` + the `changed` signal, both of which worked.

**Fix (pushed, HEAD c95ad00, 375 green):** connect each slider to its spin
(`slider.valueChanged.connect(spin.setValue)`; scale maps 1..20 → the 0..100 spin
via `round(v/20*100)`), and expose spins via `panel._spins` / `panel.saturation_spin`.
Regression test `test_slider_updates_its_number_display` checks `spin.text()`.

**How to apply:** display-only Qt widgets that mirror a control need an explicit
signal connection — a stored reference + initial `setValue` is not enough. Verify
by driving the real GUI, not just state assertions.
