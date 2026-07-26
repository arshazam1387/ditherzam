# Changelog

## Unreleased

## 0.3.0-alpha — 2026-07-26

### Added

- Editable post-Look raster masks for every spatial layer, including Reveal All,
  Hide All, transparency and Smart-derived creation, import/export, enable,
  density, invert, fill, replacement confirmation, and exact compositing.
- Bounded layer-document undo/redo history with stale-publication protection.
- Deterministic mask brush painting with Reveal/Hide, size, hardness, strength,
  document-coordinate mapping, dirty-region previews, and exact release renders.
- Normal, red-overlay, and mask-only inspection modes that never affect export.
- Smart-mask refinement with threshold, grow/shrink, feather, invert, cancellable
  previews, and exact source-resolution confirmation.
- Source-luminance, linear/radial gradient, Bayer/line/seeded-noise, and
  endpoint-safe dithered raster-mask generators.
- Exact Replace, Add, Subtract, and Intersect candidate combinations with capped
  preview, exact confirmation, undo, cancellation, and stale-result protection.
- Temporary rectangle/ellipse selections with soft document-coordinate coverage,
  From Selection, and selection-restricted raster-mask editing.

### Improved

- Raster-mask creation now uses one source-bound typed candidate boundary across
  Smart, imported, luminance, gradient, and pattern sources.
- Hilbert/Riemersma feedback remains numerically stable across advertised control
  ranges, and Triangular rendering is exact across JIT modes.
- Previously duplicated Stippling, Artifact Modulation, Uniform Modulation X,
  and Bit Tone defaults now produce distinct, deterministic styles.

## 0.2.0 — 2026-07-24

### Added

- Same-canvas spatial layers with independent pixels, Looks, masks, opacity,
  blend modes, ordering, placement, and exact flattened export.
- Photoshop-style active-layer editing and transactional canvas Transform mode.
- Eight accessible transform handles, directional cursors, keyboard nudging,
  Confirm/Cancel, Center, Fit, and aspect locking.
- Look Composer with alpha-aware transitions, asynchronous preview, playback,
  and frame export.
- Explicit active-layer Smart Mask scope and document-composite mask inspection.
- Interactive application, developer, and learning flowcharts.

### Improved

- Transform movement and resizing now use a synchronous cached visual proxy and
  perform one exact render on release.
- Layers dock density, canvas space, mask language, model-unavailable guidance,
  preview scheduling, stale-result rejection, and thumbnail behavior.
- Line Spacing behavior across diffusion and wave-family styles.

### Compatibility

- Core modules remain Qt-free.
- Exact exports remain independent from inspection overlays and interactive
  transform proxies.
- Smart Mask inference still requires an approved local model bundle; no model
  is downloaded or accessed through a cloud service.
