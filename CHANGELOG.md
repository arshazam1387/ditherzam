# Changelog

## Unreleased

## 0.3.0 — 2026-08-18

### Added

- Exact Cython acceleration for five-mode layer compositing, alpha-aware
  transitions, transformed selection-to-source mapping, and Round mask-brush
  stamping, with tested Python fallbacks and a bounded native thread policy.
- Reproducible native benchmarks, pinned Windows build inputs, standard
  PyInstaller packaging, and frozen native smoke coverage.
- Square, Diamond, and deterministic Texture mask brushes, adjustable spacing,
  polygon/freehand selections, non-destructive selection refinement, and a
  reversible alpha-aware Color Range workflow.
- Persistent bounded program diagnostics and semantic action logs for startup,
  rendering, layers, masks, palettes, media, and export operations.
- A source-backed subsystem/change-impact map and a complete masking user guide.

### Improved

- Reorganized the masking workspace into compact Layer and Mask tabs with
  explicit Pointer/Paint/Select modes and immediate cursor changes.
- Moved large selection refinement and Color Range work onto bounded latest-wins
  background execution while preserving exact authoritative coverage.
- Hardened layer-preview publication, render-worker teardown, and native/Qt
  lifecycle behavior under accumulated suite load.
- Expanded distinct-color extraction with OKLab diversity and honest percentage
  reporting, raised the Dither Scale ceiling to 50, and made zoomed Transform
  handles fully clickable.

### Compatibility

- Native acceleration is byte-exact with the reference implementation and does
  not change dither kernels, render order, palettes, effects, or exported pixels.
- Creative Square, Diamond, and Texture brush tips deliberately bypass the
  Round-only native brush seam.
- GitHub `main` contains the tested native integration.

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
