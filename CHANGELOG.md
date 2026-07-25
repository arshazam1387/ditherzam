# Changelog

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
