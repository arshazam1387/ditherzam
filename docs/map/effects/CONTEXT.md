# Change-impact routing

| If changing | Hits | Does not hit |
|---|---|---|
| Render order/settings/cache | pipeline, request snapshots, previews, media, layer Looks | layer geometry/brush stamping |
| Dither style/control | registry, kernel, controls, presets, signature | palette YAML loading |
| Palette/extraction | palette system, ColorEngine, editor/picker, render context | kernel math |
| Layer/mask/transform model | document, history, controller, cache, compositor, serialization | single-image stage order |
| Blend/alpha/layer preview | compositor, native fallback, cache, exact export | palette extraction |
| Interactive lifecycle | request, scheduler, worker retention, diagnostics | completed render pixel math |
| PNG/JPEG/SVG | exact producer, serializer, UI actions, alpha tests | proxy cap |
| Startup/packaging/model | app shell, diagnostics, offline security, memory | headless algorithms |
| Status/release/native promotion | memory and branch lineage | runtime pixels |

## High-risk invariants

- Render order is frozen at `ditherzam/render.py:135`.
- Core stays Qt-free per `docs/memory/002-qt-free-core.md`.
- Clean-room/offline constraints are in memory 001.
- Native code is live on `main`; preserve byte-exact fallbacks, the two-thread
  default, and the Round-only brush boundary (memory 093 and 101).
- Multi-layer File export uses the full compositor (memory 082).

## Human check

Verify the selected row's obvious non-hit as well as its hits before editing.
