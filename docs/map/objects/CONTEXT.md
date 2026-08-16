# Objects — durable subsystem catalog

| Object | Owns | Source | Tests |
|---|---|---|---|
| Render pipeline | fixed stage order, settings, cache, cancellation | `ditherzam/render.py:99`, `:112`, `:129`, `:135` | `test_render_order.py`, `test_render_cache.py` |
| Dither registry | style metadata and kernel lookup | `ditherzam/dithering/registry.py:6`, `:17` | `test_registry.py`, `test_pipeline.py` |
| Palette system | values, extraction, builtin/user storage | `ditherzam/color/palette.py:10`, `ditherzam/color/palette_store.py:16` | `test_palette.py`, `test_palette_store.py` |
| Layer document | canvas, sources, masks, transforms, stack | `ditherzam/layers/model.py:26`, `:38`, `:80`, `:187`, `:212`, `:259`, `:321` | `test_raster_layer_mask_model.py`, `test_layers_phase_a_integration.py` |
| Layer compositor | Looks/masks, blend, proxy, exact output | `ditherzam/layers/render.py:18`, `:414`, `:608`, `:630` | `test_layers_render.py` |
| Render request | immutable UI-to-worker snapshot | `ditherzam/ui/render_request.py:21`, `:43`, `:63` | `test_render_scheduler.py` |
| Render scheduler | latest-wins lifecycle and cancellation | `ditherzam/ui/render_scheduler.py:26`, `ditherzam/ui/main_window.py:77` | `test_render_scheduler.py`, `test_render_resilience.py` |
| Application shell | Qt startup, editor, local model, warmup | `ditherzam/app.py:9`, `ditherzam/ui/main_window.py:213` | `test_offline_security.py` |
| Export surfaces | PNG/JPEG semantics and SVG runs | `ditherzam/export/raster.py:11`, `:37`, `ditherzam/export/vector.py:9` | export-action tests |
| Diagnostics | bounded logs and semantic actions | `ditherzam/diagnostics.py:35`, `:104` | `test_diagnostics.py`, `test_semantic_action_logging.py` |
| Project memory | durable status and constraints | `docs/memory/INDEX.md`, `.codex/skills/zam-memory/SKILL.md` | relevant atomic entry |

Verified against the working tree on 2026-08-14. Re-verify the selected row before editing.

## Load boundaries

- Rendering: pipeline plus only the color/dither/effect object involved.
- Editing: layer document and compositor; add UI only for interaction changes.
- UI lifecycle: request and scheduler before `ImageEditor`.
- Delivery: exact producer plus one serializer.
- Release state: memory, never code alone.
