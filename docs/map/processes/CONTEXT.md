# Processes — implemented movements

## Interactive render

Source + UI edit → immutable `RenderRequest` (`ditherzam/ui/render_request.py:63`) →
latest-wins `RenderScheduler` (`ditherzam/ui/render_scheduler.py:26`) → context-snapshot
worker (`ditherzam/ui/main_window.py:83`) → settled QImage preview or logged failure.

## Compose layer document

Canvas + ordered layers → resolve geometry → render Look and post-Look mask →
blend RGBA → cached preview proxy or exact document output
(`ditherzam/layers/render.py:414`, `:608`).

## Export output

Exact pipeline/compositor pixels → UI export action → PNG/JPEG alpha semantics or
SVG run scan → delivery file (`ditherzam/export/raster.py:37`,
`ditherzam/export/vector.py:9`). Capped
preview pixels are never export input.

## Record project memory

Durable fact → scan `docs/memory/INDEX.md` → update or add one atomic entry → refresh
one index line → commit only when requested.

## Human check

Follow one movement into current source and verify it is wired, not merely planned.
