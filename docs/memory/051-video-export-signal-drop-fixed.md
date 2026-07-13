---
type: gotcha
phase: 7
status: done
date: 2026-07-12
---

Video export "does nothing / fast progress / occasional crash" ROOT-CAUSED+FIXED
(resolves the 050 video-debug handoff, entry deleted per its self-destruct rule). **Root cause:** in
`ui/video_controller.py` all three QRunnable workers (import/dither/assemble) were
started with `pool.start(worker)` and only referenced by local variables. QThreadPool
owns just the C++ runnable and deletes it (autoDelete) the instant `run()` returns;
that released the last Python ref, destroying the worker's `WorkerSignals` QObject.
Because every handler was a lambda/local closure, the queued `finished`/`error`
emission's delivery context was that dying signals object → Qt purged the pending
event → the chain silently stalled after the dither phase (progress emits arrive
fine mid-run — hence "runs through frames then nothing, no error, no output").
Cross-thread destruction racing event delivery explains the intermittent crashes.
NOT a regression from 4.1/Wave 4 — latent since Phase 7; unit tests masked it by
monkeypatching `pool.start` with `started.append` (which keeps the worker alive).

**Fix:** `VideoController._start_worker()` holds each started worker in
`self._active_workers` and releases it when `finished`/`error` is delivered.
Regression test `test_dither_finished_survives_worker_gc` uses the REAL pool with
no test-held refs (red before fix). Verified end-to-end: offscreen app-path export
produces a playing MP4 with audio.

**Rule:** a QRunnable's signals must outlive `run()` — either keep the wrapper
referenced until terminal-signal delivery, or connect to bound methods of a
long-lived QObject (which is why `main_window.py` render/mask/decode paths never
broke). **Latent same-pattern risk:** `timeline_panel.py` `AnimationController` is
a plain Python class whose `_launch_worker` connections are functors with the
sender as context — same hazard if worker refs are ever dropped; audit if animation
preview ever "stalls silently".
