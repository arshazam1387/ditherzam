---
type: progress
phase: 8
status: blocked
date: 2026-07-12
---

**HANDOFF — debug broken video rendering. SELF-DESTRUCT: the agent that resolves
this must DELETE this file and its INDEX.md line when done (record the root cause
and fix as a new `gotcha` entry instead).**

User report (2026-07-12): video is completely non-functional and has been for a
while (start date unknown — predates recent branches, bisect candidates below).
Symptoms:

- Usually it "just does nothing" — no output produced.
- Progress runs through frames at ~30 fps *implausibly fast*, i.e. it is almost
  certainly NOT actually dithering frames, just iterating and discarding.
- Often it crashes; lighter inputs (smaller/shorter clip) sometimes avoid the
  crash but still produce nothing useful.

Where to look (phase 7 architecture, see [[012-phase7-video-done]]):
- `ditherzam/video/` — ffmpeg command builders, limits, runner, assemble.
- `ditherzam/video/workers.py` (Qt worker threads) and
  `ditherzam/ui/video_controller.py` — wiring between UI and pipeline.
- Task 4.1 capped async media previews ([[036-capped-media-previews-shipped]])
  changed the video/animation preview path — a prime suspect window, as is the
  high-res Wave 4 threading/export-context work ([[037-high-res-wave4-complete]]).
- "Does nothing + eats frames fast" pattern smells like the render worker
  swallowing per-frame exceptions (compare the earlier wedge bug
  [[022-render-worker-wedge-fix]]) or ffmpeg never being invoked / its stderr
  being discarded.

How to work:
1. Invoke `zam-memory` recall + read this entry; use systematic-debugging.
2. Reproduce first: run the app (`.venv` at repo root, Python 3.12), load any
   short mp4, run a video export; capture worker/ffmpeg stderr and exceptions.
   Check whether ffmpeg is on PATH at all and whether the runner surfaces a
   missing-binary error to the UI (fail-silent is suspected).
3. Bisect candidates: video worked at phase 7 completion (298 green). Suspect
   ranges: optimization pass (015), task 4.1 (036), Wave 4 (037).
4. Tests: `NUMBA_DISABLE_JIT=1 pytest tests -k video` JIT-off; add a red test
   for the found root cause before fixing (TDD per [[003-python-and-tests]]).
5. When fixed: verify a real end-to-end export produces a playing video, record
   a `gotcha` entry with root cause + fix, then delete this file and its INDEX
   line.
