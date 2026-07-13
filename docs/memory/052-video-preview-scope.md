---
type: decision
phase: 7
status: n/a
date: 2026-07-13
---

Video has NO live editing preview — never built, in any commit (verified across the
full `video_controller.py` history 2026-07-13). After video import the viewport
shows one RAW original frame; moving dither/effect/palette controls renders nothing
until export. The only dithered video preview is the post-export playback loop
(`FramePlayer` over `dithered_frames/`), which fires from the dither worker's
finished signal (so it vanished with the [[051-video-export-signal-drop-fixed]] bug
and returned with the fix). The live animated preview the user remembers is the
ANIMATION timeline (play + amplitude on a still), not video.

**Why:** a whole debug session was spent re-deriving this; user reports of "video
preview broken" likely mean the animation preview or post-export playback.
**Apply:** if live per-frame video editing preview is wanted, it is a NEW feature —
candidate design: treat the detected preview frame as the editing base image so the
existing still capped-preview path drives it.
