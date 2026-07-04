# Phase 7 — Video — Implementation Plan

> REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`. Read
> [`00-ROADMAP.md`](00-ROADMAP.md); complete Phases 1–5.

**Goal:** Import a video, probe fps/duration, extract frames, dither every frame
through the render pipeline, reassemble to MP4 (preserving original audio), and
support live playback/preview in the UI.

**Architecture:** `video/ffmpeg.py` is a pure wrapper around bundled ffmpeg/ffprobe
that builds and runs command lists (no Qt). `video/workers.py` wraps those calls
in `QRunnable` workers with `Signal`s for progress/finish/error. FFmpeg command
*construction* is unit-tested without invoking ffmpeg (inject a fake runner).

**Tech Stack:** subprocess · Pillow/NumPy · PySide6 (workers only) · pytest.

## Global Constraints
See roadmap. **Do not commit ffmpeg binaries.** Locate them via
`shutil.which("ffmpeg")` or a configured `assets/ffmpeg/` dir; if absent, video
features degrade gracefully with a clear error. On Windows, subprocesses use
`CREATE_NO_WINDOW`. Temp frames under `tempfile.gettempdir()/ditherzam/<uid>/`.

---

## File structure (this phase)

- Create `ditherzam/video/__init__.py`, `ffmpeg.py`, `workers.py`
- Tests: `tests/test_ffmpeg_cmds.py`, `tests/test_video_limits.py`, `tests/test_video_frame_dither.py`

---

### Task 1: FFmpeg command construction (pure, tested)

**Files:** Create `ditherzam/video/ffmpeg.py`; Test `tests/test_ffmpeg_cmds.py`
**Interfaces:**
```python
def ffmpeg_bin() -> str; def ffprobe_bin() -> str
def cmd_probe_fps(path) -> list[str]
def cmd_probe_duration(path) -> list[str]
def cmd_probe_has_audio(path) -> list[str]
def cmd_extract_frames(video, frames_dir) -> list[str]     # frame%06d.png, -qscale:v 2
def cmd_encode(frames_dir, fps, out) -> list[str]          # libx264 yuv420p
def cmd_extract_audio(video, audio_out) -> list[str]       # -vn -acodec copy
def cmd_mux(video, audio, out) -> list[str]                # -c copy -shortest
def parse_fps(text) -> float                               # "30000/1001" -> 29.97
```

- [ ] **Step 1: Failing test**

```python
from ditherzam.video.ffmpeg import (cmd_extract_frames, cmd_encode, parse_fps, cmd_probe_fps)

def test_extract_cmd_uses_qscale_and_pattern():
    c = cmd_extract_frames("in.mp4", "/frames")
    assert "-qscale:v" in c and "2" in c
    assert any(a.endswith("frame%06d.png") for a in c)

def test_encode_cmd_libx264_yuv420p():
    c = cmd_encode("/frames", 30, "out.mp4")
    assert "libx264" in c and "yuv420p" in c and "30" in [str(x) for x in c]

def test_parse_fps_ratio():
    assert abs(parse_fps("30000/1001") - 29.97) < 0.01
    assert parse_fps("25/1") == 25.0
```

- [ ] **Step 2: Run — fail**
- [ ] **Step 3: Implement** the command builders (return `list[str]`) and `parse_fps`
  (split on `/`, divide; handle plain number). `ffmpeg_bin`/`ffprobe_bin` resolve
  from `assets/ffmpeg/` then `shutil.which`.
- [ ] **Step 4: Pass. Step 5: Commit** `feat(video): ffmpeg command builders + fps parse`

---

### Task 2: Import limits (pure, tested)

**Files:** Modify `ditherzam/video/ffmpeg.py`; Test `tests/test_video_limits.py`
**Interfaces:** `check_video_limits(fps, duration, expert: bool) -> str | None`
(returns an error message or `None` if allowed).

- [ ] **Step 1: Failing test**

```python
from ditherzam.video.ffmpeg import check_video_limits

def test_reject_high_fps():
    assert check_video_limits(75, 10, expert=False) is not None

def test_reject_long_duration():
    assert check_video_limits(30, 120, expert=False) is not None

def test_allow_within_limits():
    assert check_video_limits(30, 30, expert=False) is None

def test_expert_bypasses():
    assert check_video_limits(120, 600, expert=True) is None
```

- [ ] **Step 2–3:** implement (fps>60 or duration>60 → message unless expert).
- [ ] **Step 4: pass. Step 5: commit** `feat(video): import limit checks + expert bypass`

---

### Task 3: Per-frame dither (headless, tested)

**Files:** Modify `ditherzam/video/ffmpeg.py` (or new `frames.py`); Test `tests/test_video_frame_dither.py`
**Interfaces:**
```python
def dither_frames(in_dir, out_dir, pipeline, settings, progress=lambda i,n: None,
                  is_cancelled=lambda: False) -> int   # returns frames written
```

- [ ] **Step 1: Failing test** — write 3 tiny PNG "frames" to `in_dir`, run
  `dither_frames` with a real `RenderPipeline(registry)` and
  `RenderSettings(style="Floyd-Steinberg", scale=1)`; assert 3 outputs exist, are
  same size, and progress called 3 times; cancel after 1 → only 1 written.
- [ ] **Step 2–3:** implement (sorted glob, load gray, `pipeline.render`, save; check
  `is_cancelled()` each iter; call `progress`).
- [ ] **Step 4: pass. Step 5: commit** `feat(video): headless per-frame dithering with cancel`

---

### Task 4: Assemble with audio preservation (integration, guarded)

**Files:** Modify `ditherzam/video/ffmpeg.py`; Test `tests/test_ffmpeg_integration.py`
**Interfaces:** `assemble_video(frames_dir, fps, orig_video, out) -> None`.

- [ ] **Step 1: Failing test** — guard with `pytest.mark.skipif(ffmpeg missing)`;
  generate 5 solid-color PNG frames, call `assemble_video(..., orig_video=None, ...)`,
  assert `out` exists and `cmd_probe_duration` on it returns > 0.
- [ ] **Step 2–3:** implement: encode frames → temp mp4; if `orig_video` has audio,
  extract (copy, fallback aac) and mux `-shortest`; else move temp → out.
- [ ] **Step 4: pass (or skip if no ffmpeg). Step 5: commit** `feat(video): assemble + audio mux`

---

### Task 5: Qt workers + UI wiring

**Files:** Create `ditherzam/video/workers.py`; Modify `ditherzam/ui/main_window.py`;
(smoke) `tests/test_video_workers_smoke.py`
**Interfaces:** `VideoImportWorker`, `VideoDitherWorker(cancel())`, `VideoAssembleWorker`
(each `QRunnable` with a `Signal` object: `finished/error/progress`).

- [ ] **Step 1:** importorskip PySide6; construct each worker with fakes; assert signals exist.
- [ ] **Step 2–3:** implement workers delegating to Task 1–4 functions; wire UI menu
  Video → Import/Export with progress dialogs; add a simple frame-scrubber/playback
  preview (QTimer stepping through dithered frames).
- [ ] **Step 4: pass. Step 5: commit** `feat(video): Qt workers + import/export UI + playback`

---

## Phase 7 Self-Review
- [ ] Command builders produce correct arg lists (tested without running ffmpeg).
- [ ] Limits: >60fps / >60s rejected; expert bypasses.
- [ ] Per-frame dither writes N frames, supports cancel, reports progress.
- [ ] Assemble produces a playable MP4 (integration, skipped if ffmpeg absent); audio preserved when present.
- [ ] No ffmpeg binaries committed; core command logic Qt-free.
