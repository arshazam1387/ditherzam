# Phase 7 — Video — Completion Task List

Import a video, probe fps/duration, extract frames, dither every frame through the
render pipeline, reassemble to MP4 (preserving original audio), and drive it from
Qt workers + a UI controller with live playback — all with **ffmpeg resolved at
runtime, never committed**, command builders unit-tested **without invoking
ffmpeg**, and a hard **60fps / 60s import cap with an expert-mode bypass**.

## Prereqs
- **Phase 1** green (`ditherzam.imaging.to_gray_f32`, `ditherzam.dithering.registry`).
- **Phase 4** green (`ditherzam.render.RenderPipeline`, `ditherzam.render.RenderSettings`).
- **Phase 5** green for Task 7.6 UI wiring only (`ditherzam/ui/main_window.py`,
  a viewport widget exposing `set_image(np.ndarray | QPixmap)`). Tasks 7.0–7.5 are
  fully headless and do **not** need the UI.

## Hard rules honored by this phase
- **No ffmpeg binaries committed.** Resolve at runtime: `assets/ffmpeg/ffmpeg[.exe]`
  first, then `shutil.which("ffmpeg")`. If neither exists, command builders still
  return a valid arg list (bare name `"ffmpeg"`) so they stay pure and testable;
  actual execution degrades gracefully with a clear `FFmpegError`.
- **Core stays Qt-free.** `ditherzam/video/ffmpeg.py` and `ditherzam/video/frames.py`
  import **no** PySide6. Only `ditherzam/video/workers.py` and
  `ditherzam/ui/video_controller.py` may import PySide6.
- **Command construction is tested without running ffmpeg** (assert the arg list).
  Execution paths inject a `runner` callable so they can be faked in unit tests, and
  the one real-ffmpeg integration test is `skipif`-guarded.
- On Windows subprocesses use `CREATE_NO_WINDOW`. Temp frames live under
  `tempfile.gettempdir()/ditherzam/<uid>/`.
- TDD: red → green → commit after every green. No placeholders — every code step is
  the whole, real implementation.

## Environment note (Task 0 / bootstrap)
Python **3.12** exactly. If `python --version` is not 3.12 on PATH, use the pinned
interpreter created during Phase 0 bootstrap
(`.../scratchpad/dbwork/py312/python.exe`) and prefix commands with it. All test
commands below assume `NUMBA_DISABLE_JIT=1` is exported (already set in
`tests/conftest.py`, but the video tests never touch Numba so it is a no-op here).

## File structure produced by this phase
- Create `ditherzam/video/__init__.py`
- Create `ditherzam/video/ffmpeg.py`   (binary resolution, command builders, runner, probes, assemble)
- Create `ditherzam/video/frames.py`   (headless per-frame dithering)
- Create `ditherzam/video/workers.py`  (Qt QRunnable workers — **only** Qt file in core)
- Create `ditherzam/ui/video_controller.py`  (menu wiring + QTimer playback)
- Modify `ditherzam/ui/main_window.py`  (attach the Video menu)
- Tests: `tests/test_video_ffmpeg_resolve.py`, `tests/test_ffmpeg_cmds.py`,
  `tests/test_video_limits.py`, `tests/test_video_frame_dither.py`,
  `tests/test_ffmpeg_integration.py`, `tests/test_video_workers_smoke.py`

---

### Task 7.0: Video package scaffold + ffmpeg binary resolution

**Files:**
- Create: `ditherzam/video/__init__.py`
- Create: `ditherzam/video/ffmpeg.py` (first slice — resolution + errors only)
- Test: `tests/test_video_ffmpeg_resolve.py`

**Interfaces:**
- Produces:
  ```python
  class FFmpegError(RuntimeError): ...
  def _find(name: str) -> str | None        # assets/ffmpeg/<name>[.exe] then shutil.which
  def ffmpeg_bin() -> str                    # resolved path or bare "ffmpeg"
  def ffprobe_bin() -> str                   # resolved path or bare "ffprobe"
  def have_ffmpeg() -> bool                  # True only if BOTH resolve to real files
  ```

- [ ] **Step 1: Write `ditherzam/video/__init__.py`**

```python
"""ditherzam.video — clean-room ffmpeg wrappers, headless frame dithering, Qt workers."""
```

- [ ] **Step 2: Write the failing test — `tests/test_video_ffmpeg_resolve.py`**

```python
import os
from pathlib import Path
import ditherzam.video.ffmpeg as ff


def test_bins_fall_back_to_bare_names_when_absent(monkeypatch):
    # No assets/ffmpeg and nothing on PATH -> builders still get a usable token.
    monkeypatch.setattr(ff, "_ASSETS_FFMPEG", Path("does/not/exist"))
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    assert ff.ffmpeg_bin() == "ffmpeg"
    assert ff.ffprobe_bin() == "ffprobe"
    assert ff.have_ffmpeg() is False


def test_prefers_assets_dir_over_path(tmp_path, monkeypatch):
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    (tmp_path / exe).write_bytes(b"stub")
    monkeypatch.setattr(ff, "_ASSETS_FFMPEG", tmp_path)
    monkeypatch.setattr(ff.shutil, "which", lambda name: "/usr/bin/" + name)
    assert ff.ffmpeg_bin() == str(tmp_path / exe)   # assets dir wins


def test_falls_back_to_which_when_no_assets(monkeypatch):
    monkeypatch.setattr(ff, "_ASSETS_FFMPEG", Path("does/not/exist"))
    monkeypatch.setattr(ff.shutil, "which", lambda name: "/usr/bin/" + name)
    assert ff.ffprobe_bin() == "/usr/bin/ffprobe"
    assert ff.have_ffmpeg() is True


def test_ffmpeg_error_is_runtimeerror():
    assert issubclass(ff.FFmpegError, RuntimeError)
```

- [ ] **Step 3: Run — verify fail**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_ffmpeg_resolve.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ditherzam.video.ffmpeg'`)

- [ ] **Step 4: Implement `ditherzam/video/ffmpeg.py` (resolution slice)**

```python
from __future__ import annotations

import os
import shutil
from pathlib import Path

# assets/ffmpeg lives at repo root: <repo>/assets/ffmpeg/. This module is at
# <repo>/ditherzam/video/ffmpeg.py, so go up two parents to reach the package
# root, then one more to the repo root.
_ASSETS_FFMPEG = Path(__file__).resolve().parents[2] / "assets" / "ffmpeg"


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe invocation fails or a binary is unavailable."""


def _find(name: str) -> str | None:
    """Return an absolute path to `name`, preferring bundled assets/ffmpeg, then PATH.

    Returns None if the binary cannot be located. NEVER embeds a binary; it only
    resolves one that the user/installer placed on disk.
    """
    exe = name + (".exe" if os.name == "nt" else "")
    local = _ASSETS_FFMPEG / exe
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    return found


def ffmpeg_bin() -> str:
    """Resolved ffmpeg path, or the bare name `"ffmpeg"` for pure command building."""
    return _find("ffmpeg") or "ffmpeg"


def ffprobe_bin() -> str:
    """Resolved ffprobe path, or the bare name `"ffprobe"` for pure command building."""
    return _find("ffprobe") or "ffprobe"


def have_ffmpeg() -> bool:
    """True only when BOTH ffmpeg and ffprobe resolve to real files on disk."""
    return _find("ffmpeg") is not None and _find("ffprobe") is not None
```

- [ ] **Step 5: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_ffmpeg_resolve.py -v`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add ditherzam/video/__init__.py ditherzam/video/ffmpeg.py tests/test_video_ffmpeg_resolve.py
git commit -m "feat(video): package scaffold + runtime ffmpeg binary resolution"
```

---

### Task 7.1: FFmpeg command builders + fps parsing (pure, tested without ffmpeg)

**Files:**
- Modify: `ditherzam/video/ffmpeg.py`
- Test: `tests/test_ffmpeg_cmds.py`

**Interfaces:**
```python
def cmd_probe_fps(path) -> list[str]        # ffprobe stream=r_frame_rate
def cmd_probe_duration(path) -> list[str]   # ffprobe format=duration
def cmd_probe_has_audio(path) -> list[str]  # ffprobe -select_streams a stream=codec_type
def cmd_extract_frames(video, frames_dir) -> list[str]   # -qscale:v 2, frame%06d.png
def cmd_encode(frames_dir, fps, out) -> list[str]        # -framerate fps, libx264, yuv420p
def cmd_extract_audio(video, audio_out) -> list[str]     # -vn -acodec copy
def cmd_extract_audio_reencode(video, audio_out) -> list[str]  # -vn -c:a aac -f adts
def cmd_mux(video, audio, out) -> list[str]              # -c copy -shortest
def parse_fps(text) -> float                             # "30000/1001" -> 29.97
```
These are **pure functions**: every element of the returned list is a string, the
first element is `ffmpeg_bin()`/`ffprobe_bin()` (which falls back to a bare name so
the builders never need ffmpeg installed), and no subprocess is ever spawned. The
tests below assert on the arg lists only.

- [ ] **Step 1: Write the failing test — `tests/test_ffmpeg_cmds.py`**

```python
from ditherzam.video.ffmpeg import (
    cmd_probe_fps, cmd_probe_duration, cmd_probe_has_audio,
    cmd_extract_frames, cmd_encode, cmd_extract_audio,
    cmd_extract_audio_reencode, cmd_mux, parse_fps,
)


def test_all_builders_return_list_of_str():
    for c in (
        cmd_probe_fps("in.mp4"),
        cmd_probe_duration("in.mp4"),
        cmd_probe_has_audio("in.mp4"),
        cmd_extract_frames("in.mp4", "/frames"),
        cmd_encode("/frames", 30, "out.mp4"),
        cmd_extract_audio("in.mp4", "a.m4a"),
        cmd_extract_audio_reencode("in.mp4", "a.aac"),
        cmd_mux("v.mp4", "a.m4a", "out.mp4"),
    ):
        assert isinstance(c, list) and all(isinstance(x, str) for x in c)


def test_probe_fps_cmd():
    c = cmd_probe_fps("in.mp4")
    assert "ffprobe" in c[0]
    assert "-select_streams" in c and "v:0" in c
    assert "stream=r_frame_rate" in c
    assert "default=noprint_wrappers=1:nokey=1" in c
    assert c[-1] == "in.mp4"


def test_probe_duration_cmd():
    c = cmd_probe_duration("in.mp4")
    assert "format=duration" in c
    assert c[-1] == "in.mp4"


def test_probe_has_audio_cmd():
    c = cmd_probe_has_audio("in.mp4")
    assert "-select_streams" in c and "a" in c
    assert "stream=codec_type" in c


def test_extract_cmd_uses_qscale_and_pattern():
    c = cmd_extract_frames("in.mp4", "/frames")
    assert "-qscale:v" in c and "2" in c
    assert "-i" in c and "in.mp4" in c
    assert any(a.endswith("frame%06d.png") for a in c)


def test_encode_cmd_libx264_yuv420p_framerate():
    c = cmd_encode("/frames", 30, "out.mp4")
    assert "libx264" in c and "yuv420p" in c
    assert "-framerate" in c and "30" in c
    assert any(a.endswith("frame%06d.png") for a in c)
    assert c[-1] == "out.mp4"


def test_extract_audio_copy_then_reencode():
    c1 = cmd_extract_audio("in.mp4", "a.m4a")
    assert "-vn" in c1 and "-acodec" in c1 and "copy" in c1
    c2 = cmd_extract_audio_reencode("in.mp4", "a.aac")
    assert "-vn" in c2 and "aac" in c2 and "-f" in c2 and "adts" in c2


def test_mux_cmd_copy_shortest():
    c = cmd_mux("v.mp4", "a.m4a", "out.mp4")
    assert "-c" in c and "copy" in c and "-shortest" in c
    assert "v.mp4" in c and "a.m4a" in c and c[-1] == "out.mp4"


def test_parse_fps_ratio():
    assert abs(parse_fps("30000/1001") - 29.97) < 0.01
    assert parse_fps("25/1") == 25.0
    assert parse_fps("60/1") == 60.0


def test_parse_fps_plain_number():
    assert parse_fps("24") == 24.0
    assert abs(parse_fps("23.976") - 23.976) < 1e-6


def test_parse_fps_bad_input_returns_zero():
    assert parse_fps("") == 0.0
    assert parse_fps("0/0") == 0.0
    assert parse_fps("garbage") == 0.0
```

- [ ] **Step 2: Run — verify fail**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_ffmpeg_cmds.py -v`
Expected: FAIL (`ImportError: cannot import name 'cmd_probe_fps'`)

- [ ] **Step 3: Implement — append to `ditherzam/video/ffmpeg.py`**

```python
from pathlib import PurePath


def _frame_pattern(frames_dir) -> str:
    """`<frames_dir>/frame%06d.png` with forward slashes (ffmpeg-safe on Windows)."""
    return PurePath(frames_dir).as_posix().rstrip("/") + "/frame%06d.png"


def cmd_probe_fps(path) -> list[str]:
    return [
        ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]


def cmd_probe_duration(path) -> list[str]:
    return [
        ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]


def cmd_probe_has_audio(path) -> list[str]:
    return [
        ffprobe_bin(), "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]


def cmd_extract_frames(video, frames_dir) -> list[str]:
    return [
        ffmpeg_bin(), "-y", "-i", str(video),
        "-qscale:v", "2", _frame_pattern(frames_dir),
    ]


def cmd_encode(frames_dir, fps, out) -> list[str]:
    return [
        ffmpeg_bin(), "-y", "-framerate", str(fps),
        "-i", _frame_pattern(frames_dir),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    ]


def cmd_extract_audio(video, audio_out) -> list[str]:
    return [ffmpeg_bin(), "-y", "-i", str(video), "-vn", "-acodec", "copy", str(audio_out)]


def cmd_extract_audio_reencode(video, audio_out) -> list[str]:
    return [ffmpeg_bin(), "-y", "-i", str(video), "-vn", "-c:a", "aac", "-f", "adts", str(audio_out)]


def cmd_mux(video, audio, out) -> list[str]:
    return [ffmpeg_bin(), "-y", "-i", str(video), "-i", str(audio), "-c", "copy", "-shortest", str(out)]


def parse_fps(text) -> float:
    """Parse an ffprobe r_frame_rate token (`"num/den"` or a plain number) to float.

    Returns 0.0 on empty/garbage/zero-denominator input so callers can reject it via
    the import-limit check rather than crash.
    """
    s = str(text).strip()
    if not s:
        return 0.0
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            den_f = float(den)
            if den_f == 0.0:
                return 0.0
            return float(num) / den_f
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0
```

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_ffmpeg_cmds.py -v`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/video/ffmpeg.py tests/test_ffmpeg_cmds.py
git commit -m "feat(video): ffmpeg/ffprobe command builders + fps parse (pure, ffmpeg-free tests)"
```

---

### Task 7.2: Import limits (60fps / 60s cap) + expert bypass

**Files:**
- Modify: `ditherzam/video/ffmpeg.py`
- Test: `tests/test_video_limits.py`

**Interfaces:**
```python
FPS_LIMIT = 60
DURATION_LIMIT = 60
MSG_FPS = "Sorry, videos with a framerate above 60 fps aren't supported."
MSG_DURATION = "Sorry, videos longer than 60 seconds aren't supported."
def check_video_limits(fps: float, duration: float, expert: bool = False) -> str | None
```
Returns an error message string when the clip must be rejected, or `None` when it is
allowed. `expert=True` short-circuits to `None` (bypasses both caps). The boundary
is **strictly greater than**: exactly 60fps / 60.0s is allowed; 60.01 is not.

- [ ] **Step 1: Write the failing test — `tests/test_video_limits.py`**

```python
from ditherzam.video.ffmpeg import (
    check_video_limits, MSG_FPS, MSG_DURATION, FPS_LIMIT, DURATION_LIMIT,
)


def test_reject_high_fps():
    msg = check_video_limits(75, 10, expert=False)
    assert msg == MSG_FPS


def test_reject_long_duration():
    msg = check_video_limits(30, 120, expert=False)
    assert msg == MSG_DURATION


def test_fps_checked_before_duration_when_both_bad():
    # Both over the cap -> fps message wins (checked first).
    assert check_video_limits(90, 300, expert=False) == MSG_FPS


def test_allow_within_limits():
    assert check_video_limits(30, 30, expert=False) is None


def test_boundary_exactly_at_cap_is_allowed():
    assert check_video_limits(FPS_LIMIT, DURATION_LIMIT, expert=False) is None


def test_just_over_boundary_rejected():
    assert check_video_limits(60.01, 30, expert=False) == MSG_FPS
    assert check_video_limits(30, 60.01, expert=False) == MSG_DURATION


def test_expert_bypasses_everything():
    assert check_video_limits(120, 600, expert=True) is None
    assert check_video_limits(240, 3600, expert=True) is None
```

- [ ] **Step 2: Run — verify fail**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_limits.py -v`
Expected: FAIL (`ImportError: cannot import name 'check_video_limits'`)

- [ ] **Step 3: Implement — append to `ditherzam/video/ffmpeg.py`**

```python
FPS_LIMIT = 60
DURATION_LIMIT = 60
MSG_FPS = "Sorry, videos with a framerate above 60 fps aren't supported."
MSG_DURATION = "Sorry, videos longer than 60 seconds aren't supported."


def check_video_limits(fps: float, duration: float, expert: bool = False) -> str | None:
    """Enforce the import caps. Returns an error message, or None if allowed.

    Normal mode rejects fps > 60 (checked first) or duration > 60 s. Expert mode
    bypasses both caps entirely (spec §12.2 / §12.6).
    """
    if expert:
        return None
    if fps > FPS_LIMIT:
        return MSG_FPS
    if duration > DURATION_LIMIT:
        return MSG_DURATION
    return None
```

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_limits.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/video/ffmpeg.py tests/test_video_limits.py
git commit -m "feat(video): import limit checks (60fps/60s) with expert bypass"
```

---

### Task 7.3: Subprocess runner + probe wrappers (execution, faked in tests)

**Files:**
- Modify: `ditherzam/video/ffmpeg.py`
- Test: `tests/test_video_limits.py` (extend) — no ffmpeg needed; the runner is injected

**Interfaces:**
```python
def run_command(cmd: list[str]) -> str          # runs subprocess, CREATE_NO_WINDOW on Win, raises FFmpegError
def probe_fps(path, runner=run_command) -> float
def probe_duration(path, runner=run_command) -> float
def probe_has_audio(path, runner=run_command) -> bool
```
The three `probe_*` helpers take an injectable `runner` so they are unit-tested with
a fake that returns canned ffprobe stdout — **no ffmpeg is ever spawned in the unit
tests.**

- [ ] **Step 1: Extend the failing test — append to `tests/test_video_limits.py`**

```python
from ditherzam.video.ffmpeg import probe_fps, probe_duration, probe_has_audio


def test_probe_fps_parses_runner_output():
    fake = lambda cmd: "30000/1001\n"
    assert abs(probe_fps("in.mp4", runner=fake) - 29.97) < 0.01


def test_probe_duration_parses_runner_output():
    fake = lambda cmd: "12.480000\n"
    assert abs(probe_duration("in.mp4", runner=fake) - 12.48) < 1e-6


def test_probe_duration_bad_output_is_zero():
    fake = lambda cmd: "N/A\n"
    assert probe_duration("in.mp4", runner=fake) == 0.0


def test_probe_has_audio_true_when_codec_type_present():
    assert probe_has_audio("in.mp4", runner=lambda cmd: "audio\n") is True


def test_probe_has_audio_false_when_empty():
    assert probe_has_audio("in.mp4", runner=lambda cmd: "\n") is False
```

- [ ] **Step 2: Run — verify fail**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_limits.py -v`
Expected: FAIL (`ImportError: cannot import name 'probe_fps'`)

- [ ] **Step 3: Implement — append to `ditherzam/video/ffmpeg.py`**

```python
import subprocess

# Suppress the flashing console window ffmpeg would otherwise pop on Windows.
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def run_command(cmd: list[str]) -> str:
    """Run an ffmpeg/ffprobe command, returning captured stdout (text).

    Raises FFmpegError on a nonzero exit code, with the exit code and any stderr
    tail for diagnosis. This is the only place in the module that spawns a process.
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
            text=True,
        )
    except FileNotFoundError as e:
        raise FFmpegError(f"ffmpeg/ffprobe binary not found: {cmd[0]}") from e
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise FFmpegError(
            f"FFmpeg failed with exit code {proc.returncode}: " + " | ".join(tail)
        )
    return proc.stdout


def probe_fps(path, runner=run_command) -> float:
    return parse_fps(runner(cmd_probe_fps(path)).strip())


def probe_duration(path, runner=run_command) -> float:
    text = runner(cmd_probe_duration(path)).strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe_has_audio(path, runner=run_command) -> bool:
    return "audio" in runner(cmd_probe_has_audio(path)).strip().lower()
```

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_limits.py -v`
Expected: `12 passed` (7 from Task 7.2 + 5 new)

- [ ] **Step 5: Commit**

```bash
git add ditherzam/video/ffmpeg.py tests/test_video_limits.py
git commit -m "feat(video): subprocess runner + ffprobe wrappers (injectable, ffmpeg-free tests)"
```

---

### Task 7.4: Headless per-frame dithering with cancel + progress

**Files:**
- Create: `ditherzam/video/frames.py`
- Test: `tests/test_video_frame_dither.py`

**Interfaces:**
```python
def dither_frames(in_dir, out_dir, pipeline, settings,
                  progress=lambda i, n: None,
                  is_cancelled=lambda: False) -> int   # returns frames written
def detect_preview_frame(frames_dir, min_mean=5.0) -> str | None
```
- Consumes: `RenderPipeline.render(base_gray_f32, settings) -> uint8 HxWx3` (Phase 4)
  and `to_gray_f32` (Phase 1).
- Loads frames in **sorted** filename order, converts to gray float32, renders,
  writes `frame%06d.png` to `out_dir`. Checks `is_cancelled()` at the top of each
  iteration (cancelling before frame *k* leaves exactly *k* frames written). Calls
  `progress(done, total)` after each successful frame. `detect_preview_frame`
  returns the first frame whose mean intensity exceeds `min_mean` (skips
  mostly-black frames), or `None`.

- [ ] **Step 1: Write the failing test — `tests/test_video_frame_dither.py`**

```python
import numpy as np
from PIL import Image

from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.video.frames import dither_frames, detect_preview_frame


def _write_frames(d, n, size=(8, 8), value=180):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        Image.fromarray(arr).save(d / f"frame{i:06d}.png")


def test_dithers_all_frames_same_size(tmp_path):
    src = tmp_path / "in"; out = tmp_path / "out"
    _write_frames(src, 3, size=(8, 8))
    pipe = RenderPipeline(registry)
    settings = RenderSettings(style="Floyd-Steinberg", scale=1)

    calls = []
    written = dither_frames(
        src, out, pipe, settings, progress=lambda i, n: calls.append((i, n))
    )
    assert written == 3
    outs = sorted(out.glob("frame*.png"))
    assert len(outs) == 3
    for p in outs:
        assert Image.open(p).size == (8, 8)
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_cancel_after_one_frame(tmp_path):
    src = tmp_path / "in"; out = tmp_path / "out"
    _write_frames(src, 3)
    pipe = RenderPipeline(registry)
    settings = RenderSettings(style="Floyd-Steinberg", scale=1)

    state = {"n": 0}
    def cancel():
        # Allow the first frame, cancel before the second.
        state["n"] += 1
        return state["n"] > 1

    written = dither_frames(src, out, pipe, settings, is_cancelled=cancel)
    assert written == 1
    assert len(list(out.glob("frame*.png"))) == 1


def test_detect_preview_skips_black_frames(tmp_path):
    d = tmp_path / "frames"; d.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "frame000000.png")
    Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(d / "frame000001.png")
    picked = detect_preview_frame(d)
    assert picked is not None and picked.endswith("frame000001.png")


def test_detect_preview_none_when_all_black(tmp_path):
    d = tmp_path / "frames"; d.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "frame000000.png")
    assert detect_preview_frame(d) is None
```

- [ ] **Step 2: Run — verify fail**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_frame_dither.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ditherzam.video.frames'`)

- [ ] **Step 3: Implement `ditherzam/video/frames.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ditherzam.imaging import to_gray_f32

_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _sorted_frames(frames_dir) -> list[Path]:
    d = Path(frames_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _EXTS)


def dither_frames(
    in_dir,
    out_dir,
    pipeline,
    settings,
    progress: Callable[[int, int], None] = lambda i, n: None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> int:
    """Dither every frame in `in_dir` through `pipeline`, writing to `out_dir`.

    Returns the number of frames written. Honors cooperative cancellation: when
    `is_cancelled()` is truthy at the start of an iteration, processing stops and
    the count of frames already written is returned. `progress(done, total)` is
    emitted after each successful frame.
    """
    frames = _sorted_frames(in_dir)
    total = len(frames)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, src in enumerate(frames):
        if is_cancelled():
            break
        with Image.open(src) as im:
            gray = to_gray_f32(im.convert("L"))
        rgb_u8 = pipeline.render(gray, settings)  # uint8 HxWx3
        dst = out / f"frame{idx:06d}.png"
        Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8)).save(dst)
        written += 1
        progress(written, total)
    return written


def detect_preview_frame(frames_dir, min_mean: float = 5.0) -> str | None:
    """First frame (sorted) whose mean intensity exceeds `min_mean`, else None.

    Used to skip leading mostly-black frames when choosing a preview thumbnail.
    """
    for p in _sorted_frames(frames_dir):
        with Image.open(p) as im:
            if float(np.asarray(im.convert("L"), dtype=np.float32).mean()) > min_mean:
                return str(p)
    return None
```

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_frame_dither.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add ditherzam/video/frames.py tests/test_video_frame_dither.py
git commit -m "feat(video): headless per-frame dithering with cancel + progress"
```

---

### Task 7.5: Assemble frames + audio mux (integration, ffmpeg-guarded)

**Files:**
- Modify: `ditherzam/video/ffmpeg.py`
- Test: `tests/test_ffmpeg_integration.py`

**Interfaces:**
```python
def assemble_video(frames_dir, fps, orig_video, out, runner=run_command) -> None
```
- Encodes `frames_dir/frame%06d.png` → a temp `temp_video.mp4` (libx264, yuv420p).
- If `orig_video` is truthy **and** has an audio stream: extract audio (stream copy,
  falling back to AAC/ADTS re-encode on failure), then mux with `-c copy -shortest`
  into `out`.
- Otherwise move `temp_video.mp4` → `out`.
- The one real-ffmpeg test is guarded with `skipif(not have_ffmpeg())`. A second
  fully-headless test injects a fake `runner` to assert the **command sequence**
  without ffmpeg (covers the audio branch selection logic deterministically).

- [ ] **Step 1: Write the failing test — `tests/test_ffmpeg_integration.py`**

```python
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ditherzam.video.ffmpeg import assemble_video, have_ffmpeg, probe_duration


def _write_solid_frames(d, n, color=(120, 60, 200), size=(32, 32)):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.full((size[1], size[0], 3), 0, dtype=np.uint8)
        arr[:] = color
        Image.fromarray(arr).save(d / f"frame{i:06d}.png")


# --- Headless: verify the command SEQUENCE via an injected fake runner ---

def test_assemble_no_audio_moves_temp(tmp_path, monkeypatch):
    frames = tmp_path / "frames"; _write_solid_frames(frames, 5)
    out = tmp_path / "out.mp4"

    called = []
    def fake_runner(cmd):
        called.append(cmd)
        # Simulate the encoder producing temp_video.mp4 on disk.
        if "-c:v" in cmd and "libx264" in cmd:
            Path(cmd[-1]).write_bytes(b"FAKEMP4")
        return ""

    assemble_video(frames, 30, orig_video=None, out=out, runner=fake_runner)
    assert out.is_file()
    # Exactly one command (the encode); no probe/extract/mux when orig_video is None.
    assert len(called) == 1
    assert "libx264" in called[0]


def test_assemble_with_audio_muxes(tmp_path):
    frames = tmp_path / "frames"; _write_solid_frames(frames, 3)
    out = tmp_path / "out.mp4"

    seq = []
    def fake_runner(cmd):
        seq.append(cmd)
        if "-c:v" in cmd and "libx264" in cmd:
            Path(cmd[-1]).write_bytes(b"V")
        if "codec_type" in cmd:
            return "audio\n"            # original has audio
        if "-acodec" in cmd and "copy" in cmd:
            Path(cmd[-1]).write_bytes(b"A")  # audio extracted OK
        if "-shortest" in cmd:
            Path(cmd[-1]).write_bytes(b"MUXED")  # muxed output produced
        return ""

    assemble_video(frames, 24, orig_video="orig.mp4", out=out, runner=fake_runner)
    assert out.is_file()
    kinds = [
        "encode" if "libx264" in c else
        "probe_audio" if "codec_type" in c else
        "extract_audio" if ("-acodec" in c or "-c:a" in c) else
        "mux" if "-shortest" in c else "other"
        for c in seq
    ]
    assert kinds == ["encode", "probe_audio", "extract_audio", "mux"]


# --- Real ffmpeg: guarded end-to-end ---

@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not available")
def test_assemble_produces_playable_mp4(tmp_path):
    frames = tmp_path / "frames"; _write_solid_frames(frames, 10, size=(64, 64))
    out = tmp_path / "out.mp4"
    assemble_video(frames, 10, orig_video=None, out=out)
    assert out.is_file() and out.stat().st_size > 0
    assert probe_duration(out) > 0
```

- [ ] **Step 2: Run — verify fail**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_ffmpeg_integration.py -v`
Expected: FAIL (`ImportError: cannot import name 'assemble_video'`)

- [ ] **Step 3: Implement — append to `ditherzam/video/ffmpeg.py`**

```python
import shutil as _shutil_mod  # module-level `shutil` already imported for `which`


def assemble_video(frames_dir, fps, orig_video, out, runner=run_command) -> None:
    """Encode dithered frames to MP4, preserving original audio when present.

    Steps (spec §12.5):
      1. Encode frames_dir/frame%06d.png -> temp_video.mp4 (libx264, yuv420p).
      2. If orig_video has an audio stream: extract it (stream copy, else AAC/ADTS
         re-encode) and mux with `-c copy -shortest` into `out`.
      3. Otherwise move temp_video.mp4 -> out.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent
    temp_video = work / "temp_video.mp4"

    runner(cmd_encode(frames_dir, fps, str(temp_video)))

    has_audio = bool(orig_video) and probe_has_audio(orig_video, runner=runner)
    if has_audio:
        audio = work / "audio.m4a"
        try:
            runner(cmd_extract_audio(orig_video, str(audio)))
        except FFmpegError:
            audio = work / "audio.aac"
            runner(cmd_extract_audio_reencode(orig_video, str(audio)))
        runner(cmd_mux(str(temp_video), str(audio), str(out_path)))
        if temp_video.exists():
            temp_video.unlink()
    else:
        _shutil_mod.move(str(temp_video), str(out_path))
```

> Note: `assemble_video` reuses `probe_has_audio(orig_video, runner=runner)`, so the
> injected fake runner in the unit tests fully controls the audio branch without any
> ffmpeg present. The real end-to-end path is exercised only by the guarded test.

- [ ] **Step 4: Run — verify pass (or skip if no ffmpeg)**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_ffmpeg_integration.py -v`
Expected: `2 passed, 1 skipped` on a box without ffmpeg; `3 passed` with ffmpeg on PATH.

- [ ] **Step 5: Commit**

```bash
git add ditherzam/video/ffmpeg.py tests/test_ffmpeg_integration.py
git commit -m "feat(video): assemble frames + audio mux (guarded integration)"
```

---

### Task 7.6: Qt workers + UI wiring + playback

**Files:**
- Create: `ditherzam/video/workers.py` (the **only** Qt file under `ditherzam/video/`)
- Create: `ditherzam/ui/video_controller.py`
- Modify: `ditherzam/ui/main_window.py` (attach the Video menu via the controller)
- Test: `tests/test_video_workers_smoke.py`

**Interfaces:**
```python
# ditherzam/video/workers.py
class WorkerSignals(QObject):
    finished = Signal(object); error = Signal(str); progress = Signal(int, int)
class VideoImportWorker(QRunnable):    # extract frames
    def __init__(self, video, frames_dir, runner=run_command)
class VideoDitherWorker(QRunnable):    # dither frames; cancellable
    def __init__(self, in_dir, out_dir, pipeline, settings)
    def cancel(self) -> None
class VideoAssembleWorker(QRunnable):  # encode + mux
    def __init__(self, frames_dir, fps, orig_video, out, runner=run_command)

# ditherzam/ui/video_controller.py
class VideoController:
    def __init__(self, main_window, pipeline, settings_provider, expert_provider)
    def build_menu(self) -> QMenu       # File > Video submenu (Import/Export)
    def import_video(self) -> None
    def export_video(self) -> None
    class FramePlayer: def start(); def stop(); def step()
```
Workers delegate to the pure Task 7.1–7.5 functions and communicate via `Signal`s.
The controller enforces `check_video_limits(...)` using the caller-supplied
`expert_provider()` **before** importing, and drives a `QTimer`-based playback of the
dithered frames. Smoke test only imports PySide6 via `importorskip` and asserts the
signal/method surface — no event loop, no ffmpeg.

- [ ] **Step 1: Write the failing test — `tests/test_video_workers_smoke.py`**

```python
import pytest

pytest.importorskip("PySide6")  # skip on headless boxes without Qt

from PySide6.QtWidgets import QApplication
from ditherzam.dithering import registry
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.video.workers import (
    WorkerSignals, VideoImportWorker, VideoDitherWorker, VideoAssembleWorker,
)

_app = QApplication.instance() or QApplication([])


def test_signals_have_expected_channels():
    s = WorkerSignals()
    assert hasattr(s, "finished") and hasattr(s, "error") and hasattr(s, "progress")


def test_import_worker_constructs_with_fake_runner():
    w = VideoImportWorker("in.mp4", "/frames", runner=lambda cmd: "")
    assert hasattr(w.signals, "finished")


def test_dither_worker_is_cancellable():
    pipe = RenderPipeline(registry)
    w = VideoDitherWorker("/in", "/out", pipe, RenderSettings(style="Floyd-Steinberg"))
    assert w._is_canceled is False
    w.cancel()
    assert w._is_canceled is True


def test_assemble_worker_constructs():
    w = VideoAssembleWorker("/frames", 30, None, "out.mp4", runner=lambda cmd: "")
    assert hasattr(w.signals, "progress")


def test_import_worker_run_emits_finished(tmp_path):
    frames = tmp_path / "frames"
    got = {}
    w = VideoImportWorker("in.mp4", str(frames), runner=lambda cmd: "")
    w.signals.finished.connect(lambda payload: got.setdefault("done", payload))
    w.run()
    assert got.get("done") == str(frames)


def test_import_worker_run_emits_error_on_failure():
    def boom(cmd):
        raise RuntimeError("ffmpeg exploded")
    w = VideoImportWorker("in.mp4", "/frames", runner=boom)
    errs = []
    w.signals.error.connect(lambda m: errs.append(m))
    w.run()
    assert errs and "exploded" in errs[0]
```

- [ ] **Step 2: Run — verify fail**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_workers_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'ditherzam.video.workers'`), or
`skipped` if PySide6 is not installed.

- [ ] **Step 3a: Implement `ditherzam/video/workers.py`**

```python
"""Qt QRunnable workers for the video pipeline.

This is the ONLY module under ditherzam/video/ permitted to import PySide6 (see
roadmap "Core is Qt-free"). Each worker delegates to the pure, headless functions
in ffmpeg.py / frames.py and reports via Signals.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .ffmpeg import (
    FFmpegError, cmd_extract_frames, run_command, assemble_video, probe_fps,
)
from .frames import dither_frames


class WorkerSignals(QObject):
    finished = Signal(object)   # payload varies per worker
    error = Signal(str)
    progress = Signal(int, int)  # (done, total)


class VideoImportWorker(QRunnable):
    """Extract original frames from a video into `frames_dir` (spec §12.3)."""

    def __init__(self, video, frames_dir, runner=run_command) -> None:
        super().__init__()
        self.video = video
        self.frames_dir = str(frames_dir)
        self.runner = runner
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            import os
            os.makedirs(self.frames_dir, exist_ok=True)
            self.runner(cmd_extract_frames(self.video, self.frames_dir))
            self.signals.finished.emit(self.frames_dir)
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI thread
            self.signals.error.emit(str(e))


class VideoDitherWorker(QRunnable):
    """Dither every extracted frame; cancellable mid-run (spec §12.4)."""

    def __init__(self, in_dir, out_dir, pipeline, settings) -> None:
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.pipeline = pipeline
        self.settings = settings
        self._is_canceled = False
        self.signals = WorkerSignals()

    def cancel(self) -> None:
        self._is_canceled = True

    @Slot()
    def run(self) -> None:
        try:
            written = dither_frames(
                self.in_dir, self.out_dir, self.pipeline, self.settings,
                progress=lambda done, total: self.signals.progress.emit(done, total),
                is_cancelled=lambda: self._is_canceled,
            )
            self.signals.finished.emit(written)
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(str(e))


class VideoAssembleWorker(QRunnable):
    """Encode dithered frames + mux audio into the final MP4 (spec §12.5)."""

    def __init__(self, frames_dir, fps, orig_video, out, runner=run_command) -> None:
        super().__init__()
        self.frames_dir = frames_dir
        self.fps = fps
        self.orig_video = orig_video
        self.out = out
        self.runner = runner
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            assemble_video(
                self.frames_dir, self.fps, self.orig_video, self.out,
                runner=self.runner,
            )
            self.signals.finished.emit(self.out)
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(str(e))
```

- [ ] **Step 3b: Implement `ditherzam/ui/video_controller.py`**

```python
"""UI glue for the video pipeline: menu actions, worker orchestration, playback.

Qt-only (allowed under ditherzam/ui/). All heavy lifting is delegated to the
headless functions in ditherzam.video.* and to the QRunnable workers.
"""
from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox, QProgressDialog

from ditherzam.video.ffmpeg import check_video_limits, probe_duration, probe_fps
from ditherzam.video.frames import detect_preview_frame
from ditherzam.video.workers import (
    VideoAssembleWorker, VideoDitherWorker, VideoImportWorker,
)

_VIDEO_FILTER = "Video Files (*.mp4 *.avi *.mov *.mkv)"
_MP4_FILTER = "MP4 Files (*.mp4)"


def _new_temp_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "ditherzam" / uuid.uuid4().hex
    (d / "original_frames").mkdir(parents=True, exist_ok=True)
    (d / "dithered_frames").mkdir(parents=True, exist_ok=True)
    return d


class FramePlayer:
    """QTimer-driven playback over a directory of dithered frames."""

    def __init__(self, on_frame, fps: float = 24.0) -> None:
        self._on_frame = on_frame
        self._frames: list[Path] = []
        self._idx = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self.step)
        self.set_fps(fps)

    def set_fps(self, fps: float) -> None:
        interval = int(1000.0 / fps) if fps and fps > 0 else 42
        self._timer.setInterval(max(1, interval))

    def load(self, frames_dir) -> None:
        self._frames = sorted(Path(frames_dir).glob("frame*.png"))
        self._idx = 0

    def start(self) -> None:
        if self._frames:
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def step(self) -> None:
        if not self._frames:
            return
        p = self._frames[self._idx % len(self._frames)]
        self._on_frame(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
        self._idx += 1


class VideoController:
    """Owns video state and wires the File > Video submenu to workers."""

    def __init__(self, main_window, pipeline, settings_provider, expert_provider) -> None:
        self.win = main_window
        self.pipeline = pipeline
        self._settings_provider = settings_provider   # () -> RenderSettings
        self._expert_provider = expert_provider        # () -> bool
        self.pool = QThreadPool.globalInstance()
        self.temp_dir: Path | None = None
        self.input_file: str | None = None
        self.framerate: float = 24.0
        self.player = FramePlayer(self._show_frame)

    # --- menu construction ---
    def build_menu(self) -> QMenu:
        menu = QMenu("Video", self.win)
        menu.addAction("Import Video", self.import_video)
        self._export_action = menu.addAction("Export Video", self.export_video)
        self._export_action.setEnabled(False)
        return menu

    def _show_frame(self, rgb_u8: np.ndarray) -> None:
        # main window's viewport is expected to accept an HxWx3 uint8 array.
        self.win.viewport.set_image(rgb_u8)

    def _error(self, msg: str) -> None:
        QMessageBox.critical(self.win, "Video", msg)

    # --- import ---
    def import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.win, "Import Video", "", _VIDEO_FILTER)
        if not path:
            return
        expert = bool(self._expert_provider())
        try:
            fps = probe_fps(path)
            duration = probe_duration(path)
        except Exception as e:  # noqa: BLE001
            self._error(str(e))
            return
        limit_msg = check_video_limits(fps, duration, expert=expert)
        if limit_msg is not None:
            self._error(limit_msg)
            return

        self.temp_dir = _new_temp_dir()
        self.input_file = path
        self.framerate = fps or 24.0
        self.player.set_fps(self.framerate)

        dlg = QProgressDialog("Extracting frames...", "", 0, 0, self.win)
        dlg.setCancelButton(None)
        dlg.setWindowModality(dlg.windowModality())
        dlg.show()

        worker = VideoImportWorker(path, str(self.temp_dir / "original_frames"))
        worker.signals.error.connect(lambda m: (dlg.close(), self._error(m)))
        worker.signals.finished.connect(lambda _fd: self._on_imported(dlg))
        self.pool.start(worker)

    def _on_imported(self, dlg) -> None:
        dlg.close()
        assert self.temp_dir is not None
        preview = detect_preview_frame(self.temp_dir / "original_frames")
        if preview is None:
            self._error("Could not detect a non-black frame for preview.")
        else:
            self._show_frame(np.asarray(Image.open(preview).convert("RGB"), np.uint8))
        self._export_action.setEnabled(True)

    # --- export ---
    def export_video(self) -> None:
        if self.temp_dir is None:
            return
        out, _ = QFileDialog.getSaveFileName(self.win, "Export Video", "", _MP4_FILTER)
        if not out:
            return
        in_dir = self.temp_dir / "original_frames"
        out_dir = self.temp_dir / "dithered_frames"

        prog = QProgressDialog("Processing frames...", "Cancel", 0, 100, self.win)
        dither = VideoDitherWorker(
            str(in_dir), str(out_dir), self.pipeline, self._settings_provider()
        )
        prog.canceled.connect(dither.cancel)

        def on_dither_progress(done: int, total: int) -> None:
            prog.setMaximum(max(total, 1))
            prog.setValue(done)

        def on_dither_done(_written: int) -> None:
            prog.close()
            self.player.load(out_dir)
            self.player.start()
            reasm = QProgressDialog("Reassembling video...", "", 0, 0, self.win)
            reasm.setCancelButton(None)
            reasm.show()
            assemble = VideoAssembleWorker(
                str(out_dir), self.framerate, self.input_file, out
            )
            assemble.signals.error.connect(lambda m: (reasm.close(), self._error(m)))
            assemble.signals.finished.connect(
                lambda _o: (reasm.close(),
                            QMessageBox.information(self.win, "Video",
                                                    "Video export complete!"))
            )
            self.pool.start(assemble)

        dither.signals.progress.connect(on_dither_progress)
        dither.signals.error.connect(lambda m: (prog.close(), self._error(m)))
        dither.signals.finished.connect(on_dither_done)
        self.pool.start(dither)
```

- [ ] **Step 3c: Wire it into `ditherzam/ui/main_window.py`**

Add, inside `MainWindow._create_menus()` (or equivalent menu-building method, after
the File menu exists), the following — importing at the top of the file:

```python
from ditherzam.ui.video_controller import VideoController
```

and where menus are assembled:

```python
        # --- Video (File submenu) ---
        self.video_controller = VideoController(
            self,
            self.render_pipeline,                    # RenderPipeline built in Phase 4/5
            settings_provider=self._current_render_settings,   # () -> RenderSettings
            expert_provider=lambda: self.expert_mode,          # bool flag toggled in Extras
        )
        self.file_menu.addMenu(self.video_controller.build_menu())
```

> `self._current_render_settings` and `self.expert_mode` already exist from Phase 5
> (controls → RenderSettings mapping, and the Extras → Expert Mode toggle). If the
> Expert Mode toggle is not yet present, add a boolean `self.expert_mode = False`
> attribute and an Extras menu checkable action that flips it — both are one-liners
> and do not affect this phase's tests.

- [ ] **Step 4: Run — verify pass**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_workers_smoke.py -v`
Expected: `6 passed` (or `6 skipped` on a box without PySide6 installed).

- [ ] **Step 5: Commit**

```bash
git add ditherzam/video/workers.py ditherzam/ui/video_controller.py ditherzam/ui/main_window.py tests/test_video_workers_smoke.py
git commit -m "feat(video): Qt workers + import/export UI controller + frame playback"
```

---

## Subsystem Definition of Done (checklist)

- [ ] `ditherzam/video/` package exists; **no ffmpeg binary is committed** (`git ls-files assets/ffmpeg` is empty). `ffmpeg_bin`/`ffprobe_bin` resolve `assets/ffmpeg/` first, then `shutil.which`, and fall back to a bare name for pure command building.
- [ ] All command builders (`cmd_probe_fps/duration/has_audio`, `cmd_extract_frames`, `cmd_encode`, `cmd_extract_audio[_reencode]`, `cmd_mux`) return `list[str]` and are asserted **without spawning ffmpeg** (`tests/test_ffmpeg_cmds.py`).
- [ ] `parse_fps` handles `num/den`, plain numbers, and garbage (→ 0.0).
- [ ] `check_video_limits` rejects fps > 60 and duration > 60 (fps checked first), allows the exact boundary, and `expert=True` bypasses both (`tests/test_video_limits.py`).
- [ ] `probe_fps/duration/has_audio` are injectable-runner wrappers (ffmpeg-free unit tests).
- [ ] `dither_frames` writes N frames, preserves size, calls `progress(done,total)` per frame, and cancels cleanly leaving exactly the frames completed (`tests/test_video_frame_dither.py`).
- [ ] `assemble_video` encodes, and when the original has audio, extracts (copy→AAC fallback) and muxes `-c copy -shortest`; else moves temp→out. Command sequence verified headlessly; end-to-end verified only when ffmpeg is present (`tests/test_ffmpeg_integration.py`, `skipif` guarded).
- [ ] `workers.py` is the **only** Qt import under `ditherzam/video/`; workers delegate to the pure functions and expose `finished/error/progress` signals; `VideoDitherWorker.cancel()` sets `_is_canceled`.
- [ ] UI controller enforces the limit check pre-import, drives QTimer playback, and wires the File → Video submenu.
- [ ] Full suite green: `NUMBA_DISABLE_JIT=1 pytest tests/test_video_ffmpeg_resolve.py tests/test_ffmpeg_cmds.py tests/test_video_limits.py tests/test_video_frame_dither.py tests/test_ffmpeg_integration.py tests/test_video_workers_smoke.py -q`.

## Self-Review

**Spec-coverage map (DITHER_BOY_FULL_SPEC.md §12):**
| Spec | Task |
|---|---|
| §12.1 Probe (get_video_info) | 7.1 `cmd_probe_fps/duration`, 7.3 `probe_fps/duration` |
| §12.2 Import + constraints | 7.2 `check_video_limits`, 7.6 controller `import_video` |
| §12.3 Frame extraction (`-qscale:v 2`, `frame%06d.png`) | 7.1 `cmd_extract_frames`, 7.6 `VideoImportWorker`; preview via 7.4 `detect_preview_frame` |
| §12.4 Dither each frame (+cancel/progress) | 7.4 `dither_frames`, 7.6 `VideoDitherWorker` |
| §12.5 Reassemble + audio preservation | 7.1 `cmd_encode/extract_audio*/mux`, 7.5 `assemble_video`, 7.6 `VideoAssembleWorker` |
| §12.6 Expert Mode bypass | 7.2 `expert` flag, 7.6 `expert_provider` |
| §12.7 Temp cleanup | temp dirs under `tempfile.gettempdir()/ditherzam/<uid>/`; controller creates them (purge action belongs to the Extras menu in Phase 5) |
| §17.6 Live video playback/preview | 7.6 `FramePlayer` QTimer playback |

**Placeholder scan:** no `TODO`, no `pass`-only bodies, no `...` stubs — every code step is the complete implementation.

**Type-consistency vs FROZEN CONTRACTS:**
- Consumes `RenderPipeline.render(base_gray_f32, settings) -> uint8 HxWx3` and `RenderSettings(style=..., scale=...)` verbatim (Phase 4 contract).
- Consumes `to_gray_f32` (Phase 1) and `ditherzam.dithering.registry` (Phase 1).
- Command builders return `list[str]`; `check_video_limits(fps, duration, expert) -> str | None`; `dither_frames(...) -> int` — all match the plan's declared interfaces.
- Worker/Signals names (`VideoImportWorker`, `VideoDitherWorker`, `VideoAssembleWorker`, `WorkerSignals`) mirror spec §12 (`VideoDitherSignals`/`VideoAssembleSignals` collapsed into one reusable `WorkerSignals`, which is a clean-room simplification, not a copied name).

**Legal:** no Dither Boy strings/URLs/binaries embedded. The two user-facing limit
messages are functional error text required by the spec's behavior contract, not
copyrighted creative content; expert-mode emoji flavor text from the original is
**omitted**.
</content>
</invoke>
