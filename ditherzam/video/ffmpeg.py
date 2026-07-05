"""Clean-room ffmpeg/ffprobe wrappers: binary resolution, pure command builders,
import-limit checks, an injectable subprocess runner, probe helpers, and video
assembly. This module imports NO PySide6 — it must stay Qt-free.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path, PurePath

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


# --- pure command builders (never spawn a process; every element is a str) ----

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


# --- import limit checks (spec §12.2 / §12.6) ---------------------------------

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


# --- subprocess runner + probe wrappers (execution; injectable in tests) -------

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


# --- assemble frames + preserve audio (spec §12.5) ----------------------------

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
        shutil.move(str(temp_video), str(out_path))
