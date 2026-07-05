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
