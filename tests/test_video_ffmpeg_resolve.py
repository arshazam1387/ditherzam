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
