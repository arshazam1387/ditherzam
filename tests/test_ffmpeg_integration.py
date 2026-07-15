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
        joined = " ".join(cmd)
        if "-c:v" in cmd and "libx264" in cmd:
            Path(cmd[-1]).write_bytes(b"V")
        if "codec_type" in joined:
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
        "probe_audio" if "codec_type" in " ".join(c) else
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
