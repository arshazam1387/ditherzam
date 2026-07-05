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
