import numpy as np
from PIL import Image
from ditherzam.batch import batch_process
from ditherzam.render import RenderSettings, RenderPipeline
from ditherzam.dithering import registry


def _write_png(path, size_wh, value=127):
    w, h = size_wh
    Image.fromarray(np.full((h, w, 3), value, np.uint8)).save(path)


def test_batch_processes_matching_and_skips_others(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _write_png(src / "a.png", (8, 8))
    _write_png(src / "b.png", (8, 8))
    _write_png(src / "c.png", (16, 16))          # different size -> skipped
    (src / "notes.txt").write_text("ignore me")  # non-image -> ignored

    out = tmp_path / "out"
    pipeline = RenderPipeline(registry)
    settings = RenderSettings(style="None")

    processed, skipped = batch_process(src, out, settings, pipeline, (8, 8))
    assert (processed, skipped) == (2, 1)
    assert (out / "a.png").exists() and (out / "b.png").exists()
    assert not (out / "c.png").exists()


def test_batch_creates_output_folder(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _write_png(src / "only.png", (8, 8))
    out = tmp_path / "made" / "here"
    pipeline = RenderPipeline(registry)
    processed, skipped = batch_process(src, out, RenderSettings(style="None"),
                                       pipeline, (8, 8))
    assert out.is_dir() and processed == 1 and skipped == 0
    result = np.array(Image.open(out / "only.png").convert("RGB"))
    assert result.shape == (8, 8, 3)
