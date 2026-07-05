import numpy as np
from PIL import Image
from ditherzam.export.raster import save_raster


def test_png_round_trips_exactly(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = save_raster(a, tmp_path / "x.png")
    assert out.exists()
    back = np.array(Image.open(out).convert("RGB"))
    assert back.shape == (8, 8, 3)
    np.testing.assert_array_equal(back, a)          # PNG is lossless


def test_jpg_matches_shape(tmp_path):
    a = np.full((8, 8, 3), 120, dtype=np.uint8)
    out = save_raster(a, tmp_path / "x.jpg")
    assert out.exists()
    back = np.array(Image.open(out).convert("RGB"))
    assert back.shape == (8, 8, 3)                   # JPEG is lossy, shape stable


def test_jpeg_extension_also_works(tmp_path):
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    out = save_raster(a, tmp_path / "y.jpeg")
    assert out.exists() and out.suffix == ".jpeg"


def test_accepts_non_uint8_input(tmp_path):
    a = np.full((4, 4, 3), 200.0, dtype=np.float32)
    out = save_raster(a, tmp_path / "z.png")
    back = np.array(Image.open(out).convert("RGB"))
    assert back.dtype == np.uint8 and int(back[0, 0, 0]) == 200
