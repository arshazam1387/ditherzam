import numpy as np
import pytest
from PIL import Image
from ditherzam.export.raster import RasterExportError, save_raster
from ditherzam.masking.composite import flatten_rgba_white


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


def test_png_round_trips_straight_rgba_exactly(tmp_path):
    rgba = np.array(
        [[[5, 17, 231, 0], [9, 101, 203, 1], [250, 40, 80, 127],
          [3, 222, 91, 254], [99, 88, 77, 255]]], dtype=np.uint8,
    )
    out = save_raster(rgba, tmp_path / "hair.png")
    back = np.asarray(Image.open(out))
    assert Image.open(out).mode == "RGBA"
    np.testing.assert_array_equal(back, rgba)


def test_jpeg_rgba_uses_shared_deterministic_white_flatten(tmp_path, monkeypatch):
    rgba = np.array(
        [[[0, 0, 0, 0], [10, 20, 30, 64], [250, 100, 5, 128], [1, 2, 3, 255]]],
        dtype=np.uint8,
    )
    expected = flatten_rgba_white(rgba)
    observed = {}
    real_fromarray = Image.fromarray

    def capture(value, *args, **kwargs):
        observed["value"] = np.array(value, copy=True)
        return real_fromarray(value, *args, **kwargs)

    monkeypatch.setattr("ditherzam.export.raster.Image.fromarray", capture)
    save_raster(rgba, tmp_path / "edge.jpg")
    np.testing.assert_array_equal(observed["value"], expected)
    assert tuple(expected[0, 0]) == (255, 255, 255)


def test_jpeg_transparent_thin_edges_have_white_corners_with_codec_tolerance(tmp_path):
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[..., :3] = (15, 30, 45)
    rgba[16:48, 31:33, :3] = (220, 80, 30)
    rgba[16:48, 31:33, 3] = np.linspace(0, 255, 32, dtype=np.uint8)[:, None]
    out = save_raster(rgba, tmp_path / "thin.jpeg")
    back = np.asarray(Image.open(out).convert("RGB"))
    assert np.min(back[[0, 0, -1, -1], [0, -1, 0, -1]]) >= 250


@pytest.mark.parametrize("shape", [(2, 2, 1), (2, 2, 2), (2, 2, 5), (2,), (1, 2, 3, 4)])
def test_unsupported_channel_or_rank_fails(shape, tmp_path):
    with pytest.raises(RasterExportError, match="shape"):
        save_raster(np.zeros(shape, np.uint8), tmp_path / "bad.png")


@pytest.mark.parametrize("name", ["x.bmp", "x.webp", "x", "x.PNG.tmp"])
def test_unsupported_extension_fails(name, tmp_path):
    with pytest.raises(RasterExportError, match="unsupported raster extension"):
        save_raster(np.zeros((2, 2, 3), np.uint8), tmp_path / name)


def test_historical_grayscale_png_and_jpeg_are_accepted(tmp_path):
    gray = np.array([[0, 64], [128, 255]], np.uint8)
    png = save_raster(gray, tmp_path / "gray.png")
    jpg = save_raster(gray, tmp_path / "gray.jpg")
    np.testing.assert_array_equal(np.asarray(Image.open(png)), gray)
    assert np.asarray(Image.open(jpg).convert("RGB")).shape == (2, 2, 3)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_float_values_fail_before_cast(value, tmp_path):
    image = np.zeros((2, 2, 3), np.float32)
    image[0, 0, 0] = value
    with pytest.raises(RasterExportError, match="finite"):
        save_raster(image, tmp_path / "bad.png")


def test_boolean_values_are_not_uint8_like(tmp_path):
    with pytest.raises(RasterExportError, match="boolean"):
        save_raster(np.zeros((2, 2, 3), dtype=bool), tmp_path / "bad.png")
