import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from ditherzam.layers import (
    RasterMaskIOError,
    export_raster_mask_png,
    import_raster_mask_png,
)


@pytest.mark.parametrize("mode", ["L", "1", "RGB", "RGBA", "P"])
def test_png_import_supported_modes(tmp_path, mode):
    path = tmp_path / f"{mode}.png"
    if mode == "L":
        image = Image.fromarray(np.array([[0, 127], [200, 255]], np.uint8), "L")
    elif mode == "1":
        image = Image.fromarray(np.array([[0, 255], [255, 0]], np.uint8), "L").convert("1")
    elif mode == "P":
        image = Image.fromarray(np.array([[0, 1], [1, 0]], np.uint8), "P")
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    else:
        channels = 4 if mode == "RGBA" else 3
        data = np.zeros((2, 2, channels), np.uint8)
        data[..., :3] = [[[0, 0, 0], [255, 255, 255]],
                         [[255, 0, 0], [0, 255, 0]]]
        if mode == "RGBA":
            data[..., 3] = [[0, 255], [1, 2]]
        image = Image.fromarray(data, mode)
    image.save(path)
    actual = import_raster_mask_png(path, (2, 2))
    expected = np.asarray(
        image if mode == "L" else
        image.convert("L") if mode == "1" else
        image.convert("RGB").convert("L"),
        dtype=np.uint8,
    )
    assert np.array_equal(actual, expected)


def test_rgba_import_ignores_alpha(tmp_path):
    data = np.array([[[10, 20, 30, 0], [10, 20, 30, 255]]], np.uint8)
    path = tmp_path / "alpha.png"
    Image.fromarray(data, "RGBA").save(path)
    actual = import_raster_mask_png(path, (1, 2))
    assert actual[0, 0] == actual[0, 1]


@pytest.mark.parametrize("kind", ["corrupt", "nonpng", "unsupported", "mismatch"])
def test_png_import_rejects_invalid_inputs(tmp_path, kind):
    path = tmp_path / "mask.png"
    expected = (2, 2)
    if kind == "corrupt":
        path.write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    elif kind == "nonpng":
        Image.new("L", (2, 2)).save(path, format="BMP")
    elif kind == "unsupported":
        Image.fromarray(np.zeros((2, 2), np.int32), "I").save(path)
    else:
        Image.new("L", (3, 2)).save(path)
    with pytest.raises(RasterMaskIOError):
        import_raster_mask_png(path, expected)


def test_png_import_rejects_oversize_before_decode(tmp_path):
    path = tmp_path / "large.png"
    raw = bytearray()
    raw.extend(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", 2**13 + 1, 2**13, 8, 0, 0, 0, 0)
    raw.extend(struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr)
    raw.extend(struct.pack(">I", zlib.crc32(b"IHDR" + ihdr)))
    compressed = zlib.compress(b"\x00")
    raw.extend(struct.pack(">I", len(compressed)) + b"IDAT" + compressed)
    raw.extend(struct.pack(">I", zlib.crc32(b"IDAT" + compressed)))
    raw.extend(struct.pack(">I", 0) + b"IEND")
    raw.extend(struct.pack(">I", zlib.crc32(b"IEND")))
    path.write_bytes(raw)
    with pytest.raises(RasterMaskIOError, match="pixel limit"):
        import_raster_mask_png(path, (2**13, 2**13 + 1))


def test_atomic_exact_export_roundtrip_and_alpha_independence(tmp_path):
    pixels = np.array([[0, 1, 127, 128, 254, 255]], np.uint8)
    path = tmp_path / "mask.png"
    export_raster_mask_png(path, pixels)
    with Image.open(path) as image:
        assert image.mode == "L"
        assert image.info == {}
    assert np.array_equal(import_raster_mask_png(path, pixels.shape), pixels)
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(RasterMaskIOError):
        export_raster_mask_png(path, pixels.astype(np.float32))
