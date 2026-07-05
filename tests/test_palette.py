import numpy as np
import pytest
from ditherzam.color.palette import Palette, builtin_palettes, extract_palette


def test_from_list_shape_and_dtype():
    p = Palette.from_list("duo", [[0, 0, 0], [255, 255, 255]])
    assert p.name == "duo"
    assert p.colors.shape == (2, 3)
    assert p.colors.dtype == np.float32


def test_from_list_values_preserved():
    p = Palette.from_list("t", [[10, 20, 30], [40, 50, 60]])
    np.testing.assert_array_equal(p.colors, np.array([[10, 20, 30], [40, 50, 60]], np.float32))


def test_roundtrip_yaml(tmp_path):
    p = Palette.from_list("mypal", [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    f = tmp_path / "mypal.yaml"
    p.to_yaml(f)
    assert f.is_file()
    q = Palette.load(f)
    assert q.name == "mypal"
    assert q.colors.dtype == np.float32
    np.testing.assert_array_equal(q.colors, p.colors)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Palette.load(tmp_path / "nope.yaml")


def test_builtins_all_present():
    b = builtin_palettes()
    for name in ("grayscale", "gameboy", "cga", "pico8", "sepia"):
        assert name in b, f"missing built-in palette: {name}"


def test_builtin_counts_and_shape():
    b = builtin_palettes()
    assert b["grayscale"].colors.shape == (4, 3)
    assert b["gameboy"].colors.shape == (4, 3)
    assert b["cga"].colors.shape == (16, 3)
    assert b["pico8"].colors.shape == (16, 3)
    assert b["sepia"].colors.shape == (4, 3)
    for p in b.values():
        assert p.colors.dtype == np.float32


def test_builtin_exact_values():
    b = builtin_palettes()
    np.testing.assert_array_equal(
        b["gameboy"].colors,
        np.array([[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]], np.float32),
    )
    # PICO-8 index 8 is the signature red (#FF004D)
    np.testing.assert_array_equal(b["pico8"].colors[8], np.array([255, 0, 77], np.float32))
    # CGA index 14 is yellow
    np.testing.assert_array_equal(b["cga"].colors[14], np.array([255, 255, 85], np.float32))


def test_extract_returns_k_colors():
    img = np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    p = extract_palette(img, k=8)
    assert p.colors.shape == (8, 3)
    assert p.colors.dtype == np.float32
    assert p.name == "source"


def test_extract_default_name_and_k():
    img = np.random.RandomState(3).randint(0, 256, (16, 16, 3), dtype=np.uint8)
    p = extract_palette(img)
    assert p.colors.shape == (16, 3)


def test_extract_two_color_image():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :5] = [255, 0, 0]
    img[:, 5:] = [0, 0, 255]
    p = extract_palette(img, k=2)
    got = [tuple(int(round(v)) for v in c) for c in p.colors]
    reds = [c for c in got if c[0] > 200 and c[2] < 55]
    blues = [c for c in got if c[2] > 200 and c[0] < 55]
    assert reds and blues


def test_extract_k_larger_than_unique_colors():
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :4] = [10, 10, 10]
    img[:, 4:] = [200, 200, 200]
    p = extract_palette(img, k=4)
    # still returns exactly k rows even when the image has < k distinct colors
    assert p.colors.shape == (4, 3)
