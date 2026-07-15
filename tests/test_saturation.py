import numpy as np
from ditherzam.adjustments import apply_saturation, apply_saturation_u8


def rgb(r, g, b):
    return np.array([[[r, g, b]]], dtype=np.float32)


def test_saturation_50_is_identity():
    c = rgb(200, 50, 30)
    np.testing.assert_allclose(apply_saturation(c, 50), c, atol=1e-3)


def test_saturation_0_is_gray():
    c = rgb(200, 50, 30)
    out = apply_saturation(c, 0)
    assert abs(out[0, 0, 0] - out[0, 0, 1]) < 1e-3
    assert abs(out[0, 0, 1] - out[0, 0, 2]) < 1e-3
    # the gray value equals the luminance
    lum = 0.299 * 200 + 0.587 * 50 + 0.114 * 30
    np.testing.assert_allclose(out[0, 0, 0], lum, atol=1e-3)


def test_saturation_100_doubles_deviation():
    c = rgb(200, 50, 30)
    lum = 0.299 * 200 + 0.587 * 50 + 0.114 * 30
    out = apply_saturation(c, 100)
    # each channel deviation from luminance is doubled
    np.testing.assert_allclose(out[0, 0], lum + (c[0, 0] - lum) * 2.0, atol=1e-3)


def test_saturation_returns_float32():
    out = apply_saturation(rgb(10, 20, 30), 75)
    assert out.dtype == np.float32


def _legacy_saturation_u8(rgb, value):
    factor = value / 50.0
    lum = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )[..., None]
    adjusted = (lum + (rgb - lum) * factor).astype(np.float32)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def test_saturation_u8_is_exact_for_fractional_and_out_of_range_rgb():
    rng = np.random.default_rng(20260709)
    colors = rng.uniform(-30, 285, size=(37, 43, 3)).astype(np.float32)
    for value in (0, 1, 49, 50, 51, 73, 99, 100):
        got = apply_saturation_u8(colors, value)
        np.testing.assert_array_equal(got, _legacy_saturation_u8(colors, value))
        assert got.dtype == np.uint8


def test_saturation_u8_gray_avoids_materializing_rgb_but_keeps_legacy_bytes():
    gray = np.linspace(-10.25, 265.75, 47 * 31, dtype=np.float32).reshape(31, 47)
    legacy_rgb = np.repeat(gray[..., None], 3, axis=2)
    for value in (0, 50, 100):
        np.testing.assert_array_equal(
            apply_saturation_u8(gray, value),
            _legacy_saturation_u8(legacy_rgb, value),
        )
