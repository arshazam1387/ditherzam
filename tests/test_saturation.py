import numpy as np
from ditherzam.adjustments import apply_saturation


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
