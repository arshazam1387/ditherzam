import numpy as np
from ditherzam.adjustments import (
    apply_contrast, apply_midtones, apply_highlights, apply_blur, apply_invert,
)

def arr(v): return np.full((2, 2), v, dtype=np.float32)

def test_contrast_50_is_identity():
    np.testing.assert_allclose(apply_contrast(arr(100), 50), arr(100))  # factor 50/50=1

def test_contrast_100_doubles():
    np.testing.assert_allclose(apply_contrast(arr(100), 100), arr(200))  # factor 2

def test_midtones_50_is_identity():
    np.testing.assert_allclose(apply_midtones(arr(128), 50), arr(128), atol=1e-3)  # gamma 1

def test_midtones_gamma_formula():
    # value 90 -> gamma = 1 + (90-50)/200 = 1.2 ; out = 255*(img/255)**(1/1.2)
    out = apply_midtones(arr(128), 90)
    exp = 255 * (128/255) ** (1/1.2)
    np.testing.assert_allclose(out, arr(exp), atol=1e-3)

def test_highlights_formula():
    # value 100 -> factor 1 + (100-50)/100 = 1.5
    np.testing.assert_allclose(apply_highlights(arr(100), 100), arr(150))

def test_blur_zero_is_identity():
    np.testing.assert_allclose(apply_blur(arr(100), 0), arr(100))

def test_invert():
    np.testing.assert_allclose(apply_invert(arr(40), True), arr(215))
    np.testing.assert_allclose(apply_invert(arr(40), False), arr(40))
