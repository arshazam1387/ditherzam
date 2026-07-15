import numpy as np
from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither

def test_none_style_is_passthrough():
    img = np.full((8, 8), 100.0, dtype=np.float32)
    out = apply_dither(img, style="None", scale=1, luminance_threshold=50,
                       params={}, registry=registry)
    np.testing.assert_allclose(out, img)

def test_preview_disabled_is_passthrough():
    img = np.full((8, 8), 100.0, dtype=np.float32)
    out = apply_dither(img, style="Floyd-Steinberg", scale=1, luminance_threshold=50,
                       params={}, registry=registry, preview_disabled=True)
    np.testing.assert_allclose(out, img)

def test_dither_returns_binary_same_size():
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    out = apply_dither(img, style="Floyd-Steinberg", scale=1, luminance_threshold=50,
                       params={}, registry=registry)
    assert out.shape == img.shape
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}

def test_scale_pixelates_via_block_size():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    out = apply_dither(img, style="Bayer-Matrix 4x4", scale=4, luminance_threshold=50,
                       params={}, registry=registry)
    assert out.shape == img.shape  # upscaled back
