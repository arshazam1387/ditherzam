import numpy as np
from ditherzam.dithering import registry

NAMES = [
    "Bayer-Matrix 2x2", "Bayer-Matrix 8x8", "Bayer-Matrix 16x16",
    "Bayer-Ordered", "Bayer-Void", "Random Ordered", "Bit Tone", "Mosaic",
    "Modulated Bayer Dither", "Cluster-Dot", "Halftone-Ordered",
]


def test_ordered_registered_binary_shape():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    for n in NAMES:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 4 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_ordered_extremes():
    black = np.zeros((16, 16), dtype=np.float32)
    white = np.full((16, 16), 255.0, dtype=np.float32)
    for n in ("Bayer-Matrix 2x2", "Bayer-Matrix 8x8", "Cluster-Dot"):
        assert registry.get_entry(n).func(black.copy(), 0, 128.0).sum() == 0.0, n
        assert np.all(registry.get_entry(n).func(white.copy(), 0, 128.0) == 255.0), n


def test_random_ordered_is_deterministic():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    a = registry.get_entry("Random Ordered").func(img.copy(), 0, 128.0)
    b = registry.get_entry("Random Ordered").func(img.copy(), 0, 128.0)
    np.testing.assert_array_equal(a, b)
