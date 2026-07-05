import numpy as np
from ditherzam.dithering import registry

NAMES = [
    "Jarvis-Judice-Ninke", "Stucki", "Burkes", "Sierra", "Sierra-Lite",
    "Two-Row-Sierra", "Stevenson-Arce", "Ostromukhov", "Gaussian",
    "Fan", "Shiau-Fan", "False Floyd-Steinberg", "Atkinson-Light",
]


def test_none_registered_is_identity():
    e = registry.get_entry("None")
    assert e is not None
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    out = e.func(img.copy(), 0, 128.0)
    np.testing.assert_allclose(out, img)


def test_all_diffusion_registered_binary_and_shape():
    img = np.tile(np.linspace(0, 255, 8, dtype=np.float32), (8, 1))
    for n in NAMES:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 1 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape, n
        assert out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_diffusion_black_stays_black():
    img = np.zeros((8, 8), dtype=np.float32)
    for n in ("Stucki", "Burkes", "Sierra", "Fan"):
        out = registry.get_entry(n).func(img.copy(), 0, 128.0)
        assert out.sum() == 0.0, n


def test_gaussian_is_deterministic():
    img = np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))
    a = registry.get_entry("Gaussian").func(img.copy(), 5, 128.0)
    b = registry.get_entry("Gaussian").func(img.copy(), 5, 128.0)
    np.testing.assert_array_equal(a, b)
