import numpy as np
from ditherzam.dithering import registry

SINGLE = [
    "Artifact Modulation", "Atkinson-VHS", "Glitch", "Modulated Diffuse Y",
    "Modulated Diffuse X", "Waveform", "Waveform Alt", "Ordered Modulation",
    "Stucki Diffusion Lines", "Contrast Aware Y", "Contrast Aware X",
]
MULTI = {
    "Uniform Modulation Y": (4, 50, 20, 100, 0),
    "Uniform Modulation X": (4, 50, 20, 100, 0),
    "Smooth Diffuse": (4, 5, 100, 100, 0),
    "Atkinson Line Modulation": (5, 5, 8, 100, 100),
}


def test_single_param_glitch_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n in SINGLE:
        e = registry.get_entry(n)
        assert e is not None, n
        out = e.func(img.copy(), 4, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_multi_param_glitch_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n, param in MULTI.items():
        e = registry.get_entry(n)
        assert e is not None, n
        assert len(e.param_sliders) == len(param), n
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_glitch_is_deterministic():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    a = registry.get_entry("Glitch").func(img.copy(), 6, 128.0)
    b = registry.get_entry("Glitch").func(img.copy(), 6, 128.0)
    np.testing.assert_array_equal(a, b)
