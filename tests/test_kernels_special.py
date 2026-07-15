import numpy as np
from ditherzam.dithering import registry

SINGLE = ["Radial Burst", "Wave", "Noise", "Topography", "Thresholder",
          "Diagonal", "Vortex", "Concentric Rings", "Wireframe Alt",
          "Crosshatch Alt"]


def test_special_single_binary_shape():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    for n in SINGLE:
        e = registry.get_entry(n)
        assert e is not None, n
        param = 5 if e.param_sliders else 0
        out = e.func(img.copy(), param, 128.0)
        assert out.shape == img.shape and out.dtype == np.float32, n
        assert set(np.unique(out).tolist()) <= {0.0, 255.0}, n


def test_displace_contour_takes_native_tuple():
    e = registry.get_entry("Displace Contour")
    assert e is not None and len(e.param_sliders) == 5
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    out = e.func(img.copy(), (70, 2, 0, 3, 4), 128.0)
    assert out.shape == img.shape
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}


def test_sine_wave_modulation_takes_native_tuple():
    e = registry.get_entry("Sine Wave Modulation")
    assert e is not None and len(e.param_sliders) == 6
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    out = e.func(img.copy(), (5, 10, 10, 0, 100, 100), 128.0)
    assert out.shape == img.shape
    assert set(np.unique(out).tolist()) <= {0.0, 255.0}


def test_noise_is_deterministic():
    img = np.tile(np.linspace(0, 255, 24, dtype=np.float32), (24, 1))
    a = registry.get_entry("Noise").func(img.copy(), 0, 128.0)
    b = registry.get_entry("Noise").func(img.copy(), 0, 128.0)
    np.testing.assert_array_equal(a, b)
