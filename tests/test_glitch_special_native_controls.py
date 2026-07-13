import numpy as np
import pytest

from ditherzam.dithering import registry


DEFAULTS = {
    "Artifact Modulation": (1, 5, 10, 128, 0), "Atkinson-VHS": (1, 1, 255, 100, 0),
    "Glitch": (1, 0, 1, 50, 1), "Modulated Diffuse Y": (1, 100, 0, 0, 0),
    "Modulated Diffuse X": (1, 100, 0, 0, 0), "Uniform Modulation Y": (2, 50, 25, 100, 10),
    "Uniform Modulation X": (1, 0, 0, 100, 0), "Waveform": (1, 5, 30, 128, 0),
    "Waveform Alt": (1, 5, 10, 128, 0), "Ordered Modulation": (1, 20, 40, 50, 0),
    "Smooth Diffuse": (1, 5, 100, 100, 0), "Stucki Diffusion Lines": (5, 50, 20, 30, 128),
    "Atkinson Line Modulation": (5, 5, 8, 100, 100), "Contrast Aware Y": (1, 25, 64, 100, 1),
    "Contrast Aware X": (1, 25, 64, 100, 1), "Radial Burst": (24, 0, 0, 0, 128),
    "Wave": (15, 15, 0, 50, 128), "Noise": (255, 0, 0, 1, 100),
    "Topography": (1, 8, 10, 1, 0), "Thresholder": (1, 1, 1, 64, 0),
    "Diagonal": (1, 100, 100, 1, 0), "Displace Contour": (50, 1, 0, 1, 0),
    "Sine Wave Modulation": (5, 10, 10, 0, 100), "Vortex": (6, 15, 0, 0, 0),
    "Concentric Rings": (30, 0, 0, 0, 100), "Wireframe Alt": (1, 100, 100, 100, 1),
    "Crosshatch Alt": (4, 15, 40, 65, 85),
}

VARIANTS = {
    "Artifact Modulation": (8, 12, 25, 80, 90), "Atkinson-VHS": (4, 3, 80, 180, 3),
    "Glitch": (12, 9, 3, 20, 0), "Modulated Diffuse Y": (4, 160, 30, 40, 30),
    "Modulated Diffuse X": (4, 160, 30, 40, 30), "Uniform Modulation Y": (4, .6, 50, 150, 30),
    "Uniform Modulation X": (4, .6, 50, 150, 30), "Waveform": (8, 14, 70, 70, 80),
    "Waveform Alt": (8, 14, 25, 70, 80), "Ordered Modulation": (6, 40, 90, 80, 70),
    "Smooth Diffuse": (4, 2, 160, 40, 40), "Stucki Diffusion Lines": (9, 80, 50, 10, 180),
    "Atkinson Line Modulation": (9, 9, 3, 180, 30), "Contrast Aware Y": (4, 80, 20, 160, 3),
    "Contrast Aware X": (4, 80, 20, 160, 3), "Radial Burst": (9, 80, 4, -3, 70),
    "Wave": (35, 8, 80, 80, 70), "Noise": (100, 8, 30, 3, 70),
    "Topography": (10, 15, 30, 3, 70), "Thresholder": (6, 3, 2, 110, 60),
    "Diagonal": (8, 160, 30, 3, 30), "Displace Contour": (80, 3, 2, 4, 8),
    "Sine Wave Modulation": (12, 22, 25, 80, 160), "Vortex": (11, 35, 70, 4, -3),
    "Concentric Rings": (55, 70, 4, -3, 150), "Wireframe Alt": (8, 160, 30, 180, 3),
    "Crosshatch Alt": (7, 5, 25, 45, 70),
}


@pytest.fixture(scope="module")
def image():
    y, x = np.mgrid[:37, :41]
    return ((x * 17 + y * 29 + (x * y) % 53) % 256).astype(np.float32)


@pytest.mark.parametrize("name", DEFAULTS)
def test_each_style_declares_at_least_five_native_controls(name):
    assert len(registry.get_entry(name).param_sliders) >= 5


@pytest.mark.parametrize("name", DEFAULTS)
def test_explicit_historical_defaults_match_legacy_call_shape(name, image):
    entry = registry.get_entry(name)
    legacy = {
        "Uniform Modulation Y": DEFAULTS["Uniform Modulation Y"], "Uniform Modulation X": DEFAULTS["Uniform Modulation X"],
        "Smooth Diffuse": (1, 5), "Atkinson Line Modulation": (5, 5),
            "Displace Contour": DEFAULTS["Displace Contour"], "Sine Wave Modulation": (5, 10),
    }.get(name, DEFAULTS[name] if name in {"Artifact Modulation", "Atkinson-VHS", "Glitch", "Noise", "Topography", "Diagonal", "Displace Contour", "Wireframe Alt"} else (0 if name in {"Radial Burst", "Wave", "Vortex", "Concentric Rings"} else DEFAULTS[name][0]))
    np.testing.assert_array_equal(
        entry.func(image.copy(), DEFAULTS[name], 128.0),
        entry.func(image.copy(), legacy, 128.0),
    )


@pytest.mark.parametrize("name", DEFAULTS)
def test_native_variant_changes_actual_kernel_output(name, image):
    entry = registry.get_entry(name)
    base = entry.func(image.copy(), DEFAULTS[name], 128.0)
    changed = entry.func(image.copy(), VARIANTS[name], 128.0)
    assert np.any(base != changed), name
