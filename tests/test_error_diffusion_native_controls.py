import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.kernels import error_diffusion as ed


WEIGHTED = {
    "Jarvis-Judice-Ninke": (ed._JJN_OFF, ed._JJN_W, 48.0),
    "Stucki": (ed._STUCKI_OFF, ed._STUCKI_W, 42.0),
    "Burkes": (ed._BURKES_OFF, ed._BURKES_W, 32.0),
    "Sierra": (ed._SIERRA_OFF, ed._SIERRA_W, 32.0),
    "Sierra-Lite": (ed._SIERRA_LITE_OFF, ed._SIERRA_LITE_W, 4.0),
    "Two-Row-Sierra": (ed._TWO_ROW_OFF, ed._TWO_ROW_W, 16.0),
    "Stevenson-Arce": (ed._STEVENSON_OFF, ed._STEVENSON_W, 200.0),
    "Fan": (ed._FAN_OFF, ed._FAN_W, 16.0),
    "Shiau-Fan": (ed._SHIAU_OFF, ed._SHIAU_W, 16.0),
    "False Floyd-Steinberg": (ed._FALSE_FS_OFF, ed._FALSE_FS_W, 8.0),
    "Atkinson-Light": (ed._ATK_LIGHT_OFF, ed._ATK_LIGHT_W, 8.0),
}

DIFFUSION_STYLES = ["Floyd-Steinberg", "Atkinson", *WEIGHTED, "Ostromukhov"]


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(4).integers(0, 256, (48, 48)).astype(np.float32)


@pytest.mark.parametrize("name", DIFFUSION_STYLES)
def test_declared_defaults_preserve_historical_pixels(name, image):
    entry = registry.get_entry(name)
    actual = entry.func(image, (100, 100, 100, 50, 100), 127.5,
                        2) if entry.supports_levels else entry.func(
                            image, (100, 100, 100, 50, 100), 127.5)
    if name == "Floyd-Steinberg":
        expected = ed._floyd_steinberg(image, 127.5, 2)
    elif name == "Atkinson":
        expected = ed._atkinson(image, 127.5, 2)
    elif name == "Ostromukhov":
        expected = ed._ostromukhov(image, 127.5)
    else:
        offsets, weights, divisor = WEIGHTED[name]
        expected = ed._diffuse(image, 127.5, offsets, weights, divisor, 2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", DIFFUSION_STYLES)
def test_each_native_diffusion_control_changes_kernel_output(name, image):
    entry = registry.get_entry(name)
    defaults = (100, 100, 100, 50, 100)
    base = entry.func(image, defaults, 127.5, 2) if entry.supports_levels else entry.func(
        image, defaults, 127.5)
    alternatives = (35, 35, 35, 85, 45)
    for index, value in enumerate(alternatives):
        params = list(defaults)
        params[index] = value
        changed = entry.func(image, tuple(params), 127.5,
                             2) if entry.supports_levels else entry.func(
                                 image, tuple(params), 127.5)
        assert np.any(changed != base), (name, entry.param_sliders[index])


def test_gaussian_has_five_native_controls_and_legacy_default(image):
    entry = registry.get_entry("Gaussian")
    defaults = (1, 50, 50, 50, 0)
    base = entry.func(image, defaults, 127.5)
    np.testing.assert_array_equal(base, ed._gaussian_dither(image, 1.0, 127.5))
    for index, value in enumerate((4, 90, 90, 90, 90)):
        params = list(defaults)
        params[index] = value
        assert np.any(entry.func(image, tuple(params), 127.5) != base)


def test_every_local_error_diffusion_style_has_five_native_controls():
    excluded = {"None", "Hilbert (Riemersma)", "Spiral Path"}
    names = set(registry.by_category()["Error Diffusion"]) - excluded
    assert names
    for name in names:
        assert len(registry.get_entry(name).param_sliders) == 5, name


def test_bayer4_defaults_and_each_native_control(image):
    entry = registry.get_entry("Bayer-Matrix 4x4")
    defaults = (1, 0, 100, 0, 0)
    base = entry.func(image, defaults, 127.5, 2)
    np.testing.assert_array_equal(base, ed._ordered(image, ed._BAYER4, 2))
    assert len(entry.param_sliders) == 5
    for index, value in enumerate((3, 1, 160, 1, 1)):
        params = list(defaults)
        params[index] = value
        changed = entry.func(image, tuple(params), 127.5, 2)
        assert np.any(changed != base), entry.param_sliders[index]
