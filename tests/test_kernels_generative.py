import numpy as np

from ditherzam.dithering import registry


NAMES_AND_CATEGORIES = {
    "Hilbert (Riemersma)": "Error Diffusion",
    "Spiral Path": "Error Diffusion",
    "Flow Hatch": "Patterned",
    "Hex Bayer": "Ordered Dither",
    "Triangular": "Ordered Dither",
    "Spiral Engrave": "Special Effects",
    "Reaction-Diffusion": "Special Effects",
    "Quasicrystal": "Special Effects",
}


def test_generative_kernels_are_registered_in_existing_categories():
    for name, category in NAMES_AND_CATEGORIES.items():
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.category == category
        assert entry.param_sliders[0] == "dither_parameter_slider"
        assert len(entry.param_sliders) >= 5
        assert not entry.supports_levels


def test_generative_kernels_are_binary_and_nonconstant_on_gradient():
    image = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))
    for name in NAMES_AND_CATEGORIES:
        out = registry.get_entry(name).func(image.copy(), 4, 128.0)
        assert out.dtype == np.float32, name
        assert set(np.unique(out)).issubset({0.0, 255.0}), name
        assert np.unique(out).size == 2, name


def test_generative_kernels_are_repeatable():
    rng = np.random.default_rng(20260710)
    image = rng.integers(0, 256, size=(31, 37)).astype(np.float32)
    for name in NAMES_AND_CATEGORIES:
        func = registry.get_entry(name).func
        np.testing.assert_array_equal(func(image.copy(), 7, 143.0),
                                      func(image.copy(), 7, 143.0),
                                      err_msg=name)
