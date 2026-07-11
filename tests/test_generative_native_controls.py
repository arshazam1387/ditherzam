import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.kernels import generative as g


CASES = {
    "Hilbert (Riemersma)": ((8, 50, 100, 0, 0), (16, 35, 50, 20, 100)),
    "Spiral Path": ((70, 100, 0, 0, 100), (30, 50, 3, -2, 0)),
    "Flow Hatch": ((6, 100, 100, 100, 0), (10, 30, 160, 50, 30)),
    "Hex Bayer": ((3, 50, 35, 100, 0), (6, 20, 80, 150, 1)),
    "Triangular": ((3, 100, 2, 100, 0), (6, 50, 5, 150, 1)),
    "Spiral Engrave": ((6, 0, 0, 100, 100), (10, 4, -3, 160, 50)),
    "Reaction-Diffusion": ((20, 100, 100, 100, 5), (10, 70, 140, 60, 20)),
    "Quasicrystal": ((5, 32, 240, 100, 100), (7, 55, 180, 150, 160)),
}


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(5).integers(0, 256, (40, 40)).astype(np.float32)


def historical(name, image, threshold):
    return {
        "Hilbert (Riemersma)": lambda: g._hilbert_riemersma(image, threshold, 16),
        "Spiral Path": lambda: g._spiral_path(image, threshold, np.float32(0.7)),
        "Flow Hatch": lambda: g._flow_hatch(image, threshold, np.float32(6)),
        "Hex Bayer": lambda: g._hex_bayer(image, np.float32(3)),
        "Triangular": lambda: g._triangular(image, np.float32(3)),
        "Spiral Engrave": lambda: g._spiral_engrave(image, threshold, np.float32(6)),
        "Reaction-Diffusion": lambda: g._reaction_diffusion(
            image, threshold, 20, seed_cutoff=5),
        "Quasicrystal": lambda: g._quasicrystal(image, threshold, 5),
    }[name]()


@pytest.mark.parametrize("name", CASES)
def test_generative_declared_defaults_preserve_historical_pixels(name, image):
    threshold = np.float32(127.5)
    defaults, _ = CASES[name]
    actual = registry.get_entry(name).func(image, defaults, threshold)
    np.testing.assert_array_equal(actual, historical(name, image, threshold))


@pytest.mark.parametrize("name", CASES)
def test_each_generative_native_control_changes_pixels(name, image):
    defaults, alternatives = CASES[name]
    entry = registry.get_entry(name)
    base = entry.func(image, defaults, np.float32(127.5))
    assert len(entry.param_sliders) == 5
    for index, value in enumerate(alternatives):
        params = list(defaults)
        params[index] = value
        changed = entry.func(image, tuple(params), np.float32(127.5))
        assert np.any(changed != base), (name, entry.param_sliders[index])


def test_reaction_diffusion_default_is_not_collapsed_on_common_tones():
    entry = registry.get_entry("Reaction-Diffusion")
    defaults = CASES["Reaction-Diffusion"][0]
    gradient = np.tile(np.linspace(0, 255, 96, dtype=np.float32), (96, 1))
    midtone = np.full((96, 96), 127.5, dtype=np.float32)

    for image in (gradient, midtone):
        output = entry.func(image, defaults, np.float32(127.5))
        black_fraction = np.mean(output == 0)
        assert 0.25 < black_fraction < 0.65


def test_reaction_diffusion_iteration_control_is_literal(monkeypatch):
    seen = {}

    def fake_kernel(image, threshold, iterations, feed, kill, diffusion, seeds):
        seen["iterations"] = iterations
        return image

    monkeypatch.setattr(g, "_reaction_diffusion", fake_kernel)
    image = np.zeros((2, 2), dtype=np.float32)
    g.reaction_diffusion(image, (17, 100, 100, 100, 5), np.float32(127.5))
    assert seen["iterations"] == 17
