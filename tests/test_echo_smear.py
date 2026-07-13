import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs

# param tuple order: (count, spacing, wave, phase, streak, dissolve, breath)
DEFAULTS = (6, 10, 8, 0, 20, 30, 50)
THR = np.float32(128.0)


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(7).integers(0, 256, (48, 48)).astype(np.float32)


@pytest.fixture(scope="module")
def gradient():
    return np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))


def entry():
    e = registry.get_entry("Echo Smear")
    assert e is not None
    return e


def test_registered_in_special_effects():
    e = entry()
    assert e.category == "Special Effects"
    assert e.dims == 2
    assert e.param_sliders == (
        "echo_count_slider", "echo_spacing_slider", "echo_wave_amount_slider",
        "echo_wave_phase_slider", "echo_streak_slider", "echo_dissolve_slider",
        "echo_breath_slider",
    )


def test_parameter_metadata_exact():
    specs = {s.key: (s.label, s.minimum, s.maximum, s.default)
             for s in parameter_specs(entry())
             if not s.key.startswith("creative_")}
    assert specs == {
        "echo_count_slider": ("Echo Count", 0, 16, 6),
        "echo_spacing_slider": ("Echo Spacing", 2, 40, 10),
        "echo_wave_amount_slider": ("Wave Amount", 0, 32, 8),
        "echo_wave_phase_slider": ("Wave Phase", 0, 360, 0),
        "echo_streak_slider": ("Streak Amount", 0, 100, 20),
        "echo_dissolve_slider": ("Dissolve Amount", 0, 100, 30),
        "echo_breath_slider": ("Breath", 0, 100, 50),
    }


def test_output_contract(image):
    out = entry().func(image.copy(), DEFAULTS, THR)
    assert out.dtype == np.float32
    assert out.shape == image.shape
    assert set(np.unique(out)) <= {0.0, 255.0}


def test_ingredients_off_equals_plain_threshold(gradient):
    # count=0, wave irrelevant, streak=0, dissolve=0, breath=0 -> pure threshold body
    out = entry().func(gradient.copy(), (0, 10, 8, 0, 0, 0, 0), THR)
    expected = np.where(gradient < 128.0, 0.0, 255.0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_deterministic(image):
    a = entry().func(image.copy(), DEFAULTS, THR)
    b = entry().func(image.copy(), DEFAULTS, THR)
    np.testing.assert_array_equal(a, b)
