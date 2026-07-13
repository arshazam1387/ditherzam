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


def _subject_square(size=64):
    # Dark 20px-wide square on white: crisp silhouette with a right edge at x=30.
    img = np.full((size, size), 230.0, dtype=np.float32)
    img[22:42, 10:31] = 20.0
    return img


def test_echoes_add_ink_right_of_subject():
    img = _subject_square()
    none = entry().func(img.copy(), (0, 10, 8, 0, 0, 0, 100), THR)
    some = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 100), THR)
    right = slice(None), slice(32, None)          # strictly right of the square
    assert (some[right] == 0.0).sum() > (none[right] == 0.0).sum()
    # body region unchanged by echoes
    np.testing.assert_array_equal(some[22:42, 10:31], none[22:42, 10:31])


def test_echoes_vanish_at_breath_zero():
    img = _subject_square()
    out = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 0), THR)
    expected = np.where(img < 128.0, 0.0, 255.0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_wave_phase_moves_the_waves():
    img = _subject_square()
    a = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 100), THR)
    b = entry().func(img.copy(), (6, 10, 8, 180, 0, 0, 100), THR)
    assert np.any(a != b)


def test_echo_spacing_changes_pixels():
    img = _subject_square()
    a = entry().func(img.copy(), (6, 4, 8, 0, 0, 0, 100), THR)
    b = entry().func(img.copy(), (6, 20, 8, 0, 0, 0, 100), THR)
    assert np.any(a != b)


def _streak_columns(out):
    # Columns that are ink for their entire height.
    return {x for x in range(out.shape[1]) if np.all(out[:, x] == 0.0)}


def test_streaks_are_full_height_and_scale_with_slider():
    img = _subject_square(128)
    img[50:80, 40:90] = 20.0                     # widen subject so many columns qualify
    off = entry().func(img.copy(), (0, 10, 0, 0, 0, 0, 0), THR)
    lo = entry().func(img.copy(), (0, 10, 0, 0, 30, 0, 0), THR)
    hi = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 0), THR)
    assert _streak_columns(off) == set()
    assert len(_streak_columns(hi)) >= len(_streak_columns(lo))
    assert len(_streak_columns(hi)) >= 1


def test_streaks_only_from_subject_columns():
    img = np.full((64, 64), 230.0, dtype=np.float32)   # no subject anywhere
    out = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 0), THR)
    assert _streak_columns(out) == set()


def test_streaks_survive_breath_zero():
    img = _subject_square(128)
    img[50:80, 40:90] = 20.0
    b0 = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 0), THR)
    b100 = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 100), THR)
    assert _streak_columns(b0) == _streak_columns(b100)
