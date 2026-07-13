import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.parameters import parameter_specs

# param tuple order: (count, spacing, wave, phase, streak, dissolve, breath, wave_freq)
DEFAULTS = (6, 10, 8, 0, 20, 30, 50, 10)
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
        "echo_breath_slider", "echo_wave_frequency_slider",
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
        "echo_wave_frequency_slider": ("Wave Frequency", 1, 100, 10),
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


def test_dust_appears_left_of_subject():
    img = _subject_square()
    no_dust = entry().func(img.copy(), (0, 10, 0, 0, 0, 0, 100), THR)
    dust = entry().func(img.copy(), (6, 10, 0, 0, 0, 100, 100), THR)
    left = slice(None), slice(0, 10)             # strictly left of the square
    assert (dust[left] == 0.0).sum() > (no_dust[left] == 0.0).sum()


def test_full_breath_full_dissolve_erases_body():
    img = _subject_square()
    out = entry().func(img.copy(), (0, 10, 0, 0, 0, 100, 100), THR)
    interior = out[24:40, 12:29]                 # deep inside the square
    assert np.all(interior == 255.0)


def test_partial_dissolve_erodes_partially():
    img = _subject_square()
    out = entry().func(img.copy(), (0, 10, 0, 0, 0, 50, 50), THR)
    interior = out[24:40, 12:29]
    frac = (interior == 0.0).mean()
    assert 0.2 < frac < 0.9                      # eroded but present


def test_each_native_control_changes_pixels(image):
    e = entry()
    base = e.func(image.copy(), DEFAULTS, THR)
    alternatives = (12, 20, 20, 180, 80, 90, 100, 60)
    assert len(e.param_sliders) == 8
    for index, value in enumerate(alternatives):
        params = list(DEFAULTS)
        params[index] = value
        changed = e.func(image.copy(), tuple(params), THR)
        assert np.any(changed != base), e.param_sliders[index]


def test_default_output_not_collapsed(gradient):
    out = entry().func(gradient.copy(), DEFAULTS, THR)
    ink = (out == 0.0).mean()
    assert 0.02 < ink < 0.98


def test_not_a_duplicate_of_contour_family(image):
    from tests.golden_harness import default_param
    ours = entry().func(image.copy(), DEFAULTS, THR)
    for name in ("Topography", "Topography Alt", "Displace Contour"):
        other = registry.get_entry(name)
        theirs = other.func(image.copy(), default_param(other), THR)
        assert np.any(ours != theirs), name


def test_echoes_are_continuous_lines_at_full_breath():
    img = _subject_square()
    out = entry().func(img.copy(), (3, 10, 0, 0, 0, 0, 100), THR)
    # echo 1 of the square's right edge (x=30) with wave=0 lands at column 40
    col = out[22:42, 40]
    assert np.all(col == 0.0)


def test_echoes_are_continuous_at_partial_breath():
    img = _subject_square()
    out = entry().func(img.copy(), (3, 10, 0, 0, 0, 0, 68), THR)
    col = out[22:42, 40]   # echo 1 solid: visible = 0.68 * 3 = 2.04 >= 1
    assert np.all(col == 0.0)


def test_wave_frequency_changes_pixels():
    img = _subject_square()
    a = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 100, 10), THR)
    b = entry().func(img.copy(), (6, 10, 8, 0, 0, 0, 100, 60), THR)
    assert np.any(a != b)


def test_default_wave_frequency_is_backward_compatible():
    img = _subject_square()
    seven = entry().func(img.copy(), (6, 10, 8, 0, 20, 30, 68), THR)
    eight = entry().func(img.copy(), (6, 10, 8, 0, 20, 30, 68, 10), THR)
    np.testing.assert_array_equal(seven, eight)
