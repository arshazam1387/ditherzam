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


def _drip_scene():
    img = _subject_square(128)
    img[50:80, 40:90] = 20.0
    return img


def test_drips_fall_below_subject_and_fade():
    out = entry().func(_drip_scene(), (0, 10, 0, 0, 100, 0, 0, 10), THR)
    near = out[82:100, :]                       # just below the wide block
    tail = out[110:, :]
    assert (near == 0.0).sum() > 0
    assert (near == 0.0).mean() > (tail == 0.0).mean()   # dissolves with distance


def test_drips_never_above_subject():
    out = entry().func(_drip_scene(), (0, 10, 8, 0, 100, 0, 0, 10), THR)
    assert np.all(out[:22, :] == 255.0)         # nothing above the topmost subject row


def test_wave_phase_sways_drips():
    a = entry().func(_drip_scene(), (0, 10, 8, 0, 100, 0, 0, 10), THR)
    b = entry().func(_drip_scene(), (0, 10, 8, 180, 100, 0, 0, 10), THR)
    assert np.any(a[82:, :] != b[82:, :])       # drips ride Wave Phase


def test_straight_drips_at_wave_zero_confined_near_origin_columns():
    img = _drip_scene()
    out = entry().func(img.copy(), (0, 10, 0, 0, 100, 0, 0, 10), THR)
    ink_cols = {x for x in range(128) if (out[82:, x] == 0.0).any()}
    subject_cols = {x for x in range(128) if (img[:, x] < 128.0).any()}
    widened = set()
    for x in subject_cols:
        widened.update((x - 1, x, x + 1))       # 3px taper near the body
    assert ink_cols <= widened


def test_drips_off_at_zero_and_only_from_subject_columns():
    blank = np.full((64, 64), 230.0, dtype=np.float32)
    out = entry().func(blank.copy(), (0, 10, 0, 0, 100, 0, 0, 10), THR)
    assert np.all(out == 255.0)
    off = entry().func(_drip_scene(), (0, 10, 0, 0, 0, 0, 0, 10), THR)
    assert np.all(off[82:, :] == 255.0)


def test_drips_independent_of_breath():
    b0 = entry().func(_drip_scene(), (0, 10, 0, 0, 100, 0, 0, 10), THR)
    b100 = entry().func(_drip_scene(), (0, 10, 0, 0, 100, 0, 100, 10), THR)
    np.testing.assert_array_equal(b0[82:, :], b100[82:, :])


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


def test_each_native_control_changes_pixels():
    rng = np.random.default_rng(7)
    img = np.full((48, 48), 235.0, dtype=np.float32)
    img[4:24, :] = rng.integers(0, 256, (20, 48)).astype(np.float32)  # textured subject band, 24 rows of drop below
    e = entry()
    base = e.func(img.copy(), DEFAULTS, THR)
    alternatives = (12, 20, 20, 180, 80, 90, 100, 60)
    assert len(e.param_sliders) == 8
    for index, value in enumerate(alternatives):
        params = list(DEFAULTS)
        params[index] = value
        changed = e.func(img.copy(), tuple(params), THR)
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


def test_echoes_only_right_of_trailing_edge():
    # Discriminator: the old any-boundary kernel echoed BOTH edges rightward.
    # With spacing 4, count 6, wave 0, its left-edge boundaries at sx=8,9
    # (bg vs subject at sx+2=10,11) landed echoes at x = sx + 4n, inking the
    # background band x=31..33 in subject rows (x=32: n=6, sx=8; x=33: n=6,
    # sx=9 / n=1, sx=29).  Trailing-edge semantics fire only at sx=30,
    # echoing to x = 30 + 4n = 34,38,... so 31..33 must stay white.
    img = _subject_square()
    out = entry().func(img.copy(), (6, 4, 0, 0, 0, 0, 100), THR)
    assert np.all(out[22:42, 31:34] == 255.0)
    # Forward pin (white under the old kernel too): echoes displace
    # rightward only — min echo x = 8 + spacing >= 12 — so ink may never
    # appear strictly left of the square.
    assert np.all(out[:, 0:10] == 255.0)


def test_echoes_hug_subject_vertical_extent():
    img = _subject_square()
    out = entry().func(img.copy(), (3, 10, 0, 0, 0, 0, 100), THR)
    # Discriminator: the old kernel's yd=y+2 top-boundary test fired at
    # y=20,21 (bg vs subject at y+2=22,23) for sx=10..30, inking
    # x = sx + 10n within 31: (e.g. y=20, x=40, n=1, sx=30).  Trailing-edge
    # echoes need img[y,sx] < thr, impossible above the square.
    assert np.all(out[20:22, 31:] == 255.0)
    # Forward pins (old kernel was already white here): far field above,
    # and rows below the square — the old bottom boundary fired at y=40,41
    # (inside the body), so rows 42+ were clean for it too.
    assert np.all(out[0:20, 31:] == 255.0)
    assert np.all(out[42:, 31:] == 255.0)


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


def test_wave_phase_travels_echoes_toward_subject():
    img = _subject_square()
    a = entry().func(img.copy(), (3, 10, 0, 0, 0, 0, 100, 10), THR)
    b = entry().func(img.copy(), (3, 10, 0, 180, 0, 0, 100, 10), THR)
    assert np.any(a != b)                       # RED today: wave=0 makes phase a dead slider
    assert np.all(a[22:42, 40] == 0.0)          # phase 0: first echo one spacing out
    assert np.all(a[22:42, 35] == 255.0)
    assert np.all(b[22:42, 35] == 0.0)          # phase 180: line traveled half a spacing inward


def test_wave_phase_full_cycle_is_seamless():
    img = _subject_square()
    a = entry().func(img.copy(), (3, 10, 0, 0, 0, 0, 100, 10), THR)
    c = entry().func(img.copy(), (3, 10, 0, 360, 0, 0, 100, 10), THR)
    np.testing.assert_array_equal(a, c)         # wave=0: 360 wraps exactly to 0
