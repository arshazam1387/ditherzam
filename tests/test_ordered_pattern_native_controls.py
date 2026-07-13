import numpy as np
import pytest

from ditherzam.dithering import registry
from ditherzam.dithering.kernels import ordered, pattern  # noqa: F401


@pytest.fixture(scope="module")
def texture():
    rng = np.random.default_rng(817)
    ramp_x = np.linspace(20, 235, 37, dtype=np.float32)[None, :]
    ramp_y = np.linspace(-75, 75, 31, dtype=np.float32)[:, None]
    return np.clip(ramp_x + ramp_y + rng.normal(0, 38, (31, 37)), 0, 255).astype(np.float32)


ORDERED_DEFAULTS = {
    "Bayer-Matrix 2x2": (100, 0, 0, 0, 0),
    "Bayer-Matrix 8x8": (100, 0, 0, 0, 0),
    "Bayer-Matrix 16x16": (100, 0, 0, 0, 0),
    "Bayer-Ordered": (100, 0, 0, 0, 0),
    "Bayer-Void": (10, 100, 0, 0, 0),
    "Random Ordered": (0, 100, 0, 1, 1),
    "Bit Tone": (1, 100, 0, 0, 0),
    "Mosaic": (10, 100, 0, 0, 0),
    "Modulated Bayer Dither": (2, 100, 0, 0, 0),
    "Cluster-Dot": (100, 0, 0, 0, 0),
    "Halftone-Ordered": (6, 100, 0, 0, 0),
}

PATTERN_DEFAULTS = {
    "Checkers - Small": (2, 100, 0, 0, 0),
    "Checkers - Medium": (4, 100, 0, 0, 0),
    "Checkers - Large": (8, 100, 0, 0, 0),
    "Diamond": (8, 100, 0, 0, 0),
    "Gridlock/Traffic": (6, 100, 0, 0, 0),
    "Print Pattern": (6, 100, 0, 0, 0),
    "Block Tone": (4, 100, 0, 0, 0),
    "Stippling": (1, 100, 0, 0, 0),
    "Crosshatch": (4, 100, 0, 0, 0),
    "Dot Screen": (6, 100, 0, 0, 0),
    "Line Screen": (6, 100, 0, 0, 0),
}


@pytest.mark.parametrize("name,cell", [
    ("Checkers - Small", 2),
    ("Checkers - Medium", 4),
    ("Checkers - Large", 8),
])
def test_checkers_default_threshold_produces_visible_checkerboard(name, cell):
    flat = np.full((cell * 4, cell * 4), 127.5, np.float32)
    out = registry.get_entry(name).func(flat, PATTERN_DEFAULTS[name], 127.5)
    assert set(np.unique(out)) == {0.0, 255.0}
    assert np.all(out[:cell, :cell] == 255.0)
    assert np.all(out[:cell, cell:cell * 2] == 0.0)


@pytest.mark.parametrize("name, defaults", list(ORDERED_DEFAULTS.items()) +
                         list(PATTERN_DEFAULTS.items()))
def test_explicit_defaults_preserve_legacy_none_output(name, defaults, texture):
    entry = registry.get_entry(name)
    assert len(entry.param_sliders) == 5
    np.testing.assert_array_equal(
        entry.func(texture, defaults, 128), entry.func(texture, None, 128))


CHANGES = {
    "pattern_scale_slider": 11,
    "pattern_contrast_slider": 45,
    "pattern_bias_slider": 55,
    "pattern_skew_slider": 1,
    "pattern_phase_y_slider": 2,
    "dither_parameter_slider": 3,
    "matrix_size_slider": 4,
    "threshold_contrast_slider": 40,
    "threshold_bias_slider": 55,
    "matrix_rotation_slider": 2,
    "matrix_offset_x_slider": 3,
    "matrix_offset_y_slider": 1,
    "tile_threshold_scale_slider": 40,
    "random_seed_slider": 19,
    "grain_width_slider": 4,
    "grain_height_slider": 3,
}


@pytest.mark.parametrize("name, defaults", list(ORDERED_DEFAULTS.items()) +
                         list(PATTERN_DEFAULTS.items()))
def test_every_native_control_changes_pixels(name, defaults, texture):
    entry = registry.get_entry(name)
    baseline = entry.func(texture, defaults, 128)
    for index, key in enumerate(entry.param_sliders):
        changed = list(defaults)
        changed[index] = CHANGES[key]
        actual = entry.func(texture, tuple(changed), 128)
        assert np.any(actual != baseline), (name, key)


@pytest.mark.parametrize("name,defaults", list(ORDERED_DEFAULTS.items()) +
                         list(PATTERN_DEFAULTS.items()))
def test_default_patterns_preserve_black_and_white_endpoints(name, defaults):
    entry = registry.get_entry(name)
    black = np.zeros((64, 64), np.float32)
    white = np.full((64, 64), 255.0, np.float32)
    if name != "Block Tone":
        # Block Tone's classic round-dot leaves cell corners white on pure
        # black (inscribed-circle geometry); that look is kept by request.
        assert np.all(entry.func(black, defaults, 127.5) == 0.0), name
    assert np.all(entry.func(white, defaults, 127.5) == 255.0), name


def test_block_tone_default_has_useful_tonal_progression():
    entry = registry.get_entry("Block Tone")
    densities = []
    for tone in (0.0, 64.0, 127.5, 192.0, 255.0):
        flat = np.full((64, 64), tone, np.float32)
        densities.append(entry.func(flat, PATTERN_DEFAULTS["Block Tone"], 127.5).mean())
    # Classic round-dot look: pure black keeps white cell corners, so the
    # floor is dark grey rather than 0, and the small default cell only
    # resolves three density steps across these five tones.
    assert densities[0] < 96.0
    assert densities[-1] == 255.0
    assert densities == sorted(densities)
    assert len(set(densities)) >= 3


def test_related_default_styles_are_not_duplicate_effects(texture):
    # Bayer-Ordered / Bayer-Matrix 4x4 / Bit Tone intentionally share the
    # classic 4x4 Bayer output at defaults (restored by request); they only
    # diverge once their native sliders move.
    halftone = registry.get_entry("Halftone-Ordered").func(
        texture, ORDERED_DEFAULTS["Halftone-Ordered"], 128)
    dot_screen = registry.get_entry("Dot Screen").func(
        texture, PATTERN_DEFAULTS["Dot Screen"], 128)
    assert np.any(halftone != dot_screen)
