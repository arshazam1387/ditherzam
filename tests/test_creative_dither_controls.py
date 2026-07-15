import numpy as np

from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither


def _render(img, style, params=None):
    return apply_dither(
        img, style=style, scale=1, luminance_threshold=50,
        params=params or {}, registry=registry, levels=2,
    )


def test_creative_defaults_are_byte_identical_for_every_style():
    img = np.random.default_rng(12).uniform(0, 255, (13, 17)).astype(np.float32)
    defaults = {
        "creative_mix": 100, "creative_orientation": 0,
        "creative_offset_x": 0, "creative_offset_y": 0,
        "creative_jitter": 0, "creative_seed": 0,
    }
    for style in registry.list_dithers():
        np.testing.assert_array_equal(_render(img, style), _render(img, style, defaults))


def test_mix_has_true_dry_and_wet_endpoints():
    img = np.arange(99, dtype=np.float32).reshape(9, 11) * np.float32(2.0)
    wet = _render(img, "Bayer-Matrix 4x4")
    np.testing.assert_array_equal(
        _render(img, "Bayer-Matrix 4x4", {"creative_mix": 0}), img)
    np.testing.assert_array_equal(
        _render(img, "Bayer-Matrix 4x4", {"creative_mix": 100}), wet)


def test_orientation_and_offsets_keep_shape_and_change_pattern():
    img = np.tile(np.linspace(20, 235, 19, dtype=np.float32), (15, 1))
    base = _render(img, "Line Screen", {"dither_parameter_slider": 5})
    changed = _render(img, "Line Screen", {
        "dither_parameter_slider": 5, "creative_orientation": 1,
        "creative_offset_x": 3, "creative_offset_y": -2,
    })
    assert changed.shape == img.shape
    assert not np.array_equal(changed, base)


def test_jitter_is_seeded_repeatable_and_seed_changes_result():
    img = np.full((32, 32), 127.0, np.float32)
    a = _render(img, "Bayer-Matrix 4x4", {"creative_jitter": 80, "creative_seed": 4})
    b = _render(img, "Bayer-Matrix 4x4", {"creative_jitter": 80, "creative_seed": 4})
    c = _render(img, "Bayer-Matrix 4x4", {"creative_jitter": 80, "creative_seed": 5})
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
