import numpy as np
from ditherzam.dithering import registry
from ditherzam.dithering.pipeline import apply_dither
from ditherzam.render import RenderPipeline, RenderSettings
from ditherzam.animation.temporal import temporal_noise


def _ramp():
    return np.tile(np.linspace(0, 255, 16, dtype=np.float32), (16, 1))


def _flat():
    return np.full((16, 16), 128.0, dtype=np.float32)


def test_apply_dither_none_field_is_byte_identical():
    img = _ramp()
    base = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                        luminance_threshold=50, params={}, registry=registry)
    with_none = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                             luminance_threshold=50, params={}, registry=registry,
                             threshold_field=None)
    np.testing.assert_array_equal(base, with_none)             # backward compatible


def test_field_changes_dither_output():
    img = _flat()
    fld = temporal_noise(0, (16, 16), "static", 80.0, seed=1)
    base = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                        luminance_threshold=50, params={}, registry=registry)
    perturbed = apply_dither(img, style="Bayer-Matrix 4x4", scale=1,
                             luminance_threshold=50, params={}, registry=registry,
                             threshold_field=fld)
    assert not np.array_equal(base, perturbed)


def test_field_resized_to_downscaled_grid():
    # scale=4 -> 16/4 = 4x4 small grid; pass a full-res 16x16 field, must not error.
    img = _ramp()
    fld = temporal_noise(0, (16, 16), "plasma", 40.0)
    out = apply_dither(img, style="Bayer-Matrix 4x4", scale=4,
                       luminance_threshold=50, params={}, registry=registry,
                       threshold_field=fld)
    assert out.shape == img.shape


def test_render_none_matches_no_field():
    p = RenderPipeline(registry)
    img = _ramp()
    s = RenderSettings(style="Bayer-Matrix 4x4", scale=1)
    a = p.render(img, s)
    b = p.render(img, s, temporal_field=None)
    np.testing.assert_array_equal(a, b)                        # backward compatible


def test_render_field_changes_output():
    p = RenderPipeline(registry)
    img = _flat()
    s = RenderSettings(style="Bayer-Matrix 4x4", scale=1)
    fld = temporal_noise(0, (16, 16), "static", 80.0, seed=1)
    a = p.render(img, s)
    b = p.render(img, s, temporal_field=fld)
    assert not np.array_equal(a, b)
