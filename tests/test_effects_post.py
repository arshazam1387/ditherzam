import numpy as np
import pytest
from ditherzam.effects.post import (
    EFFECTS, blur, sharpen, chromatic_aberration, jpeg_glitch, epsilon_glow,
)


def rand_img():
    return np.random.RandomState(0).randint(0, 256, (16, 16, 3), np.uint8)


def gray_img(v):
    return np.full((16, 16, 3), v, np.uint8)


def test_all_five_effects_registered():
    for k in ("Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch", "Epsilon Glow"):
        assert k in EFFECTS
    assert set(EFFECTS) == {
        "Blur", "Sharpen", "Chromatic Aberration", "JPEG Glitch", "Epsilon Glow"}
    assert all(callable(fn) for fn in EFFECTS.values())


def test_shape_and_dtype_preserved():
    x = rand_img()
    for fn, kw in [
        (blur, {"radius": 2}),
        (sharpen, {"amount": 1.5}),
        (chromatic_aberration, {"shift": 2}),
        (jpeg_glitch, {"quality": 10}),
        (epsilon_glow, {"radius": 3, "strength": 0.5}),
    ]:
        out = fn(x, **kw)
        assert out.shape == x.shape and out.dtype == np.uint8


def test_blur_zero_is_identity():
    x = rand_img()
    np.testing.assert_array_equal(blur(x, 0), x)


def test_blur_uniform_unchanged():
    # Gaussian blur of a flat field is the same flat field.
    x = gray_img(100)
    np.testing.assert_array_equal(blur(x, 3), x)


def test_sharpen_uniform_is_identity():
    # unsharp mask on a flat field: (a - b) == 0, so output == input.
    x = gray_img(100)
    np.testing.assert_array_equal(sharpen(x, 1.5), x)


def test_chromatic_aberration_shifts_red_right():
    x = np.zeros((4, 8, 3), np.uint8)
    x[:, 3, 0] = 255                                  # red column at x=3
    out = chromatic_aberration(x, shift=2)
    assert out[:, 5, 0].max() == 255                  # red moved +2 (right)
    assert out[:, 3, 0].max() == 0                    # vacated by the roll


def test_chromatic_aberration_shifts_blue_left():
    x = np.zeros((4, 8, 3), np.uint8)
    x[:, 5, 2] = 255                                  # blue column at x=5
    out = chromatic_aberration(x, shift=2)
    assert out[:, 3, 2].max() == 255                  # blue moved -2 (left)


def test_chromatic_aberration_leaves_green_untouched():
    x = rand_img()
    out = chromatic_aberration(x, shift=3)
    np.testing.assert_array_equal(out[..., 1], x[..., 1])


def test_jpeg_glitch_preserves_shape_and_degrades():
    x = rand_img()
    out = jpeg_glitch(x, quality=5)
    assert out.shape == x.shape and out.dtype == np.uint8
    assert not np.array_equal(out, x)                 # lossy round-trip changed it


def test_jpeg_glitch_clamps_quality():
    x = rand_img()
    # out-of-range quality must not raise (clamped into 1..100)
    assert jpeg_glitch(x, quality=0).shape == x.shape
    assert jpeg_glitch(x, quality=999).shape == x.shape


def test_epsilon_glow_brightens_uniform_field():
    # glow(blur)==100 on a flat field; out = clip(100 + 100*0.5) = 150
    x = gray_img(100)
    out = epsilon_glow(x, radius=3, strength=0.5)
    assert np.all(out == 150)


def test_epsilon_glow_clips_to_255():
    x = gray_img(200)
    out = epsilon_glow(x, radius=2, strength=1.0)      # 200 + 200 -> clipped 255
    assert out.max() == 255 and out.dtype == np.uint8
