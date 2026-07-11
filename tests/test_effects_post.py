import numpy as np
import pytest
from PIL import Image, ImageFilter
from ditherzam.effects.post import (
    EFFECTS, blur, sharpen, chromatic_aberration, jpeg_glitch, epsilon_glow,
)


def rand_img():
    return np.random.RandomState(0).randint(0, 256, (16, 16, 3), np.uint8)


def gray_img(v):
    return np.full((16, 16, 3), v, np.uint8)


def _epsilon_glow_reference(rgb_u8, threshold=64.0, smoothing=32.0,
                            radius=8.0, intensity=1.0, epsilon=0.4,
                            falloff=0.5, distance_scale=1.0, aspect=1.0):
    """Frozen pre-optimization implementation used as a differential oracle."""
    if intensity <= 0:
        return rgb_u8
    base = rgb_u8.astype(np.float32)
    lum = 0.299 * base[..., 0] + 0.587 * base[..., 1] + 0.114 * base[..., 2]
    edge0, edge1 = float(threshold), float(threshold) + float(smoothing)
    if edge1 <= edge0:
        mask = (lum >= edge0).astype(np.float32)
    else:
        t = np.clip((lum - edge0) / (edge1 - edge0), 0.0, 1.0)
        mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    src = base * mask[..., None]

    def aniso_blur(src_f, sigma):
        h, w = src_f.shape[:2]
        ax = max(float(aspect), 1e-3)
        new_w = max(1, int(round(w / ax)))
        img = Image.fromarray(np.clip(src_f, 0, 255).astype(np.uint8))
        if new_w != w:
            img = img.resize((new_w, h), Image.BILINEAR)
        img = img.filter(ImageFilter.GaussianBlur(float(max(sigma, 0.0))))
        if new_w != w:
            img = img.resize((w, h), Image.BILINEAR)
        return np.asarray(img, np.float32)

    r = max(float(radius) * float(distance_scale), 0.0)
    weights = np.array([1.0, 0.6, 0.35], np.float32)
    f = float(np.clip(falloff, 0.0, 1.0))
    weights *= np.array([1.0 + f, 1.0, 1.0 - 0.5 * f], np.float32)
    weights /= weights.sum()
    glow = np.zeros_like(base)
    for scale, weight in zip((0.5, 1.0, 2.0), weights):
        glow += aniso_blur(src, r * scale + 1e-3) * float(weight)
    eps = float(np.clip(epsilon, 0.0, 1.0))
    core = base * (mask[..., None] ** (1.0 + eps * 8.0))
    core_glow = aniso_blur(core, max(r * 0.25, 1e-3)) * eps
    return np.clip(base + (glow + core_glow) * float(intensity), 0, 255).astype(np.uint8)


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
        (epsilon_glow, {"threshold": 32, "radius": 3, "intensity": 1.0}),
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


def test_epsilon_glow_intensity_zero_is_identity():
    x = rand_img()
    np.testing.assert_array_equal(epsilon_glow(x, intensity=0.0), x)


def test_epsilon_glow_shape_dtype_all_params():
    x = rand_img()
    out = epsilon_glow(x, threshold=40, smoothing=20, radius=5, intensity=1.0,
                       epsilon=0.5, falloff=0.4, distance_scale=1.2, aspect=1.5)
    assert out.shape == x.shape and out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_epsilon_glow_higher_threshold_fewer_glow_pixels():
    # vertical luminance gradient 0..255 across the width
    grad = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    x = np.stack([grad, grad, grad], axis=-1)
    changed_low = int(np.count_nonzero(np.any(
        epsilon_glow(x, threshold=32, radius=4, intensity=1.0) != x, axis=-1)))
    changed_high = int(np.count_nonzero(np.any(
        epsilon_glow(x, threshold=200, radius=4, intensity=1.0) != x, axis=-1)))
    assert changed_high < changed_low          # higher threshold => fewer glowing pixels


def test_epsilon_glow_smoothing_zero_hard_cut_no_error():
    x = rand_img()
    out = epsilon_glow(x, threshold=100, smoothing=0, radius=3, intensity=1.0)
    assert out.dtype == np.uint8 and np.isfinite(out.astype(np.float64)).all()


def test_epsilon_glow_epsilon_raises_core_brightness():
    # one bright pixel over a dark field; brighter epsilon => brighter center
    x = np.zeros((32, 32, 3), np.uint8)
    x[16, 16] = 255
    dim = epsilon_glow(x, threshold=50, radius=6, intensity=1.0, epsilon=0.0)
    hot = epsilon_glow(x, threshold=50, radius=6, intensity=1.0, epsilon=1.0)
    assert int(hot[16, 16].sum()) >= int(dim[16, 16].sum())


def test_epsilon_glow_grayscale_input_ok():
    x = gray_img(180)
    out = epsilon_glow(x, threshold=50, radius=3, intensity=0.8)
    assert out.shape == x.shape and out.dtype == np.uint8


@pytest.mark.parametrize("params", [
    {},
    {"threshold": 40, "smoothing": 20, "radius": 5, "intensity": 1.0,
     "epsilon": 0.5, "falloff": 0.4, "distance_scale": 1.2, "aspect": 1.5},
    {"threshold": -20, "smoothing": 0, "radius": 0, "intensity": 2.5,
     "epsilon": 1.5, "falloff": -1, "distance_scale": -2, "aspect": 0.01},
])
def test_epsilon_glow_matches_frozen_reference(params):
    x = np.random.RandomState(81).randint(0, 256, (19, 23, 3), np.uint8)
    before = x.copy()
    np.testing.assert_array_equal(epsilon_glow(x, **params),
                                  _epsilon_glow_reference(x, **params))
    np.testing.assert_array_equal(x, before)


def test_epsilon_glow_matches_reference_for_noncontiguous_input():
    backing = np.random.RandomState(82).randint(0, 256, (24, 34, 3), np.uint8)
    x = backing[::2, 1::2]
    assert not x.flags.c_contiguous
    params = {"threshold": 91, "smoothing": 17, "radius": 3.5,
              "intensity": 0.7, "epsilon": 0.2, "falloff": 0.8,
              "distance_scale": 1.4, "aspect": 2.0}
    before = backing.copy()
    np.testing.assert_array_equal(epsilon_glow(x, **params),
                                  _epsilon_glow_reference(x, **params))
    np.testing.assert_array_equal(backing, before)


def test_chromatic_before_glow_differs_from_after():
    # emergent stack-order interaction: CA then Glow != Glow then CA
    x = rand_img()
    ca_then_glow = epsilon_glow(chromatic_aberration(x, shift=2),
                                threshold=40, radius=3, intensity=1.0)
    glow_then_ca = chromatic_aberration(
        epsilon_glow(x, threshold=40, radius=3, intensity=1.0), shift=2)
    assert not np.array_equal(ca_then_glow, glow_then_ca)
