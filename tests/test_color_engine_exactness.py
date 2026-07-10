import numpy as np
import pytest

from ditherzam.color.engine import (
    ColorEngine,
    _BAYER4,
    _floyd_steinberg_rgb_njit,
    _ordered_rgb_njit,
)
from ditherzam.color.palette import Palette, builtin_palettes
from ditherzam.color.ramp import RAMP_MODES, build_ramp


def _ramp_reference(image: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    """Frozen pre-optimization ramp expression used as a differential oracle."""
    arr = np.asarray(image, dtype=np.float32)
    gray = (arr if arr.ndim == 2 else
            arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    depth = ramp.shape[0]
    if depth == 1:
        level = np.zeros(gray.shape, dtype=np.int64)
    else:
        level = np.clip(np.round(gray / 255.0 * (depth - 1)), 0, depth - 1).astype(np.int64)
    return np.clip(ramp[level], 0, 255).astype(np.uint8)


def _floyd_steinberg_rgb_reference(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Frozen pre-optimization implementation used as a differential oracle."""
    h, w = rgb.shape[:2]
    work = rgb.astype(np.float32).copy()
    out = np.empty((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            old = work[y, x].copy()
            diff = pal - old
            idx = int((diff * diff).sum(axis=1).argmin())
            new = pal[idx]
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                work[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x - 1 >= 0:
                    work[y + 1, x - 1] += err * (3.0 / 16.0)
                work[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    work[y + 1, x + 1] += err * (1.0 / 16.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _palette(colors: np.ndarray) -> Palette:
    return Palette.from_list("exactness", colors.tolist())


def _ordered_rgb_reference(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Frozen pre-optimization ordered implementation."""
    k = pal.shape[0]
    spread = 255.0 / max(1, k - 1)
    h, w = rgb.shape[:2]
    mh, mw = _BAYER4.shape
    offset = _BAYER4[np.arange(h)[:, None] % mh,
                     np.arange(w)[None, :] % mw]
    biased = rgb + offset[:, :, None] * spread
    diff = biased.astype(np.float32)[:, :, None, :] - pal[None, None, :, :]
    dist = (diff * diff).sum(axis=-1)
    idx = dist.argmin(axis=-1)
    return np.clip(pal[idx], 0, 255).astype(np.uint8)


@pytest.mark.parametrize("mapping", RAMP_MODES)
@pytest.mark.parametrize("depth", [1, 2, 64])
@pytest.mark.parametrize("kind", ["gray", "rgb", "noncontiguous"])
def test_ramp_matches_frozen_reference(mapping, depth, kind):
    rng = np.random.default_rng(4100 + depth)
    pal = builtin_palettes()["pico8"]
    if kind == "gray":
        image = rng.uniform(-40, 295, size=(13, 17)).astype(np.float32)
    else:
        backing = rng.uniform(-40, 295, size=(13, 34, 4)).astype(np.float32)
        image = backing[:, ::2, :] if kind == "noncontiguous" else backing[:, :17, :]
    ramp = build_ramp(pal, depth, mapping)

    got = ColorEngine(pal, mode="ramp", depth=depth, mapping=mapping).map(image)

    np.testing.assert_array_equal(got, _ramp_reference(image, ramp))


@pytest.mark.parametrize("depth", [2, 3, 64])
def test_ramp_preserves_bankers_rounding_at_half_boundaries(depth):
    pal = builtin_palettes()["pico8"]
    boundaries = ((np.arange(depth - 1, dtype=np.float32) + np.float32(0.5))
                  * np.float32(255.0 / (depth - 1)))
    gray = np.stack([np.nextafter(boundaries, -np.inf), boundaries,
                     np.nextafter(boundaries, np.inf)], axis=1)
    ramp = build_ramp(pal, depth, "interpolated")

    got = ColorEngine(pal, mode="ramp", depth=depth,
                      mapping="interpolated").map(gray)

    np.testing.assert_array_equal(got, _ramp_reference(gray, ramp))


def test_ramp_fused_path_skips_rgb_repeat_and_post_map_clamp(monkeypatch):
    import ditherzam.color.engine as engine_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ramp mapping must return directly from its fused kernel")

    monkeypatch.setattr(engine_module, "_to_rgb", forbidden)
    monkeypatch.setattr(engine_module, "clamp_u8", forbidden)
    gray = np.linspace(0, 255, 63, dtype=np.float32).reshape(7, 9)

    out = ColorEngine(builtin_palettes()["gameboy"], mode="ramp", depth=4).map(gray)

    assert out.shape == (7, 9, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (13, 11)])
@pytest.mark.parametrize("palette_size", [1, 2, 4, 16, 64])
def test_ordered_matches_frozen_reference(shape, palette_size):
    rng = np.random.default_rng(9000 + palette_size + shape[0] * 100 + shape[1])
    rgb = rng.uniform(-40, 295, size=(*shape, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(palette_size, 3)).astype(np.float32)
    pal = _palette(colors)

    got = ColorEngine(pal, mode="ordered").map(rgb)
    expected = _ordered_rgb_reference(rgb, pal.colors.astype(np.float32))

    np.testing.assert_array_equal(got, expected)


def test_ordered_matches_reference_for_non_contiguous_input():
    rng = np.random.default_rng(93)
    backing = rng.uniform(-20, 275, size=(15, 22, 3)).astype(np.float32)
    rgb = backing[:, ::2, :]
    assert not rgb.flags.c_contiguous
    pal = builtin_palettes()["pico8"]

    got = ColorEngine(pal, mode="ordered").map(rgb)
    expected = _ordered_rgb_reference(rgb, pal.colors.astype(np.float32))

    np.testing.assert_array_equal(got, expected)


def test_ordered_keeps_first_palette_color_on_distance_tie():
    pal = _palette(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32))
    rgb = np.array([[[1, 0, 0]]], dtype=np.float32)
    # Bayer (0, 0) is -119.53125 for a two-color palette. Compensate so
    # the biased red channel is exactly halfway between the two swatches.
    rgb[0, 0, 0] -= _BAYER4[0, 0] * np.float32(255.0)

    assert ColorEngine(pal, mode="ordered").map(rgb).tolist() == [[[0, 0, 0]]]


def test_compiled_ordered_kernel_is_directly_exact():
    rgb = np.array([[[1.5, 127.25, 254.75], [90.5, -2.0, 280.0]]], np.float32)
    pal = np.array([[0, 0, 0], [127, 128, 129], [255, 255, 255]], np.float32)

    got = _ordered_rgb_njit(rgb, pal, _BAYER4)
    expected = _ordered_rgb_reference(rgb, pal)

    np.testing.assert_array_equal(got, expected)


def test_ordered_fused_path_skips_index_frame_and_post_map_clamp(monkeypatch):
    import ditherzam.color.engine as engine_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ordered mapping must return directly from its fused kernel")

    monkeypatch.setattr(engine_module, "nearest_indices", forbidden)
    monkeypatch.setattr(engine_module, "clamp_u8", forbidden)
    rgb = np.full((7, 9, 3), 127.0, dtype=np.float32)

    out = ColorEngine(builtin_palettes()["gameboy"], mode="ordered").map(rgb)

    assert out.shape == (7, 9, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (13, 11)])
@pytest.mark.parametrize("palette_size", [1, 2, 4, 16, 64])
def test_diffused_matches_frozen_reference(shape, palette_size):
    rng = np.random.default_rng(7000 + palette_size + shape[0] * 100 + shape[1])
    rgb = rng.uniform(-40, 295, size=(*shape, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(palette_size, 3)).astype(np.float32)
    pal = _palette(colors)

    got = ColorEngine(pal, mode="diffused").map(rgb)
    expected = _floyd_steinberg_rgb_reference(rgb, pal.colors.astype(np.float32))

    np.testing.assert_array_equal(got, expected)


def test_diffused_matches_reference_for_non_contiguous_input():
    rng = np.random.default_rng(91)
    backing = rng.uniform(-20, 275, size=(15, 22, 3)).astype(np.float32)
    rgb = backing[:, ::2, :]
    assert not rgb.flags.c_contiguous
    pal = builtin_palettes()["pico8"]

    got = ColorEngine(pal, mode="diffused").map(rgb)
    expected = _floyd_steinberg_rgb_reference(rgb, pal.colors.astype(np.float32))

    np.testing.assert_array_equal(got, expected)


def test_diffused_keeps_first_palette_color_on_distance_tie():
    pal = _palette(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32))
    rgb = np.array([[[1, 0, 0]]], dtype=np.float32)

    assert ColorEngine(pal, mode="diffused").map(rgb).tolist() == [[[0, 0, 0]]]


def test_compiled_diffusion_kernel_is_directly_exact():
    rgb = np.array([[[1.5, 127.25, 254.75], [90.5, -2.0, 280.0]]], np.float32)
    pal = np.array([[0, 0, 0], [127, 128, 129], [255, 255, 255]], np.float32)

    got = _floyd_steinberg_rgb_njit(rgb, pal)
    expected = _floyd_steinberg_rgb_reference(rgb, pal)

    np.testing.assert_array_equal(np.clip(got, 0, 255).astype(np.uint8), expected)


@pytest.mark.parametrize("palette_name", ["grayscale", "gameboy", "cga", "pico8"])
def test_diffused_builtin_palette_matches_reference(palette_name):
    rng = np.random.default_rng(122)
    rgb = rng.uniform(0, 255, size=(9, 14, 3)).astype(np.float32)
    pal = builtin_palettes()[palette_name]

    got = ColorEngine(pal, mode="diffused").map(rgb)
    expected = _floyd_steinberg_rgb_reference(rgb, pal.colors.astype(np.float32))

    np.testing.assert_array_equal(got, expected)
