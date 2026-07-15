from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ditherzam.masking.composite import (
    CompositeContext,
    MaskCompositeError,
    composite_masked,
    flatten_rgba_white,
)
from ditherzam.masking.settings import OutsideMode


def _rgba(rgb, alpha=255):
    value = np.empty((1, 1, 4), dtype=np.uint8)
    value[0, 0, :3] = rgb
    value[0, 0, 3] = alpha
    return value


@pytest.mark.parametrize(
    ("outside", "expected"),
    [
        (OutsideMode.ORIGINAL, [100, 110, 120]),
        (OutsideMode.WHITE, [255, 255, 255]),
        (OutsideMode.BLACK, [0, 0, 0]),
    ],
)
def test_zero_mask_returns_opaque_outside_as_rgb(outside, expected):
    result = composite_masked(
        _rgba([10, 20, 30])[..., :3], _rgba([100, 110, 120]),
        np.zeros((1, 1), np.float32), outside,
    )
    assert result.shape == (1, 1, 3)
    assert result[0, 0].tolist() == expected


@pytest.mark.parametrize("outside", list(OutsideMode))
def test_one_mask_is_exact_render_for_every_outside(outside):
    rendered = np.array([[[1, 127, 254], [9, 20, 31]]], dtype=np.uint8)
    source = np.array([[[7, 8, 9, 0], [1, 2, 3, 99]]], dtype=np.uint8)
    mask = np.ones((1, 2), np.float32)
    result = composite_masked(rendered, source, mask, outside)
    assert np.array_equal(result[..., :3], rendered)
    if result.shape[2] == 4:
        assert np.array_equal(result[..., 3], np.full((1, 2), 255, np.uint8))


def test_transparent_outside_preserves_hidden_and_edge_straight_rgb():
    rendered = np.array([[[220, 40, 10], [17, 33, 99]]], dtype=np.uint8)
    source = _rgba([1, 2, 3])
    mask = np.array([[0.5, 0.0]], dtype=np.float32)
    result = composite_masked(rendered, np.repeat(source, 2, axis=1), mask,
                              OutsideMode.TRANSPARENT)
    assert result.tolist() == [[[220, 40, 10, 128], [17, 33, 99, 0]]]


def test_original_source_alpha_uses_straight_alpha_compositing():
    # 50% opaque red render over a 50%-alpha blue outside: alpha=191,
    # straight RGB = (red*128 + blue*64) / 192.
    result = composite_masked(
        _rgba([255, 0, 0])[..., :3], _rgba([0, 0, 255], 128),
        np.array([[0.5]], np.float32), OutsideMode.ORIGINAL,
    )
    assert result.tolist() == [[[170, 0, 85, 192]]]


def test_half_up_rounding_is_deterministic_in_byte_domain():
    result = composite_masked(
        _rgba([1, 2, 254])[..., :3], _rgba([0, 1, 255]),
        np.array([[0.5]], np.float32), OutsideMode.ORIGINAL,
    )
    assert result.tolist() == [[[1, 2, 254]]]


def test_flatten_rgba_white_has_no_dark_halo_and_uses_same_rounding():
    rgba = np.array([[[220, 40, 10, 128], [17, 33, 99, 0], [3, 4, 5, 255]]],
                    dtype=np.uint8)
    result = flatten_rgba_white(rgba)
    assert result.tolist() == [[[237, 147, 132], [255, 255, 255], [3, 4, 5]]]


def test_transparent_edge_flattens_without_halo_on_dark_background():
    rendered = _rgba([240, 120, 60])[..., :3]
    transparent = composite_masked(rendered, _rgba([0, 0, 0]),
                                   np.array([[0.25]], np.float32),
                                   OutsideMode.TRANSPARENT)
    alpha = int(transparent[0, 0, 3])
    dark = ((transparent[0, 0, :3].astype(np.uint32) * alpha + 127) // 255).astype(np.uint8)
    assert transparent[0, 0, :3].tolist() == [240, 120, 60]
    assert dark.tolist() == [60, 30, 15]


def test_grayscale_derived_canonical_source_composites_exactly():
    gray = np.array([[37]], dtype=np.uint8)
    source = np.dstack((gray, gray, gray, np.full_like(gray, 255)))
    result = composite_masked(_rgba([237, 137, 37])[..., :3], source,
                              np.zeros((1, 1), np.float32), OutsideMode.ORIGINAL)
    assert result.tolist() == [[[37, 37, 37]]]


def test_inputs_are_never_mutated_and_output_is_independent():
    rendered = _rgba([4, 5, 6])[..., :3].copy()
    source = _rgba([7, 8, 9], 10)
    mask = np.array([[0.25]], np.float32)
    copies = [a.copy() for a in (rendered, source, mask)]
    result = composite_masked(rendered, source, mask, OutsideMode.ORIGINAL)
    result[...] = 0
    for actual, expected in zip((rendered, source, mask), copies):
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("rendered", "source", "mask", "outside"),
    [
        (np.zeros((1, 1, 4), np.uint8), _rgba([0, 0, 0]), np.ones((1, 1), np.float32), OutsideMode.BLACK),
        (np.zeros((1, 1, 3), np.float32), _rgba([0, 0, 0]), np.ones((1, 1), np.float32), OutsideMode.BLACK),
        (np.zeros((1, 2, 3), np.uint8), _rgba([0, 0, 0]), np.ones((1, 2), np.float32), OutsideMode.BLACK),
        (np.zeros((1, 1, 3), np.uint8), _rgba([0, 0, 0]), np.ones((2, 1), np.float32), OutsideMode.BLACK),
        (np.zeros((1, 1, 3), np.uint8), _rgba([0, 0, 0]), np.ones((1, 1), np.float64), OutsideMode.BLACK),
        (np.zeros((1, 1, 3), np.uint8), _rgba([0, 0, 0]), np.array([[1.1]], np.float32), OutsideMode.BLACK),
        (np.zeros((1, 1, 3), np.uint8), _rgba([0, 0, 0]), np.ones((1, 1), np.float32), "black"),
    ],
)
def test_invalid_contracts_fail_closed(rendered, source, mask, outside):
    with pytest.raises(MaskCompositeError):
        composite_masked(rendered, source, mask, outside)


def test_flatten_rejects_non_rgba():
    with pytest.raises(MaskCompositeError):
        flatten_rgba_white(np.zeros((1, 1, 3), np.uint8))


def test_composite_context_is_frozen_and_validated():
    source = _rgba([1, 2, 3])
    mask = np.ones((1, 1), np.float32)
    source.flags.writeable = False
    mask.flags.writeable = False
    context = CompositeContext(source, mask,
                               OutsideMode.ORIGINAL)
    with pytest.raises(FrozenInstanceError):
        context.outside_mode = OutsideMode.BLACK
    with pytest.raises(MaskCompositeError):
        CompositeContext(_rgba([1, 2, 3]), np.ones((2, 1), np.float32),
                         OutsideMode.ORIGINAL)
    with pytest.raises(ValueError):
        context.mask[0, 0] = 0.0
    with pytest.raises(ValueError):
        context.source_rgba[0, 0, 0] = 0


def test_composite_context_rejects_writable_payloads_without_copying():
    source = _rgba([1, 2, 3])
    mask = np.ones((1, 1), np.float32)
    with pytest.raises(MaskCompositeError, match="immutable"):
        CompositeContext(source, mask, OutsideMode.ORIGINAL)
    assert source.flags.writeable and mask.flags.writeable


@pytest.mark.parametrize("outside", list(OutsideMode))
def test_compiled_compositor_matches_scalar_byte_contract(outside):
    rng = np.random.default_rng(8472)
    rendered = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    source = rng.integers(0, 256, (5, 7, 4), dtype=np.uint8)
    mask = rng.random((5, 7), dtype=np.float32)
    actual = composite_masked(rendered, source, mask, outside)
    expected = np.empty_like(actual)
    for y in range(5):
        for x in range(7):
            coverage = int(float(mask[y, x]) * 255.0 + 0.5)
            inverse = 255 - coverage
            if outside is OutsideMode.ORIGINAL:
                outside_rgb = source[y, x, :3]
                outside_alpha = int(source[y, x, 3])
            elif outside is OutsideMode.TRANSPARENT:
                outside_rgb = rendered[y, x]
                outside_alpha = 0
            elif outside is OutsideMode.WHITE:
                outside_rgb = (255, 255, 255)
                outside_alpha = 255
            else:
                outside_rgb = (0, 0, 0)
                outside_alpha = 255
            alpha_numerator = coverage * 255 + inverse * outside_alpha
            for channel in range(3):
                if alpha_numerator == 0:
                    value = int(rendered[y, x, channel])
                else:
                    numerator = (
                        int(rendered[y, x, channel]) * coverage * 255
                        + int(outside_rgb[channel]) * inverse * outside_alpha
                    )
                    value = (numerator + alpha_numerator // 2) // alpha_numerator
                expected[y, x, channel] = value
            if expected.shape[2] == 4:
                expected[y, x, 3] = (alpha_numerator + 127) // 255
    assert np.array_equal(actual, expected)
