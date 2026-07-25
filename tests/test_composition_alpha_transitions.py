from __future__ import annotations

import numpy as np
import pytest


def test_straight_alpha_blend_ignores_hidden_rgb_and_preserves_endpoints():
    from ditherzam.composition.transitions import blend_straight_rgba

    # Transparent red must not contaminate a half-faded opaque blue pixel.
    a = np.array([[[255, 0, 0, 0]]], np.uint8)
    b = np.array([[[0, 0, 255, 255]]], np.uint8)
    assert np.array_equal(blend_straight_rgba(a, b, 0.0), a)
    assert np.array_equal(blend_straight_rgba(a, b, 1.0), b)
    assert np.array_equal(
        blend_straight_rgba(a, b, 0.5),
        np.array([[[0, 0, 255, 128]]], np.uint8),
    )


def test_blend_accepts_rgb_rgba_and_two_dimensional_weights():
    from ditherzam.composition.transitions import blend_straight_rgba

    a = np.zeros((2, 2, 3), np.uint8)
    b = np.full((2, 2, 4), 255, np.uint8)
    weights = np.array([[0.0, 1.0], [0.25, 0.75]], np.float32)
    out = blend_straight_rgba(a, b, weights)
    assert out.shape == (2, 2, 4)
    assert np.array_equal(out[0, 0], [0, 0, 0, 255])
    assert np.array_equal(out[0, 1], [255, 255, 255, 255])


def test_hard_select_chooses_complete_pixels_without_channel_blending():
    from ditherzam.composition.transitions import select_rgba

    a = np.array([[[10, 20, 30, 40], [1, 2, 3, 4]]], np.uint8)
    b = np.array([[[90, 80, 70, 60], [9, 8, 7, 6]]], np.uint8)
    out = select_rgba(a, b, np.array([[False, True]]))
    assert np.array_equal(out, np.array([[[10, 20, 30, 40], [9, 8, 7, 6]]], np.uint8))


def test_transition_registry_has_four_frozen_transition_names():
    from ditherzam.composition.transitions import TRANSITIONS

    assert set(TRANSITIONS) == {
        "crossfade", "dither-dissolve", "spatial-wipe", "param-morph",
    }


def test_blend_rejects_shape_dtype_and_weight_contract_violations():
    from ditherzam.composition.transitions import blend_straight_rgba

    good = np.zeros((2, 2, 4), np.uint8)
    with pytest.raises(ValueError):
        blend_straight_rgba(good.astype(np.float32), good, 0.5)
    with pytest.raises(ValueError):
        blend_straight_rgba(good, np.zeros((3, 2, 4), np.uint8), 0.5)
    with pytest.raises(ValueError):
        blend_straight_rgba(good, good, 1.01)
