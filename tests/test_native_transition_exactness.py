from __future__ import annotations

import importlib

import numpy as np
import pytest

from benchmarks.native_pixels import compare_exact_outputs, deterministic_u8


@pytest.mark.parametrize("shape", ((1, 1, 4), (3, 5, 4), (7, 2, 4)))
@pytest.mark.parametrize("weight", (0.0, 1.0, 0.25, 0.5, 0.999))
def test_scalar_transition_blend_is_byte_exact(shape, weight):
    from ditherzam.composition.transitions import (
        _blend_straight_rgba_native,
        _blend_straight_rgba_reference,
    )

    a = deterministic_u8(shape, seed=11)
    b = deterministic_u8(shape, seed=12)
    compare_exact_outputs(
        _blend_straight_rgba_reference,
        _blend_straight_rgba_native,
        a,
        b,
        weight,
    )


def test_weight_plane_transition_blend_is_byte_exact_at_boundaries():
    from ditherzam.composition.transitions import (
        _blend_straight_rgba_native,
        _blend_straight_rgba_reference,
    )

    values = np.array([0, 1, 2, 63, 64, 127, 128, 191, 254, 255], np.uint8)
    a = np.empty((values.size, values.size, 4), np.uint8)
    b = np.empty_like(a)
    a[..., 0] = values[:, None]
    a[..., 1] = values[None, :]
    a[..., 2] = values[::-1, None]
    a[..., 3] = values[None, :]
    b[..., 0] = values[None, :]
    b[..., 1] = values[::-1, None]
    b[..., 2] = values[:, None]
    b[..., 3] = values[:, None]
    weights = np.linspace(0.0, 1.0, values.size**2, dtype=np.float64).reshape(
        values.size, values.size
    )
    compare_exact_outputs(
        _blend_straight_rgba_reference,
        _blend_straight_rgba_native,
        a,
        b,
        weights,
    )


def test_zero_alpha_preserves_interpolated_hidden_rgb():
    from ditherzam.composition.transitions import (
        _blend_straight_rgba_native,
        _blend_straight_rgba_reference,
    )

    a = np.array([[[9, 17, 33, 0]]], np.uint8)
    b = np.array([[[200, 199, 198, 0]]], np.uint8)
    expected = np.array([[[105, 108, 116, 0]]], np.uint8)
    for call in (_blend_straight_rgba_reference, _blend_straight_rgba_native):
        assert np.array_equal(call(a, b, 0.5), expected)


def test_public_transition_blend_uses_environment_forced_fallback(monkeypatch):
    import ditherzam.composition.transitions as transitions

    a = deterministic_u8((3, 5, 4), seed=13)
    b = deterministic_u8((3, 5, 4), seed=14)
    expected = transitions._blend_straight_rgba_reference(a, b, 0.37)
    monkeypatch.setenv("DITHERZAM_DISABLE_NATIVE", "1")
    fallback = importlib.reload(transitions)
    try:
        assert fallback._composite is None
        assert np.array_equal(fallback.blend_straight_rgba(a, b, 0.37), expected)
        with pytest.raises(RuntimeError, match="unavailable"):
            fallback._blend_straight_rgba_native(a, b, 0.37)
    finally:
        monkeypatch.delenv("DITHERZAM_DISABLE_NATIVE")
        importlib.reload(transitions)


def test_noncontiguous_rgb_inputs_and_float32_plane_preserve_contract():
    from ditherzam.composition.transitions import (
        _blend_straight_rgba_native,
        _blend_straight_rgba_reference,
    )

    a = deterministic_u8((6, 8, 4), seed=15)[::2, ::2, :3]
    b = deterministic_u8((6, 8, 4), seed=16)[::2, ::2, :3]
    weights = np.linspace(0, 1, 48, dtype=np.float32).reshape(6, 8)[::2, ::2]
    comparison = compare_exact_outputs(
        _blend_straight_rgba_reference,
        _blend_straight_rgba_native,
        a,
        b,
        weights,
    )
    assert comparison["shape"] == (3, 4, 4)
