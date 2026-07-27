from __future__ import annotations

import importlib

import numpy as np
import pytest

from benchmarks.native_pixels import compare_exact_outputs, deterministic_u8


MODES = ("normal", "multiply", "screen", "overlay", "difference")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("opacity", (0, 1, 49, 50, 51, 99, 100))
@pytest.mark.parametrize("shape", ((1, 1, 4), (3, 5, 4), (7, 2, 4)))
def test_reference_and_native_compositors_are_byte_exact(mode, opacity, shape):
    from ditherzam.layers.blend import _blend_layer_native, _blend_layer_reference

    backdrop = deterministic_u8(shape, seed=0xBACC)
    source = deterministic_u8(shape, seed=0x50AC)
    compare_exact_outputs(
        _blend_layer_reference,
        _blend_layer_native,
        backdrop,
        source,
        mode,
        opacity,
    )


@pytest.mark.parametrize("mode", MODES)
def test_compositor_exactness_at_rounding_and_alpha_boundaries(mode):
    from ditherzam.layers.blend import _blend_layer_native, _blend_layer_reference

    values = np.array([0, 1, 2, 63, 64, 127, 128, 191, 254, 255], np.uint8)
    backdrop = np.empty((values.size, values.size, 4), np.uint8)
    source = np.empty_like(backdrop)
    backdrop[..., 0] = values[:, None]
    backdrop[..., 1] = values[None, :]
    backdrop[..., 2] = values[::-1, None]
    backdrop[..., 3] = values[None, :]
    source[..., 0] = values[None, :]
    source[..., 1] = values[::-1, None]
    source[..., 2] = values[:, None]
    source[..., 3] = values[:, None]

    for opacity in range(101):
        assert np.array_equal(
            _blend_layer_native(backdrop, source, mode, opacity),
            _blend_layer_reference(backdrop, source, mode, opacity),
        )


def test_zero_output_alpha_preserves_hidden_source_rgb():
    from ditherzam.layers.blend import _blend_layer_native, _blend_layer_reference

    backdrop = np.array([[[9, 17, 33, 0], [200, 199, 198, 0]]], np.uint8)
    source = np.array([[[1, 127, 255, 0], [44, 55, 66, 255]]], np.uint8)
    expected = np.array([[[1, 127, 255, 0], [44, 55, 66, 0]]], np.uint8)
    for call in (_blend_layer_reference, _blend_layer_native):
        assert np.array_equal(call(backdrop, source, "overlay", 0), expected)


def test_rgb_inputs_and_noncontiguous_views_preserve_wrapper_contract():
    from ditherzam.layers.blend import _blend_layer_native, _blend_layer_reference

    backdrop = deterministic_u8((6, 8, 4), seed=1)[::2, ::2, :3]
    source = deterministic_u8((6, 8, 4), seed=2)[::2, ::2, :3]
    assert not backdrop.flags.c_contiguous
    comparison = compare_exact_outputs(
        _blend_layer_reference,
        _blend_layer_native,
        backdrop,
        source,
        "screen",
        37,
    )
    assert comparison["shape"] == (3, 4, 4)


def test_public_compositor_uses_environment_forced_fallback(monkeypatch):
    import ditherzam.layers.blend as blend

    backdrop = deterministic_u8((3, 5, 4), seed=3)
    source = deterministic_u8((3, 5, 4), seed=4)
    expected = blend._blend_layer_reference(backdrop, source, "difference", 63)
    monkeypatch.setenv("DITHERZAM_DISABLE_NATIVE", "1")
    fallback = importlib.reload(blend)
    try:
        assert fallback._composite is None
        assert np.array_equal(
            fallback.blend_layer(backdrop, source, "difference", 63), expected
        )
        with pytest.raises(RuntimeError, match="unavailable"):
            fallback._blend_layer_native(
                backdrop, source, "difference", 63
            )
    finally:
        monkeypatch.delenv("DITHERZAM_DISABLE_NATIVE")
        importlib.reload(blend)


@pytest.mark.parametrize(
    ("backdrop", "source", "mode", "opacity", "message"),
    (
        (np.zeros((0, 1, 4), np.uint8), np.zeros((1, 1, 4), np.uint8), "normal", 100, "image must"),
        (np.zeros((1, 1, 4), np.uint8), np.zeros((2, 1, 4), np.uint8), "normal", 100, "equal height"),
        (np.zeros((1, 1, 4), np.uint8), np.zeros((1, 1, 4), np.uint8), "bad", 100, "mode is invalid"),
        (np.zeros((1, 1, 4), np.uint8), np.zeros((1, 1, 4), np.uint8), "normal", True, "opacity must"),
        (np.zeros((1, 1, 4), np.uint8), np.zeros((1, 1, 4), np.uint8), "normal", 101, "within 0..100"),
    ),
)
def test_reference_native_and_public_seams_retain_validation(
    backdrop, source, mode, opacity, message
):
    from ditherzam.layers.blend import (
        _blend_layer_native,
        _blend_layer_reference,
        blend_layer,
    )

    for call in (_blend_layer_reference, _blend_layer_native, blend_layer):
        with pytest.raises(ValueError, match=message):
            call(backdrop, source, mode, opacity)
