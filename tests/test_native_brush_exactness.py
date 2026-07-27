from __future__ import annotations

import numpy as np
import pytest

from ditherzam.layers.mask_brush import (
    BrushMode,
    BrushSettings,
    _stamp_mask_brush_native,
    _stamp_mask_brush_reference,
    stamp_mask_brush,
)


def _compare(
    source: np.ndarray,
    document_x: float,
    document_y: float,
    settings: BrushSettings,
    **transform,
) -> None:
    reference = source.copy()
    native = source.copy()
    reference_dirty = _stamp_mask_brush_reference(
        reference, document_x, document_y, settings, **transform
    )
    native_dirty = _stamp_mask_brush_native(
        native, document_x, document_y, settings, **transform
    )
    assert native_dirty == reference_dirty
    assert np.array_equal(native, reference)


@pytest.mark.parametrize("mode", list(BrushMode))
@pytest.mark.parametrize("hardness", [0, 1, 50, 99, 100])
@pytest.mark.parametrize("strength", [0, 1, 50, 99, 100])
def test_native_brush_matches_reference_at_endpoints(mode, hardness, strength):
    source = np.array(
        [[0, 1, 127, 128, 254, 255], [255, 254, 128, 127, 1, 0]],
        dtype=np.uint8,
    )
    _compare(
        source, 2.75, 1.0, BrushSettings(5.5, hardness, strength, mode)
    )


@pytest.mark.parametrize(
    "transform",
    [
        dict(layer_x=0.0, layer_y=0.0, scale_x=1.0, scale_y=1.0),
        dict(layer_x=-4.25, layer_y=3.75, scale_x=0.5, scale_y=2.25),
        dict(layer_x=20.0, layer_y=-30.0, scale_x=3.0, scale_y=0.25),
    ],
)
def test_native_brush_matches_reference_for_clipping_and_transforms(transform):
    source = np.arange(13 * 17, dtype=np.uint8).reshape(13, 17)
    for point in [(-20.0, -20.0), (0.0, 0.0), (8.5, 6.5), (50.0, 50.0)]:
        _compare(
            source, *point, BrushSettings(11.25, 37, 73, BrushMode.HIDE),
            **transform,
        )


def test_native_brush_matches_reference_for_randomized_cases():
    rng = np.random.default_rng(0xB205)
    for _case in range(200):
        shape = (int(rng.integers(1, 40)), int(rng.integers(1, 40)))
        source = rng.integers(0, 256, shape, dtype=np.uint8)
        settings = BrushSettings(
            float(rng.uniform(0.1, 60.0)),
            int(rng.integers(0, 101)),
            int(rng.integers(0, 101)),
            rng.choice(list(BrushMode)),
        )
        _compare(
            source,
            float(rng.uniform(-30.0, 70.0)),
            float(rng.uniform(-30.0, 70.0)),
            settings,
            layer_x=float(rng.uniform(-10.0, 10.0)),
            layer_y=float(rng.uniform(-10.0, 10.0)),
            scale_x=float(rng.uniform(0.05, 4.0)),
            scale_y=float(rng.uniform(0.05, 4.0)),
        )


def test_dispatch_can_force_reference_fallback(monkeypatch):
    source = np.zeros((9, 11), dtype=np.uint8)
    expected = source.copy()
    settings = BrushSettings(7, 25, 80, BrushMode.REVEAL)
    expected_dirty = _stamp_mask_brush_reference(expected, 4.25, 5.75, settings)
    monkeypatch.setenv("DITHERZAM_DISABLE_NATIVE", "1")
    actual_dirty = stamp_mask_brush(source, 4.25, 5.75, settings)
    assert actual_dirty == expected_dirty
    assert np.array_equal(source, expected)


def test_native_seam_preserves_validation_and_strided_in_place_mutation():
    invalid = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="writable 2D uint8"):
        _stamp_mask_brush_native(
            invalid, 1.5, 1.5, BrushSettings(2, 100, 100, BrushMode.REVEAL)
        )

    owner = np.zeros((8, 10), dtype=np.uint8)
    view = owner[::2, ::2]
    _compare(view, 2.5, 2.0, BrushSettings(4, 50, 100, BrushMode.REVEAL))
    dirty = _stamp_mask_brush_native(
        view, 2.5, 2.0, BrushSettings(4, 50, 100, BrushMode.REVEAL)
    )
    assert dirty is not None
    assert owner.any()
