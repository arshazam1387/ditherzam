from __future__ import annotations

import numpy as np
import pytest

from benchmarks.native_pixels import compare_exact_outputs, deterministic_u8
from ditherzam.layers.selection import (
    SelectionError,
    TemporarySelection,
    _selection_to_source_native,
    _selection_to_source_reference,
    selection_to_source,
)


TRANSFORMS = (
    pytest.param(0.0, 0.0, 1.0, 1.0, id="identity"),
    pytest.param(2.0, -1.0, 1.0, 1.0, id="translation"),
    pytest.param(0.25, -0.75, 1.0, 1.0, id="fractional"),
    pytest.param(-1.5, 0.75, 0.5, 2.25, id="scale"),
    pytest.param(-20.0, 30.0, 1.25, 0.75, id="out-of-document"),
)


@pytest.mark.parametrize("layer_x,layer_y,scale_x,scale_y", TRANSFORMS)
def test_native_selection_matches_reference_for_fixed_transforms(
    layer_x, layer_y, scale_x, scale_y
):
    selection = TemporarySelection(deterministic_u8((7, 11), seed=901))
    comparison = compare_exact_outputs(
        _selection_to_source_reference,
        _selection_to_source_native,
        selection,
        (9, 5),
        layer_x=layer_x,
        layer_y=layer_y,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    assert comparison["shape"] == (9, 5)
    assert comparison["dtype"] == np.dtype(np.uint8).str


def test_native_selection_matches_reference_at_boundaries():
    selection = TemporarySelection(
        np.array([[0, 1, 127], [128, 254, 255]], dtype=np.uint8)
    )
    for transform in (
        (-0.5, -0.5, 1.0, 1.0),
        (2.5, 1.5, 1.0, 1.0),
        (-1.0, -1.0, 2.0, 2.0),
    ):
        compare_exact_outputs(
            _selection_to_source_reference,
            _selection_to_source_native,
            selection,
            (4, 5),
            layer_x=transform[0],
            layer_y=transform[1],
            scale_x=transform[2],
            scale_y=transform[3],
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_native_selection_handles_extreme_finite_out_of_document_mapping():
    selection = TemporarySelection(np.array([[255]], dtype=np.uint8))
    for offset in (-1e200, 1e200):
        compare_exact_outputs(
            _selection_to_source_reference,
            _selection_to_source_native,
            selection,
            (2, 3),
            layer_x=offset,
            layer_y=offset,
            scale_x=1.0,
            scale_y=1.0,
        )


def test_native_selection_matches_reference_for_random_irregular_transforms():
    rng = np.random.default_rng(0x5E1EC7)
    for case in range(40):
        document_shape = (
            int(rng.integers(1, 18)),
            int(rng.integers(1, 18)),
        )
        output_shape = (
            int(rng.integers(1, 16)),
            int(rng.integers(1, 16)),
        )
        selection = TemporarySelection(
            deterministic_u8(document_shape, seed=case + 300)
        )
        compare_exact_outputs(
            _selection_to_source_reference,
            _selection_to_source_native,
            selection,
            output_shape,
            layer_x=float(rng.uniform(-8.0, 8.0)),
            layer_y=float(rng.uniform(-8.0, 8.0)),
            scale_x=float(rng.uniform(0.05, 4.0)),
            scale_y=float(rng.uniform(0.05, 4.0)),
        )


def test_selection_dispatch_can_force_the_reference_fallback(monkeypatch):
    selection = TemporarySelection(deterministic_u8((5, 7), seed=123))
    kwargs = dict(layer_x=0.25, layer_y=-0.75, scale_x=1.5, scale_y=0.6)
    monkeypatch.setenv("DITHERZAM_DISABLE_NATIVE", "1")
    actual = selection_to_source(selection, (6, 9), **kwargs)
    expected = _selection_to_source_reference(selection, (6, 9), **kwargs)
    assert np.array_equal(actual, expected)
    assert actual.flags.owndata and actual.flags.c_contiguous


@pytest.mark.parametrize(
    "source_shape,kwargs",
    (
        ((0, 1), dict(layer_x=0, layer_y=0, scale_x=1, scale_y=1)),
        ((1, 1), dict(layer_x=np.inf, layer_y=0, scale_x=1, scale_y=1)),
        ((1, 1), dict(layer_x=0, layer_y=0, scale_x=0, scale_y=1)),
        ((1, 1), dict(layer_x=0, layer_y=0, scale_x=1, scale_y=np.nan)),
    ),
)
def test_reference_and_native_seams_preserve_validation(source_shape, kwargs):
    selection = TemporarySelection(np.zeros((2, 2), dtype=np.uint8))
    for seam in (_selection_to_source_reference, _selection_to_source_native):
        with pytest.raises(SelectionError):
            seam(selection, source_shape, **kwargs)
