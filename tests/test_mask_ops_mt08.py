import numpy as np
import pytest

from ditherzam.layers import (
    RasterMaskOperationError,
    fill_raster_mask,
    freeze_smart_mask,
    invert_raster_mask,
)
from ditherzam.masking.contracts import (
    InferenceIdentity,
    ModelIdentity,
    ProbabilityMap,
    source_identity,
)
from ditherzam.masking.settings import MaskTarget, SmartMaskSettings


def _probability(values, rgba):
    source = source_identity(rgba)
    identity = InferenceIdentity(
        source,
        ModelIdentity("test", "1", "0" * 64),
        "v1",
        "subject",
    )
    return ProbabilityMap(identity, np.asarray(values, dtype=np.float32))


def test_strict_exact_invert_and_fills():
    pixels = np.array([[0, 1, 127, 255]], dtype=np.uint8)
    assert invert_raster_mask(pixels).tolist() == [[255, 254, 128, 0]]
    assert np.all(fill_raster_mask(pixels.shape, 255) == 255)
    assert np.all(fill_raster_mask(pixels.shape, 0) == 0)
    for invalid in (pixels.astype(np.int16), pixels[..., None]):
        with pytest.raises(RasterMaskOperationError):
            invert_raster_mask(invalid)
    with pytest.raises(RasterMaskOperationError):
        fill_raster_mask((1, 4), 1)


def test_freeze_smart_uses_derive_master_mask_and_half_up_u8():
    rgba = np.full((1, 4, 4), 255, dtype=np.uint8)
    probability = _probability([[0.0, 0.5, 0.501, 1.0]], rgba)
    settings = SmartMaskSettings(
        enabled=True, target=MaskTarget.SUBJECT, sensitivity=50,
        expansion_px=0, feather_px=0,
    )
    assert freeze_smart_mask(
        probability, settings, rgba=rgba
    ).tolist() == [[0, 255, 255, 255]]


def test_freeze_whole_image_needs_no_probability_but_other_targets_do():
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    whole = SmartMaskSettings(enabled=True, target=MaskTarget.WHOLE_IMAGE)
    assert np.all(freeze_smart_mask(None, whole, rgba=rgba) == 255)
    with pytest.raises(RasterMaskOperationError, match="unavailable"):
        freeze_smart_mask(
            None, SmartMaskSettings(enabled=True, target=MaskTarget.SUBJECT),
            rgba=rgba,
        )


def test_freeze_rejects_probability_owned_by_another_source():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    other = np.ones((2, 2, 4), dtype=np.uint8)
    probability = _probability(np.ones((2, 2)), other)
    with pytest.raises(RasterMaskOperationError, match="does not match"):
        freeze_smart_mask(
            probability,
            SmartMaskSettings(enabled=True, target=MaskTarget.SUBJECT),
            rgba=rgba,
        )
