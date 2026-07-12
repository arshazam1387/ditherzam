import numpy as np
import pytest

from ditherzam.masking.geometry import (
    FEATHER_ALGORITHM_VERSION,
    GEOMETRY_ALGORITHM_VERSION,
    RESIZE_ALGORITHM_VERSION,
    MaskGeometryError,
    derive_master_mask,
    expand_contract,
    feather,
    resize_mask_area,
    sensitivity_threshold,
)
from ditherzam.masking.settings import MaskTarget


def _confidence() -> np.ndarray:
    return np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)


def test_versions_are_nonempty_and_sensitivity_is_monotonic() -> None:
    assert all((GEOMETRY_ALGORITHM_VERSION, FEATHER_ALGORITHM_VERSION, RESIZE_ALGORITHM_VERSION))
    thresholds = [sensitivity_threshold(value) for value in range(101)]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[0] == 1.0
    assert thresholds[-1] == 0.0


def test_higher_sensitivity_selects_superset() -> None:
    low = derive_master_mask(_confidence(), sensitivity=25, target=MaskTarget.SUBJECT)
    high = derive_master_mask(_confidence(), sensitivity=75, target=MaskTarget.SUBJECT)
    assert np.all(low <= high)
    assert int(low.sum()) < int(high.sum())


def test_target_invert_geometry_feather_order_and_whole_image() -> None:
    confidence = np.zeros((7, 7), dtype=np.float32)
    confidence[3, 3] = 1.0
    subject = derive_master_mask(confidence, sensitivity=50, target=MaskTarget.SUBJECT)
    background = derive_master_mask(confidence, sensitivity=50, target=MaskTarget.BACKGROUND)
    assert np.array_equal(background, 1.0 - subject)
    # Invert occurs before expansion: inverted one-pixel subject becomes background,
    # then expansion fills the remaining central hole.
    expanded_invert = derive_master_mask(
        confidence, sensitivity=50, target=MaskTarget.SUBJECT, invert=True, expansion_px=1
    )
    assert np.all(expanded_invert == 1.0)
    whole = derive_master_mask(
        None, sensitivity=50, target=MaskTarget.WHOLE_IMAGE, source_shape=(3, 4)
    )
    assert np.array_equal(whole, np.ones((3, 4), dtype=np.float32))


def test_whole_image_ignores_all_edit_controls_after_validating_them() -> None:
    confidence = np.zeros((5, 6), dtype=np.float32)
    whole = derive_master_mask(
        confidence,
        sensitivity=100,
        target=MaskTarget.WHOLE_IMAGE,
        invert=True,
        expansion_px=-64,
        feather_px=99,
    )
    assert np.array_equal(whole, np.ones((5, 6), dtype=np.float32))
    assert not whole.flags.writeable


def test_expand_contract_are_signed_and_preserve_structural_behavior() -> None:
    mask = np.zeros((9, 9), dtype=np.float32)
    mask[2:7, 2:7] = 1.0
    mask[4, 4] = 0.0
    expanded = expand_contract(mask, 1)
    contracted = expand_contract(mask, -1)
    assert expanded.sum() > mask.sum() > contracted.sum()
    assert expanded[1, 4] == 1.0
    assert contracted[2, 4] == 0.0
    assert expand_contract(mask, 0) is not mask


def test_geometry_uses_exact_square_source_pixels_and_retains_hole_semantics() -> None:
    point = np.zeros((11, 11), dtype=np.float32)
    point[5, 5] = 1.0
    expanded = expand_contract(point, 2)
    assert expanded.sum() == 25
    assert np.all(expanded[3:8, 3:8] == 1.0)

    solid_with_hole = np.ones((11, 11), dtype=np.float32)
    solid_with_hole[5, 5] = 0.0
    contracted = expand_contract(solid_with_hole, -2)
    # Contracting foreground expands the hole by exactly two source pixels.
    assert np.all(contracted[3:8, 3:8] == 0.0)

    thin = np.zeros((11, 11), dtype=np.float32)
    thin[:, 5] = 1.0
    assert np.all(expand_contract(thin, 1)[:, 4:7] == 1.0)


@pytest.mark.parametrize("amount", [-4, -2, -1, 1, 2, 4])
def test_fast_morphology_matches_brute_force_at_borders_and_low_density(amount: int) -> None:
    rng = np.random.default_rng(20260711)
    source = (rng.random((13, 17)) < 0.12).astype(np.float32)
    source[0, 0] = source[-1, -1] = 1.0
    radius = abs(amount)
    padded = np.pad(source, radius, mode="constant")
    expected = np.empty_like(source)
    width = 2 * radius + 1
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            window = padded[y : y + width, x : x + width]
            expected[y, x] = np.any(window) if amount > 0 else np.all(window)
    assert np.array_equal(expand_contract(source, amount), expected)


def test_zero_feather_is_hard_and_positive_feather_is_symmetric() -> None:
    mask = np.zeros((1, 9), dtype=np.float32)
    mask[0, 4:] = 1.0
    hard = feather(mask, 0)
    soft = feather(mask, 2)
    assert set(np.unique(hard)) == {0.0, 1.0}
    assert soft[0, 3] == pytest.approx(1.0 - soft[0, 4], abs=1 / 255)
    assert 0.0 < soft[0, 3] < soft[0, 4] < 1.0


def test_area_resize_is_deterministic_immutable_and_preserves_thin_coverage() -> None:
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:, 1] = 1.0
    first = resize_mask_area(mask, (2, 2))
    second = resize_mask_area(mask, (2, 2))
    assert np.array_equal(first, second)
    assert np.any((first > 0.0) & (first < 1.0))
    assert first.dtype == np.float32 and first.flags.c_contiguous
    assert not first.flags.writeable


@pytest.mark.parametrize("call", [
    lambda: sensitivity_threshold(True),
    lambda: sensitivity_threshold(-1),
    lambda: expand_contract(np.zeros((2, 2), dtype=np.float64), 1),
    lambda: expand_contract(np.zeros((2, 2), dtype=np.float32), 65),
    lambda: feather(np.zeros((2, 2), dtype=np.float32), -1),
    lambda: resize_mask_area(np.zeros((2, 2), dtype=np.float32), (0, 2)),
    lambda: derive_master_mask(None, sensitivity=50, target=MaskTarget.SUBJECT),
])
def test_invalid_inputs_fail_closed(call) -> None:
    with pytest.raises((MaskGeometryError, TypeError)):
        call()
