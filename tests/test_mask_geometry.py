import numpy as np
import pytest

from ditherzam.masking.geometry import (
    FEATHER_ALGORITHM_VERSION,
    GEOMETRY_ALGORITHM_VERSION,
    RESIZE_ALGORITHM_VERSION,
    MaskGeometryError,
    derive_master_mask,
    derive_preview_mask,
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


def test_derive_upsamples_capped_probability_to_source_shape() -> None:
    # A capped probability map (bounded retained resolution for very large
    # sources) must derive a master mask at full source resolution, identical
    # to deriving from the explicitly bilinear-upsampled probability.
    rng = np.random.default_rng(20260716)
    capped = rng.random((9, 12)).astype(np.float32)
    mask = derive_master_mask(
        capped, sensitivity=50, target=MaskTarget.SUBJECT, source_shape=(27, 36)
    )
    assert mask.shape == (27, 36)
    assert not mask.flags.writeable

    from PIL import Image

    upsampled = np.asarray(
        Image.fromarray(capped, mode="F").resize((36, 27), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    expected = derive_master_mask(
        np.clip(upsampled, 0.0, 1.0), sensitivity=50, target=MaskTarget.SUBJECT
    )
    assert np.array_equal(mask, expected)

    # Exact-shape probability is untouched: no resample, byte-identical path.
    exact = derive_master_mask(
        capped, sensitivity=50, target=MaskTarget.SUBJECT, source_shape=(9, 12)
    )
    assert np.array_equal(exact, derive_master_mask(capped, sensitivity=50, target=MaskTarget.SUBJECT))


def test_preview_derivation_targets_preview_shape_and_scales_radii() -> None:
    prob = np.zeros((32, 32), dtype=np.float32)
    prob[8:24, 8:24] = 1.0
    source_shape = (320, 320)
    target_shape = (32, 32)  # ratio 0.1
    preview = derive_preview_mask(
        prob, sensitivity=50, target=MaskTarget.SUBJECT,
        expansion_px=40, feather_px=8,
        source_shape=source_shape, target_shape=target_shape,
    )
    assert preview.shape == target_shape
    assert preview.dtype == np.float32
    assert not preview.flags.writeable
    # ratio 0.1: expansion 40 -> round(4.0)=4, feather 8 -> round(0.8)=1.
    direct = derive_master_mask(
        prob, sensitivity=50, target=MaskTarget.SUBJECT,
        expansion_px=4, feather_px=1, source_shape=target_shape,
    )
    assert np.array_equal(preview, direct)


def test_preview_derivation_rounds_subpixel_radius_to_zero() -> None:
    prob = np.zeros((32, 32), dtype=np.float32)
    prob[8:24, 8:24] = 1.0
    # 8 px feather at a 3750 px source is sub-pixel at an 8 px preview.
    preview = derive_preview_mask(
        prob, sensitivity=50, target=MaskTarget.SUBJECT,
        expansion_px=0, feather_px=8,
        source_shape=(3750, 3750), target_shape=(8, 8),
    )
    hard = derive_master_mask(
        prob, sensitivity=50, target=MaskTarget.SUBJECT,
        expansion_px=0, feather_px=0, source_shape=(8, 8),
    )
    assert np.array_equal(preview, hard)
    assert set(np.unique(preview)).issubset({0.0, 1.0})


def test_preview_derivation_whole_image_is_ones_at_target_shape() -> None:
    whole = derive_preview_mask(
        None, sensitivity=50, target=MaskTarget.WHOLE_IMAGE,
        source_shape=(400, 400), target_shape=(20, 40),
    )
    assert whole.shape == (20, 40)
    assert np.all(whole == 1.0)


def test_preview_derivation_clamps_expansion_to_valid_range() -> None:
    prob = np.zeros((16, 16), dtype=np.float32)
    prob[4:12, 4:12] = 1.0
    # ratio 2.0 would scale +64 to +128, outside EXPANSION_MAX_PX; must clamp.
    preview = derive_preview_mask(
        prob, sensitivity=50, target=MaskTarget.SUBJECT,
        expansion_px=64, feather_px=0,
        source_shape=(16, 16), target_shape=(32, 32),
    )
    assert preview.shape == (32, 32)


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
