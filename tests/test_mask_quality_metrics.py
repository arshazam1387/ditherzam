"""Segmentation-quality metrics + deterministic model-selection policy (SM-02).

Freezes the decision math *before* any real model is measured: pure-NumPy
Dice/IoU/boundary-F oracle, immutable aggregate-quality records, and the
U2NET-vs-U2NETP winner policy. All masks here are synthetic NumPy arrays —
no real model weights or fixtures are involved.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ditherzam.masking.quality import (
    BOUNDARY_F_THRESHOLD,
    CATEGORY_DICE_FLOOR,
    COLD_LATENCY_MS,
    DICE_AGGREGATE_THRESHOLD,
    IOU_AGGREGATE_THRESHOLD,
    MODEL_DELTA_THRESHOLD,
    WARM_LATENCY_MEDIAN_MS,
    WARM_LATENCY_P95_MS,
    AggregateQuality,
    CandidateReport,
    ImageQualityResult,
    QualityMetricError,
    SelectionResult,
    aggregate_quality,
    boundary_f_score,
    dice_score,
    iou_score,
    select_model_candidate,
)


# -- module-level constants: locked verbatim from the SM-02 brief -------------


def test_thresholds_match_brief_exactly():
    assert DICE_AGGREGATE_THRESHOLD == 0.90
    assert IOU_AGGREGATE_THRESHOLD == 0.82
    assert CATEGORY_DICE_FLOOR == 0.82
    assert BOUNDARY_F_THRESHOLD == 0.80
    assert WARM_LATENCY_MEDIAN_MS == 500.0
    assert WARM_LATENCY_P95_MS == 800.0
    assert COLD_LATENCY_MS == 2000.0
    assert MODEL_DELTA_THRESHOLD == 0.03


# -- dice_score -----------------------------------------------------------------


def test_dice_score_both_empty_is_perfect():
    pred = np.zeros((4, 4), dtype=bool)
    truth = np.zeros((4, 4), dtype=bool)
    assert dice_score(pred, truth) == 1.0


def test_dice_score_both_full_is_perfect():
    pred = np.ones((4, 4), dtype=bool)
    truth = np.ones((4, 4), dtype=bool)
    assert dice_score(pred, truth) == 1.0


def test_dice_score_disjoint_is_zero():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    truth = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert dice_score(pred, truth) == 0.0


def test_dice_score_partial_overlap_exact_value():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    truth = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    # intersection=1, |pred|=2, |truth|=2 -> 2*1/(2+2) = 0.5
    assert dice_score(pred, truth) == pytest.approx(0.5)


# -- iou_score --------------------------------------------------------------------


def test_iou_score_both_empty_is_perfect():
    pred = np.zeros((3, 3), dtype=bool)
    truth = np.zeros((3, 3), dtype=bool)
    assert iou_score(pred, truth) == 1.0


def test_iou_score_disjoint_is_zero():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    truth = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    assert iou_score(pred, truth) == 0.0


def test_iou_score_partial_overlap_exact_value():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    truth = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    # intersection=1, union=3 -> 1/3
    assert iou_score(pred, truth) == pytest.approx(1.0 / 3.0)


# -- invalid shapes / values ----------------------------------------------------


def test_dice_score_rejects_shape_mismatch():
    pred = np.zeros((2, 2), dtype=bool)
    truth = np.zeros((3, 3), dtype=bool)
    with pytest.raises(QualityMetricError):
        dice_score(pred, truth)


def test_iou_score_rejects_shape_mismatch():
    pred = np.zeros((2, 2), dtype=bool)
    truth = np.zeros((3, 3), dtype=bool)
    with pytest.raises(QualityMetricError):
        iou_score(pred, truth)


def test_dice_score_rejects_non_2d_array():
    pred = np.array([1, 0, 1], dtype=np.uint8)
    truth = np.array([1, 0, 1], dtype=np.uint8)
    with pytest.raises(QualityMetricError):
        dice_score(pred, truth)


def test_dice_score_rejects_out_of_range_values():
    pred = np.array([[1, 2], [0, 0]], dtype=np.float32)
    truth = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(QualityMetricError):
        dice_score(pred, truth)


def test_dice_score_rejects_nan_values():
    pred = np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32)
    truth = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(QualityMetricError):
        dice_score(pred, truth)


def test_dice_score_rejects_non_array_input():
    with pytest.raises(QualityMetricError):
        dice_score([[1, 0], [0, 1]], np.zeros((2, 2), dtype=bool))  # type: ignore[arg-type]


# -- boundary_f_score: hand-derivable on a 1-row strip ---------------------------
#
# Shape (1, N): every foreground pixel is its own boundary (no interior cell can
# have all 4 neighbours in-mask when the array is only 1 row tall), which makes
# the expected precision/recall/F values exactly computable by hand.


def test_boundary_f_score_identical_masks_is_perfect():
    mask = np.zeros((1, 6), dtype=bool)
    mask[0, 2:4] = True
    assert boundary_f_score(mask, mask, tolerance_px=0) == 1.0


def test_boundary_f_score_one_pixel_shift_zero_tolerance():
    truth = np.zeros((1, 6), dtype=bool)
    truth[0, 2:4] = True  # cols 2,3
    pred = np.zeros((1, 6), dtype=bool)
    pred[0, 3:5] = True  # cols 3,4 (shifted right by 1)
    # precision = |{3,4} & {2,3}| / 2 = 1/2; recall = |{2,3} & {3,4}| / 2 = 1/2
    # F = 2*0.5*0.5/(0.5+0.5) = 0.5
    assert boundary_f_score(pred, truth, tolerance_px=0) == pytest.approx(0.5)


def test_boundary_f_score_one_pixel_shift_tolerance_one_is_perfect():
    truth = np.zeros((1, 6), dtype=bool)
    truth[0, 2:4] = True
    pred = np.zeros((1, 6), dtype=bool)
    pred[0, 3:5] = True
    # dilate-by-1 absorbs a 1px shift entirely on both sides -> F = 1.0
    assert boundary_f_score(pred, truth, tolerance_px=1) == pytest.approx(1.0)


def test_boundary_f_score_both_empty_is_perfect():
    pred = np.zeros((4, 4), dtype=bool)
    truth = np.zeros((4, 4), dtype=bool)
    assert boundary_f_score(pred, truth) == 1.0


def test_boundary_f_score_one_empty_one_not_is_zero():
    pred = np.zeros((4, 4), dtype=bool)
    truth = np.zeros((4, 4), dtype=bool)
    truth[1, 1] = True
    assert boundary_f_score(pred, truth) == 0.0


def test_boundary_f_score_rejects_shape_mismatch():
    pred = np.zeros((2, 2), dtype=bool)
    truth = np.zeros((3, 3), dtype=bool)
    with pytest.raises(QualityMetricError):
        boundary_f_score(pred, truth)


def test_boundary_f_score_rejects_negative_tolerance():
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(QualityMetricError):
        boundary_f_score(mask, mask, tolerance_px=-1)


def test_boundary_f_score_rejects_bool_tolerance():
    mask = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(QualityMetricError, match="non-negative int"):
        boundary_f_score(mask, mask, tolerance_px=True)


# -- aggregate_quality ------------------------------------------------------------


def _high_quality_results():
    return [
        ImageQualityResult(category="portrait", dice=0.95, iou=0.90, boundary_f=0.90),
        ImageQualityResult(category="product", dice=0.95, iou=0.90, boundary_f=0.90),
        ImageQualityResult(category="animal", dice=0.95, iou=0.90, boundary_f=0.90),
    ]


def test_aggregate_quality_meets_thresholds_when_all_high():
    agg = aggregate_quality(_high_quality_results())
    assert isinstance(agg, AggregateQuality)
    assert agg.meets_dice_threshold
    assert agg.meets_iou_threshold
    assert agg.meets_boundary_f_threshold
    assert agg.meets_category_floor
    assert agg.meets_thresholds


def test_aggregate_quality_category_floor_can_fail_independently_of_aggregate_dice():
    # 15 high-scoring images (dice 0.95) + 1 low full_body image (dice 0.75):
    # aggregate dice = (15*0.95 + 0.75) / 16 = 0.9375 >= 0.90 (passes)
    # but full_body's own category mean (0.75) is below the 0.82 floor.
    results = [
        ImageQualityResult(category="portrait", dice=0.95, iou=0.90, boundary_f=0.90)
        for _ in range(5)
    ] + [
        ImageQualityResult(category="product", dice=0.95, iou=0.90, boundary_f=0.90)
        for _ in range(5)
    ] + [
        ImageQualityResult(category="animal", dice=0.95, iou=0.90, boundary_f=0.90)
        for _ in range(5)
    ] + [
        ImageQualityResult(category="full_body", dice=0.75, iou=0.90, boundary_f=0.90)
    ]
    agg = aggregate_quality(results)
    assert agg.aggregate_dice == pytest.approx(0.9375)
    assert agg.meets_dice_threshold  # aggregate passes
    assert agg.category_dice["full_body"] == pytest.approx(0.75)
    assert not agg.meets_category_floor  # but the category floor does not
    assert not agg.meets_thresholds


def test_aggregate_quality_fails_on_low_aggregate_dice():
    results = [ImageQualityResult(category="portrait", dice=0.5, iou=0.5, boundary_f=0.5)]
    agg = aggregate_quality(results)
    assert not agg.meets_dice_threshold
    assert not agg.meets_thresholds


def test_aggregate_quality_rejects_empty_results():
    with pytest.raises(QualityMetricError):
        aggregate_quality([])


def test_aggregate_quality_is_frozen():
    agg = aggregate_quality(_high_quality_results())
    with pytest.raises(dataclasses.FrozenInstanceError):
        agg.aggregate_dice = 0.0  # type: ignore[misc]


def test_image_quality_result_rejects_out_of_range_value():
    with pytest.raises(QualityMetricError):
        ImageQualityResult(category="portrait", dice=1.5, iou=0.9, boundary_f=0.9)


# -- select_model_candidate: deterministic winner policy -------------------------


def _agg(dice=0.95, iou=0.95, boundary_f=0.95):
    return aggregate_quality([ImageQualityResult(category="all", dice=dice, iou=iou, boundary_f=boundary_f)])


def test_full_wins_when_eligible_approved_and_iou_delta_sufficient():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(iou=0.90, boundary_f=0.85),
        within_budgets=True, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=True, manually_approved=False,
    )
    result = select_model_candidate(full, lite)
    assert isinstance(result, SelectionResult)
    assert result.winner == "u2net"


def test_full_wins_via_boundary_f_delta_alone():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(iou=0.86, boundary_f=0.90),
        within_budgets=True, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=True, manually_approved=False,
    )
    result = select_model_candidate(full, lite)
    assert result.winner == "u2net"


def test_lite_wins_when_full_delta_insufficient():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(iou=0.86, boundary_f=0.81),
        within_budgets=True, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=True, manually_approved=False,
    )
    result = select_model_candidate(full, lite)
    assert result.winner == "u2netp"


def test_lite_wins_when_full_not_manually_approved():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(iou=0.95, boundary_f=0.95),
        within_budgets=True, manually_approved=False,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=True, manually_approved=False,
    )
    result = select_model_candidate(full, lite)
    assert result.winner == "u2netp"


def test_lite_wins_when_full_outside_budgets():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(iou=0.95, boundary_f=0.95),
        within_budgets=False, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=True, manually_approved=False,
    )
    result = select_model_candidate(full, lite)
    assert result.winner == "u2netp"


def test_unavailable_when_neither_eligible():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(dice=0.5, iou=0.5, boundary_f=0.5),
        within_budgets=True, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=False, manually_approved=False,
    )
    result = select_model_candidate(full, lite)
    assert result.winner is None


def test_unavailable_when_full_delta_insufficient_and_lite_ineligible():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(iou=0.86, boundary_f=0.81),
        within_budgets=True, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=False, manually_approved=False,
    )
    result = select_model_candidate(full, lite)
    assert result.winner is None


def test_select_model_candidate_rejects_swapped_model_ids():
    full = CandidateReport(
        model_id="u2netp", aggregate=_agg(), within_budgets=True, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2net", aggregate=_agg(), within_budgets=True, manually_approved=False,
    )
    with pytest.raises(QualityMetricError):
        select_model_candidate(full, lite)


def test_candidate_report_rejects_unapproved_model_id():
    with pytest.raises(QualityMetricError):
        CandidateReport(
            model_id="not-a-model", aggregate=_agg(), within_budgets=True, manually_approved=False,
        )


def test_selection_result_is_frozen():
    result = select_model_candidate(
        CandidateReport(model_id="u2net", aggregate=_agg(dice=0.5, iou=0.5, boundary_f=0.5),
                         within_budgets=True, manually_approved=True),
        CandidateReport(model_id="u2netp", aggregate=_agg(), within_budgets=True, manually_approved=False),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.winner = "u2net"  # type: ignore[misc]


def test_selection_is_deterministic_pure_function():
    full = CandidateReport(
        model_id="u2net", aggregate=_agg(iou=0.90, boundary_f=0.85),
        within_budgets=True, manually_approved=True,
    )
    lite = CandidateReport(
        model_id="u2netp", aggregate=_agg(iou=0.85, boundary_f=0.80),
        within_budgets=True, manually_approved=False,
    )
    results = {select_model_candidate(full, lite).winner for _ in range(10)}
    assert results == {"u2net"}


# -- benchmarks/smart_mask.py: CLI skeleton + report schema (SM-02 only) --------
#
# This is arg-parsing and a report shape, not a real bakeoff runner (SM-16
# completes that). These are smoke tests that the schema is stable and
# importable, not measurements of anything real.


def test_benchmark_report_skeleton_has_one_slot_per_approved_model():
    from benchmarks.smart_mask import APPROVED_MODEL_IDS, build_report_skeleton

    report = build_report_skeleton(fixtures_manifest=None, asset_root=None)
    model_ids = {c["model_id"] for c in report["candidates"]}
    assert model_ids == set(APPROVED_MODEL_IDS)
    assert all(c["measured"] is False for c in report["candidates"])
    assert report["selection"] is None


def test_benchmark_cli_parses_expected_flags():
    from benchmarks.smart_mask import parse_args

    args = parse_args([
        "--fixtures-manifest", "some/manifest.yaml",
        "--asset-root", "some/root",
        "--output", "some/out.json",
    ])
    assert args.fixtures_manifest == "some/manifest.yaml"
    assert args.asset_root == "some/root"
    assert args.output == "some/out.json"


def test_benchmark_main_writes_valid_json_report(tmp_path):
    import json

    from benchmarks.smart_mask import main

    output_path = tmp_path / "report.json"
    exit_code = main(["--output", str(output_path)])
    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["selection"] is None
