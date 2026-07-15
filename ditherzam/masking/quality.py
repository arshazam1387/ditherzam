"""Segmentation-quality metric oracle and model-selection policy for Smart Mask.

Freezes the decision math *before* any real model is measured (SM-02): pure
NumPy Dice/IoU/boundary-F metrics, immutable aggregate-quality records, and
the deterministic U2NET-vs-U2NETP winner policy. Every threshold below is
recorded verbatim from the approved design spec's acceptance budgets and must
not be lowered to approve a model without explicit user review.

No real model weights, inference, or fixtures are involved: callers supply
synthetic or measured binary masks as plain NumPy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ditherzam.masking.model_assets import APPROVED_MODEL_IDS


class QualityMetricError(Exception):
    """Raised on invalid metric inputs or an invalid model-selection input."""


# -- Acceptance budgets (verbatim; see design spec "Acceptance budgets") --------

DICE_AGGREGATE_THRESHOLD = 0.90
IOU_AGGREGATE_THRESHOLD = 0.82
CATEGORY_DICE_FLOOR = 0.82
BOUNDARY_F_THRESHOLD = 0.80

WARM_LATENCY_MEDIAN_MS = 500.0
WARM_LATENCY_P95_MS = 800.0
COLD_LATENCY_MS = 2000.0

# Full U2NET must beat U2NETP by at least this absolute margin in aggregate
# IoU or boundary F to be worth its extra size/latency.
MODEL_DELTA_THRESHOLD = 0.03

# Default Chebyshev-distance boundary-matching tolerance in pixels for
# boundary_f_score. A documented, fixed default; callers may override it.
DEFAULT_BOUNDARY_TOLERANCE_PX = 2

_FULL_MODEL_ID = "u2net"
_LITE_MODEL_ID = "u2netp"


# -- Metric oracle ----------------------------------------------------------------


def _as_mask_array(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise QualityMetricError(f"{name} must be a numpy array, got {type(value).__name__}")
    if value.ndim != 2:
        raise QualityMetricError(f"{name} must be a 2-D array, got shape {value.shape}")
    if value.size == 0:
        raise QualityMetricError(f"{name} must not be a zero-size array: shape {value.shape}")
    arr = value.astype(np.float64)
    if not np.isfinite(arr).all():
        raise QualityMetricError(f"{name} must not contain NaN or Inf values")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise QualityMetricError(f"{name} values must be within [0, 1]")
    return arr


def _validate_and_binarize(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = _as_mask_array(pred, "pred")
    truth_arr = _as_mask_array(truth, "truth")
    if pred_arr.shape != truth_arr.shape:
        raise QualityMetricError(
            f"pred and truth shapes must match: {pred_arr.shape} != {truth_arr.shape}"
        )
    return pred_arr > 0.5, truth_arr > 0.5


def dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """Sorensen-Dice coefficient of two binary masks.

    Empty-vs-empty (both zero foreground pixels) is defined as a perfect
    match (1.0) rather than the undefined 0/0.
    """
    pred_b, truth_b = _validate_and_binarize(pred, truth)
    intersection = np.count_nonzero(pred_b & truth_b)
    total = np.count_nonzero(pred_b) + np.count_nonzero(truth_b)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def iou_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """Intersection-over-union of two binary masks.

    Empty-vs-empty (zero union) is defined as a perfect match (1.0).
    """
    pred_b, truth_b = _validate_and_binarize(pred, truth)
    intersection = np.count_nonzero(pred_b & truth_b)
    union = np.count_nonzero(pred_b | truth_b)
    if union == 0:
        return 1.0
    return intersection / union


def _erode4(mask: np.ndarray) -> np.ndarray:
    """4-connected binary erosion; out-of-bounds neighbours count as background."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return center & up & down & left & right


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Foreground pixels with at least one background 4-neighbour (or image edge)."""
    return mask & ~_erode4(mask)


def _dilate3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev-distance dilation by `radius` pixels (iterated 3x3 Moore dilation)."""
    result = mask
    for _ in range(radius):
        result = _dilate3x3(result)
    return result


def boundary_f_score(
    pred: np.ndarray,
    truth: np.ndarray,
    tolerance_px: int = DEFAULT_BOUNDARY_TOLERANCE_PX,
) -> float:
    """Boundary F-score: harmonic mean of boundary precision/recall within a tolerance.

    Boundary pixels are the foreground pixels of each mask that touch a
    background 4-neighbour (or the image edge). A predicted boundary pixel
    counts as a true positive if it lies within `tolerance_px` (Chebyshev
    distance) of some ground-truth boundary pixel, and symmetrically for
    recall. Both-empty is a perfect match (1.0); exactly one empty is 0.0.
    """
    if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, int) or tolerance_px < 0:
        raise QualityMetricError(f"tolerance_px must be a non-negative int, got {tolerance_px!r}")

    pred_b, truth_b = _validate_and_binarize(pred, truth)
    pred_boundary = _mask_boundary(pred_b)
    truth_boundary = _mask_boundary(truth_b)

    pred_count = int(np.count_nonzero(pred_boundary))
    truth_count = int(np.count_nonzero(truth_boundary))

    if pred_count == 0 and truth_count == 0:
        return 1.0
    if pred_count == 0 or truth_count == 0:
        return 0.0

    truth_dilated = _dilate(truth_boundary, tolerance_px)
    pred_dilated = _dilate(pred_boundary, tolerance_px)

    precision = np.count_nonzero(pred_boundary & truth_dilated) / pred_count
    recall = np.count_nonzero(truth_boundary & pred_dilated) / truth_count

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# -- Immutable per-image / aggregate result records --------------------------------


def _validate_unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise QualityMetricError(f"{name} must be a finite value within [0, 1], got {value!r}")
    return result


@dataclass(frozen=True)
class ImageQualityResult:
    """One fixture image's measured metrics against its ground-truth mask."""

    category: str
    dice: float
    iou: float
    boundary_f: float

    def __post_init__(self) -> None:
        if not self.category or not self.category.strip():
            raise QualityMetricError("ImageQualityResult.category must not be empty")
        _validate_unit_interval(self.dice, "dice")
        _validate_unit_interval(self.iou, "iou")
        _validate_unit_interval(self.boundary_f, "boundary_f")


@dataclass(frozen=True)
class AggregateQuality:
    """Immutable aggregate-quality record with each acceptance gate pre-evaluated."""

    aggregate_dice: float
    aggregate_iou: float
    aggregate_boundary_f: float
    category_dice: Mapping[str, float]
    meets_dice_threshold: bool
    meets_iou_threshold: bool
    meets_boundary_f_threshold: bool
    meets_category_floor: bool

    @property
    def meets_thresholds(self) -> bool:
        return (
            self.meets_dice_threshold
            and self.meets_iou_threshold
            and self.meets_boundary_f_threshold
            and self.meets_category_floor
        )


def aggregate_quality(results: Sequence[ImageQualityResult]) -> AggregateQuality:
    """Aggregate per-image results into gate-evaluated aggregate quality.

    Aggregate Dice/IoU/boundary-F are unweighted means over all results.
    Per-category Dice is the unweighted mean within each category; the
    category floor gate fails if any category's mean Dice is below
    ``CATEGORY_DICE_FLOOR``.
    """
    if not results:
        raise QualityMetricError("aggregate_quality requires at least one ImageQualityResult")

    dice_values = [r.dice for r in results]
    iou_values = [r.iou for r in results]
    boundary_values = [r.boundary_f for r in results]

    by_category: dict[str, list[float]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r.dice)
    category_dice = MappingProxyType(
        {category: float(np.mean(values)) for category, values in by_category.items()}
    )

    aggregate_dice = float(np.mean(dice_values))
    aggregate_iou = float(np.mean(iou_values))
    aggregate_boundary_f = float(np.mean(boundary_values))

    return AggregateQuality(
        aggregate_dice=aggregate_dice,
        aggregate_iou=aggregate_iou,
        aggregate_boundary_f=aggregate_boundary_f,
        category_dice=category_dice,
        meets_dice_threshold=aggregate_dice >= DICE_AGGREGATE_THRESHOLD,
        meets_iou_threshold=aggregate_iou >= IOU_AGGREGATE_THRESHOLD,
        meets_boundary_f_threshold=aggregate_boundary_f >= BOUNDARY_F_THRESHOLD,
        meets_category_floor=all(v >= CATEGORY_DICE_FLOOR for v in category_dice.values()),
    )


# -- Deterministic model-selection policy -------------------------------------------


@dataclass(frozen=True)
class CandidateReport:
    """One model's measured aggregate quality plus its operating-budget gates."""

    model_id: str
    aggregate: AggregateQuality
    within_budgets: bool
    manually_approved: bool

    def __post_init__(self) -> None:
        if self.model_id not in APPROVED_MODEL_IDS:
            raise QualityMetricError(f"model_id is not approved for this release: {self.model_id!r}")

    @property
    def eligible(self) -> bool:
        """Meets quality acceptance thresholds AND stays within latency/memory budgets."""
        return self.aggregate.meets_thresholds and self.within_budgets


@dataclass(frozen=True)
class SelectionResult:
    """The deterministic outcome of `select_model_candidate`."""

    winner: str | None
    reason: str


def select_model_candidate(full: CandidateReport, lite: CandidateReport) -> SelectionResult:
    """Deterministic U2NET-vs-U2NETP winner policy, frozen ahead of any bakeoff.

    Full U2NET (`full`) wins only if it is eligible (meets quality thresholds
    and stays within budgets), manually approved, AND beats U2NETP (`lite`)
    by at least `MODEL_DELTA_THRESHOLD` absolute in aggregate IoU or boundary
    F. Otherwise an eligible U2NETP wins. If neither wins, the result reports
    no model available -- never a silent fallback to an ineligible candidate.
    """
    if full.model_id != _FULL_MODEL_ID:
        raise QualityMetricError(f"full candidate must be {_FULL_MODEL_ID!r}, got {full.model_id!r}")
    if lite.model_id != _LITE_MODEL_ID:
        raise QualityMetricError(f"lite candidate must be {_LITE_MODEL_ID!r}, got {lite.model_id!r}")

    if full.eligible and full.manually_approved:
        delta_iou = full.aggregate.aggregate_iou - lite.aggregate.aggregate_iou
        delta_boundary_f = full.aggregate.aggregate_boundary_f - lite.aggregate.aggregate_boundary_f
        if delta_iou >= MODEL_DELTA_THRESHOLD or delta_boundary_f >= MODEL_DELTA_THRESHOLD:
            return SelectionResult(
                winner=full.model_id,
                reason="full model eligible, manually approved, and beats lite by "
                       f">={MODEL_DELTA_THRESHOLD} absolute in aggregate IoU or boundary F",
            )

    if lite.eligible:
        return SelectionResult(
            winner=lite.model_id,
            reason="lite model eligible; full model did not qualify to replace it",
        )

    return SelectionResult(
        winner=None,
        reason="no candidate is eligible within quality and operating budgets",
    )
