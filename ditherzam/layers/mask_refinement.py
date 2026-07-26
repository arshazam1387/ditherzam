"""Qt-free Smart-derived raster-mask refinement transactions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ditherzam.masking.contracts import ProbabilityMap, source_identity
from ditherzam.masking.geometry import (
    MaskGeometryError, derive_master_mask, expand_contract, feather)
from ditherzam.render import RenderCancelled
from ditherzam.masking.settings import MaskTarget


class MaskRefinementError(ValueError):
    """A Smart refinement request violates the frozen MT-12 contract."""


class MaskRefinementKind(Enum):
    NONE = "none"
    GROW = "grow"
    SHRINK = "shrink"


@dataclass(frozen=True)
class SmartRefinementSpec:
    """Transaction-local Smart threshold and repair settings."""

    target: MaskTarget
    sensitivity: int
    invert: bool = False
    morphology: MaskRefinementKind = MaskRefinementKind.NONE
    morphology_radius: int = 0
    feather_radius: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.target, MaskTarget):
            raise MaskRefinementError("target must be a MaskTarget")
        if (
            isinstance(self.sensitivity, bool)
            or not isinstance(self.sensitivity, int)
            or not 0 <= self.sensitivity <= 100
        ):
            raise MaskRefinementError("sensitivity must be within [0, 100]")
        if not isinstance(self.invert, bool):
            raise MaskRefinementError("invert must be a bool")
        if (
            not isinstance(self.morphology, MaskRefinementKind)
        ):
            raise MaskRefinementError("morphology must be None, Grow, or Shrink")
        for name, value in (
            ("morphology_radius", self.morphology_radius),
            ("feather_radius", self.feather_radius),
        ):
            if (
                isinstance(value, bool) or not isinstance(value, int)
                or not 0 <= value <= 64
            ):
                raise MaskRefinementError(f"{name} must be within [0, 64]")


def _half_up(value: float) -> int:
    return int(value + 0.5)


def capped_refinement_shape(
    source_shape: tuple[int, int], max_side: int = 720
) -> tuple[int, int]:
    if (
        not isinstance(source_shape, tuple)
        or len(source_shape) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in source_shape)
        or isinstance(max_side, bool)
        or not isinstance(max_side, int)
        or max_side <= 0
    ):
        raise MaskRefinementError("source shape and preview cap must be positive")
    height, width = source_shape
    if max(height, width) <= max_side:
        return source_shape
    ratio = max_side / float(max(height, width))
    return max(1, _half_up(height * ratio)), max(1, _half_up(width * ratio))


def scaled_refinement_radius(
    radius: int, source_shape: tuple[int, int], preview_shape: tuple[int, int]
) -> int:
    SmartRefinementSpec(
        MaskTarget.SUBJECT, 50, morphology_radius=radius)
    if source_shape == preview_shape:
        return radius
    return _half_up(radius * max(preview_shape) / float(max(source_shape)))


def _derive(
    probability: ProbabilityMap | None,
    spec: SmartRefinementSpec,
    source_shape: tuple[int, int],
    *,
    morphology_radius: int,
    feather_radius: int,
    is_cancelled=None,
) -> np.ndarray:
    cancelled = is_cancelled or (lambda: False)
    if cancelled():
        raise RenderCancelled
    try:
        result = derive_master_mask(
            probability,
            sensitivity=spec.sensitivity,
            target=spec.target,
            invert=False, expansion_px=0, feather_px=0,
            source_shape=source_shape,
        )
        if cancelled():
            raise RenderCancelled
        if spec.invert:
            result = np.float32(1.0) - result
        if cancelled():
            raise RenderCancelled
        expansion = (
            morphology_radius if spec.morphology is MaskRefinementKind.GROW
            else -morphology_radius
            if spec.morphology is MaskRefinementKind.SHRINK else 0)
        result = expand_contract(result, expansion)
        if cancelled():
            raise RenderCancelled
        result = feather(result, feather_radius)
        if cancelled():
            raise RenderCancelled
    except MaskGeometryError as exc:
        raise MaskRefinementError(str(exc)) from exc
    return np.floor(
        np.asarray(result, dtype=np.float32) * np.float32(255.0)
        + np.float32(0.5)
    ).astype(np.uint8)


def derive_refined_smart_mask(
    probability: ProbabilityMap | None,
    spec: SmartRefinementSpec,
    *,
    rgba: np.ndarray,
    is_cancelled=None,
) -> np.ndarray:
    """Derive exact source-resolution uint8 coverage for Confirm."""
    if not isinstance(spec, SmartRefinementSpec):
        raise MaskRefinementError("refinement spec is unavailable")
    if (
        not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.ndim != 3
        or rgba.shape[2] != 4
        or not rgba.size
    ):
        raise MaskRefinementError("source RGBA is unavailable")
    if spec.target is not MaskTarget.WHOLE_IMAGE:
        if probability is None:
            raise MaskRefinementError(
                "Smart Mask probability is unavailable or pending")
        if probability.identity.source != source_identity(rgba):
            raise MaskRefinementError(
                "Smart Mask probability does not match the selected layer source")
    return _derive(
        probability, spec, rgba.shape[:2],
        morphology_radius=spec.morphology_radius,
        feather_radius=spec.feather_radius, is_cancelled=is_cancelled)


def derive_refined_smart_preview(
    probability: ProbabilityMap | None,
    spec: SmartRefinementSpec,
    *,
    rgba: np.ndarray,
    max_side: int = 720,
    is_cancelled=None,
) -> np.ndarray:
    """Derive display-only coverage at a never-upscaled capped resolution."""
    if not isinstance(spec, SmartRefinementSpec):
        raise MaskRefinementError("refinement spec is unavailable")
    if (
        not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.ndim != 3
        or rgba.shape[2] != 4
        or not rgba.size
    ):
        raise MaskRefinementError("source RGBA is unavailable")
    if spec.target is not MaskTarget.WHOLE_IMAGE:
        if probability is None:
            raise MaskRefinementError(
                "Smart Mask probability is unavailable or pending")
        if probability.identity.source != source_identity(rgba):
            raise MaskRefinementError(
                "Smart Mask probability does not match the selected layer source")
    preview_shape = capped_refinement_shape(rgba.shape[:2], max_side)
    morphology_radius = scaled_refinement_radius(
        spec.morphology_radius, rgba.shape[:2], preview_shape)
    feather_radius = scaled_refinement_radius(
        spec.feather_radius, rgba.shape[:2], preview_shape)
    return _derive(
        probability, spec, preview_shape,
        morphology_radius=morphology_radius,
        feather_radius=feather_radius, is_cancelled=is_cancelled)
