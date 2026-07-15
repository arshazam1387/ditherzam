"""Deterministic source-resolution derivation and preview resizing for masks."""
from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image, ImageFilter

from ditherzam.masking.contracts import ProbabilityMap, validate_confidence_array
from ditherzam.masking.settings import (
    EXPANSION_MAX_PX,
    EXPANSION_MIN_PX,
    MaskTarget,
    SENSITIVITY_MAX,
    SENSITIVITY_MIN,
)

GEOMETRY_ALGORITHM_VERSION: Final = "separable-square-morphology-v1"
FEATHER_ALGORITHM_VERSION: Final = "pillow-gaussian-v1"
RESIZE_ALGORITHM_VERSION: Final = "pillow-box-v1"


class MaskGeometryError(ValueError):
    """Raised when mask geometry input is outside the frozen contract."""


def _int_in_range(value: object, name: str, minimum: int, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MaskGeometryError(f"{name} must be an int")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f", {maximum}"
        raise MaskGeometryError(f"{name} must be within [{minimum}{upper}]")
    return value


def _mask(array: object, name: str = "mask") -> np.ndarray:
    try:
        return validate_confidence_array(array, name=name)
    except Exception as exc:
        raise MaskGeometryError(str(exc)) from exc


def _immutable(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    result.flags.writeable = False
    return result


def sensitivity_threshold(sensitivity: int) -> float:
    """Map 0..100 sensitivity linearly to an inclusive confidence threshold."""
    value = _int_in_range(sensitivity, "sensitivity", SENSITIVITY_MIN, SENSITIVITY_MAX)
    return (SENSITIVITY_MAX - value) / float(SENSITIVITY_MAX)


def expand_contract(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate (positive) or erode (negative) a hard mask by source pixels."""
    source = _mask(mask)
    amount = _int_in_range(pixels, "pixels", EXPANSION_MIN_PX, EXPANSION_MAX_PX)
    binary = (source >= 0.5).astype(np.uint8)
    radius = abs(amount)
    if radius:
        # A square max/min filter is exactly two 1-D filters. Rolling uint32
        # counts keep work O(HW), avoid a uint64 2-D summed-area table, and retain
        # exact binary behavior even for isolated pixels and low-density masks.
        width = 2 * radius + 1
        horizontal = _rolling_counts(binary, radius, axis=1)
        horizontal = horizontal > 0 if amount > 0 else horizontal == width
        vertical = _rolling_counts(horizontal.astype(np.uint8), radius, axis=0)
        binary = vertical > 0 if amount > 0 else vertical == width
    return _immutable(binary.astype(np.float32))


def _rolling_counts(binary: np.ndarray, radius: int, *, axis: int) -> np.ndarray:
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(binary, padding, mode="constant", constant_values=0)
    cumulative_shape = list(padded.shape)
    cumulative_shape[axis] += 1
    cumulative = np.zeros(cumulative_shape, dtype=np.uint32)
    destination = [slice(None), slice(None)]
    destination[axis] = slice(1, None)
    np.cumsum(padded, axis=axis, dtype=np.uint32, out=cumulative[tuple(destination)])
    width = 2 * radius + 1
    upper = [slice(None), slice(None)]
    lower = [slice(None), slice(None)]
    upper[axis] = slice(width, None)
    lower[axis] = slice(None, -width)
    return cumulative[tuple(upper)] - cumulative[tuple(lower)]


def feather(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Apply a symmetric Gaussian soft edge with radius in source pixels."""
    source = _mask(mask)
    radius = _int_in_range(pixels, "pixels", 0)
    if radius == 0:
        return _immutable((source >= 0.5).astype(np.float32))
    encoded = np.rint(source * 255.0).astype(np.uint8)
    image = Image.fromarray(encoded, mode="L")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    return _immutable(np.clip(blurred / 255.0, 0.0, 1.0))


def derive_master_mask(
    probability: ProbabilityMap | np.ndarray | None,
    *,
    sensitivity: int,
    target: MaskTarget,
    invert: bool = False,
    expansion_px: int = 0,
    feather_px: int = 0,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Derive the immutable master mask in the frozen operation order."""
    if not isinstance(target, MaskTarget):
        raise MaskGeometryError("target must be a MaskTarget")
    if not isinstance(invert, bool):
        raise MaskGeometryError("invert must be a bool")
    threshold = sensitivity_threshold(sensitivity)
    _int_in_range(expansion_px, "expansion_px", EXPANSION_MIN_PX, EXPANSION_MAX_PX)
    _int_in_range(feather_px, "feather_px", 0)

    values = probability.values if isinstance(probability, ProbabilityMap) else probability
    if values is not None:
        values = _mask(values, "probability")
        shape = values.shape
        if source_shape is not None and tuple(source_shape) != shape:
            raise MaskGeometryError("source_shape does not match probability shape")
    else:
        if target is not MaskTarget.WHOLE_IMAGE:
            raise MaskGeometryError("probability is required unless target is Whole Image")
        if (
            not isinstance(source_shape, tuple)
            or len(source_shape) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in source_shape)
        ):
            raise MaskGeometryError("source_shape must be a positive (height, width) tuple")
        shape = source_shape

    if target is MaskTarget.WHOLE_IMAGE:
        # Whole Image is semantic, not geometry: controls disabled by the UI must
        # not alter it when settings are restored from a preset or stale snapshot.
        return _immutable(np.ones(shape, dtype=np.float32))
    subject = (values >= threshold).astype(np.float32)
    selected = subject if target is MaskTarget.SUBJECT else 1.0 - subject
    if invert:
        selected = 1.0 - selected
    shaped = expand_contract(selected, expansion_px)
    return feather(shaped, feather_px)


def resize_mask_area(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a master mask to preview shape using deterministic area sampling."""
    source = _mask(mask)
    if (
        not isinstance(target_shape, tuple)
        or len(target_shape) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in target_shape)
    ):
        raise MaskGeometryError("target_shape must be a positive (height, width) tuple")
    height, width = target_shape
    resized = Image.fromarray(source, mode="F").resize((width, height), resample=Image.Resampling.BOX)
    return _immutable(np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0))
