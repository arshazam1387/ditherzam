"""Immutable identity and value contracts shared across Smart Mask's core,
UI, workers, caches, and export.

Identity here is always content-based: a stable digest of decoded pixel
bytes plus explicit dimensions/alpha participation, never a filename or
path. Every identity type is a frozen dataclass with default equality/hash
so it can key a cache directly. Confidence/mask arrays are immutable
C-contiguous ``float32[H,W]`` in ``[0, 1]`` -- validated fail-closed and
published read-only; a shape mismatch is always an error, never silently
resized.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ditherzam.masking.settings import (
    EXPANSION_MAX_PX,
    EXPANSION_MIN_PX,
    FEATHER_MIN_PX,
    MaskTarget,
    SENSITIVITY_MAX,
    SENSITIVITY_MIN,
)

_SHA256_HEX_LENGTH = 64
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


class MaskContractError(Exception):
    """Raised when source, model, inference, mask, or probability data is invalid."""


def _validate_sha256_hex(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH or any(
        ch not in _SHA256_HEX_DIGITS for ch in value
    ):
        raise MaskContractError(f"{name} must be a lowercase SHA-256 hex digest, got {value!r}")


def _validate_non_empty_str(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MaskContractError(f"{name} must be a non-empty str, got {value!r}")


def _validate_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise MaskContractError(f"{name} must be a positive int, got {value!r}")


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise MaskContractError(f"{name} must be a bool, got {type(value).__name__}")


def _validate_int_range(value: object, name: str, minimum: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MaskContractError(f"{name} must be an int, got {type(value).__name__}")
    if not (minimum <= value <= maximum):
        raise MaskContractError(f"{name} must be within [{minimum}, {maximum}], got {value}")


def _validate_int_min(value: object, name: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MaskContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise MaskContractError(f"{name} must be >= {minimum}, got {value}")


@dataclass(frozen=True)
class SourceIdentity:
    """Content-based identity of a decoded source image.

    Never a filename or path: reopening changed bytes produces a different
    identity, and identical bytes from any origin produce the same one.
    """

    content_hash: str
    width: int
    height: int
    has_alpha: bool

    def __post_init__(self) -> None:
        _validate_sha256_hex(self.content_hash, "content_hash")
        _validate_positive_int(self.width, "width")
        _validate_positive_int(self.height, "height")
        _validate_bool(self.has_alpha, "has_alpha")


@dataclass(frozen=True)
class ModelIdentity:
    """Exact logical model identity: id, version, and content hash."""

    model_id: str
    model_version: str
    model_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.model_id, "model_id")
        _validate_non_empty_str(self.model_version, "model_version")
        _validate_sha256_hex(self.model_hash, "model_hash")


@dataclass(frozen=True)
class InferenceIdentity:
    """Identity of one probability-map-producing inference run.

    Keys the inference cache: source content + exact model + preprocessing
    version + which detected candidate. Sensitivity/target/invert/geometry
    are deliberately not part of this -- they derive a mask from the cached
    probability map without rerunning inference.
    """

    source: SourceIdentity
    model: ModelIdentity
    preprocessing_version: str
    candidate_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise MaskContractError("InferenceIdentity.source must be a SourceIdentity")
        if not isinstance(self.model, ModelIdentity):
            raise MaskContractError("InferenceIdentity.model must be a ModelIdentity")
        _validate_non_empty_str(self.preprocessing_version, "preprocessing_version")
        _validate_non_empty_str(self.candidate_id, "candidate_id")


@dataclass(frozen=True)
class MaskIdentity:
    """Identity of one derived master mask.

    Everything that can change the mask array's pixel values: the inference
    it derives from, plus every geometry/sensitivity/target/invert setting
    and the feather algorithm version. Outside-region compositing is
    deliberately not part of this -- it is layered on top by a separate
    outer-composite cache key.
    """

    inference: InferenceIdentity
    sensitivity: int
    target: MaskTarget
    invert: bool
    expansion_px: int
    feather_px: int
    feather_algorithm_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.inference, InferenceIdentity):
            raise MaskContractError("MaskIdentity.inference must be an InferenceIdentity")
        _validate_int_range(self.sensitivity, "sensitivity", SENSITIVITY_MIN, SENSITIVITY_MAX)
        if not isinstance(self.target, MaskTarget):
            raise MaskContractError(f"MaskIdentity.target must be a MaskTarget, got {self.target!r}")
        _validate_bool(self.invert, "invert")
        _validate_int_range(self.expansion_px, "expansion_px", EXPANSION_MIN_PX, EXPANSION_MAX_PX)
        _validate_int_min(self.feather_px, "feather_px", FEATHER_MIN_PX)
        _validate_non_empty_str(self.feather_algorithm_version, "feather_algorithm_version")


def validate_rgba_u8(array: object) -> np.ndarray:
    """Validate canonical straight ``uint8`` RGBA; return it unchanged."""
    if not isinstance(array, np.ndarray):
        raise MaskContractError(f"source array must be a numpy ndarray, got {type(array).__name__}")
    if array.dtype != np.uint8:
        raise MaskContractError(f"source array dtype must be uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 4:
        raise MaskContractError(f"source array shape must be (H, W, 4), got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise MaskContractError(f"source array must not be zero-size, got shape {array.shape}")
    return array


def validate_confidence_array(array: object, *, name: str = "confidence array") -> np.ndarray:
    """Validate an immutable C-contiguous ``float32[H,W]`` array in ``[0, 1]``."""
    if not isinstance(array, np.ndarray):
        raise MaskContractError(f"{name} must be a numpy ndarray, got {type(array).__name__}")
    if array.dtype != np.float32:
        raise MaskContractError(f"{name} dtype must be float32, got {array.dtype}")
    if array.ndim != 2:
        raise MaskContractError(f"{name} must be 2-D (H, W), got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise MaskContractError(f"{name} must not be zero-size, got shape {array.shape}")
    if not array.flags["C_CONTIGUOUS"]:
        raise MaskContractError(f"{name} must be C-contiguous")
    if not np.isfinite(array).all():
        raise MaskContractError(f"{name} must not contain NaN or Inf values")
    if array.min() < 0.0 or array.max() > 1.0:
        raise MaskContractError(f"{name} values must be within [0, 1]")
    return array


@dataclass(frozen=True, eq=False)
class ProbabilityMap:
    """One inference run's immutable adapter output.

    ``values`` is the raw per-pixel foreground confidence at source
    resolution -- 1.0 means foreground. Cached separately from any derived
    mask so sensitivity/target/invert/geometry/feather edits reuse it
    without rerunning inference.

    Equality/hash are keyed on ``identity`` alone: an inference identity fully
    determines the confidence payload, and a raw ndarray field cannot support
    default dataclass eq/hash anyway. ``values`` is stored as an owned,
    read-only C-contiguous copy so no external view can mutate it.
    """

    identity: InferenceIdentity
    values: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.identity, InferenceIdentity):
            raise MaskContractError("ProbabilityMap.identity must be an InferenceIdentity")
        validate_confidence_array(self.values, name="ProbabilityMap.values")
        expected_shape = (self.identity.source.height, self.identity.source.width)
        if self.values.shape != expected_shape:
            raise MaskContractError(
                f"ProbabilityMap.values shape {self.values.shape} does not match "
                f"source identity dimensions {expected_shape}; shape mismatch is "
                "never silently resized"
            )
        # Own the memory: a defensive C-contiguous copy so a caller's view/slice
        # of a larger writable base buffer can never mutate the stored payload.
        owned = np.array(self.values, dtype=np.float32, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "values", owned)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProbabilityMap):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.identity)


def source_identity(rgba_u8: np.ndarray) -> SourceIdentity:
    """Build a content-based ``SourceIdentity`` from canonical straight RGBA.

    ``rgba_u8`` must be ``uint8`` shape ``(H, W, 4)``. Identity is a SHA-256
    digest of the exact pixel bytes plus decoded dimensions and whether the
    alpha channel participates (any pixel with alpha != 255). This is
    content identity only -- there is no path or filename input.
    """
    array = validate_rgba_u8(rgba_u8)
    height, width = int(array.shape[0]), int(array.shape[1])
    content_hash = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    has_alpha = bool(np.any(array[..., 3] != 255))
    return SourceIdentity(content_hash=content_hash, width=width, height=height, has_alpha=has_alpha)
