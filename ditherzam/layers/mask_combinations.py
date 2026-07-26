"""Typed, deterministic composition of raster-mask candidates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from ditherzam.masking.contracts import SourceIdentity


class MaskCombinationError(ValueError):
    """A mask-combination request violates the MT-15 contract."""


class MaskCombinationMode(Enum):
    REPLACE = "Replace"
    ADD = "Add"
    SUBTRACT = "Subtract"
    INTERSECT = "Intersect"


class MaskCandidateOrigin(Enum):
    SMART = "Smart"
    IMPORTED = "Imported"
    LUMINANCE = "Luminance"
    GRADIENT = "Gradient"
    PATTERN = "Pattern"


@dataclass(frozen=True)
class MaskCandidate:
    """Owned candidate crossing the common Smart/import/generator boundary."""

    pixels: np.ndarray
    origin: MaskCandidateOrigin
    label: str
    source_identity: SourceIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.origin, MaskCandidateOrigin):
            raise MaskCombinationError("candidate origin is unsupported")
        if not isinstance(self.source_identity, SourceIdentity):
            raise MaskCombinationError("candidate source identity is unavailable")
        if not isinstance(self.label, str) or not self.label.strip():
            raise MaskCombinationError("candidate label cannot be empty")
        pixels = self.pixels
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                or pixels.ndim != 2 or not pixels.size):
            raise MaskCombinationError(
                "candidate pixels must be a non-empty 2-D uint8 array")
        owned = np.array(pixels, dtype=np.uint8, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "pixels", owned)
        object.__setattr__(self, "label", self.label.strip())


def combine_mask_candidate(
    current: np.ndarray,
    candidate: MaskCandidate,
    mode: MaskCombinationMode,
) -> MaskCandidate:
    """Combine soft coverage using the four exact public operations."""
    if not isinstance(candidate, MaskCandidate):
        raise MaskCombinationError("a typed mask candidate is required")
    if not isinstance(mode, MaskCombinationMode):
        raise MaskCombinationError("combination mode is unsupported")
    if (not isinstance(current, np.ndarray) or current.dtype != np.uint8
            or current.ndim != 2 or current.shape != candidate.pixels.shape):
        raise MaskCombinationError(
            "current mask must be uint8 and match the candidate shape")
    a = current.astype(np.uint16)
    b = candidate.pixels.astype(np.uint16)
    if mode is MaskCombinationMode.REPLACE:
        result = candidate.pixels
    elif mode is MaskCombinationMode.ADD:
        result = np.minimum(a + b, 255).astype(np.uint8)
    elif mode is MaskCombinationMode.SUBTRACT:
        result = np.maximum(a.astype(np.int16) - b.astype(np.int16), 0).astype(
            np.uint8)
    else:
        result = ((a * b + 127) // 255).astype(np.uint8)
    return MaskCandidate(
        result, candidate.origin, f"{mode.value} · {candidate.label}",
        candidate.source_identity)


def capped_combination_preview(
    current: np.ndarray,
    candidate: MaskCandidate,
    mode: MaskCombinationMode,
    cap: int = 720,
) -> np.ndarray:
    """Return an immutable nearest-sampled preview without retaining extra planes."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise MaskCombinationError("preview cap must be a positive integer")
    if current.shape != candidate.pixels.shape:
        raise MaskCombinationError("current mask and candidate must match")
    height, width = current.shape
    scale = min(1.0, cap / max(height, width))
    out_height = max(1, int(round(height * scale)))
    out_width = max(1, int(round(width * scale)))
    rows = np.minimum(
        (np.arange(out_height, dtype=np.int64) * height) // out_height,
        height - 1)
    columns = np.minimum(
        (np.arange(out_width, dtype=np.int64) * width) // out_width,
        width - 1)
    sampled = MaskCandidate(
        candidate.pixels[np.ix_(rows, columns)],
        candidate.origin, candidate.label, candidate.source_identity)
    preview = combine_mask_candidate(
        current[np.ix_(rows, columns)], sampled, mode).pixels
    preview.flags.writeable = False
    return preview
