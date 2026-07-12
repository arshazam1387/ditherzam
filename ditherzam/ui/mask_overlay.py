"""Display-only Smart Mask overlay (never cached or exported)."""
from __future__ import annotations

import numpy as np

from ditherzam.masking.contracts import validate_confidence_array


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray,
                       color: tuple[int, int, int] = (255, 48, 96),
                       opacity: float = 0.35) -> np.ndarray:
    """Return a tinted copy showing selected-mask coverage.

    Alpha, when present, is copied unchanged. Inputs are never mutated.
    """
    src = np.asarray(image)
    coverage = validate_confidence_array(mask, name="mask")
    if (src.dtype != np.uint8 or src.ndim != 3 or src.shape[2] not in (3, 4)
            or src.shape[:2] != coverage.shape):
        raise ValueError("image and mask must be aligned uint8 RGB/RGBA and float32 mask")
    if not (0.0 <= float(opacity) <= 1.0):
        raise ValueError("opacity must be within [0, 1]")
    tint = np.asarray(color, dtype=np.uint8)
    if tint.shape != (3,):
        raise ValueError("color must contain three uint8 channels")
    out = src.copy()
    weight = coverage[..., None] * np.float32(opacity)
    out[..., :3] = np.floor(
        src[..., :3].astype(np.float32) * (1.0 - weight)
        + tint.astype(np.float32) * weight + 0.5
    ).astype(np.uint8)
    return out
