"""Qt-free exact operations for source-sized raster layer masks."""
from __future__ import annotations

import numpy as np

from ditherzam.masking.contracts import ProbabilityMap, source_identity
from ditherzam.masking.geometry import MaskGeometryError, derive_master_mask
from ditherzam.masking.settings import MaskTarget, SmartMaskSettings


class RasterMaskOperationError(ValueError):
    """A raster-mask operation could not produce a valid candidate."""


def _pixels_u8(pixels: object) -> np.ndarray:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 2
        or not pixels.size
    ):
        raise RasterMaskOperationError(
            "raster mask pixels must be a non-empty 2-D uint8 array")
    return pixels


def invert_raster_mask(pixels: object) -> np.ndarray:
    return np.subtract(255, _pixels_u8(pixels), dtype=np.uint8)


def fill_raster_mask(shape: object, value: object) -> np.ndarray:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in shape)
    ):
        raise RasterMaskOperationError(
            "raster mask shape must be a positive (height, width) tuple")
    if value not in (0, 255) or isinstance(value, bool):
        raise RasterMaskOperationError("raster mask fill must be 0 or 255")
    return np.full(shape, value, dtype=np.uint8)


def freeze_smart_mask(
    probability: ProbabilityMap | None,
    settings: SmartMaskSettings,
    *,
    rgba: np.ndarray,
) -> np.ndarray:
    if not isinstance(settings, SmartMaskSettings):
        raise RasterMaskOperationError("Smart Mask settings are unavailable")
    if (
        not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.ndim != 3
        or rgba.shape[2] != 4
        or not rgba.size
    ):
        raise RasterMaskOperationError("source RGBA is unavailable")
    if settings.target is not MaskTarget.WHOLE_IMAGE:
        if probability is None:
            raise RasterMaskOperationError(
                "Smart Mask probability is unavailable or pending")
        if probability.identity.source != source_identity(rgba):
            raise RasterMaskOperationError(
                "Smart Mask probability does not match the selected layer source")
    try:
        master = derive_master_mask(
            probability,
            sensitivity=settings.sensitivity,
            target=settings.target,
            invert=settings.invert,
            expansion_px=settings.expansion_px,
            feather_px=settings.feather_px,
            source_shape=rgba.shape[:2],
        )
    except MaskGeometryError as exc:
        raise RasterMaskOperationError(str(exc)) from exc
    return np.floor(
        np.asarray(master, dtype=np.float32) * np.float32(255.0)
        + np.float32(0.5)
    ).astype(np.uint8)
