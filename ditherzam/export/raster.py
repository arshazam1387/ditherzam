from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ditherzam.masking.composite import flatten_rgba_white


class RasterExportError(ValueError):
    """Raised when an array or file extension cannot be exported safely."""


def _canonical_u8(image: object) -> np.ndarray:
    """Validate raster geometry, then perform the historical uint8 coercion."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        if 0 in arr.shape:
            raise RasterExportError("raster image must not be empty")
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        if 0 in arr.shape[:2]:
            raise RasterExportError("raster image must not be empty")
    else:
        raise RasterExportError(
            "raster image shape must be (H, W), (H, W, 3), or (H, W, 4)"
        )
    if arr.dtype.kind == "b":
        raise RasterExportError("boolean raster values are not meaningful uint8-like data")
    if arr.dtype.kind not in "uif":
        raise RasterExportError("raster image values must be numeric")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise RasterExportError("raster image values must be finite")
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_raster(image_u8: np.ndarray, path) -> Path:
    """Save grayscale/RGB/RGBA raster data with explicit format semantics.

    PNG preserves straight RGBA bytes exactly. JPEG cannot store transparency,
    so straight RGBA is deterministically flattened onto white before encoding.
    Numeric uint8-like inputs retain the historical clip-and-cast behavior.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise RasterExportError(f"unsupported raster extension: {ext or '<none>'}")
    arr = _canonical_u8(image_u8)
    if ext in (".jpg", ".jpeg"):
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = flatten_rgba_white(arr)
        Image.fromarray(arr).convert("RGB").save(path, "JPEG", quality=95)
    else:
        Image.fromarray(arr).save(path, "PNG")
    return path
