"""Exact still-frame export for completed composition frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ditherzam.export.raster import save_raster
from ditherzam.masking.composite import flatten_rgba_white


def export_frame(frame, path) -> Path:
    """Export one canonical RGB or RGBA composition frame."""
    if not isinstance(frame, np.ndarray):
        raise ValueError("frame must be a numpy ndarray")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(
            f"frame shape must be (H, W, 3) or (H, W, 4), got {frame.shape}"
        )
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame must not be empty")

    destination = Path(path)
    if destination.suffix.lower() in (".jpg", ".jpeg") and frame.shape[2] == 4:
        return save_raster(flatten_rgba_white(frame), destination)
    return save_raster(frame, destination)
