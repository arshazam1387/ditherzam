"""Temporary document-coordinate selections for raster-mask editing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os

import numpy as np

try:
    from ditherzam._native import _selection as _native_selection
except ImportError:
    _native_selection = None


class SelectionError(ValueError):
    pass


class SelectionShape(Enum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"


class SelectionOperation(Enum):
    REPLACE = "replace"
    ADD = "add"
    SUBTRACT = "subtract"


@dataclass(frozen=True)
class TemporarySelection:
    """Immutable grayscale coverage in document pixel coordinates."""

    pixels: np.ndarray

    def __post_init__(self) -> None:
        value = self.pixels
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.uint8
            or value.ndim != 2
            or not value.size
        ):
            raise SelectionError("selection must be a non-empty 2-D uint8 array")
        owned = np.array(value, dtype=np.uint8, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "pixels", owned)


def _bounds(bounds: object) -> tuple[float, float, float, float]:
    if not isinstance(bounds, tuple) or len(bounds) != 4:
        raise SelectionError("bounds must be (x0, y0, x1, y1)")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in bounds
    ):
        raise SelectionError("selection bounds must be finite numbers")
    x0, y0, x1, y1 = map(float, bounds)
    if x0 == x1 or y0 == y1:
        raise SelectionError("selection bounds must have non-zero area")
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def rasterize_selection(
    document_shape: tuple[int, int],
    shape: SelectionShape,
    bounds: tuple[float, float, float, float],
    *,
    samples: int = 4,
) -> np.ndarray:
    """Rasterize a soft selection deterministically using fixed subpixel samples."""
    if (
        not isinstance(document_shape, tuple)
        or len(document_shape) != 2
        or any(type(value) is not int or value <= 0 for value in document_shape)
    ):
        raise SelectionError("document shape must be positive (height, width)")
    if not isinstance(shape, SelectionShape):
        raise SelectionError("shape must be Rectangle or Ellipse")
    if type(samples) is not int or not 1 <= samples <= 16:
        raise SelectionError("samples must be an integer within 1..16")
    x0, y0, x1, y1 = _bounds(bounds)
    height, width = document_shape
    coverage = np.zeros((height, width), dtype=np.uint16)
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    for oy in offsets:
        ys = np.arange(height, dtype=np.float64)[:, None] + oy
        for ox in offsets:
            xs = np.arange(width, dtype=np.float64)[None, :] + ox
            if shape is SelectionShape.RECTANGLE:
                inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            else:
                cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                rx, ry = (x1 - x0) * 0.5, (y1 - y0) * 0.5
                inside = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
            coverage += inside
    divisor = samples * samples
    return ((coverage * 255 + divisor // 2) // divisor).astype(np.uint8)


def combine_selection(
    current: TemporarySelection | None,
    candidate: np.ndarray,
    operation: SelectionOperation,
) -> TemporarySelection:
    if not isinstance(operation, SelectionOperation):
        raise SelectionError("operation must be Replace, Add, or Subtract")
    value = TemporarySelection(candidate).pixels
    if operation is SelectionOperation.REPLACE:
        return TemporarySelection(value)
    if current is None:
        current = TemporarySelection(np.zeros(value.shape, dtype=np.uint8))
    elif current.pixels.shape != value.shape:
        raise SelectionError("selection dimensions must match the document")
    if operation is SelectionOperation.ADD:
        result = np.minimum(
            current.pixels.astype(np.uint16) + value.astype(np.uint16), 255)
    else:
        result = np.maximum(
            current.pixels.astype(np.int16) - value.astype(np.int16), 0)
    return TemporarySelection(result.astype(np.uint8))


def _validate_selection_to_source(
    selection: TemporarySelection,
    source_shape: tuple[int, int],
    *,
    layer_x: float,
    layer_y: float,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, float, float, float, float]:
    if (
        not isinstance(source_shape, tuple)
        or len(source_shape) != 2
        or any(type(value) is not int or value <= 0 for value in source_shape)
    ):
        raise SelectionError("source shape must be positive (height, width)")
    values = (layer_x, layer_y, scale_x, scale_y)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ) or scale_x <= 0 or scale_y <= 0:
        raise SelectionError("layer mapping must use finite positive scales")
    height, width = source_shape
    return (
        height,
        width,
        float(layer_x),
        float(layer_y),
        float(scale_x),
        float(scale_y),
    )


def _selection_to_source_reference(
    selection: TemporarySelection,
    source_shape: tuple[int, int],
    *,
    layer_x: float,
    layer_y: float,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Independently callable NumPy reference for differential verification."""
    height, width, layer_x, layer_y, scale_x, scale_y = (
        _validate_selection_to_source(
            selection,
            source_shape,
            layer_x=layer_x,
            layer_y=layer_y,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    )
    # A source pixel centre is transformed into document coordinates, then
    # sampled from document coverage with bilinear interpolation.
    dx = layer_x + (np.arange(width) + 0.5) * scale_x - 0.5
    dy = layer_y + (np.arange(height) + 0.5) * scale_y - 0.5
    x0 = np.floor(dx).astype(np.int64)
    y0 = np.floor(dy).astype(np.int64)
    fx = dx - x0
    fy = dy - y0
    source = selection.pixels
    result = np.zeros((height, width), dtype=np.float64)
    for yy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
        valid_y = (yy >= 0) & (yy < source.shape[0])
        for xx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
            valid_x = (xx >= 0) & (xx < source.shape[1])
            valid = valid_y[:, None] & valid_x[None, :]
            clipped_y = np.clip(yy, 0, source.shape[0] - 1)
            clipped_x = np.clip(xx, 0, source.shape[1] - 1)
            sample = source[np.ix_(clipped_y, clipped_x)]
            result += sample * wy[:, None] * wx[None, :] * valid
    return np.floor(result + 0.5).astype(np.uint8)


def _selection_to_source_native(
    selection: TemporarySelection,
    source_shape: tuple[int, int],
    *,
    layer_x: float,
    layer_y: float,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Independently callable native seam; it never silently falls back."""
    height, width, layer_x, layer_y, scale_x, scale_y = (
        _validate_selection_to_source(
            selection,
            source_shape,
            layer_x=layer_x,
            layer_y=layer_y,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    )
    if _native_selection is None:
        raise RuntimeError("native selection extension is unavailable")
    return _native_selection.selection_to_source_u8(
        selection.pixels,
        height,
        width,
        layer_x,
        layer_y,
        scale_x,
        scale_y,
    )


def selection_to_source(
    selection: TemporarySelection,
    source_shape: tuple[int, int],
    *,
    layer_x: float,
    layer_y: float,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Map document coverage to source pixels through inverse x/y/scale."""
    kwargs = {
        "layer_x": layer_x,
        "layer_y": layer_y,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    if _native_selection is None or os.environ.get("DITHERZAM_DISABLE_NATIVE") == "1":
        return _selection_to_source_reference(selection, source_shape, **kwargs)
    return _selection_to_source_native(selection, source_shape, **kwargs)


def restrict_mask_edit(
    before: np.ndarray, edited: np.ndarray, selection_coverage: np.ndarray
) -> np.ndarray:
    """Blend an edit through soft selection coverage with exact uint8 rounding."""
    values = (before, edited, selection_coverage)
    if any(
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 2
        for value in values
    ) or not (before.shape == edited.shape == selection_coverage.shape):
        raise SelectionError("mask edit and selection must be matching 2-D uint8 arrays")
    weight = selection_coverage.astype(np.uint32)
    total = (
        before.astype(np.uint32) * (255 - weight)
        + edited.astype(np.uint32) * weight
        + 127
    )
    return (total // 255).astype(np.uint8)
