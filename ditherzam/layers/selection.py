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


def source_selection_to_document(
    selection: TemporarySelection,
    document_shape: tuple[int, int],
    *,
    layer_x: float,
    layer_y: float,
    scale_x: float,
    scale_y: float,
) -> TemporarySelection:
    """Map source coverage into document coordinates through x/y/scale."""
    height, width = _document_shape(document_shape)
    if not isinstance(selection, TemporarySelection):
        raise SelectionError("selection is required")
    values = (layer_x, layer_y, scale_x, scale_y)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ) or scale_x <= 0 or scale_y <= 0:
        raise SelectionError("layer mapping must use finite positive scales")

    source = selection.pixels
    dx = np.arange(width, dtype=np.float64) + 0.5
    dy = np.arange(height, dtype=np.float64) + 0.5
    inside_x = (dx >= layer_x) & (dx < layer_x + source.shape[1] * scale_x)
    inside_y = (dy >= layer_y) & (dy < layer_y + source.shape[0] * scale_y)
    sx = (dx - float(layer_x)) / float(scale_x) - 0.5
    sy = (dy - float(layer_y)) / float(scale_y) - 0.5
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx = sx - x0
    fy = sy - y0
    result = np.zeros((height, width), dtype=np.float64)
    for yy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
        clipped_y = np.clip(yy, 0, source.shape[0] - 1)
        for xx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
            clipped_x = np.clip(xx, 0, source.shape[1] - 1)
            result += (
                source[np.ix_(clipped_y, clipped_x)]
                * wy[:, None]
                * wx[None, :]
            )
    result *= inside_y[:, None] & inside_x[None, :]
    return TemporarySelection(np.floor(result + 0.5).astype(np.uint8))


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

# Creative selection primitives intentionally live in the Qt-free core.
def _document_shape(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple) or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise SelectionError("document shape must be positive (height, width)")
    return value


def _points(value: object, *, minimum: int) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (tuple, list)) or len(value) < minimum:
        raise SelectionError(f"selection requires at least {minimum} point(s)")
    result = []
    for point in value:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise SelectionError("each point must be (x, y)")
        x, y = point
        if any(isinstance(item, bool) or not isinstance(item, (int, float))
               or not math.isfinite(float(item)) for item in point):
            raise SelectionError("selection points must be finite numbers")
        result.append((float(x), float(y)))
    return tuple(result)


def _supersampled(document_shape, samples: int):
    height, width = _document_shape(document_shape)
    if type(samples) is not int or not 1 <= samples <= 16:
        raise SelectionError("samples must be an integer within 1..16")
    return height, width, samples


def rasterize_polygon_selection(
    document_shape: tuple[int, int],
    points,
    *,
    samples: int = 4,
) -> np.ndarray:
    """Rasterize a closed polygon with deterministic supersampled coverage."""
    from PIL import Image, ImageDraw

    height, width, scale = _supersampled(document_shape, samples)
    vertices = _points(points, minimum=3)
    image = Image.new("L", (width * scale, height * scale), 0)
    # Pixel centres at the supersampled resolution approximate area coverage.
    scaled = [(int(round(x * scale)), int(round(y * scale)))
              for x, y in vertices]
    ImageDraw.Draw(image).polygon(scaled, fill=255)
    values = np.asarray(image, dtype=np.uint16).reshape(
        height, scale, width, scale).sum(axis=(1, 3))
    divisor = scale * scale
    return ((values + divisor // 2) // divisor).astype(np.uint8)


def rasterize_freehand_selection(
    document_shape: tuple[int, int],
    points,
    *,
    diameter: float,
    samples: int = 4,
) -> np.ndarray:
    """Rasterize a round-capped freehand/lasso stroke into soft coverage."""
    from PIL import Image, ImageDraw

    height, width, scale = _supersampled(document_shape, samples)
    vertices = _points(points, minimum=1)
    if (isinstance(diameter, bool) or not isinstance(diameter, (int, float))
            or not math.isfinite(float(diameter)) or diameter <= 0):
        raise SelectionError("diameter must be a finite positive number")
    image = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(image)
    scaled = [(int(round(x * scale)), int(round(y * scale)))
              for x, y in vertices]
    line_width = max(1, int(round(float(diameter) * scale)))
    radius = line_width / 2.0
    if len(scaled) > 1:
        draw.line(scaled, fill=255, width=line_width, joint="curve")
    for x, y in (scaled if len(scaled) == 1 else (scaled[0], scaled[-1])):
        draw.ellipse((int(x - radius), int(y - radius),
                      int(x + radius), int(y + radius)), fill=255)
    values = np.asarray(image, dtype=np.uint16).reshape(
        height, scale, width, scale).sum(axis=(1, 3))
    divisor = scale * scale
    return ((values + divisor // 2) // divisor).astype(np.uint8)


def select_all(document_shape: tuple[int, int]) -> TemporarySelection:
    """Create full document coverage."""
    height, width = _document_shape(document_shape)
    return TemporarySelection(np.full((height, width), 255, dtype=np.uint8))


def invert_selection(selection: TemporarySelection) -> TemporarySelection:
    if not isinstance(selection, TemporarySelection):
        raise SelectionError("selection is required")
    return TemporarySelection(255 - selection.pixels)


def _selection_radius(selection, radius):
    if not isinstance(selection, TemporarySelection):
        raise SelectionError("selection is required")
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 64:
        raise SelectionError("radius must be an integer within 0..64")
    return selection, radius


def grow_selection(selection: TemporarySelection, radius: int) -> TemporarySelection:
    """Grow coverage from its >=50% contour by a square pixel radius."""
    selection, radius = _selection_radius(selection, radius)
    if radius == 0:
        return TemporarySelection(selection.pixels)
    from ditherzam.masking.geometry import expand_contract
    result = expand_contract(selection.pixels.astype(np.float32) / 255.0, radius)
    return TemporarySelection(np.floor(result * 255.0 + 0.5).astype(np.uint8))


def shrink_selection(selection: TemporarySelection, radius: int) -> TemporarySelection:
    """Shrink coverage from its >=50% contour by a square pixel radius."""
    selection, radius = _selection_radius(selection, radius)
    if radius == 0:
        return TemporarySelection(selection.pixels)
    from ditherzam.masking.geometry import expand_contract
    result = expand_contract(selection.pixels.astype(np.float32) / 255.0, -radius)
    return TemporarySelection(np.floor(result * 255.0 + 0.5).astype(np.uint8))


def feather_selection(selection: TemporarySelection, radius: int) -> TemporarySelection:
    """Gaussian-feather current soft coverage without hardening radius zero."""
    selection, radius = _selection_radius(selection, radius)
    if radius == 0:
        return TemporarySelection(selection.pixels)
    from PIL import Image, ImageFilter
    image = Image.fromarray(selection.pixels, mode="L")
    return TemporarySelection(np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.uint8))


def select_color_range(
    source: np.ndarray,
    target_rgb,
    *,
    tolerance: float = 0,
    softness: float = 0,
    respect_alpha: bool = True,
) -> TemporarySelection:
    """Select pixels by Euclidean RGB distance, with an optional soft falloff."""
    if (not isinstance(source, np.ndarray) or source.dtype != np.uint8
            or source.ndim != 3 or source.shape[2] not in (3, 4) or not source.size):
        raise SelectionError("source must be a non-empty uint8 RGB or RGBA array")
    target = np.asarray(target_rgb)
    if (target.shape != (3,) or not np.issubdtype(target.dtype, np.number)
            or np.any(~np.isfinite(target.astype(np.float64)))
            or np.any(target < 0) or np.any(target > 255)):
        raise SelectionError("target_rgb must contain three values within 0..255")
    for name, value in (("tolerance", tolerance), ("softness", softness)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or not 0 <= value <= 441.673):
            raise SelectionError(f"{name} must be finite within 0..441.673")
    if not isinstance(respect_alpha, bool):
        raise SelectionError("respect_alpha must be a bool")
    delta = source[..., :3].astype(np.float32) - target.astype(np.float32)
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    tolerance = float(tolerance)
    softness = float(softness)
    if softness == 0:
        coverage = np.where(distance <= tolerance, 255, 0).astype(np.uint8)
    else:
        coverage = np.floor(np.clip(
            (tolerance + softness - distance) / softness, 0.0, 1.0
        ) * 255.0 + 0.5).astype(np.uint8)
    if respect_alpha and source.shape[2] == 4:
        coverage = ((coverage.astype(np.uint16) * source[..., 3].astype(np.uint16)
                     + 127) // 255).astype(np.uint8)
    return TemporarySelection(coverage)
