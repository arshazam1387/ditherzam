"""Qt-free raster-mask inspection views derived from accepted render inputs."""
from __future__ import annotations

from enum import Enum

import numpy as np

from .model import RasterLayerMask


class InspectionMode(str, Enum):
    NORMAL = "Normal"
    RED_OVERLAY = "Red Overlay"
    MASK_ONLY = "Mask Only"


INSPECTION_RED = (255, 0, 0)
INSPECTION_RED_OPACITY = 128


def _selected_coverage(proxy, raster_mask: RasterLayerMask) -> np.ndarray:
    selected = proxy.layer_before_raster_mask_rgba
    map_x = proxy.mask_source_x
    map_y = proxy.mask_source_y
    if selected is None or map_x is None or map_y is None:
        raise ValueError("proxy must retain accepted pre-raster selected-layer data")
    source = raster_mask.pixels
    if (
        map_x.shape != (selected.shape[1],)
        or map_y.shape != (selected.shape[0],)
        or int(map_x.min(initial=0)) < 0
        or int(map_x.max(initial=0)) >= source.shape[1]
        or int(map_y.min(initial=0)) < 0
        or int(map_y.max(initial=0)) >= source.shape[0]
    ):
        raise ValueError("proxy source-index maps do not match the raster mask")
    base_alpha = selected[..., 3].astype(np.uint16)
    if not proxy.visible:
        return np.zeros(base_alpha.shape, dtype=np.uint8)
    if not raster_mask.enabled or raster_mask.density == 0:
        coverage = base_alpha
    else:
        effective = source[np.ix_(map_y, map_x)].astype(np.uint16)
        np.subtract(255, effective, out=effective)
        np.multiply(effective, raster_mask.density, out=effective)
        np.add(effective, 50, out=effective)
        np.floor_divide(effective, 100, out=effective)
        np.subtract(255, effective, out=effective)
        np.multiply(effective, base_alpha, out=effective)
        np.add(effective, 127, out=effective)
        np.floor_divide(effective, 255, out=effective)
        coverage = effective
    opacity = int(proxy.opacity)
    if opacity == 100:
        return coverage.astype(np.uint8)
    weighted = coverage.astype(np.uint16)
    np.multiply(weighted, opacity, out=weighted)
    np.add(weighted, 50, out=weighted)
    np.floor_divide(weighted, 100, out=weighted)
    return weighted.astype(np.uint8)


def _place_channel(canvas: np.ndarray, layer: np.ndarray, x: int, y: int) -> None:
    left, top = max(0, x), max(0, y)
    right = min(canvas.shape[1], x + layer.shape[1])
    bottom = min(canvas.shape[0], y + layer.shape[0])
    if left < right and top < bottom:
        canvas[top:bottom, left:right] = layer[
            top - y:bottom - y, left - x:right - x]


def inspect_mask(
    normal_composite: np.ndarray,
    proxy,
    raster_mask: RasterLayerMask,
    mode: InspectionMode,
) -> np.ndarray:
    """Create an inspection display without mutating accepted render inputs."""
    if not isinstance(mode, InspectionMode):
        raise ValueError("mode must be an InspectionMode")
    composite = np.asarray(normal_composite)
    if (
        composite.dtype != np.uint8 or composite.ndim != 3
        or composite.shape[2] != 4
    ):
        raise ValueError("normal_composite must be an RGBA uint8 array")
    if mode is InspectionMode.NORMAL:
        return normal_composite
    if not isinstance(raster_mask, RasterLayerMask):
        raise ValueError("raster_mask must be present")

    selected_coverage = _selected_coverage(proxy, raster_mask)
    coverage = np.zeros(composite.shape[:2], dtype=np.uint8)
    _place_channel(coverage, selected_coverage, proxy.x, proxy.y)
    if mode is InspectionMode.MASK_ONLY:
        result = np.empty_like(composite)
        result[..., :3] = coverage[..., None]
        result[..., 3] = 255
        return result

    base = proxy.layer_before_raster_mask_rgba[..., 3]
    if proxy.opacity != 100:
        base = (
            base.astype(np.uint16) * int(proxy.opacity) + 50
        ) // 100
        base = base.astype(np.uint8)
    placed_base = np.zeros(composite.shape[:2], dtype=np.uint8)
    if proxy.visible:
        _place_channel(placed_base, base, proxy.x, proxy.y)
    hidden = placed_base.astype(np.int16) - coverage.astype(np.int16)
    np.maximum(hidden, 0, out=hidden)
    amount = (
        hidden.astype(np.uint16) * INSPECTION_RED_OPACITY + 127
    ) // 255
    result = np.array(composite, copy=True, order="C")
    rgb = result[..., :3].astype(np.uint16)
    red = np.asarray(INSPECTION_RED, dtype=np.uint16)
    rgb = (
        rgb * (255 - amount[..., None])
        + red * amount[..., None] + 127
    ) // 255
    result[..., :3] = rgb.astype(np.uint8)
    return result
