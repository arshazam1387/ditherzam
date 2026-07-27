"""Deterministic, Qt-free painting for source-sized raster layer masks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os

import numpy as np

from .mask_contracts import document_to_mask_point

try:
    from ditherzam._native import _brush
except ImportError:
    _brush = None


class BrushMode(Enum):
    REVEAL = "reveal"
    HIDE = "hide"


def _strict_percent(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be an integer within 0..100")
    return value


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class BrushSettings:
    """A circular brush whose ``size`` is its diameter in document units."""

    size: float
    hardness: int
    strength: int
    mode: BrushMode

    def __post_init__(self) -> None:
        size = _finite_number(self.size, "size")
        if size <= 0:
            raise ValueError("size must be positive")
        if not isinstance(self.mode, BrushMode):
            raise ValueError("mode must be a BrushMode")
        _strict_percent(self.hardness, "hardness")
        _strict_percent(self.strength, "strength")
        object.__setattr__(self, "size", size)

    @property
    def spacing(self) -> float:
        """Fixed center-to-center stamp spacing in document units."""
        return max(1.0, self.size * 0.25)


@dataclass(frozen=True)
class DirtyRect:
    """A non-empty, half-open source-pixel rectangle."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("dirty rectangle coordinates must be integers")
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("dirty rectangle must be non-empty and nonnegative")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def union(self, other: DirtyRect | None) -> DirtyRect:
        if other is None:
            return self
        if not isinstance(other, DirtyRect):
            raise ValueError("other must be a DirtyRect or None")
        return DirtyRect(
            min(self.x0, other.x0), min(self.y0, other.y0),
            max(self.x1, other.x1), max(self.y1, other.y1),
        )


def _validate_buffer(buffer: np.ndarray) -> np.ndarray:
    if (
        not isinstance(buffer, np.ndarray)
        or buffer.ndim != 2
        or buffer.dtype != np.uint8
        or not buffer.flags.writeable
    ):
        raise ValueError("buffer must be a writable 2D uint8 NumPy array")
    return buffer


def _half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _stamp_mask_brush_reference(
    buffer: np.ndarray,
    document_x: float,
    document_y: float,
    settings: BrushSettings,
    *,
    layer_x: float = 0.0,
    layer_y: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotation_degrees: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> DirtyRect | None:
    """Apply one circular stamp and return its minimal changed-pixel bounds."""
    work = _validate_buffer(buffer)
    if not isinstance(settings, BrushSettings):
        raise ValueError("settings must be BrushSettings")
    # Reuse the architecture contract as the canonical transform validator.
    document_to_mask_point(
        document_x, document_y, layer_x=layer_x, layer_y=layer_y,
        scale_x=scale_x, scale_y=scale_y,
        rotation_degrees=rotation_degrees, flip_x=flip_x, flip_y=flip_y,
    )
    if settings.strength == 0 or work.size == 0:
        return None

    cx = float(document_x)
    cy = float(document_y)
    lx = float(layer_x)
    ly = float(layer_y)
    sx = float(scale_x)
    sy = float(scale_y)
    radius = settings.size * 0.5
    height, width = work.shape
    x0 = max(0, math.ceil((cx - radius - lx) / sx - 0.5))
    x1 = min(width, math.floor((cx + radius - lx) / sx - 0.5) + 1)
    y0 = max(0, math.ceil((cy - radius - ly) / sy - 0.5))
    y1 = min(height, math.floor((cy + radius - ly) / sy - 0.5) + 1)
    if x0 >= x1 or y0 >= y1:
        return None

    x_distance_sq = (
        lx + (np.arange(x0, x1, dtype=np.float64) + 0.5) * sx - cx
    ) ** 2
    hard_radius = radius * settings.hardness / 100.0
    changed_x0, changed_y0 = width, height
    changed_x1 = changed_y1 = 0

    for y in range(y0, y1):
        dy = ly + (y + 0.5) * sy - cy
        distance = np.sqrt(x_distance_sq + dy * dy)
        inside = distance <= radius
        if not np.any(inside):
            continue
        if settings.hardness == 100:
            coverage = np.where(inside, 255, 0).astype(np.int32)
        else:
            coverage = _half_up(
                np.clip((radius - distance) / (radius - hard_radius), 0.0, 1.0) * 255.0
            )
            if hard_radius > 0:
                coverage[distance <= hard_radius] = 255
        amount = (coverage * settings.strength + 50) // 100
        row = work[y, x0:x1]
        old = row.astype(np.int32)
        if settings.mode is BrushMode.REVEAL:
            new = old + ((255 - old) * amount + 127) // 255
        else:
            new = old - (old * amount + 127) // 255
        changed = new != old
        if not np.any(changed):
            continue
        row[changed] = new[changed].astype(np.uint8)
        indices = np.flatnonzero(changed)
        changed_x0 = min(changed_x0, x0 + int(indices[0]))
        changed_x1 = max(changed_x1, x0 + int(indices[-1]) + 1)
        changed_y0 = min(changed_y0, y)
        changed_y1 = max(changed_y1, y + 1)

    if changed_x1 == 0:
        return None
    return DirtyRect(changed_x0, changed_y0, changed_x1, changed_y1)


def _stamp_mask_brush_native(
    buffer: np.ndarray,
    document_x: float,
    document_y: float,
    settings: BrushSettings,
    *,
    layer_x: float = 0.0,
    layer_y: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotation_degrees: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> DirtyRect | None:
    """Validated native-only seam used by differential tests."""
    work = _validate_buffer(buffer)
    if not isinstance(settings, BrushSettings):
        raise ValueError("settings must be BrushSettings")
    document_to_mask_point(
        document_x, document_y, layer_x=layer_x, layer_y=layer_y,
        scale_x=scale_x, scale_y=scale_y,
        rotation_degrees=rotation_degrees, flip_x=flip_x, flip_y=flip_y,
    )
    if settings.strength == 0 or work.size == 0:
        return None
    if _brush is None:
        raise RuntimeError("native brush extension is unavailable")

    cx = float(document_x)
    cy = float(document_y)
    lx = float(layer_x)
    ly = float(layer_y)
    sx = float(scale_x)
    sy = float(scale_y)
    radius = settings.size * 0.5
    height, width = work.shape
    x0 = max(0, math.ceil((cx - radius - lx) / sx - 0.5))
    x1 = min(width, math.floor((cx + radius - lx) / sx - 0.5) + 1)
    y0 = max(0, math.ceil((cy - radius - ly) / sy - 0.5))
    y1 = min(height, math.floor((cy + radius - ly) / sy - 0.5) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    bounds = _brush.stamp_mask_brush_u8(
        work, x0, x1, y0, y1, cx, cy, lx, ly, sx, sy, radius,
        settings.hardness, settings.strength,
        0 if settings.mode is BrushMode.REVEAL else 1,
    )
    return None if bounds is None else DirtyRect(*bounds)


def stamp_mask_brush(
    buffer: np.ndarray,
    document_x: float,
    document_y: float,
    settings: BrushSettings,
    *,
    layer_x: float = 0.0,
    layer_y: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotation_degrees: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> DirtyRect | None:
    """Apply one circular stamp, preferring the exact native backend."""
    kwargs = dict(
        layer_x=layer_x, layer_y=layer_y, scale_x=scale_x, scale_y=scale_y,
        rotation_degrees=rotation_degrees, flip_x=flip_x, flip_y=flip_y,
    )
    implementation = (
        _stamp_mask_brush_reference
        if _brush is None or os.environ.get("DITHERZAM_DISABLE_NATIVE") == "1"
        else _stamp_mask_brush_native
    )
    return implementation(
        buffer, document_x, document_y, settings, **kwargs
    )


# A concise alternate name for callers that already know they are editing a mask.
stamp_brush = stamp_mask_brush


class BrushStroke:
    """Incrementally sample a polyline using fixed spacing and carried residual."""

    def __init__(
        self,
        buffer: np.ndarray,
        settings: BrushSettings,
        *,
        layer_x: float = 0.0,
        layer_y: float = 0.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        rotation_degrees: float = 0.0,
        flip_x: bool = False,
        flip_y: bool = False,
    ) -> None:
        self.buffer = _validate_buffer(buffer)
        if not isinstance(settings, BrushSettings):
            raise ValueError("settings must be BrushSettings")
        # Validate even before the first event.
        document_to_mask_point(
            0.0, 0.0, layer_x=layer_x, layer_y=layer_y,
            scale_x=scale_x, scale_y=scale_y,
            rotation_degrees=rotation_degrees, flip_x=flip_x, flip_y=flip_y,
        )
        self.settings = settings
        self._transform = {
            "layer_x": layer_x, "layer_y": layer_y,
            "scale_x": scale_x, "scale_y": scale_y,
            "rotation_degrees": rotation_degrees,
            "flip_x": flip_x, "flip_y": flip_y,
        }
        self._last: tuple[float, float] | None = None
        self._distance_to_next = settings.spacing
        self.dirty_rect: DirtyRect | None = None

    def _stamp(self, x: float, y: float) -> DirtyRect | None:
        dirty = stamp_mask_brush(
            self.buffer, x, y, self.settings, **self._transform
        )
        if dirty is not None:
            self.dirty_rect = dirty if self.dirty_rect is None else self.dirty_rect.union(dirty)
        return dirty

    def start(self, document_x: float, document_y: float) -> DirtyRect | None:
        if self._last is not None:
            raise ValueError("stroke has already started")
        x = _finite_number(document_x, "document_x")
        y = _finite_number(document_y, "document_y")
        self._last = (x, y)
        return self._stamp(x, y)

    def add_point(self, document_x: float, document_y: float) -> DirtyRect | None:
        if self._last is None:
            raise ValueError("stroke must be started first")
        x = _finite_number(document_x, "document_x")
        y = _finite_number(document_y, "document_y")
        x0, y0 = self._last
        dx, dy = x - x0, y - y0
        length = math.hypot(dx, dy)
        event_dirty: DirtyRect | None = None
        travelled = 0.0
        while length - travelled + 1e-12 >= self._distance_to_next:
            travelled += self._distance_to_next
            fraction = travelled / length
            dirty = self._stamp(x0 + dx * fraction, y0 + dy * fraction)
            if dirty is not None:
                event_dirty = dirty if event_dirty is None else event_dirty.union(dirty)
            self._distance_to_next = self.settings.spacing
        self._distance_to_next -= length - travelled
        self._last = (x, y)
        return event_dirty

    # Common event-oriented spelling.
    move_to = add_point
