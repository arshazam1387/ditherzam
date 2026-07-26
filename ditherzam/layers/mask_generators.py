"""Qt-free source and geometric raster-mask generators.

All coordinates are normalized source-image coordinates.  This makes canvas
direct-manipulation and numeric controls two views of the same exact state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ditherzam.masking.contracts import SourceIdentity, source_identity


class MaskGeneratorError(ValueError):
    """A creative mask generator request is invalid."""


class LuminancePreset(Enum):
    SHADOWS = "shadows"
    MIDTONES = "midtones"
    HIGHLIGHTS = "highlights"


class GradientKind(Enum):
    LINEAR = "linear"
    RADIAL = "radial"


@dataclass(frozen=True)
class LuminanceRange:
    """Inclusive full-strength band with soft outside transitions."""

    lower: int
    upper: int
    softness: int = 0

    def __post_init__(self) -> None:
        values = (self.lower, self.upper, self.softness)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise MaskGeneratorError("luminance bounds must be integers")
        if not 0 <= self.lower <= self.upper <= 255:
            raise MaskGeneratorError("luminance bounds must satisfy 0 <= lower <= upper <= 255")
        if not 0 <= self.softness <= 255:
            raise MaskGeneratorError("luminance softness must be within 0..255")


LUMINANCE_PRESETS = {
    LuminancePreset.SHADOWS: LuminanceRange(0, 64, 64),
    LuminancePreset.MIDTONES: LuminanceRange(96, 159, 64),
    LuminancePreset.HIGHLIGHTS: LuminanceRange(191, 255, 64),
}


@dataclass(frozen=True)
class GradientSpec:
    kind: GradientKind
    start_x: float
    start_y: float
    end_x: float
    end_y: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GradientKind):
            raise MaskGeneratorError("gradient kind must be Linear or Radial")
        values = (self.start_x, self.start_y, self.end_x, self.end_y)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) or not 0.0 <= float(v) <= 1.0
               for v in values):
            raise MaskGeneratorError(
                "gradient coordinates must be finite numbers within 0..1")
        if self.start_x == self.end_x and self.start_y == self.end_y:
            raise MaskGeneratorError("gradient geometry must have non-zero length")


@dataclass(frozen=True)
class MaskGeneratorCandidate:
    """Only boundary accepted by the controller for generated masks."""

    pixels: np.ndarray
    label: str
    source_identity: object

    def __post_init__(self) -> None:
        pixels = self.pixels
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                or pixels.ndim != 2 or not pixels.size):
            raise MaskGeneratorError(
                "candidate pixels must be a non-empty 2-D uint8 array")
        if not isinstance(self.label, str) or not self.label.strip():
            raise MaskGeneratorError("candidate label cannot be empty")
        if not isinstance(self.source_identity, SourceIdentity):
            raise MaskGeneratorError("candidate source identity is unavailable")
        owned = np.array(pixels, dtype=np.uint8, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "pixels", owned)
        object.__setattr__(self, "label", self.label.strip())


def _source_rgba(rgba: object) -> np.ndarray:
    if (not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8
            or rgba.ndim != 3 or rgba.shape[2] != 4 or not rgba.size):
        raise MaskGeneratorError(
            "source must be a non-empty HxWx4 uint8 RGBA array")
    return rgba


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, np.float32(0.0), np.float32(1.0))
    return value * value * (np.float32(3.0) - np.float32(2.0) * value)


def source_luminance_u8(rgba: object) -> np.ndarray:
    """Return exact Rec.709 source-RGB luminance; rendered pixels are never used."""
    source = _source_rgba(rgba)
    rgb = source[..., :3].astype(np.float32)
    luminance = (rgb[..., 0] * np.float32(0.2126)
                 + rgb[..., 1] * np.float32(0.7152)
                 + rgb[..., 2] * np.float32(0.0722))
    return np.floor(luminance + np.float32(0.5)).astype(np.uint8)


def luminance_mask(
    rgba: object,
    bounds: LuminanceRange | LuminancePreset,
) -> MaskGeneratorCandidate:
    """Select a source-RGB luminance band, independent of source alpha."""
    source = _source_rgba(rgba)
    if isinstance(bounds, LuminancePreset):
        bounds = LUMINANCE_PRESETS[bounds]
        label = f"Luminance · {bounds_for_label(bounds)}"
    elif isinstance(bounds, LuminanceRange):
        label = f"Luminance · {bounds_for_label(bounds)}"
    else:
        raise MaskGeneratorError("luminance bounds are unavailable")
    y = source_luminance_u8(source).astype(np.float32)
    coverage = np.ones(y.shape, dtype=np.float32)
    if bounds.lower > 0:
        if bounds.softness:
            coverage *= _smoothstep01(
                (y - np.float32(bounds.lower - bounds.softness))
                / np.float32(bounds.softness))
        else:
            coverage *= y >= bounds.lower
    if bounds.upper < 255:
        if bounds.softness:
            coverage *= _smoothstep01(
                (np.float32(bounds.upper + bounds.softness) - y)
                / np.float32(bounds.softness))
        else:
            coverage *= y <= bounds.upper
    pixels = np.floor(coverage * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)
    return MaskGeneratorCandidate(pixels, label, source_identity(source))


def bounds_for_label(bounds: LuminanceRange) -> str:
    return f"{bounds.lower}–{bounds.upper}, soft {bounds.softness}"


def gradient_mask(
    rgba: object, spec: GradientSpec
) -> MaskGeneratorCandidate:
    source = _source_rgba(rgba)
    if not isinstance(spec, GradientSpec):
        raise MaskGeneratorError("gradient spec is unavailable")
    height, width = source.shape[:2]
    xs = (np.arange(width, dtype=np.float32)
          / np.float32(max(1, width - 1)))
    ys = (np.arange(height, dtype=np.float32)
          / np.float32(max(1, height - 1)))[:, None]
    sx, sy = np.float32(spec.start_x), np.float32(spec.start_y)
    ex, ey = np.float32(spec.end_x), np.float32(spec.end_y)
    dx, dy = ex - sx, ey - sy
    if spec.kind is GradientKind.LINEAR:
        value = ((xs - sx) * dx + (ys - sy) * dy) / (dx * dx + dy * dy)
    else:
        radius = np.sqrt(dx * dx + dy * dy)
        value = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2) / radius
    pixels = np.floor(
        np.clip(value, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)
    ).astype(np.uint8)
    return MaskGeneratorCandidate(
        pixels, "Linear Gradient" if spec.kind is GradientKind.LINEAR
        else "Radial Gradient", source_identity(source))
