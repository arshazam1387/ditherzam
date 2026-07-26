"""Deterministic, Qt-free patterns for raster-mask generation and dithering."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from ditherzam.masking.contracts import SourceIdentity, source_identity as identify_source

from .mask_generators import (
    MaskGeneratorCandidate,
    MaskGeneratorError,
    _source_rgba,
)


class MaskPatternKind(Enum):
    BAYER = "bayer"
    LINES = "lines"
    NOISE = "noise"


@dataclass(frozen=True)
class MaskPatternSpec:
    kind: MaskPatternKind = MaskPatternKind.BAYER
    scale: int = 1
    orientation: int = 0
    offset_x: int = 0
    offset_y: int = 0
    mix: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        values = (self.scale, self.orientation, self.offset_x, self.offset_y,
                  self.mix, self.seed)
        if not isinstance(self.kind, MaskPatternKind):
            raise MaskGeneratorError("pattern kind must be Bayer, Lines, or Noise")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise MaskGeneratorError("pattern parameters must be integers")
        if not 1 <= self.scale <= 4096:
            raise MaskGeneratorError("pattern scale must be within 1..4096")
        orientations = {
            MaskPatternKind.BAYER: {0, 90, 180, 270},
            MaskPatternKind.LINES: {0, 45, 90, 135},
            MaskPatternKind.NOISE: {0},
        }[self.kind]
        if self.orientation not in orientations:
            raise MaskGeneratorError("pattern orientation is unavailable")
        if not -1_000_000 <= self.offset_x <= 1_000_000:
            raise MaskGeneratorError("pattern x offset is out of range")
        if not -1_000_000 <= self.offset_y <= 1_000_000:
            raise MaskGeneratorError("pattern y offset is out of range")
        if not 0 <= self.mix <= 100:
            raise MaskGeneratorError("pattern mix must be within 0..100")
        if not -(2 ** 31) <= self.seed < 2 ** 31:
            raise MaskGeneratorError("pattern seed must be a signed 32-bit integer")


_BAYER_4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.uint8,
)
_BAYER_THRESHOLDS = ((2 * _BAYER_4.astype(np.uint16) + 1) * 255 + 16) // 32


def _shape(value: object) -> tuple[int, int]:
    if (not isinstance(value, tuple) or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int)
                   or item <= 0 for item in value)):
        raise MaskGeneratorError("pattern shape must be positive (height, width)")
    return value


def _coordinates(shape: tuple[int, int], spec: MaskPatternSpec):
    height, width = shape
    x = np.arange(width, dtype=np.int64) + spec.offset_x
    y = np.arange(height, dtype=np.int64)[:, None] + spec.offset_y
    return x, y


def _oriented(x: np.ndarray, y: np.ndarray, orientation: int):
    if orientation == 0:
        return x, y
    if orientation == 90:
        return y, -x
    if orientation == 180:
        return -x, -y
    if orientation == 270:
        return -y, x
    if orientation == 135:
        return y - x, -(x + y)
    return x + y, y - x


def _bayer_thresholds(shape: tuple[int, int], spec: MaskPatternSpec) -> np.ndarray:
    x, y = _coordinates(shape, spec)
    u, v = _oriented(x, y, spec.orientation)
    columns = np.floor_divide(u, spec.scale) % 4
    rows = np.floor_divide(v, spec.scale) % 4
    return _BAYER_THRESHOLDS[rows, columns].astype(np.uint8)


def _line_thresholds(shape: tuple[int, int], spec: MaskPatternSpec) -> np.ndarray:
    x, y = _coordinates(shape, spec)
    radians = math.radians(spec.orientation)
    u = np.rint(x * math.cos(radians) + y * math.sin(radians)).astype(np.int64)
    period = max(2, spec.scale)
    phase = u % period
    return ((phase * 255 + (period - 1) // 2) // (period - 1)).astype(np.uint8)


def _noise_thresholds(shape: tuple[int, int], spec: MaskPatternSpec) -> np.ndarray:
    x, y = _coordinates(shape, spec)
    cell_x = np.floor_divide(x, spec.scale).astype(np.uint32)
    cell_y = np.floor_divide(y, spec.scale).astype(np.uint32)
    value = (
        cell_x * np.uint32(0x9E3779B1)
        ^ cell_y * np.uint32(0x85EBCA77)
        ^ np.uint32(spec.seed & 0xFFFFFFFF)
    )
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7FEB352D)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846CA68B)
    value ^= value >> np.uint32(16)
    return (value >> np.uint32(24)).astype(np.uint8)


def pattern_thresholds(
    shape: tuple[int, int], spec: MaskPatternSpec = MaskPatternSpec()
) -> np.ndarray:
    """Return an owned uint8 threshold field for an approved pattern."""
    checked_shape = _shape(shape)
    if not isinstance(spec, MaskPatternSpec):
        raise MaskGeneratorError("pattern spec is unavailable")
    if spec.kind is MaskPatternKind.BAYER:
        return _bayer_thresholds(checked_shape, spec)
    if spec.kind is MaskPatternKind.LINES:
        return _line_thresholds(checked_shape, spec)
    return _noise_thresholds(checked_shape, spec)


def pattern_mask(
    rgba: object, spec: MaskPatternSpec = MaskPatternSpec()
) -> MaskGeneratorCandidate:
    source = _source_rgba(rgba)
    if spec.mix != 100:
        raise MaskGeneratorError("mix applies only to Dither the Mask")
    pixels = pattern_thresholds(source.shape[:2], spec)
    label = {
        MaskPatternKind.BAYER: "Bayer Pattern",
        MaskPatternKind.LINES: "Lines Pattern",
        MaskPatternKind.NOISE: "Seeded Noise Pattern",
    }[spec.kind]
    return MaskGeneratorCandidate(pixels, label, identify_source(source))


def dither_mask(
    source: object,
    spec: MaskPatternSpec = MaskPatternSpec(),
    source_identity: SourceIdentity | None = None,
) -> MaskGeneratorCandidate:
    """Threshold a soft mask and mix the binary result back with integer math."""
    if (not isinstance(source, np.ndarray) or source.dtype != np.uint8
            or source.ndim != 2 or not source.size):
        raise MaskGeneratorError(
            "dither source must be a non-empty 2-D uint8 array")
    if not isinstance(spec, MaskPatternSpec):
        raise MaskGeneratorError("pattern spec is unavailable")
    if not isinstance(source_identity, SourceIdentity):
        raise MaskGeneratorError("dither source identity is unavailable")
    thresholds = pattern_thresholds(source.shape, spec)
    binary = np.where(source >= thresholds, 255, 0).astype(np.uint16)
    binary[source == 0] = 0
    binary[source == 255] = 255
    values = source.astype(np.uint16)
    mixed = (values * (100 - spec.mix) + binary * spec.mix + 50) // 100
    return MaskGeneratorCandidate(
        mixed.astype(np.uint8), f"Dither Mask · {spec.kind.value.title()}",
        source_identity)
