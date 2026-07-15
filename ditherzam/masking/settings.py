"""Frozen Smart Mask settings shared across UI, core, workers, and presets.

Every default and range below is recorded verbatim from the approved design
spec's state table and must not change without a spec revision.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MaskSettingsError(Exception):
    """Raised when a Smart Mask setting value or type is invalid."""


class MaskTarget(Enum):
    """What the derived mask selects."""

    SUBJECT = "subject"
    BACKGROUND = "background"
    WHOLE_IMAGE = "whole_image"


class OutsideMode(Enum):
    """What appears outside the selected mask target after compositing."""

    ORIGINAL = "original"
    TRANSPARENT = "transparent"
    WHITE = "white"
    BLACK = "black"


# Sensitivity is a 0-100 slider mapped monotonically to the adapter's
# documented foreground-confidence calibration.
SENSITIVITY_MIN = 0
SENSITIVITY_MAX = 100

# Edge feather in source pixels; zero is hard-edged. No documented upper
# bound in the design spec -- only non-negative is enforced here.
FEATHER_MIN_PX = 0

# Expand/contract in source pixels; signed, provisionally +/-64 per spec.
EXPANSION_MIN_PX = -64
EXPANSION_MAX_PX = 64

DEFAULT_ENABLED = False
DEFAULT_TARGET = MaskTarget.SUBJECT
DEFAULT_SENSITIVITY = 50
DEFAULT_FEATHER_PX = 8
DEFAULT_EXPANSION_PX = 0
DEFAULT_INVERT = False
DEFAULT_OUTSIDE = OutsideMode.ORIGINAL
DEFAULT_BAKE_FILL = False


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise MaskSettingsError(f"{name} must be a bool, got {type(value).__name__}")


def _validate_int_range(value: object, name: str, minimum: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MaskSettingsError(f"{name} must be an int, got {type(value).__name__}")
    if not (minimum <= value <= maximum):
        raise MaskSettingsError(f"{name} must be within [{minimum}, {maximum}], got {value}")


def _validate_int_min(value: object, name: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MaskSettingsError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise MaskSettingsError(f"{name} must be >= {minimum}, got {value}")


@dataclass(frozen=True)
class SmartMaskSettings:
    """Reusable, preset-storable Smart Mask settings.

    Excludes anything derived or session-scoped (probability maps,
    master-mask pixels, source identity, candidate ID, overlay/progress
    state) -- those live in ``contracts.py`` and worker state, not here.
    """

    enabled: bool = DEFAULT_ENABLED
    target: MaskTarget = DEFAULT_TARGET
    sensitivity: int = DEFAULT_SENSITIVITY
    feather_px: int = DEFAULT_FEATHER_PX
    expansion_px: int = DEFAULT_EXPANSION_PX
    invert: bool = DEFAULT_INVERT
    outside: OutsideMode = DEFAULT_OUTSIDE
    # Bake the White/Black outside fill into the pipeline input so dither and
    # effects render across it; ignored for Original/Transparent outsides.
    bake_fill: bool = DEFAULT_BAKE_FILL

    def __post_init__(self) -> None:
        _validate_bool(self.enabled, "enabled")
        if not isinstance(self.target, MaskTarget):
            raise MaskSettingsError(f"target must be a MaskTarget, got {self.target!r}")
        _validate_int_range(self.sensitivity, "sensitivity", SENSITIVITY_MIN, SENSITIVITY_MAX)
        _validate_int_min(self.feather_px, "feather_px", FEATHER_MIN_PX)
        _validate_int_range(self.expansion_px, "expansion_px", EXPANSION_MIN_PX, EXPANSION_MAX_PX)
        _validate_bool(self.invert, "invert")
        if not isinstance(self.outside, OutsideMode):
            raise MaskSettingsError(f"outside must be an OutsideMode, got {self.outside!r}")
        _validate_bool(self.bake_fill, "bake_fill")
