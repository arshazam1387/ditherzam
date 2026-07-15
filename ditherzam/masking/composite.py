"""Qt-free straight-alpha outer compositor for completed Smart Mask renders.

All operations use a single byte-domain rule: non-negative integer division is
rounded to nearest with ties upward.  A float mask is first quantized the same
way to byte coverage (``floor(mask * 255 + 0.5)``).  RGB contributions are
combined in premultiplied form, then converted back to canonical straight RGB;
this keeps soft transparent edges free of dark/white matte contamination.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from ditherzam.masking.contracts import MaskContractError, validate_confidence_array, validate_rgba_u8
from ditherzam.masking.settings import OutsideMode


class MaskCompositeError(Exception):
    """Raised when compositor inputs violate the canonical array contract."""


def _validate_mask(mask: object, shape: tuple[int, int]) -> np.ndarray:
    try:
        value = validate_confidence_array(mask, name="mask")
    except MaskContractError as exc:
        raise MaskCompositeError(str(exc)) from exc
    if value.shape != shape:
        raise MaskCompositeError(f"mask shape {value.shape} does not match image shape {shape}")
    return value


def _validate_source(source_rgba: object) -> np.ndarray:
    try:
        return validate_rgba_u8(source_rgba)
    except MaskContractError as exc:
        raise MaskCompositeError(str(exc)) from exc


@dataclass(frozen=True)
class CompositeContext:
    """One frozen outer-composite snapshot for a render request.

    Array payloads are references to the request's immutable snapshots; no live
    editor state is reread and constructing a context does not copy a full image.
    """

    source_rgba: np.ndarray
    mask: np.ndarray
    outside_mode: OutsideMode

    def __post_init__(self) -> None:
        source = _validate_source(self.source_rgba)
        mask = _validate_mask(self.mask, source.shape[:2])
        if source.flags.writeable or mask.flags.writeable:
            raise MaskCompositeError(
                "CompositeContext arrays must be immutable (writeable=False) request snapshots"
            )
        if not isinstance(self.outside_mode, OutsideMode):
            raise MaskCompositeError(
                f"outside_mode must be an OutsideMode, got {self.outside_mode!r}"
            )


def _validate_rendered(rendered_rgb: object) -> np.ndarray:
    if not isinstance(rendered_rgb, np.ndarray):
        raise MaskCompositeError(
            f"rendered_rgb must be a numpy ndarray, got {type(rendered_rgb).__name__}"
        )
    if rendered_rgb.dtype != np.uint8:
        raise MaskCompositeError(f"rendered_rgb dtype must be uint8, got {rendered_rgb.dtype}")
    if rendered_rgb.ndim != 3 or rendered_rgb.shape[2] != 3:
        raise MaskCompositeError(
            f"rendered_rgb shape must be (H, W, 3), got {rendered_rgb.shape}"
        )
    if rendered_rgb.shape[0] == 0 or rendered_rgb.shape[1] == 0:
        raise MaskCompositeError("rendered_rgb must not be zero-size")
    return rendered_rgb


_MODE_ORIGINAL = 0
_MODE_TRANSPARENT = 1
_MODE_WHITE = 2
_MODE_BLACK = 3
_MODE_CODES = {
    OutsideMode.ORIGINAL: _MODE_ORIGINAL,
    OutsideMode.TRANSPARENT: _MODE_TRANSPARENT,
    OutsideMode.WHITE: _MODE_WHITE,
    OutsideMode.BLACK: _MODE_BLACK,
}


@njit(cache=True, parallel=True)
def _composite_u8(rendered, source, mask, mode, channels):
    """One allocation, exact byte-domain compositor (compiled when JIT is on)."""
    height, width = mask.shape
    out = np.empty((height, width, channels), dtype=np.uint8)
    for y in prange(height):
        for x in range(width):
            coverage = int(float(mask[y, x]) * 255.0 + 0.5)
            inverse = 255 - coverage
            if mode == _MODE_ORIGINAL:
                outside_alpha = int(source[y, x, 3])
            elif mode == _MODE_TRANSPARENT:
                outside_alpha = 0
            else:
                outside_alpha = 255
            alpha_numerator = coverage * 255 + inverse * outside_alpha
            alpha = (alpha_numerator + 127) // 255
            for channel in range(3):
                rendered_value = int(rendered[y, x, channel])
                if mode == _MODE_ORIGINAL:
                    outside_value = int(source[y, x, channel])
                elif mode == _MODE_WHITE:
                    outside_value = 255
                elif mode == _MODE_BLACK:
                    outside_value = 0
                else:
                    outside_value = rendered_value
                if alpha_numerator == 0:
                    value = rendered_value
                else:
                    numerator = (
                        rendered_value * coverage * 255
                        + outside_value * inverse * outside_alpha
                    )
                    value = (numerator + alpha_numerator // 2) // alpha_numerator
                out[y, x, channel] = value
            if channels == 4:
                out[y, x, 3] = alpha
    return out


@njit(cache=True, parallel=True)
def _flatten_white_u8(source):
    height, width = source.shape[:2]
    out = np.empty((height, width, 3), dtype=np.uint8)
    for y in prange(height):
        for x in range(width):
            alpha = int(source[y, x, 3])
            inverse = 255 - alpha
            for channel in range(3):
                numerator = int(source[y, x, channel]) * alpha + 255 * inverse
                out[y, x, channel] = (numerator + 127) // 255
    return out


def composite_masked(
    rendered_rgb: np.ndarray,
    source_rgba: np.ndarray,
    mask: np.ndarray,
    outside_mode: OutsideMode,
) -> np.ndarray:
    """Composite one completed opaque RGB render over the selected outside.

    Returns RGB when every result pixel is necessarily opaque (Original with an
    opaque source, White, or Black), otherwise canonical straight RGBA. Inputs
    are read-only from this function's perspective and are never mutated.
    """
    rendered = _validate_rendered(rendered_rgb)
    source = _validate_source(source_rgba)
    if rendered.shape[:2] != source.shape[:2]:
        raise MaskCompositeError(
            f"rendered image shape {rendered.shape[:2]} does not match source shape {source.shape[:2]}"
        )
    coverage = _validate_mask(mask, rendered.shape[:2])
    if not isinstance(outside_mode, OutsideMode):
        raise MaskCompositeError(f"outside_mode must be an OutsideMode, got {outside_mode!r}")

    opaque = outside_mode in (OutsideMode.WHITE, OutsideMode.BLACK) or (
        outside_mode is OutsideMode.ORIGINAL and bool(np.all(source[..., 3] == 255))
    )
    return _composite_u8(rendered, source, coverage, _MODE_CODES[outside_mode], 3 if opaque else 4)


def bake_outside_base(base_gray_f32: np.ndarray, mask: np.ndarray,
                      fill_value: float) -> np.ndarray:
    """Blend the outside of ``mask`` toward ``fill_value`` on a grayscale base.

    Runs BEFORE the render pipeline so dither/effects texture the fill.
    Feathered (fractional) coverage blends softly; inside pixels (coverage 1)
    are untouched. The input base is never mutated.
    """
    base = np.asarray(base_gray_f32, dtype=np.float32)
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise MaskCompositeError("mask must be a 2-D ndarray")
    if base.ndim != 2:
        raise MaskCompositeError(f"base must be 2-D grayscale, got shape {base.shape}")
    if mask.shape != base.shape:
        raise MaskCompositeError(
            f"mask shape {mask.shape} does not match base shape {base.shape}")
    coverage = np.asarray(mask, dtype=np.float32)
    fill = np.float32(fill_value)
    return base * coverage + fill * (np.float32(1.0) - coverage)


def flatten_rgba_white(rgba: np.ndarray) -> np.ndarray:
    """Flatten canonical straight RGBA onto opaque white with byte-exact math."""
    source = _validate_source(rgba)
    return _flatten_white_u8(source)
