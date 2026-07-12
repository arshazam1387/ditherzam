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
        _validate_mask(self.mask, source.shape[:2])
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


def _round_divide(numerator: np.ndarray, denominator: np.ndarray | int) -> np.ndarray:
    """Nearest integer division with exact half ties rounded upward."""
    return (numerator + denominator // 2) // denominator


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

    # uint32 safely holds the maximum 255*255*255 contributions summed here.
    cov = np.floor(coverage.astype(np.float64) * 255.0 + 0.5).astype(np.uint32)
    inv = np.uint32(255) - cov
    rendered_u32 = rendered.astype(np.uint32)

    if outside_mode is OutsideMode.ORIGINAL:
        outside_rgb = source[..., :3].astype(np.uint32)
        outside_alpha = source[..., 3].astype(np.uint32)
    elif outside_mode is OutsideMode.WHITE:
        outside_rgb = np.full(rendered.shape, 255, dtype=np.uint32)
        outside_alpha = np.full(rendered.shape[:2], 255, dtype=np.uint32)
    elif outside_mode is OutsideMode.BLACK:
        outside_rgb = np.zeros(rendered.shape, dtype=np.uint32)
        outside_alpha = np.full(rendered.shape[:2], 255, dtype=np.uint32)
    else:  # Transparent: hidden RGB deterministically follows the rendered branch.
        outside_rgb = rendered_u32
        outside_alpha = np.zeros(rendered.shape[:2], dtype=np.uint32)

    alpha_numerator = cov * np.uint32(255) + inv * outside_alpha
    alpha = _round_divide(alpha_numerator, 255).astype(np.uint8)
    premultiplied = (
        rendered_u32 * (cov * np.uint32(255))[..., None]
        + outside_rgb * (inv * outside_alpha)[..., None]
    )
    safe_denominator = np.where(alpha_numerator == 0, 1, alpha_numerator)
    straight = _round_divide(premultiplied, safe_denominator[..., None])
    # Both contributions have zero alpha here. The completed branch is the one
    # deterministic, useful hidden color for later compositing.
    straight[alpha_numerator == 0] = rendered_u32[alpha_numerator == 0]
    rgb = straight.astype(np.uint8)

    opaque = outside_mode in (OutsideMode.WHITE, OutsideMode.BLACK) or (
        outside_mode is OutsideMode.ORIGINAL and bool(np.all(source[..., 3] == 255))
    )
    if opaque:
        return rgb
    return np.concatenate((rgb, alpha[..., None]), axis=2)


def flatten_rgba_white(rgba: np.ndarray) -> np.ndarray:
    """Flatten canonical straight RGBA onto opaque white with byte-exact math."""
    source = _validate_source(rgba)
    rgb = source[..., :3].astype(np.uint32)
    alpha = source[..., 3].astype(np.uint32)
    numerator = rgb * alpha[..., None] + np.uint32(255) * (255 - alpha[..., None])
    return _round_divide(numerator, 255).astype(np.uint8)
