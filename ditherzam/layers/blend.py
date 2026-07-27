from __future__ import annotations

import importlib
import os

import numpy as np

from .model import BLEND_MODES


try:
    if os.environ.get("DITHERZAM_DISABLE_NATIVE") == "1":
        raise ImportError("native extensions disabled by environment")
    _composite = importlib.import_module("ditherzam._native._composite")
except ImportError:
    _composite = None

_MODE_CODES = {
    "normal": 0,
    "multiply": 1,
    "screen": 2,
    "overlay": 3,
    "difference": 4,
}


def _rgba(image) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
        or image.shape[2] not in (3, 4)
    ):
        raise ValueError("image must be a non-empty uint8 RGB or RGBA array")
    if image.shape[2] == 4:
        return image
    result = np.empty((*image.shape[:2], 4), dtype=np.uint8)
    result[..., :3] = image
    result[..., 3] = 255
    return result


def _validated_inputs(backdrop, source, mode: str, opacity: int):
    backdrop = _rgba(backdrop)
    source = _rgba(source)
    if backdrop.shape[:2] != source.shape[:2]:
        raise ValueError("images must have equal height and width")
    if mode not in BLEND_MODES:
        raise ValueError("mode is invalid")
    if isinstance(opacity, bool) or not isinstance(opacity, int):
        raise ValueError("opacity must be an integer")
    if not 0 <= opacity <= 100:
        raise ValueError("opacity must be within 0..100")
    return backdrop, source


def _blend_layer_reference(
    backdrop, source, mode: str = "normal", opacity: int = 100
) -> np.ndarray:
    """Independently callable NumPy reference for exactness verification."""
    backdrop, source = _validated_inputs(backdrop, source, mode, opacity)
    cb = backdrop[..., :3].astype(np.float64) / 255.0
    cs = source[..., :3].astype(np.float64) / 255.0
    ab = backdrop[..., 3:4].astype(np.float64) / 255.0
    # Opacity becomes effective source alpha in the uint8 image domain.  This
    # mirrors the rest of the renderer's deterministic half-up alpha policy
    # before the normalized W3C calculation.
    source_alpha = source[..., 3:4].astype(np.float64)
    source_alpha = np.floor(source_alpha * (opacity / 100.0) + 0.5)
    ass = source_alpha / 255.0

    if mode == "normal":
        blended = cs
    elif mode == "multiply":
        blended = cb * cs
    elif mode == "screen":
        blended = 1.0 - (1.0 - cb) * (1.0 - cs)
    elif mode == "overlay":
        blended = np.where(
            cb <= 0.5,
            2.0 * cb * cs,
            1.0 - 2.0 * (1.0 - cb) * (1.0 - cs),
        )
    else:
        blended = np.abs(cb - cs)

    ao = ass + ab * (1.0 - ass)
    premul = (
        (1.0 - ass) * ab * cb
        + (1.0 - ab) * ass * cs
        + ab * ass * blended
    )
    rgb = np.empty_like(premul)
    np.divide(premul, ao, out=rgb, where=ao > 0.0)
    np.copyto(rgb, cs, where=np.broadcast_to(ao == 0.0, rgb.shape))
    straight = np.concatenate((rgb, ao), axis=2) * 255.0
    return np.clip(np.floor(straight + 0.5), 0.0, 255.0).astype(np.uint8)


def _blend_layer_native(
    backdrop, source, mode: str = "normal", opacity: int = 100
) -> np.ndarray:
    """Independently callable native seam; raises instead of falling back."""
    backdrop, source = _validated_inputs(backdrop, source, mode, opacity)
    if _composite is None:
        raise RuntimeError("native compositor extension is unavailable")
    return _composite.blend_layer_u8(
        np.ascontiguousarray(backdrop),
        np.ascontiguousarray(source),
        _MODE_CODES[mode],
        opacity,
    )


def blend_layer(backdrop, source, mode: str = "normal", opacity: int = 100):
    """Composite source over backdrop using the W3C separable blend formula."""
    if _composite is None:
        return _blend_layer_reference(backdrop, source, mode, opacity)
    return _blend_layer_native(backdrop, source, mode, opacity)
